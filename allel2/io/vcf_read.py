# -*- coding: utf-8 -*-
"""
Extract data from VCF files.

This module contains Functions for extracting data from Variant Call Format (VCF) files
and loading into NumPy arrays, NumPy files, HDF5 files or Zarr array stores.

"""
import gzip
import logging
from numbers import Integral, Real
import os
import re
from collections import namedtuple, defaultdict
import warnings
import time
import subprocess
import tempfile
import textwrap
from collections import OrderedDict


import numpy as np


from allel2.opt.io_vcf_read import VCFChunkIterator, FileInputStream
from allel2.opt.io_vcf_read_rs import VCFChunkIterator as RustVCFChunkIterator
# expose some names from cython extension
# noinspection PyUnresolvedReferences
from allel2.opt.io_vcf_read import (  # noqa: F401
    ANNTransformer, ANN_AA_LENGTH_FIELD, ANN_AA_POS_FIELD, ANN_ANNOTATION_FIELD,
    ANN_ANNOTATION_IMPACT_FIELD, ANN_CDNA_LENGTH_FIELD, ANN_CDNA_POS_FIELD,
    ANN_CDS_LENGTH_FIELD, ANN_CDS_POS_FIELD, ANN_DISTANCE_FIELD, ANN_FEATURE_ID_FIELD,
    ANN_FEATURE_TYPE_FIELD, ANN_FIELD, ANN_FIELDS, ANN_GENE_ID_FIELD,
    ANN_GENE_NAME_FIELD, ANN_HGVS_C_FIELD, ANN_HGVS_P_FIELD, ANN_RANK_FIELD,
    ANN_TRANSCRIPT_BIOTYPE_FIELD
)


DEFAULT_BUFFER_SIZE = 2**14
DEFAULT_CHUNK_LENGTH = 2**16
DEFAULT_CHUNK_WIDTH = 2**6
DEFAULT_ALT_NUMBER = 3


# names for computed fields
FIELD_NUMALT = 'numalt'
FIELD_ALTLEN = 'altlen'
FIELD_IS_SNP = 'is_snp'
COMPUTED_FIELDS = [FIELD_NUMALT, FIELD_ALTLEN, FIELD_IS_SNP]


logger = logging.getLogger(__name__)

# environment variable choosing the VCF parser, while the Cython one exists
BACKEND_VARIABLE = 'ALLEL2_BACKEND'
BACKENDS = ('rust', 'cython')


def _prep_fields_param(fields):
    """Prepare the `fields` parameter, and determine whether or not to store samples."""

    store_samples = False

    if fields is None:
        # add samples by default
        return True, None

    if isinstance(fields, str):
        fields = [fields]
    else:
        fields = list(fields)

    if 'samples' in fields:
        fields.remove('samples')
        store_samples = True
    elif '*' in fields:
        store_samples = True

    return store_samples, fields


def _chunk_iter_progress(it, log, prefix):
    """Wrap a chunk iterator for progress logging."""
    n_variants = 0
    before_all = time.time()
    before_chunk = before_all
    for chunk, chunk_length, chrom, pos in it:
        after_chunk = time.time()
        elapsed_chunk = after_chunk - before_chunk
        elapsed = after_chunk - before_all
        n_variants += chunk_length
        chrom = str(chrom, 'utf8')
        message = (
            '%s %s rows in %.2fs; chunk in %.2fs (%s rows/s)' %
            (prefix, n_variants, elapsed, elapsed_chunk,
             int(chunk_length // elapsed_chunk))
        )
        if chrom:
            message += '; %s:%s' % (chrom, pos)
        print(message, file=log)
        log.flush()
        yield chunk, chunk_length, chrom, pos
        before_chunk = after_chunk
    after_all = time.time()
    elapsed = after_all - before_all
    print('%s all done (%s rows/s)' %
          (prefix, int(n_variants // elapsed)), file=log)
    log.flush()


def _chunk_iter_to_string_dtype(it, types):
    for chunk, chunk_length, chrom, pos in it:
        for f, t in types.items():
            chunk[f] = chunk[f].astype(t)
        yield chunk, chunk_length, chrom, pos


def _chunk_iter_transform(it, transformers):
    for chunk, chunk_length, chrom, pos in it:
        for transformer in transformers:
            transformer.transform_chunk(chunk)
        yield chunk, chunk_length, chrom, pos


def _do_rename(it, fields, rename_fields, headers):

    # check no duplicate values
    found = set()
    for v in rename_fields.values():
        if v.lower() in found:
            raise ValueError('rename clash: {!r}'.format(v))
        found.add(v.lower())

    # check no parent clashes
    for v in rename_fields.values():
        segments = v.split('/')
        for i in range(1, len(segments)):
            prefix = '/'.join(segments[:i]).lower()
            if prefix in found:
                raise ValueError(
                    'rename clash: {!r} versus {!r}'.format(v, prefix))

    # normalise keys
    rename_fields = {_normalize_field_prefix(k, headers): v
                     for k, v in rename_fields.items()}

    # check all keys match selected fields
    for k in rename_fields.keys():
        if k not in fields:
            raise ValueError('key {!r} in rename_fields does not match any selected ' \
                             'fields {!r}'.format(k, fields))

    # wrap iterator
    it = _chunk_iter_rename(it, rename_fields=rename_fields)

    return rename_fields, it


def _chunk_iter_rename(it, rename_fields):
    for chunk, chunk_length, chrom, pos in it:
        renamed_chunk = dict()
        for k, v in chunk.items():
            k = rename_fields.get(k, k)
            renamed_chunk[k] = v
        yield renamed_chunk, chunk_length, chrom, pos


_doc_param_input = \
    """Path to VCF file on the local file system. May be uncompressed or gzip-compatible
        compressed file. May also be a file-like object (e.g., `io.BytesIO`)."""

_doc_param_fields = \
    """Fields to extract data for. Should be a list of strings, e.g.,
        ``['variants/CHROM', 'variants/POS', 'variants/DP', 'calldata/GT']``. If you are feeling lazy,
        you can drop the 'variants/' and 'calldata/' prefixes, in which case the fields
        will be matched against fields declared in the VCF header, with variants taking
        priority over calldata if a field with the same ID exists both in INFO and
        FORMAT headers. I.e., ``['CHROM', 'POS', 'DP', 'GT']`` will work, although
        watch out for fields like 'DP' which can be both INFO and FORMAT. For
        convenience, some special string values are also recognized. To extract all
        fields, provide just the string ``'*'``. To extract all variants fields
        (including all INFO fields) provide ``'variants/*'``. To extract all calldata
        fields (i.e., defined in FORMAT headers) provide ``'calldata/*'``."""

_doc_param_exclude_fields = \
    """Fields to exclude. E.g., for use in combination with ``fields='*'``."""

_doc_param_rename_fields = \
    """Fields to be renamed. Should be a dictionary mapping old to new names,
    giving the complete path, e.g., ``{'variants/FOO': 'variants/bar'}``."""

_doc_param_types = \
    """Overide data types. Should be a dictionary mapping field names to NumPy data types.
        E.g., providing the dictionary ``{'variants/DP': 'i8', 'calldata/GQ': 'i2'}`` will
        mean the 'variants/DP' field is stored in a 64-bit integer array, and the
        'calldata/GQ' field is stored in a 16-bit integer array."""

_doc_param_numbers = \
    """Override the expected number of values. Should be a dictionary mapping field names
        to integers. E.g., providing the dictionary
        ``{'variants/ALT': 5, 'variants/AC': 5, 'calldata/HQ': 2}`` will mean that, for each variant, 5 values
        are stored for the 'variants/ALT' field, 5 values are stored for the
        'variants/AC' field, and for each sample, 2 values are stored for the
        'calldata/HQ' field."""

_doc_param_alt_number = \
    """Assume this number of alternate alleles and set expected number of values
        accordingly for any field declared with number 'A' or 'R' in the VCF
        meta-information."""

_doc_param_fills = \
    """Override the fill value used for empty values. Should be a dictionary mapping
        field names to fill values."""

_doc_param_region = \
    """Genomic region to extract variants for. If provided, should be a tabix-style
        region string, which can be either just a chromosome name (e.g., '2L'),
        or a chromosome name followed by 1-based beginning and end coordinates (e.g.,
        '2L:100000-200000'). Note that only variants whose start position (POS) is
        within the requested range will be included. This is slightly different from
        the default tabix behaviour, where a variant (e.g., deletion) may be included
        if its position (POS) occurs before the requested region but its reference allele
        overlaps the region - such a variant will *not* be included in the data
        returned by this function."""

_doc_param_tabix = \
    """Name or path to tabix executable. Only required if `region` is given. Setting
        `tabix` to `None` will cause a fall-back to scanning through the VCF file from
        the beginning, which may be much slower than tabix but the only option if tabix
        is not available on your system and/or the VCF file has not been tabix-indexed."""

_doc_param_samples = \
    """Selection of samples to extract calldata for. If provided, should be a list of
        strings giving sample identifiers. May also be a list of integers giving
        indices of selected samples."""

_doc_param_transformers = \
    """Transformers for post-processing data. If provided, should be a list of Transformer
        objects, each of which must implement a "transform()" method that accepts a dict
        containing the chunk of data to be transformed. See also the
        :class:`ANNTransformer` class which implements post-processing of data from
        SNPEFF."""

_doc_param_buffer_size = \
    """Size in bytes of the I/O buffer used when reading data from the underlying file or
        tabix stream."""

_doc_param_chunk_length = \
    """Length (number of variants) of chunks in which data are processed."""

_doc_param_n_threads = \
    """Number of threads parsing the VCF file, and decompressing it when the
        format allows. By default, the CPU cores this process may run on, at most
        the value of the ``OMP_NUM_THREADS`` environment variable if it is set.
        Ignored when the Cython parser reads the file (see ``ALLEL2_BACKEND``)."""

_doc_param_log = \
    """A file-like object (e.g., `sys.stderr`) to print progress information."""


# noinspection PyShadowingBuiltins
def read_vcf(input,
             fields=None,
             exclude_fields=None,
             rename_fields=None,
             types=None,
             numbers=None,
             alt_number=DEFAULT_ALT_NUMBER,
             fills=None,
             region=None,
             tabix='tabix',
             samples=None,
             transformers=None,
             buffer_size=DEFAULT_BUFFER_SIZE,
             chunk_length=DEFAULT_CHUNK_LENGTH,
             n_threads=None,
             log=None):
    """Read data from a VCF file into NumPy arrays.

    .. versionchanged:: 1.12.0
        Now returns None if no variants are found in the VCF file or matching the
        requested region.

    Parameters
    ----------
    input : string or file-like
        {input}
    fields : list of strings, optional
        {fields}
    exclude_fields : list of strings, optional
        {exclude_fields}
    rename_fields : dict[str -> str], optional
        {rename_fields}
    types : dict, optional
        {types}
    numbers : dict, optional
        {numbers}
    alt_number : int, optional
        {alt_number}
    fills : dict, optional
        {fills}
    region : string, optional
        {region}
    tabix : string, optional
        {tabix}
    samples : list of strings
        {samples}
    transformers : list of transformer objects, optional
        {transformers}
    buffer_size : int, optional
        {buffer_size}
    chunk_length : int, optional
        {chunk_length}
    n_threads : int, optional
        {n_threads}
    log : file-like, optional
        {log}

    Returns
    -------
    data : dict[str, ndarray]
        A dictionary holding arrays, or None if no variants were found.

    """

    # samples requested?
    # noinspection PyTypeChecker
    store_samples, fields = _prep_fields_param(fields)

    # setup
    fields, samples, headers, it = iter_vcf_chunks(
        input=input, fields=fields, exclude_fields=exclude_fields, types=types,
        numbers=numbers, alt_number=alt_number, buffer_size=buffer_size,
        chunk_length=chunk_length, n_threads=n_threads, fills=fills, region=region,
        tabix=tabix,
        samples=samples, transformers=transformers
    )

    # handle field renaming
    if rename_fields:
        rename_fields, it = _do_rename(it, fields=fields,
                                       rename_fields=rename_fields,
                                       headers=headers)

    # setup progress logging
    if log is not None:
        it = _chunk_iter_progress(it, log, prefix='[read_vcf]')

    # read all chunks into a list
    chunks = [d[0] for d in it]

    if chunks:

        # setup output
        output = dict()

        if len(samples) > 0 and store_samples:
            output['samples'] = samples

        # find array keys
        keys = sorted(chunks[0].keys())

        # concatenate chunks
        for k in keys:
            output[k] = np.concatenate([chunk[k] for chunk in chunks], axis=0)

    else:

        output = None

    return output


read_vcf.__doc__ = read_vcf.__doc__.format(
    input=_doc_param_input,
    fields=_doc_param_fields,
    exclude_fields=_doc_param_exclude_fields,
    rename_fields=_doc_param_rename_fields,
    types=_doc_param_types,
    numbers=_doc_param_numbers,
    alt_number=_doc_param_alt_number,
    fills=_doc_param_fills,
    region=_doc_param_region,
    tabix=_doc_param_tabix,
    samples=_doc_param_samples,
    transformers=_doc_param_transformers,
    buffer_size=_doc_param_buffer_size,
    chunk_length=_doc_param_chunk_length,
    n_threads=_doc_param_n_threads,
    log=_doc_param_log,
)


_doc_param_output = \
    """File-system path to write output to."""

_doc_param_overwrite = \
    """If False (default), do not overwrite an existing file."""


# noinspection PyShadowingBuiltins
def vcf_to_npz(input, output,
               compressed=True,
               overwrite=False,
               fields=None,
               exclude_fields=None,
               rename_fields=None,
               types=None,
               numbers=None,
               alt_number=DEFAULT_ALT_NUMBER,
               fills=None,
               region=None,
               tabix=True,
               samples=None,
               transformers=None,
               buffer_size=DEFAULT_BUFFER_SIZE,
               chunk_length=DEFAULT_CHUNK_LENGTH,
               n_threads=None,
               log=None):
    """Read data from a VCF file into NumPy arrays and save as a .npz file.

    .. versionchanged:: 1.12.0
        Now will not create any output file if no variants are found in the VCF file or
        matching the requested region.

    Parameters
    ----------
    input : string
        {input}
    output : string
        {output}
    compressed : bool, optional
        If True (default), save with compression.
    overwrite : bool, optional
        {overwrite}
    fields : list of strings, optional
        {fields}
    exclude_fields : list of strings, optional
        {exclude_fields}
    rename_fields : dict[str -> str], optional
        {rename_fields}
    types : dict, optional
        {types}
    numbers : dict, optional
        {numbers}
    alt_number : int, optional
        {alt_number}
    fills : dict, optional
        {fills}
    region : string, optional
        {region}
    tabix : string, optional
        {tabix}
    samples : list of strings
        {samples}
    transformers : list of transformer objects, optional
        {transformers}
    buffer_size : int, optional
        {buffer_size}
    chunk_length : int, optional
        {chunk_length}
    n_threads : int, optional
        {n_threads}
    log : file-like, optional
        {log}

    """

    # guard condition
    if not overwrite and os.path.exists(output):
        raise ValueError(
            'file exists at path %r; use overwrite=True to replace' % output)

    # read all data into memory
    data = read_vcf(
        input=input, fields=fields, exclude_fields=exclude_fields,
        rename_fields=rename_fields, types=types, numbers=numbers,
        alt_number=alt_number, buffer_size=buffer_size, chunk_length=chunk_length,
        n_threads=n_threads,
        log=log, fills=fills, region=region, tabix=tabix, samples=samples,
        transformers=transformers
    )

    if data is None:
        # no data, bail out
        return

    # setup save function
    if compressed:
        savez = np.savez_compressed
    else:
        savez = np.savez

    # save as npz; StringDType would be pickled as a custom dtype, so store
    # strings as object arrays, as scikit-allel did
    data = {k: a.astype(object) if a.dtype.kind == 'T' else a
            for k, a in data.items()}
    savez(output, **data)


vcf_to_npz.__doc__ = vcf_to_npz.__doc__.format(
    input=_doc_param_input,
    output=_doc_param_output,
    overwrite=_doc_param_overwrite,
    fields=_doc_param_fields,
    exclude_fields=_doc_param_exclude_fields,
    rename_fields=_doc_param_rename_fields,
    types=_doc_param_types,
    numbers=_doc_param_numbers,
    alt_number=_doc_param_alt_number,
    fills=_doc_param_fills,
    region=_doc_param_region,
    tabix=_doc_param_tabix,
    samples=_doc_param_samples,
    transformers=_doc_param_transformers,
    buffer_size=_doc_param_buffer_size,
    chunk_length=_doc_param_chunk_length,
    n_threads=_doc_param_n_threads,
    log=_doc_param_log,
)


def _h5like_copy_metadata(k, headers, ds):
    # copy metadata from VCF headers
    meta = None
    if k.startswith('variants/'):
        _, name = k.split('/', 1)
        if name in headers.infos:
            meta = headers.infos[name]
    elif k.startswith('calldata/'):
        _, name = k.split('/', 1)
        if name in headers.formats:
            meta = headers.formats[name]
    if meta is not None:
        if hasattr(ds.attrs, 'put'):
            # optimisation for zarr, put all attributes in one operation
            ds.attrs.put(meta)
        else:
            if 'ID' in meta:
                ds.attrs['ID'] = meta['ID']
            if 'Number' in meta:
                ds.attrs['Number'] = meta['Number']
            if 'Type' in meta:
                ds.attrs['Type'] = meta['Type']
            if 'Description' in meta:
                ds.attrs['Description'] = meta['Description']


def _strings_to_bytes(a):
    """Convert an object or StringDType array of strings to fixed-width bytes."""
    # StringDType cannot be cast directly to a fixed-width dtype of unspecified size
    return a.astype(object).astype('S')


def _hdf5_setup_datasets(chunk, root, chunk_length, chunk_width, compression,
                         compression_opts, shuffle, overwrite, headers, vlen):
    import h5py

    # handle no input
    if chunk is None:
        raise RuntimeError('input file has no data?')

    # obtain dataset keys
    keys = sorted(chunk.keys())

    # deal with overwriting existing data
    _h5like_handle_overwrite(root, keys, overwrite)

    # setup datasets
    for k in keys:

        # obtain initial data
        data = chunk[k]

        # determine chunk shape
        if data.ndim == 1:
            chunk_shape = (chunk_length,)
        else:
            chunk_shape = (chunk_length, min(
                chunk_width, data.shape[1])) + data.shape[2:]

        # create dataset
        shape = (0,) + data.shape[1:]
        maxshape = (None,) + data.shape[1:]
        if data.dtype.kind in 'OT':
            if vlen:
                dt = h5py.special_dtype(vlen=str)
            else:
                data = _strings_to_bytes(data)
                dt = data.dtype
        else:
            dt = data.dtype
        ds = root.create_dataset(
            k, shape=shape, maxshape=maxshape, chunks=chunk_shape, dtype=dt,
            compression=compression, compression_opts=compression_opts, shuffle=shuffle
        )

        # copy metadata from VCF headers
        _h5like_copy_metadata(k, headers, ds)

    return keys


def _hdf5_store_chunk(root, keys, chunk, vlen):

    # compute length of current chunk
    current_chunk_length = chunk[keys[0]].shape[0]

    # find current length of datasets
    old_length = root[keys[0]].shape[0]

    # new length of all arrays after loading this chunk
    new_length = old_length + current_chunk_length

    # load arrays
    for k in keys:

        # data to be loaded
        data = chunk[k]

        # obtain dataset
        dataset = root[k]

        # handle variable length strings
        if data.dtype.kind in 'OT' and not vlen:
            data = _strings_to_bytes(data)
            if data.dtype.itemsize > dataset.dtype.itemsize:
                warnings.warn(
                    'found string length %s longer than %s guessed for field %r, values ' \
                    'will be truncated; recommend rerunning, setting type to at least ' \
                    '"S%s"' % (data.dtype.itemsize, dataset.dtype.itemsize, k,
                               data.dtype.itemsize)
                )

        # ensure dataset is long enough
        dataset.resize(new_length, axis=0)

        # store the data
        dataset[old_length:new_length, ...] = data


_doc_param_chunk_width = \
    """Width (number of samples) to use when storing chunks in output."""


# noinspection PyShadowingBuiltins
def vcf_to_hdf5(input, output,
                group='/',
                compression='gzip',
                compression_opts=1,
                shuffle=False,
                overwrite=False,
                vlen=True,
                fields=None,
                exclude_fields=None,
                rename_fields=None,
                types=None,
                numbers=None,
                alt_number=DEFAULT_ALT_NUMBER,
                fills=None,
                region=None,
                tabix='tabix',
                samples=None,
                transformers=None,
                buffer_size=DEFAULT_BUFFER_SIZE,
                chunk_length=DEFAULT_CHUNK_LENGTH,
                n_threads=None,
                chunk_width=DEFAULT_CHUNK_WIDTH,
                log=None):
    """Read data from a VCF file and load into an HDF5 file.

    .. versionchanged:: 1.12.0
        Now will not create any output file if no variants are found in the VCF file or
        matching the requested region.

    Parameters
    ----------
    input : string
        {input}
    output : string
        {output}
    group : string
        Group within destination HDF5 file to store data in.
    compression : string
        Compression algorithm, e.g., 'gzip' (default).
    compression_opts : int
        Compression level, e.g., 1 (default).
    shuffle : bool
        Use byte shuffling, which may improve compression (default is False).
    overwrite : bool
        {overwrite}
    vlen : bool
        If True, store variable length strings. Note that there is considerable storage
        overhead for variable length strings in HDF5, and leaving this option as True (
        default) may lead to large file sizes. If False, all strings will be stored in
        the HDF5 file as fixed length strings, even if they are specified as 'object'
        type. In this case, the string length for any field with 'object' type will be
        determined based on the maximum length of strings found in the first chunk,
        and this may cause values to be truncated if longer values are found in later
        chunks. To avoid truncation and large file sizes, manually set the type for all
        string fields to an explicit fixed length string type, e.g., 'S10' for a field
        where you know at most 10 characters are required.
    fields : list of strings, optional
        {fields}
    exclude_fields : list of strings, optional
        {exclude_fields}
    rename_fields : dict[str -> str], optional
        {rename_fields}
    types : dict, optional
        {types}
    numbers : dict, optional
        {numbers}
    alt_number : int, optional
        {alt_number}
    fills : dict, optional
        {fills}
    region : string, optional
        {region}
    tabix : string, optional
        {tabix}
    samples : list of strings
        {samples}
    transformers : list of transformer objects, optional
        {transformers}
    buffer_size : int, optional
        {buffer_size}
    chunk_length : int, optional
        {chunk_length}
    n_threads : int, optional
        {n_threads}
    chunk_width : int, optional
        {chunk_width}
    log : file-like, optional
        {log}

    """

    import h5py

    # samples requested?
    # noinspection PyTypeChecker
    store_samples, fields = _prep_fields_param(fields)

    # setup chunk iterator
    fields, samples, headers, it = iter_vcf_chunks(
        input, fields=fields, exclude_fields=exclude_fields, types=types,
        numbers=numbers, alt_number=alt_number, buffer_size=buffer_size,
        chunk_length=chunk_length, n_threads=n_threads, fills=fills, region=region,
        tabix=tabix,
        samples=samples, transformers=transformers
    )

    # handle field renaming
    if rename_fields:
        rename_fields, it = _do_rename(it, fields=fields,
                                       rename_fields=rename_fields,
                                       headers=headers)

    # setup progress logging
    if log is not None:
        it = _chunk_iter_progress(it, log, prefix='[vcf_to_hdf5]')

    # read first chunk
    try:
        chunk, _, _, _ = next(it)
    except StopIteration:
        # no data, bail out
        return

    with h5py.File(output, mode='a') as h5f:

        # obtain root group that data will be stored into
        root = h5f.require_group(group)

        if len(samples) > 0 and store_samples:
            # store samples
            name = 'samples'
            if name in root:
                if overwrite:
                    del root[name]
                else:
                    raise ValueError(
                        'dataset exists at path %r; use overwrite=True to replace' % name)
            if samples.dtype.kind in 'OT':
                if vlen:
                    t = h5py.special_dtype(vlen=str)
                else:
                    samples = _strings_to_bytes(samples)
                    t = samples.dtype
            else:
                t = samples.dtype
            root.create_dataset(name, data=samples, chunks=None, dtype=t)

        # setup datasets
        # noinspection PyTypeChecker
        keys = _hdf5_setup_datasets(
            chunk=chunk, root=root, chunk_length=chunk_length, chunk_width=chunk_width,
            compression=compression, compression_opts=compression_opts, shuffle=shuffle,
            overwrite=overwrite, headers=headers, vlen=vlen
        )

        # store first chunk
        _hdf5_store_chunk(root, keys, chunk, vlen)

        # store remaining chunks
        for chunk, _, _, _ in it:

            _hdf5_store_chunk(root, keys, chunk, vlen)


vcf_to_hdf5.__doc__ = vcf_to_hdf5.__doc__.format(
    input=_doc_param_input,
    output=_doc_param_output,
    overwrite=_doc_param_overwrite,
    fields=_doc_param_fields,
    exclude_fields=_doc_param_exclude_fields,
    rename_fields=_doc_param_rename_fields,
    types=_doc_param_types,
    numbers=_doc_param_numbers,
    alt_number=_doc_param_alt_number,
    fills=_doc_param_fills,
    region=_doc_param_region,
    tabix=_doc_param_tabix,
    samples=_doc_param_samples,
    transformers=_doc_param_transformers,
    buffer_size=_doc_param_buffer_size,
    chunk_length=_doc_param_chunk_length,
    n_threads=_doc_param_n_threads,
    chunk_width=_doc_param_chunk_width,
    log=_doc_param_log,
)


def _h5like_handle_overwrite(root, keys, overwrite):
    # deal with overwriting existing data, do this up front
    for k in keys:
        if k in root:
            if overwrite:
                del root[k]
            else:
                raise ValueError('object exists at path %r; use overwrite=True to ' \
                                 'replace' % k)


_blosc_shuffle_names = {0: 'noshuffle', 1: 'shuffle', 2: 'bitshuffle'}


def _zarr_compressors(compressor, zarr_format):
    """Translate the `compressor` argument of :func:`vcf_to_zarr` into the
    `compressors` argument of :meth:`zarr.Group.create_array`."""
    import zarr.codecs
    from zarr.abc.codec import BytesBytesCodec
    from numcodecs import Blosc, Zstd

    if compressor is None:
        return None
    if zarr_format == 2:
        if isinstance(compressor, BytesBytesCodec):
            raise ValueError(
                'compressor %r cannot be used with zarr format 2; use a numcodecs '
                'codec' % (compressor,))
        # zarr format 2 stores numcodecs codecs as they are
        return compressor
    if isinstance(compressor, BytesBytesCodec):
        return compressor
    if isinstance(compressor, Blosc):
        return zarr.codecs.BloscCodec(
            cname=compressor.cname, clevel=compressor.clevel,
            # AUTOSHUFFLE (-1) lets zarr choose from the data type
            shuffle=_blosc_shuffle_names.get(compressor.shuffle),
            blocksize=compressor.blocksize)
    if isinstance(compressor, Zstd):
        return zarr.codecs.ZstdCodec(level=compressor.level, checksum=compressor.checksum)
    raise ValueError(
        'compressor %r cannot be used with zarr format %s; use a codec from '
        'zarr.codecs, numcodecs.Blosc or numcodecs.Zstd' % (compressor, zarr_format))


def _zarr_check_existing_format(output, zarr_format):
    """Refuse to write into an existing store of another zarr format, which zarr
    would otherwise turn into a store mixing both formats."""
    import zarr
    try:
        existing = zarr.open_group(output, mode='r').metadata.zarr_format
    except (FileNotFoundError, zarr.errors.GroupNotFoundError):
        return
    if existing != zarr_format:
        raise ValueError(
            'cannot write zarr format %s into %r, which has existing zarr format %s'
            % (zarr_format, output, existing))


def _zarr_setup_datasets(chunk, root, chunk_length, chunk_width, compressors, overwrite,
                         headers):

    # handle no input
    if chunk is None:
        raise RuntimeError('input file has no data?')

    # obtain dataset keys
    keys = sorted(chunk.keys())

    # deal with overwriting existing data
    _h5like_handle_overwrite(root, keys, overwrite)

    # create datasets
    for k in keys:

        # obtain initial data
        data = chunk[k]

        # determine chunk shape
        if data.ndim == 1:
            chunk_shape = (chunk_length,)
        else:
            chunk_shape = (chunk_length, min(
                chunk_width, data.shape[1])) + data.shape[2:]

        # create dataset
        shape = (0,) + data.shape[1:]
        if data.dtype.kind in 'OT':
            dtype = 'str'
        else:
            dtype = data.dtype
        ds = root.create_array(k, shape=shape, chunks=chunk_shape, dtype=dtype,
                               compressors=compressors, overwrite=overwrite)

        # copy metadata from VCF headers
        _h5like_copy_metadata(k, headers, ds)

    return keys


def _zarr_store_chunk(root, keys, chunk):

    # load arrays
    for k in keys:

        # append data
        root[k].append(chunk[k], axis=0)


# noinspection PyShadowingBuiltins
def vcf_to_zarr(input, output,
                group='/',
                compressor='default',
                overwrite=False,
                fields=None,
                exclude_fields=None,
                rename_fields=None,
                types=None,
                numbers=None,
                alt_number=DEFAULT_ALT_NUMBER,
                fills=None,
                region=None,
                tabix='tabix',
                samples=None,
                transformers=None,
                buffer_size=DEFAULT_BUFFER_SIZE,
                chunk_length=DEFAULT_CHUNK_LENGTH,
                n_threads=None,
                chunk_width=DEFAULT_CHUNK_WIDTH,
                log=None,
                zarr_format=3):
    """Read data from a VCF file and load into a Zarr on-disk store.

    .. versionchanged:: 1.12.0
        Now will not create any output files if no variants are found in the VCF file or
        matching the requested region.

    .. versionchanged:: 2.0.0
        Writes zarr format 3 by default (see `zarr_format`); the compressor is
        applied as requested, and strings are stored as variable-width UTF-8.

    Parameters
    ----------
    input : string
        {input}
    output : string
        {output}
    group : string
        Group within destination Zarr hierarchy to store data in.
    compressor : codec, dict or None, optional
        Compression codec. Defaults to Blosc with LZ4, level 5 and byte shuffle.
        Accepts a codec from :mod:`zarr.codecs`, a :class:`numcodecs.Blosc` or
        :class:`numcodecs.Zstd` codec (translated to their :mod:`zarr.codecs`
        equivalents), a numcodecs configuration such as `{{'id': 'blosc', 'cname':
        'zstd', 'clevel': 1, 'shuffle': 1}}`, or a dict of :class:`numcodecs.Blosc`
        arguments. Any other codec raises a ValueError. None disables compression.
    overwrite : bool
        {overwrite}
    fields : list of strings, optional
        {fields}
    exclude_fields : list of strings, optional
        {exclude_fields}
    rename_fields : dict[str -> str], optional
        {rename_fields}
    types : dict, optional
        {types}
    numbers : dict, optional
        {numbers}
    alt_number : int, optional
        {alt_number}
    fills : dict, optional
        {fills}
    region : string, optional
        {region}
    tabix : string, optional
        {tabix}
    samples : list of strings
        {samples}
    transformers : list of transformer objects, optional
        {transformers}
    buffer_size : int, optional
        {buffer_size}
    chunk_length : int, optional
        {chunk_length}
    n_threads : int, optional
        {n_threads}
    chunk_width : int, optional
        {chunk_width}
    log : file-like, optional
        {log}
    zarr_format : {{2, 3}}, optional
        Zarr format to write, 3 by default. Files in zarr format 3 cannot be read
        with zarr-python 2; use `zarr_format=2` for tools that still depend on it.
        In zarr format 3, strings requested as fixed-width bytes (e.g., `types=
        {{'CHROM': 'S10'}}`) are stored with the `null_terminated_bytes` data type,
        which has no zarr specification yet; the default variable-width strings
        use the specified `string` data type.

    """

    import zarr
    import numcodecs
    from numcodecs import Blosc

    if type(zarr_format) is not int or zarr_format not in (2, 3):
        raise ValueError('zarr_format must be 2 or 3, found %r' % (zarr_format,))

    # samples requested?
    # noinspection PyTypeChecker
    store_samples, fields = _prep_fields_param(fields)

    # setup chunk iterator
    fields, samples, headers, it = iter_vcf_chunks(
        input, fields=fields, exclude_fields=exclude_fields, types=types,
        numbers=numbers, alt_number=alt_number, buffer_size=buffer_size,
        chunk_length=chunk_length, n_threads=n_threads, fills=fills, region=region,
        tabix=tabix,
        samples=samples, transformers=transformers
    )
    # setup compressor
    if compressor == 'default':
        compressor = Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE)
    elif isinstance(compressor, dict):
        if 'id' in compressor:
            # numcodecs codec configuration
            compressor = numcodecs.get_codec(compressor)
        else:
            compressor = Blosc(**compressor)

    # handle field renaming
    if rename_fields:
        rename_fields, it = _do_rename(it, fields=fields,
                                       rename_fields=rename_fields,
                                       headers=headers)

    # check for any case-insensitive duplicate fields
    # https://github.com/cggh/scikit-allel/issues/215
    ci_field_index = defaultdict(list)
    for f in fields:
        if rename_fields:
            f = rename_fields.get(f, f)
        ci_field_index[f.lower()].append(f)
    for k, v in ci_field_index.items():
        if len(v) > 1:
            msg = textwrap.fill(
                'Found two or more fields with the same name when compared ' 
                'case-insensitive: {!r}; this is not supported because it causes ' 
                'problems on platforms with a case-insensitive file system, which is ' 
                'usually the default on Windows and Mac OS. Please rename fields so they ' 
                'are distinct under a case-insensitive comparison via the ' 
                'rename_fields argument.'.format(v), width=80)
            raise ValueError(msg)

    # setup progress logging
    if log is not None:
        it = _chunk_iter_progress(it, log, prefix='[vcf_to_zarr]')

    # read first chunk
    try:
        chunk, _, _, _ = next(it)
    except StopIteration:
        # no data, bail out
        return

    # check the compressor and the format of any existing store before writing anything
    compressors = _zarr_compressors(compressor, zarr_format)
    _zarr_check_existing_format(output, zarr_format)

    # open root group
    # ensure path exists
    if not os.path.exists(output):
        os.makedirs(output)
    root = zarr.open_group(output, mode='a', path=group, zarr_format=zarr_format)

    if samples.shape[0] > 0 and store_samples:
        # store samples, uncompressed
        if samples.dtype.kind in 'OT':
            dtype = 'str'
        else:
            dtype = samples.dtype
        ds = root.create_array('samples', shape=samples.shape, dtype=dtype,
                               compressors=None, overwrite=overwrite)
        ds[:] = samples

    # setup datasets
    # noinspection PyTypeChecker
    keys = _zarr_setup_datasets(
        chunk, root=root, chunk_length=chunk_length, chunk_width=chunk_width,
        compressors=compressors, overwrite=overwrite, headers=headers
    )

    # store first chunk
    _zarr_store_chunk(root, keys, chunk)

    # store remaining chunks
    for chunk, _, _, _ in it:

        _zarr_store_chunk(root, keys, chunk)

vcf_to_zarr.__doc__ = vcf_to_zarr.__doc__.format(
    input=_doc_param_input,
    output=_doc_param_output,
    overwrite=_doc_param_overwrite,
    fields=_doc_param_fields,
    exclude_fields=_doc_param_exclude_fields,
    rename_fields=_doc_param_rename_fields,
    types=_doc_param_types,
    numbers=_doc_param_numbers,
    alt_number=_doc_param_alt_number,
    fills=_doc_param_fills,
    region=_doc_param_region,
    tabix=_doc_param_tabix,
    samples=_doc_param_samples,
    transformers=_doc_param_transformers,
    buffer_size=_doc_param_buffer_size,
    chunk_length=_doc_param_chunk_length,
    n_threads=_doc_param_n_threads,
    chunk_width=_doc_param_chunk_width,
    log=_doc_param_log,
)


class _TabixStream:
    """Standard output of a running tabix process, read by FileInputStream.

    Messages printed by tabix are kept out of the VCF stream and relayed as
    warnings; closing the stream waits for the process.
    """

    def __init__(self, tabix, input, region):
        # a file rather than a pipe, so that tabix never blocks on its stderr
        self._err = tempfile.TemporaryFile()
        self._relayed = 0
        self._process = subprocess.Popen([tabix, '-h', input, region],
                                         stdout=subprocess.PIPE, stderr=self._err)
        # wait for the first output, or for tabix to exit without output
        if not self._process.stdout.peek(1) and self._process.wait() != 0:
            message = self._read_messages()
            self.close()
            raise RuntimeError(message)
        self._relay_messages()

    def _read_messages(self):
        self._err.seek(self._relayed)
        text = self._err.read()
        self._relayed += len(text)
        return str(text, 'utf-8', 'replace').strip()

    def _relay_messages(self):
        for message in self._read_messages().splitlines():
            if message.strip():
                warnings.warn('tabix: ' + message.strip())

    def readinto(self, b):
        return self._process.stdout.readinto(b)

    def close(self):
        if self._err.closed:
            return
        self._process.stdout.close()
        self._process.wait()
        self._relay_messages()
        self._err.close()


# noinspection PyShadowingBuiltins
def _setup_input_stream(input, region=None, tabix=None, buffer_size=DEFAULT_BUFFER_SIZE):

    # obtain a file-like object
    close = False
    if isinstance(input, str) and input.endswith('gz'):

        if region and tabix and os.name != 'nt':

            try:
                # try tabix
                fileobj = _TabixStream(tabix, input, region)
                close = True
                # N.B., still pass the region parameter through so we get strictly only
                # variants that start within the requested region. See also
                # https://github.com/alimanfoo/vcfnp/issues/54

            except FileNotFoundError:
                # no tabix, fall back to scanning
                warnings.warn(
                    'tabix not found, falling back to scanning to region')
                fileobj = gzip.open(input, mode='rb')
                close = True

            except Exception as e:
                warnings.warn('error occurred attempting tabix (%s); falling back to ' \
                              'scanning to region' % e)
                fileobj = gzip.open(input, mode='rb')
                close = True

        else:
            fileobj = gzip.open(input, mode='rb')
            close = True

    elif isinstance(input, str):
        # assume no compression
        fileobj = open(input, mode='rb', buffering=0)
        close = True

    elif hasattr(input, 'readinto'):
        fileobj = input

    else:
        raise ValueError('path must be string or file-like, found %r' % input)

    return FileInputStream(fileobj, buffer_size=buffer_size, close=close)


# noinspection PyShadowingBuiltins
def iter_vcf_chunks(input,
                    fields=None,
                    exclude_fields=None,
                    types=None,
                    numbers=None,
                    alt_number=DEFAULT_ALT_NUMBER,
                    fills=None,
                    region=None,
                    tabix='tabix',
                    samples=None,
                    transformers=None,
                    buffer_size=DEFAULT_BUFFER_SIZE,
                    chunk_length=DEFAULT_CHUNK_LENGTH,
                    n_threads=None):
    """Iterate over chunks of data from a VCF file as NumPy arrays.

    Parameters
    ----------
    input : string
        {input}
    fields : list of strings, optional
        {fields}
    exclude_fields : list of strings, optional
        {exclude_fields}
    types : dict, optional
        {types}
    numbers : dict, optional
        {numbers}
    alt_number : int, optional
        {alt_number}
    fills : dict, optional
        {fills}
    region : string, optional
        {region}
    tabix : string, optional
        {tabix}
    samples : list of strings
        {samples}
    transformers : list of transformer objects, optional
        {transformers}
    buffer_size : int, optional
        {buffer_size}
    chunk_length : int, optional
        {chunk_length}
    n_threads : int, optional
        {n_threads}

    Returns
    -------
    fields : list of strings
        Normalised names of fields that will be extracted.
    samples : ndarray
        Samples for which data will be extracted.
    headers : VCFHeaders
        Tuple of metadata extracted from VCF headers.
    it : iterator
        Chunk iterator.

    """

    n_threads = _normalize_n_threads(n_threads)

    # setup commmon keyword args
    kwds = dict(fields=fields, exclude_fields=exclude_fields, types=types,
                numbers=numbers, alt_number=alt_number, chunk_length=chunk_length,
                fills=fills, samples=samples, region=region, n_threads=n_threads)

    # choose the parser: Rust unless the input needs the Cython one
    reason = _cython_input_reason(input, region, transformers)
    if reason is not None:
        _log_fallback(input, reason)

    # setup input stream
    stream = _setup_input_stream(input=input, region=region, tabix=tabix,
                                 buffer_size=buffer_size)

    # setup iterator
    rust_path = input if reason is None else None
    fields, samples, headers, it = _iter_vcf_stream(stream, rust_path=rust_path, **kwds)

    # setup transformers
    if transformers is not None:
        # API flexibility
        if not isinstance(transformers, (list, tuple)):
            transformers = [transformers]
        for trans in transformers:
            fields = trans.transform_fields(fields)
        it = _chunk_iter_transform(it, transformers)

    return fields, samples, headers, it


iter_vcf_chunks.__doc__ = iter_vcf_chunks.__doc__.format(
    input=_doc_param_input,
    fields=_doc_param_fields,
    exclude_fields=_doc_param_exclude_fields,
    types=_doc_param_types,
    numbers=_doc_param_numbers,
    alt_number=_doc_param_alt_number,
    fills=_doc_param_fills,
    region=_doc_param_region,
    tabix=_doc_param_tabix,
    samples=_doc_param_samples,
    transformers=_doc_param_transformers,
    buffer_size=_doc_param_buffer_size,
    chunk_length=_doc_param_chunk_length,
    n_threads=_doc_param_n_threads,
    log=_doc_param_log,
)


FIXED_VARIANTS_FIELDS = (
    'CHROM',
    'POS',
    'ID',
    'REF',
    'ALT',
    'QUAL',
)


def _normalize_field_prefix(field, headers):

    # already contains prefix?
    if field.startswith('variants/') or field.startswith('calldata/'):
        return field

    # try to find in fixed fields
    elif field in FIXED_VARIANTS_FIELDS:
        return 'variants/' + field

    # try to find in FILTER
    elif field.startswith('FILTER_'):
        return 'variants/' + field

    # try to find in FILTER
    elif field in headers.filters:
        return 'variants/FILTER_' + field

    # try to find in INFO
    elif field in headers.infos:
        return 'variants/' + field

    # try to find in computed fields
    elif field in COMPUTED_FIELDS:
        return 'variants/' + field

    # try to find in FORMAT
    elif field in headers.formats:
        return 'calldata/' + field

    else:
        # assume anything else in variants, even if header not found
        return 'variants/' + field


def _check_field(field, headers):

    # assume field is already normalized for prefix
    group, name = field.split('/')

    if group == 'variants':

        if name in FIXED_VARIANTS_FIELDS:
            pass

        elif name in COMPUTED_FIELDS:
            # computed fields
            pass

        elif name.startswith('FILTER_'):
            filter_name = name[7:]
            if filter_name in headers.filters:
                pass
            else:
                warnings.warn('%r FILTER header not found' % filter_name)

        elif name in headers.infos:
            pass

        else:
            warnings.warn('%r INFO header not found' % name)

    elif group == 'calldata':

        if name in headers.formats:
            pass

        else:
            warnings.warn('%r FORMAT header not found' % name)

    else:
        # should never be reached
        raise ValueError('invalid field specification: %r' % field)


def _add_all_fields(fields, headers, samples):
    _add_all_variants_fields(fields, headers)
    if len(samples) > 0:
        _add_all_calldata_fields(fields, headers)


def _add_all_variants_fields(fields, headers):
    _add_all_fixed_variants_fields(fields)
    _add_all_info_fields(fields, headers)
    _add_all_filter_fields(fields, headers)
    _add_all_computed_fields(fields)


def _add_all_fixed_variants_fields(fields):
    for k in FIXED_VARIANTS_FIELDS:
        f = 'variants/' + k
        if f not in fields:
            fields.append(f)


def _add_all_info_fields(fields, headers):
    for k in headers.infos:
        f = 'variants/' + k
        if f not in fields:
            fields.append(f)


def _add_all_filter_fields(fields, headers):
    fields.append('variants/FILTER_PASS')
    for k in headers.filters:
        f = 'variants/FILTER_' + k
        if f not in fields:
            fields.append(f)


def _add_all_computed_fields(fields):
    for k in COMPUTED_FIELDS:
        f = 'variants/' + k
        if f not in fields:
            fields.append(f)


def _add_all_calldata_fields(fields, headers):
    # only add calldata fields if there are samples
    if headers.samples:
        for k in headers.formats:
            f = 'calldata/' + k
            if f not in fields:
                fields.append(f)


def _normalize_fields(fields, headers, samples):

    # setup normalized fields
    normed_fields = list()

    # special case, single field specification
    if isinstance(fields, str):
        fields = [fields]

    for f in fields:

        # special cases: be lenient about how to specify

        if f in ['*', 'kitchen sink']:
            _add_all_fields(normed_fields, headers, samples)

        elif f in ['variants', 'variants*', 'variants/*']:
            _add_all_variants_fields(normed_fields, headers)

        elif f in ['calldata', 'calldata*', 'calldata/*'] and len(samples) > 0:
            _add_all_calldata_fields(normed_fields, headers)

        elif f in ['INFO', 'INFO*', 'INFO/*', 'variants/INFO', 'variants/INFO*', \
                   'variants/INFO/*']:
            _add_all_info_fields(normed_fields, headers)

        elif f in ['FILTER', 'FILTER*', 'FILTER/*', 'FILTER_*', 'variants/FILTER', \
                   'variants/FILTER*', 'variants/FILTER/*', 'variants/FILTER_*']:
            _add_all_filter_fields(normed_fields, headers)

        # exact field specification

        else:

            # normalize field specification
            f = _normalize_field_prefix(f, headers)
            _check_field(f, headers)
            if f.startswith('calldata/') and len(samples) == 0:
                # only add calldata fields if there are samples
                pass
            elif f not in normed_fields:
                normed_fields.append(f)

    return normed_fields


default_integer_dtype = 'i4'
default_float_dtype = 'f4'
default_string_dtype = np.dtypes.StringDType()


def _is_string_dtype(t):
    """Whether `t` requests NumPy's variable-width ``StringDType``."""
    try:
        return np.dtype(t).kind == 'T'
    except TypeError:
        # custom types such as 'genotype/i1'
        return False


def _normalize_type(t):
    if t == 'Integer':
        return np.dtype(default_integer_dtype)
    elif t == 'Float':
        return np.dtype(default_float_dtype)
    elif t == 'String':
        return default_string_dtype
    elif t == 'Character':
        return default_string_dtype
    elif t == 'Flag':
        return np.dtype(bool)
    elif isinstance(t, str) and t.startswith('genotype/'):
        # custom genotype dtype
        return t
    elif isinstance(t, str) and t.startswith('genotype_ac/'):
        # custom genotype allele counts dtype
        return t
    else:
        return np.dtype(t)


default_types = {
    'variants/CHROM': default_string_dtype,
    'variants/POS': 'i4',
    'variants/ID': default_string_dtype,
    'variants/REF': default_string_dtype,
    'variants/ALT': default_string_dtype,
    'variants/QUAL': 'f4',
    'variants/DP': 'i4',
    'variants/AN': 'i4',
    'variants/AC': 'i4',
    'variants/AF': 'f4',
    'variants/MQ': 'f4',
    'variants/ANN': default_string_dtype,
    'calldata/GT': 'genotype/i1',
    'calldata/GQ': 'i1',
    'calldata/HQ': 'i1',
    'calldata/DP': 'i2',
    'calldata/AD': 'i2',
    'calldata/MQ0': 'i2',
    'calldata/MQ': 'f2',
}


def _normalize_types(types, fields, headers):

    # normalize user-provided types
    if types is None:
        types = dict()
    types = {_normalize_field_prefix(f, headers): _normalize_type(t)
             for f, t in types.items()}

    # setup output
    normed_types = dict()

    for f in fields:

        group, name = f.split('/')
        default_type = default_types.get(f)
        if default_type:
            default_type = _normalize_type(default_type)

        if f in types:
            # user had manually specified the type
            normed_types[f] = types[f]

        elif group == 'variants':

            if name in COMPUTED_FIELDS:
                # computed fields, special case
                continue

            elif name.startswith('FILTER_'):
                normed_types[f] = np.dtype(bool)

            elif name in headers.infos:
                header_type = _normalize_type(headers.infos[name]['Type'])
                if isinstance(default_type, np.dtype):
                    # check that default is compatible with header
                    if (
                            default_type.kind in 'ifb' and
                            default_type.kind != header_type.kind):
                        # default is not compatible with header, fall back to header
                        t = header_type
                    else:
                        t = default_type
                elif default_type:
                    t = default_type
                else:
                    t = header_type
                normed_types[f] = t

            elif default_type:
                normed_types[f] = default_type

            else:
                # fall back to string
                normed_types[f] = _normalize_type('String')
                warnings.warn('no type for field %r, assuming %s' % \
                              (f, normed_types[f]))

        elif group == 'calldata':

            if name in headers.formats:
                header_type = _normalize_type(headers.formats[name]['Type'])
                if isinstance(default_type, np.dtype):
                    # check that default is compatible with header
                    if (
                            default_type.kind in 'ifb' and
                            default_type.kind != header_type.kind):
                        # default is not compatible with header, fall back to header
                        t = header_type
                    else:
                        t = default_type
                elif default_type:
                    t = default_type
                else:
                    t = header_type
                normed_types[f] = t

            elif default_type:
                normed_types[f] = default_type

            else:
                # fall back to string
                normed_types[f] = _normalize_type('String')
                warnings.warn('no type for field %r, assuming %s' % \
                              (f, normed_types[f]))

        else:
            raise RuntimeError('unpected field: %r' % f)

    return normed_types


default_numbers = {
    'variants/CHROM': 1,
    'variants/POS': 1,
    'variants/ID': 1,
    'variants/REF': 1,
    'variants/ALT': 'A',
    'variants/QUAL': 1,
    'variants/DP': 1,
    'variants/AN': 1,
    'variants/AC': 'A',
    'variants/AF': 'A',
    'variants/MQ': 1,
    'variants/ANN': 1,
    'calldata/DP': 1,
    'calldata/GT': 2,
    'calldata/GQ': 1,
    'calldata/HQ': 2,
    'calldata/AD': 'R',
    'calldata/MQ0': 1,
    'calldata/MQ': 1,
}


def _normalize_number(field, n, alt_number):
    if n == '.':
        return 1
    elif n == 'A':
        return alt_number
    elif n == 'R':
        return alt_number + 1
    elif n == 'G':
        return 3
    else:
        try:
            return int(n)
        except ValueError:
            warnings.warn(
                'error parsing %r as number for field %r' % (n, field))
        return 1


def _normalize_numbers(numbers, fields, headers, alt_number):

    # normalize field prefixes
    if numbers is None:
        numbers = dict()
    numbers = {_normalize_field_prefix(
        f, headers): n for f, n in numbers.items()}

    # setup output
    normed_numbers = dict()

    for f in fields:

        group, name = f.split('/')

        if f in numbers:
            normed_numbers[f] = _normalize_number(f, numbers[f], alt_number)

        elif f in default_numbers:
            normed_numbers[f] = _normalize_number(
                f, default_numbers[f], alt_number)

        elif group == 'variants':

            if name in COMPUTED_FIELDS:
                # computed fields, special case (for altlen, number depends on ALT)
                continue

            elif name.startswith('FILTER_'):
                normed_numbers[f] = 0

            elif name in headers.infos:
                normed_numbers[f] = _normalize_number(f, headers.infos[name]['Number'],
                                                      alt_number)

            else:
                # fall back to 1
                normed_numbers[f] = 1
                warnings.warn('no number for field %r, assuming 1' % f)

        elif group == 'calldata':

            if name in headers.formats:
                normed_numbers[f] = _normalize_number(f, headers.formats[name]['Number'],
                                                      alt_number)

            else:
                # fall back to 1
                normed_numbers[f] = 1
                warnings.warn('no number for field %r, assuming 1' % f)

        else:
            raise RuntimeError('unexpected field: %r' % f)

    return normed_numbers


def _normalize_fills(fills, fields, headers):

    if fills is None:
        fills = dict()
    fills = {_normalize_field_prefix(f, headers): v
             for f, v in fills.items()}

    # setup output
    normed_fills = dict()

    for f in fields:

        if f in fills:
            normed_fills[f] = fills[f]

    return normed_fills


def _normalize_samples(samples, headers, types):
    loc_samples = np.zeros(len(headers.samples), dtype='u1')

    if samples is None:
        normed_samples = list(headers.samples)
        loc_samples.fill(1)

    else:
        samples = set(samples)
        normed_samples = []
        for i, s in enumerate(headers.samples):
            if i in samples:
                normed_samples.append(s)
                samples.remove(i)
                loc_samples[i] = 1
            elif s in samples:
                normed_samples.append(s)
                samples.remove(s)
                loc_samples[i] = 1
        if len(samples) > 0:
            warnings.warn('some samples not found, will be ignored: ' \
                          ', '.join(map(repr, sorted(samples))))

    t = default_string_dtype
    if types is not None:
        t = types.get('samples', t)
    normed_samples = np.array(normed_samples, dtype=t)

    return normed_samples, loc_samples


# size in bytes of the batches of lines that the Rust parser parses in
# parallel
_RUST_BATCH_BYTES = 1 << 22


def _available_cores():
    """The CPU cores this process may run on."""
    if hasattr(os, 'sched_getaffinity'):
        return len(os.sched_getaffinity(0))
    if hasattr(os, 'process_cpu_count'):
        return os.process_cpu_count() or 1
    return os.cpu_count() or 1


def _normalize_n_threads(n_threads):
    """The number of threads to read with: `n_threads`, or by default the
    available cores, at most ``OMP_NUM_THREADS`` if set."""
    if n_threads is None:
        n = _available_cores()
        try:
            omp = int(os.environ.get('OMP_NUM_THREADS', ''))
        except ValueError:
            omp = 0
        return min(n, omp) if omp >= 1 else n
    if isinstance(n_threads, bool) or not isinstance(n_threads, Integral) or n_threads < 1:
        raise ValueError('n_threads must be None or an integer >= 1, found %r' % (n_threads,))
    return int(n_threads)


def _backend():
    backend = os.environ.get(BACKEND_VARIABLE, 'rust')
    if backend not in BACKENDS:
        raise ValueError('%s must be one of %s, found %r'
                         % (BACKEND_VARIABLE, ', '.join(map(repr, BACKENDS)), backend))
    return backend


def _log_fallback(input, reason):
    logger.debug('reading %r: falling back to the Cython VCF parser (%s)', input, reason)


# noinspection PyShadowingBuiltins
def _cython_input_reason(input, region, transformers):
    """Why `input` must be read by the Cython parser, or None if the Rust parser
    can read it."""
    if _backend() == 'cython':
        return '%s=cython' % BACKEND_VARIABLE
    if transformers is not None:
        return 'transformers are not handled by the Rust parser'
    if not isinstance(input, str):
        return 'file objects are not handled by the Rust parser'
    if region is not None:
        return 'regions are not yet handled by the Rust parser'
    return None


# fields read by the Rust parser whatever their type: numbers, computed fields
# (FILTER flags, strings and calldata/GT are checked separately)
_RUST_FIELDS = ('variants/POS', 'variants/QUAL', 'variants/numalt', 'variants/altlen',
                'variants/is_snp')
_RUST_STRING_FIELDS = ('variants/CHROM', 'variants/ID', 'variants/REF', 'variants/ALT')


def _rust_string_code(t):
    """The dtype code ('O' or 'S<width>') the Rust parser stores strings of
    type `t` with (objects for StringDType, converted afterwards), or None if
    it does not handle `t`."""
    t = np.dtype(t)
    if t.kind in 'OT':
        return 'O'
    if t.kind == 'S' and t.itemsize > 0:
        return 'S%d' % t.itemsize
    return None


def _is_info_field(f):
    # as in the Cython parser, the variants fields left are INFO fields
    return (f.startswith('variants/') and f not in _RUST_FIELDS and
            f not in _RUST_STRING_FIELDS and not f.startswith('variants/FILTER_'))


def _rust_number_code(t):
    """The dtype code ('i1' to 'u8', 'f4' or 'f8') of number type `t` if the
    Rust parser handles it, else None. The Cython parser skips the types of
    other byte orders."""
    if t.isnative and (t.kind in 'iu' or t in (np.dtype('f4'), np.dtype('f8'))):
        return '%s%d' % (t.kind, t.itemsize)
    return None


def _rust_info_code(t, number):
    """The dtype code the Rust parser stores an INFO field of type `t` with:
    'b1' for flags, as for any field of number 0, a number code or a string
    code; None if it does not handle `t`."""
    try:
        t = np.dtype(t)
    except TypeError:
        return None
    if number < 0:
        # the Cython parser rejects these
        return None
    if t == np.dtype(bool) or number == 0:
        return 'b1'
    return _rust_number_code(t) or _rust_string_code(t)


def _rust_format_code(f, t, number):
    """The dtype code the Rust parser stores a FORMAT field `f` of type `t`
    with: 'gt:' or 'ac:' before an integer code for genotypes, of type
    'genotype/<dtype>' or an integer type for GT, and their allele counts, of
    type 'genotype_ac/<dtype>'; a number code or a string code otherwise.
    None if it does not handle `t`."""
    if number < 0:
        return None
    prefix = ''
    if isinstance(t, str) and t.startswith(('genotype/', 'genotype_ac/')):
        kind, t = t.split('/', 1)
        prefix = 'gt:' if kind == 'genotype' else 'ac:'
    try:
        t = np.dtype(t)
    except TypeError:
        return None
    if f == 'calldata/GT' and t.kind in 'iu':
        prefix = prefix or 'gt:'
    if prefix:
        code = _rust_number_code(t)
        return prefix + code if code and t.kind in 'iu' else None
    if t == np.dtype(bool):
        # the Cython parser skips these
        return None
    return _rust_number_code(t) or _rust_string_code(t)


def _rust_fill(code, fills, f):
    """The fill of INFO or FORMAT field `f` of dtype `code` as the Rust parser
    takes it, None for the default; raises TypeError if the Rust parser does
    not handle it."""
    dtype = code[len('gt:'):] if code.startswith('gt:') else code
    if f not in fills or dtype[0] not in 'iuf':
        # the Cython parser ignores fills of flags, strings and allele counts
        # ('ac:' codes, whose dtype starts with 'a')
        return None
    fill = fills[f]
    if isinstance(fill, Integral):
        return int(fill)
    if code[0] == 'f' and isinstance(fill, Real):
        return float(fill)
    raise TypeError('fill %r' % (fill,))


def _cython_fields_reason(fields, types, numbers, fills):
    """Why `fields` must be read by the Cython parser, or None if the Rust
    parser can read them."""
    for f in fields:
        if f in _RUST_FIELDS or f.startswith('variants/FILTER_'):
            continue
        if f in _RUST_STRING_FIELDS:
            if _rust_string_code(types[f]) is None:
                return ('type %r of field %r is not handled by the Rust parser'
                        % (types[f], f))
            continue
        if _is_info_field(f):
            code = _rust_info_code(types[f], numbers[f])
        else:
            code = _rust_format_code(f, types[f], numbers[f])
        if code is None:
            return ('type %r or number %r of field %r is not handled by the Rust parser'
                    % (types[f], numbers[f], f))
        try:
            _rust_fill(code, fills, f)
        except TypeError:
            return 'fill %r of field %r is not handled by the Rust parser' % (fills[f], f)
    return None


def _rust_chunks(path, headers, fields, types, numbers, fills, chunk_length,
                 loc_samples, n_threads):
    # as the Cython parser does
    for f, t in ('POS', 'int32'), ('QUAL', 'float32'):
        requested = types.get('variants/' + f)
        if 'variants/' + f in fields and requested is not None and requested != np.dtype(t):
            warnings.warn('only %s supported for %s field, ignoring requested type: %r'
                          % (t, f, requested))
    string_types = {f: _rust_string_code(types[f])
                    for f in fields if f in _RUST_STRING_FIELDS}
    infos = list()
    # sorted as the Cython parser warns about them
    for f in sorted((f for f in fields if _is_info_field(f)), key=lambda f: f.encode()):
        t, n = types[f], numbers[f]
        code = _rust_info_code(t, n)
        if code == 'b1' and t != np.dtype(bool):
            # as the Cython parser does, which shows the type it parses with
            t = np.dtype(object) if np.dtype(t).kind == 'T' else t
            warnings.warn('cannot have non-bool dtype for field with number 0, '
                          'ignoring type %r' % t)
        infos.append((f[len('variants/'):], code, n, _rust_fill(code, fills, f)))
    formats = list()
    for f in fields:
        if f.startswith('calldata/'):
            code = _rust_format_code(f, types[f], numbers[f])
            formats.append((f[len('calldata/'):], code, numbers[f], _rust_fill(code, fills, f)))
    return RustVCFChunkIterator(path, chunk_length=chunk_length, samples=headers.samples,
                                loc_samples=loc_samples.astype(bool).tolist(),
                                fields=fields, string_types=string_types,
                                alt_number=numbers.get('variants/ALT', 1), infos=infos,
                                formats=formats, n_threads=n_threads,
                                batch_bytes=_RUST_BATCH_BYTES)


def _iter_vcf_stream(stream, fields, exclude_fields, types, numbers, alt_number,
                     chunk_length, fills, region, samples, n_threads=1, rust_path=None):

    # read VCF headers
    headers = _read_vcf_headers(stream)

    # setup samples
    samples, loc_samples = _normalize_samples(samples=samples, headers=headers,
                                              types=types)

    # setup fields to read
    if fields is None:

        # choose default fields
        fields = list()
        _add_all_fixed_variants_fields(fields)
        fields.append('variants/FILTER_PASS')
        if len(samples) > 0 and 'GT' in headers.formats:
            fields.append('calldata/GT')

    else:
        fields = _normalize_fields(
            fields=fields, headers=headers, samples=samples)

    # deal with field exclusions
    if exclude_fields:
        exclude_fields = _normalize_fields(fields=exclude_fields, headers=headers,
                                           samples=samples)
        fields = [f for f in fields if f not in exclude_fields]

    # setup data types
    types = _normalize_types(types=types, fields=fields, headers=headers)

    # setup numbers (a.k.a., arity)
    numbers = _normalize_numbers(numbers=numbers, fields=fields, headers=headers,
                                 alt_number=alt_number)

    # setup fills
    fills = _normalize_fills(fills=fills, fields=fields, headers=headers)

    # the parser produces object arrays for variable-width strings, which are
    # then converted chunk by chunk to the requested StringDType
    # (computed fields such as numalt have no entry in types)
    string_fields = [f for f in fields if f in types and _is_string_dtype(types[f])]
    parser_types = {f: np.dtype(object) if f in string_fields else t
                    for f, t in types.items()}

    # setup chunks iterator
    if rust_path is not None:
        reason = _cython_fields_reason(fields, types, numbers, fills)
        if reason is not None:
            _log_fallback(rust_path, reason)
            rust_path = None
    if rust_path is not None:
        chunks = _rust_chunks(rust_path, headers=headers, fields=fields, types=types,
                              numbers=numbers, fills=fills, chunk_length=chunk_length,
                              loc_samples=loc_samples, n_threads=n_threads)
        logger.debug('reading %r with the Rust VCF parser on %d threads', rust_path,
                     chunks.n_threads)
    else:
        chunks = VCFChunkIterator(
            stream, chunk_length=chunk_length, headers=headers, fields=fields,
            types=parser_types, numbers=numbers, fills=fills, region=region,
            loc_samples=loc_samples
        )
    if string_fields:
        chunks = _chunk_iter_to_string_dtype(
            chunks, {f: np.dtype(types[f]) for f in string_fields})

    return fields, samples, headers, chunks


# pre-compile some regular expressions
_re_filter_header = \
    re.compile('##FILTER=<ID=([^,]+),Description="([^"]*)">')
_re_info_header = \
    re.compile(
        '##INFO=<ID=([^,]+),Number=([^,]+),Type=([^,]+),Description="([^"]*)">')
_re_format_header = \
    re.compile(
        '##FORMAT=<ID=([^,]+),Number=([^,]+),Type=([^,]+),Description="([^"]*)">')


VCFHeaders = namedtuple('VCFHeaders', ['headers', 'filters', 'infos', 'formats',
                                       'samples'])


# noinspection PyShadowingBuiltins
def read_vcf_headers(input):
    """Read headers from a VCF file."""
    stream = _setup_input_stream(input)
    return _read_vcf_headers(stream)


def _read_vcf_headers(stream):

    # setup
    headers = []
    samples = None
    filters = dict()
    infos = dict()
    formats = dict()

    # read first header line
    header = stream.readline()
    header = str(header, 'utf8')

    while header and header[0] == '#':

        headers.append(header)

        if header.startswith('##FILTER'):

            match = _re_filter_header.match(header)
            if match is None:
                warnings.warn('invalid FILTER header: %r' % header)
            else:
                k, d = match.groups()
                if k in filters:
                    warnings.warn('multiple FILTER headers for %r' % k)
                filters[k] = {'ID': k, 'Description': d}

        elif header.startswith('##INFO'):

            match = _re_info_header.match(header)
            if match is None:
                warnings.warn('invalid INFO header: %r' % header)
            else:
                k, n, t, d = match.groups()
                if k in infos:
                    warnings.warn('multiple INFO headers for %r' % k)
                infos[k] = {'ID': k, 'Number': n, 'Type': t, 'Description': d}

        elif header.startswith('##FORMAT'):

            match = _re_format_header.match(header)
            if match is None:
                warnings.warn('invalid FORMAT header: %r' % header)
            else:
                k, n, t, d = match.groups()
                if k in formats:
                    warnings.warn('multiple FORMAT headers for %r' % k)
                formats[k] = {'ID': k, 'Number': n,
                              'Type': t, 'Description': d}

        elif header.startswith('#CHROM'):

            # parse out samples
            samples = header.strip().split('\t')[9:]
            break

        # read next header line
        header = stream.readline()
        header = str(header, 'utf8')

    # check if we saw the mandatory header line or not
    if samples is None:
        # can't warn about this, it's fatal
        raise RuntimeError(
            'VCF file is missing mandatory header line ("#CHROM...")')

    return VCFHeaders(headers, filters, infos, formats, samples)


def _chunk_to_dataframe(fields, chunk):
    import pandas
    items = list()
    string_columns = list()
    for f in fields:
        a = chunk[f]
        group, name = f.split('/')
        assert group == 'variants'
        is_string = a.dtype.kind in 'SOT'
        if a.dtype.kind == 'S':
            a = a.astype('U')
        if is_string:
            # empty string means missing
            a = a.astype(object)
            a[a == ''] = np.nan
        if a.ndim == 1:
            columns = [(name, a)]
        elif a.ndim == 2:
            columns = [('%s_%s' % (name, i + 1), a[:, i]) for i in range(a.shape[1])]
        else:
            warnings.warn(
                'cannot handle array %r with >2 dimensions, skipping' % name)
            continue
        items.extend(columns)
        if is_string:
            string_columns.extend(c for c, _ in columns)
    df = pandas.DataFrame.from_dict(OrderedDict(items))
    # always use the pandas str dtype for strings, whatever the NumPy dtype
    return df.astype({c: 'str' for c in string_columns})


# noinspection PyShadowingBuiltins
def vcf_to_dataframe(input,
                     fields=None,
                     exclude_fields=None,
                     types=None,
                     numbers=None,
                     alt_number=DEFAULT_ALT_NUMBER,
                     fills=None,
                     region=None,
                     tabix='tabix',
                     transformers=None,
                     buffer_size=DEFAULT_BUFFER_SIZE,
                     chunk_length=DEFAULT_CHUNK_LENGTH,
                     n_threads=None,
                     log=None):
    """Read data from a VCF file into a pandas DataFrame.

    Parameters
    ----------
    input : string
        {input}
    fields : list of strings, optional
        {fields}
    exclude_fields : list of strings, optional
        {exclude_fields}
    types : dict, optional
        {types}
    numbers : dict, optional
        {numbers}
    alt_number : int, optional
        {alt_number}
    fills : dict, optional
        {fills}
    region : string, optional
        {region}
    tabix : string, optional
        {tabix}
    transformers : list of transformer objects, optional
        {transformers}
    buffer_size : int, optional
        {buffer_size}
    chunk_length : int, optional
        {chunk_length}
    n_threads : int, optional
        {n_threads}
    log : file-like, optional
        {log}

    Returns
    -------
    df : pandas.DataFrame

    """

    import pandas

    # samples requested?
    # noinspection PyTypeChecker
    _, fields = _prep_fields_param(fields)

    # setup
    fields, _, _, it = iter_vcf_chunks(
        input=input, fields=fields, exclude_fields=exclude_fields, types=types,
        numbers=numbers, alt_number=alt_number, buffer_size=buffer_size,
        chunk_length=chunk_length, n_threads=n_threads, fills=fills, region=region,
        tabix=tabix, samples=[],
        transformers=transformers
    )

    # setup progress logging
    if log is not None:
        it = _chunk_iter_progress(it, log, prefix='[vcf_to_dataframe]')

    # read all chunks into a list
    chunks = [d[0] for d in it]

    # setup output
    output = None

    if chunks:

        # concatenate chunks
        output = pandas.concat([_chunk_to_dataframe(fields, chunk)
                                for chunk in chunks])

    return output


vcf_to_dataframe.__doc__ = vcf_to_dataframe.__doc__.format(
    input=_doc_param_input,
    fields=_doc_param_fields,
    exclude_fields=_doc_param_exclude_fields,
    types=_doc_param_types,
    numbers=_doc_param_numbers,
    alt_number=_doc_param_alt_number,
    fills=_doc_param_fills,
    region=_doc_param_region,
    tabix=_doc_param_tabix,
    transformers=_doc_param_transformers,
    buffer_size=_doc_param_buffer_size,
    chunk_length=_doc_param_chunk_length,
    n_threads=_doc_param_n_threads,
    log=_doc_param_log,
)


# noinspection PyShadowingBuiltins
def vcf_to_csv(input, output,
               fields=None,
               exclude_fields=None,
               types=None,
               numbers=None,
               alt_number=DEFAULT_ALT_NUMBER,
               fills=None,
               region=None,
               tabix='tabix',
               transformers=None,
               buffer_size=DEFAULT_BUFFER_SIZE,
               chunk_length=DEFAULT_CHUNK_LENGTH,
               n_threads=None,
               log=None,
               **kwargs):
    r"""Read data from a VCF file and write out to a comma-separated values (CSV) file.

    Parameters
    ----------
    input : string
        {input}
    output : string
        {output}
    fields : list of strings, optional
        {fields}
    exclude_fields : list of strings, optional
        {exclude_fields}
    types : dict, optional
        {types}
    numbers : dict, optional
        {numbers}
    alt_number : int, optional
        {alt_number}
    fills : dict, optional
        {fills}
    region : string, optional
        {region}
    tabix : string, optional
        {tabix}
    transformers : list of transformer objects, optional
        {transformers}
    buffer_size : int, optional
        {buffer_size}
    chunk_length : int, optional
        {chunk_length}
    n_threads : int, optional
        {n_threads}
    log : file-like, optional
        {log}
    kwargs : keyword arguments
        All remaining keyword arguments are passed through to pandas.DataFrame.to_csv().
        E.g., to write a tab-delimited file, provide `sep='\t'`.

    """

    # samples requested?
    # noinspection PyTypeChecker
    _, fields = _prep_fields_param(fields)

    # setup
    fields, _, _, it = iter_vcf_chunks(
        input=input, fields=fields, exclude_fields=exclude_fields, types=types,
        numbers=numbers, alt_number=alt_number, buffer_size=buffer_size,
        chunk_length=chunk_length, n_threads=n_threads, fills=fills, region=region,
        tabix=tabix, samples=[],
        transformers=transformers
    )

    # setup progress logging
    if log is not None:
        it = _chunk_iter_progress(it, log, prefix='[vcf_to_csv]')

    kwargs['index'] = False
    for i, (chunk, _, _, _) in enumerate(it):
        df = _chunk_to_dataframe(fields, chunk)
        if i == 0:
            kwargs['header'] = True
            kwargs['mode'] = 'w'
        else:
            kwargs['header'] = False
            kwargs['mode'] = 'a'
        df.to_csv(output, **kwargs)


vcf_to_csv.__doc__ = vcf_to_csv.__doc__.format(
    input=_doc_param_input,
    output=_doc_param_output,
    fields=_doc_param_fields,
    exclude_fields=_doc_param_exclude_fields,
    types=_doc_param_types,
    numbers=_doc_param_numbers,
    alt_number=_doc_param_alt_number,
    fills=_doc_param_fills,
    region=_doc_param_region,
    tabix=_doc_param_tabix,
    transformers=_doc_param_transformers,
    buffer_size=_doc_param_buffer_size,
    chunk_length=_doc_param_chunk_length,
    n_threads=_doc_param_n_threads,
    log=_doc_param_log,
)


def _chunk_to_recarray(fields, chunk):
    arrays = list()
    names = list()
    for f in fields:
        a = chunk[f]
        group, name = f.split('/')
        if a.dtype.kind == 'T':
            # structured dtypes do not support StringDType fields
            a = a.astype(object)
        if a.ndim == 1:
            arrays.append(a)
            names.append(name)
        elif a.ndim == 2:
            for i in range(a.shape[1]):
                arrays.append(a[:, i])
                names.append('%s_%s' % (name, i + 1))
        else:
            warnings.warn(
                'cannot handle arrays with >2 dimensions, ignoring %r' % name)
    ra = np.rec.fromarrays(arrays, names=names)
    return ra


# noinspection PyShadowingBuiltins
def vcf_to_recarray(input,
                    fields=None,
                    exclude_fields=None,
                    types=None,
                    numbers=None,
                    alt_number=DEFAULT_ALT_NUMBER,
                    fills=None,
                    region=None,
                    tabix='tabix',
                    transformers=None,
                    buffer_size=DEFAULT_BUFFER_SIZE,
                    chunk_length=DEFAULT_CHUNK_LENGTH,
                    n_threads=None,
                    log=None):
    """Read data from a VCF file into a NumPy recarray.

    Parameters
    ----------
    input : string
        {input}
    fields : list of strings, optional
        {fields}
    exclude_fields : list of strings, optional
        {exclude_fields}
    types : dict, optional
        {types}
    numbers : dict, optional
        {numbers}
    alt_number : int, optional
        {alt_number}
    fills : dict, optional
        {fills}
    region : string, optional
        {region}
    tabix : string, optional
        {tabix}
    transformers : list of transformer objects, optional
        {transformers}
    buffer_size : int, optional
        {buffer_size}
    chunk_length : int, optional
        {chunk_length}
    n_threads : int, optional
        {n_threads}
    log : file-like, optional
        {log}

    Returns
    -------
    ra : np.rec.array

    """

    # samples requested?
    # noinspection PyTypeChecker
    _, fields = _prep_fields_param(fields)

    # setup chunk iterator
    # N.B., set samples to empty list so we don't get any calldata fields
    fields, _, _, it = iter_vcf_chunks(
        input=input, fields=fields, exclude_fields=exclude_fields, types=types,
        numbers=numbers, alt_number=alt_number, buffer_size=buffer_size,
        chunk_length=chunk_length, n_threads=n_threads, fills=fills, region=region,
        tabix=tabix, samples=[],
        transformers=transformers
    )

    # setup progress logging
    if log is not None:
        it = _chunk_iter_progress(it, log, prefix='[vcf_to_recarray]')

    # read all chunks into a list
    chunks = [d[0] for d in it]

    # setup output
    output = None

    if chunks:

        # concatenate chunks
        output = np.concatenate(
            [_chunk_to_recarray(fields, chunk) for chunk in chunks])

    return output


vcf_to_recarray.__doc__ = vcf_to_recarray.__doc__.format(
    input=_doc_param_input,
    fields=_doc_param_fields,
    exclude_fields=_doc_param_exclude_fields,
    types=_doc_param_types,
    numbers=_doc_param_numbers,
    alt_number=_doc_param_alt_number,
    fills=_doc_param_fills,
    region=_doc_param_region,
    tabix=_doc_param_tabix,
    transformers=_doc_param_transformers,
    buffer_size=_doc_param_buffer_size,
    chunk_length=_doc_param_chunk_length,
    n_threads=_doc_param_n_threads,
    log=_doc_param_log,
)
