# -*- coding: utf-8 -*-
"""The Rust and Cython VCF parsers give the same output.

The Cython parser is the reference for as long as it exists: each read is done
once with ``ALLEL2_BACKEND=cython`` and once with the default backend, and the
chunks must match. Fields not yet handled by the Rust parser fall back to the
Cython parser, with a debug message giving the reason.
"""
import io
import logging
import os
import warnings

import numpy as np
import pytest

import allel2.io.vcf_read
from allel2.io.vcf_read import (iter_vcf_chunks, read_vcf, read_vcf_headers,
                                ANNTransformer, vcf_to_npz, vcf_to_hdf5, vcf_to_zarr,
                                vcf_to_dataframe, vcf_to_csv, vcf_to_recarray)


DATA = os.path.join(os.path.dirname(__file__), os.pardir, 'data')
LOGGER = 'allel2.io.vcf_read'
FALLBACK = 'falling back to the Cython VCF parser'
RUST = 'with the Rust VCF parser'

# every test VCF that has a header line
FILES = ['altlen.vcf', 'ann.vcf', 'eff.vcf', 'empty.vcf', 'region.vcf', 'region.vcf.gz',
         'region_csi.vcf.gz', 'sample.utf8.vcf', 'sample.vcf', 'sample.vcf.gz', 'test1.vcf',
         'test14.vcf', 'test16.vcf', 'test176.vcf', 'test32.vcf', 'test4.vcf',
         'test48b.vcf', 'test54.vcf.gz', 'test63.vcf', 'unsorted.vcf']

HEADER = (b'##fileformat=VCFv4.2\n'
          b'##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
          b'#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\tS3\n')

# lines that exercise the corners of POS and GT parsing
EDGE_CASES = {
    'crlf': HEADER.replace(b'\n', b'\r\n') +
    b'2L\t12\t.\tA\tC\t.\t.\t.\tGT\t0/1\t1|1\t./.\r\n'
    b'2L\t15\t.\tA\tC\t.\t.\t.\tGT\t0/0\t0|1\t1/.\r\n',
    'no_final_newline': HEADER +
    b'2L\t12\t.\tA\tC\t.\t.\t.\tGT\t0/1\t1|1\t./.\n'
    b'2L\t15\t.\tA\tC\t.\t.\t.\tGT\t0/0\t0|1\t1/.',
    'gt_not_first': HEADER +
    b'2L\t12\t.\tA\tC\t.\t.\t.\tDP:GT\t3:0/1\t4:1|1\t.:./.\n'
    b'2L\t15\t.\tA\tC\t.\t.\t.\tDP:.:GT\t3:x:0/1\t4\t\n',
    'missing_and_short': HEADER +
    b'2L\t12\t.\tA\tC\t.\t.\t.\tGT\t.\t0\t0/1/2/3\n'
    b'2L\t13\t.\tA\tC\t.\t.\t.\t.\t0/1\t1/1\t1/1\n'
    b'2L\t14\t.\tA\tC\t.\t.\t.\tGT\t0/1\n'
    b'2L\t15\t.\tA\tC\t.\t.\t.\n'
    b'2L\t16\n',
    'large_alleles': HEADER +
    b'2L\t12\t.\tA\tC\t.\t.\t.\tGT\t127/128\t255/256\t-1/-2\n'
    b'2L\t13\t.\tA\tC\t.\t.\t.\tGT\t+3/ 4\t99999999999999999999/0\t1/1\n',
    'dodgy_values': HEADER +
    b'2L\taaa\t.\tA\tC\t.\t.\t.\tGT\taa/bb\t12aa/22\t1/1\n'
    b'2L\t12aaa\t.\tA\tC\t.\t.\t.\tGT\t0/1\t1/1\t1/1\n'
    b'\t\t.\tA\tC\t.\t.\t.\tGT\t0/1\t1/1\t1/1\n'
    b'2L\t99999999999\t.\tA\tC\t.\t.\t.\tGT\t0/1\t1/1\t1/1\n',
    'extra_samples': HEADER +
    b'2L\t12\t.\tA\tC\t.\t.\t.\tGT\t0/1\t1/1\t1/1\t0/0\t1/0\n',
    'empty_line': HEADER +
    b'2L\t12\t.\tA\tC\t.\t.\t.\tGT\t0/1\t1/1\t1/1\n'
    b'\n'
    b'2L\t14\t.\tA\tC\t.\t.\t.\tGT\t0/1\t1/1\t1/1\n',
    # line terminators and null bytes, as the Cython parser reads them
    'lone_cr': HEADER +
    b'2L\t12\t.\tA\tC\t1\tPASS\t.\tGT\t0/1\t1|1\t./.\r'
    b'2L\t15\t.\tA\tC\t2\tPASS\t.\tGT\t0/0\t0|1\t1/.\r\r\n'
    b'2L\t16\t.\tA\tC\t3\tPASS\t.\tGT\t0/0\t0|1\t1/.\n',
    'final_cr': HEADER.replace(b'\n', b'\r\n') +
    b'2L\t12\t.\tA\tC\t1\tPASS\t.\tGT\t0/1\t1|1\t./.\r\n'
    b'2L\t15\t.\tA\tC\t2\tPASS\r',
    'null_byte': HEADER +
    b'2L\t12\t.\tA\tC\t1\tPASS\t.\tGT\t0/1\t1|1\t./.\n'
    b'2L\t1\x005\t.\tA\tC\t2\tPASS\t.\tGT\t0/0\t0|1\t1/.\n'
    b'2L\t16\t.\tA\tC\t3\tPASS\t.\tGT\t0/0\t0|1\t1/.\n',
    'null_line': HEADER +
    b'2L\t12\t.\tA\tC\t1\tPASS\t.\tGT\t0/1\t1|1\t./.\n'
    b'\x002L\t16\t.\tA\tC\t3\tPASS\t.\tGT\t0/0\t0|1\t1/.\n',
    # lines that exercise the fixed columns
    'qual': HEADER +
    b'2L\t1\t.\tA\tC\t12.5\t.\t.\tGT\t0/1\t1/1\t1/1\n'
    b'2L\t2\t.\tA\tC\tabc\t.\t.\tGT\t0/1\t1/1\t1/1\n'
    b'2L\t3\t.\tA\tC\t12x\t.\t.\tGT\t0/1\t1/1\t1/1\n'
    b'2L\t4\t.\tA\tC\t 3e2\t.\t.\tGT\t0/1\t1/1\t1/1\n'
    b'2L\t5\t.\tA\tC\tinf\t.\n'
    b'2L\t6\t.\tA\tC\t-NaN\n'
    b'2L\t7\t.\tA\tC\t1e400\t\t.\n'
    b'2L\t8\t.\tA\tC\t.5\n'
    b'2L\t9\t.\tA\tC\t0x1.8p3\t.\n'
    b'2L\t10\t.\tA\tC\t0xg\t.\n'
    b'2L\t11\t.\tA\tC\tInfinity\t.\n'
    b'2L\t12\t.\tA\tC\tnan(123)x\t.\n'
    b'2L\t13\t.\tA\tC\t1e\t.\n'
    b'2L\t14\t.\tA\tC\t-.\t.\n'
    b'2L\t15\t.\tA\tC\t1.e-3x',
    'filter': HEADER +
    b'2L\t1\t.\tA\tC\t.\tPASS\t.\n'
    b'2L\t2\t.\tA\tC\t.\tq10;s50\t.\n'
    b'2L\t3\t.\tA\tC\t.\ts50,q10:PASS\t.\n'
    b'2L\t4\t.\tA\tC\t.\t.PASS\t.\n'
    b'2L\t5\t.\tA\tC\t.\t;;q10;\t.\n'
    b'2L\t6\t.\tA\tC\t.\tPASSq10\t.\n'
    b'2L\t7\t.\tA\tC\t.\t\t.\n'
    b'2L\t8\t.\tA\tC\t.\tq10\n'
    b'2L\t9\t.\tA\tC\t.\n',
    'alleles': HEADER +
    b'2L\t1\tid1\tA\tC,G,T,AC\t.\t.\t.\n'
    b'2L\t2\tid2;id3\tAT\tA,*\t.\t.\t.\n'
    b'2L\t3\t.\tA\t.\t.\t.\t.\n'
    b'2L\t4\t.\tA\t.,C\t.\t.\t.\n'
    b'2L\t5\t.\tA\t\t.\t.\t.\n'
    b'2L\t6\t.\tA.\tC\t.\t.\t.\n'
    b'2L\t7\t.\t.\tC\t.\t.\t.\n'
    b'2L\t8\t.\tA\tC,,G\t.\t.\t.\n'
    b'2L\t9\t.\tACGTACGT\tACGTACGTAC,<DEL>\t.\t.\t.\n'
    b'2L\t10\t.\tA\tC,G\n'
    b'2L\t11\t.\tA\n'
    b'2L\t12\tid4\n'
    b'\xc3\xa9\t13\t\xc3\xa9\t\xc3\xa9\t\xc3\xa9,T\t.\t.\t.\n'
    b'2L\t14\t.\t\xc3\xa9\tT\t.\t.\t.\n',
}

FIXED_FIELDS = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER_PASS',
                'numalt', 'altlen', 'is_snp']

FIELDS = [
    ['variants/POS'],
    ['calldata/GT'],
    ['variants/POS', 'calldata/GT'],
    ['calldata/GT', 'POS'],
    ['CHROM'],
    ['ID', 'REF'],
    ['ALT', 'numalt', 'altlen', 'is_snp'],
    ['numalt', 'altlen', 'is_snp'],
    ['QUAL'],
    ['FILTER_PASS'],
    None,
    'filters',
    'infos',
    'formats',
    ['GT', 'GQ', 'calldata/DP', 'HQ'],
]

GT_OPTIONS = (
    [{'types': {'calldata/GT': t}}
     for t in ('i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8')] +
    [{},
     {'types': {'calldata/GT': 'genotype/u8'}},
     {'numbers': {'calldata/GT': 1}},
     {'numbers': {'calldata/GT': 3}},
     {'fills': {'calldata/GT': -9}},
     {'types': {'calldata/GT': 'u2'}, 'fills': {'calldata/GT': 7}},
     {'alt_number': 1}]
)


@pytest.fixture
def edge_vcf(tmp_path):
    def write(name, gz=False):
        data = EDGE_CASES[name]
        if gz:
            import gzip
            path = tmp_path / (name + '.vcf.gz')
            with gzip.open(path, 'wb') as f:
                f.write(data)
        else:
            path = tmp_path / (name + '.vcf')
            path.write_bytes(data)
        return str(path)
    return write


def _fields(path, fields):
    # 'filters': the fixed and computed fields, and every declared filter;
    # 'infos': POS and every declared INFO field; 'formats': POS and every
    # declared FORMAT field
    if fields == 'filters':
        filters = ['FILTER_' + f for f in read_vcf_headers(path).filters]
        return FIXED_FIELDS + filters + ['calldata/GT']
    if fields == 'infos':
        return ['POS'] + ['variants/' + k for k in read_vcf_headers(path).infos]
    if fields == 'formats':
        return ['POS'] + ['calldata/' + k for k in read_vcf_headers(path).formats]
    return fields


def _chunks(path, backend, monkeypatch, caplog, **kwargs):
    """Read all chunks with the given backend, returning chunks, warnings and
    whether the read fell back to the Cython parser."""
    monkeypatch.setenv('ALLEL2_BACKEND', backend)
    caplog.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            fields, samples, headers, it = iter_vcf_chunks(path, **kwargs)
            chunks = list(it)
    used_rust = any(RUST in r.getMessage() for r in caplog.records)
    return (fields, samples, chunks), [str(w.message) for w in caught], used_rust


def assert_same_chunks(expected, actual):
    e_fields, e_samples, e_chunks = expected
    a_fields, a_samples, a_chunks = actual
    assert e_fields == a_fields
    np.testing.assert_array_equal(e_samples, a_samples)
    assert len(e_chunks) == len(a_chunks)
    for (e, e_len, e_chrom, e_pos), (a, a_len, a_chrom, a_pos) in zip(e_chunks, a_chunks):
        assert (e_len, e_chrom, e_pos) == (a_len, a_chrom, a_pos)
        assert sorted(e) == sorted(a)
        for k in e:
            assert e[k].dtype == a[k].dtype, k
            assert e[k].shape == a[k].shape, k
            np.testing.assert_array_equal(e[k], a[k], err_msg=k)


def assert_rust_matches_cython(path, monkeypatch, caplog, **kwargs):
    cython, cython_warnings, _ = _chunks(path, 'cython', monkeypatch, caplog, **kwargs)
    rust, rust_warnings, used_rust = _chunks(path, 'rust', monkeypatch, caplog, **kwargs)
    assert used_rust, caplog.text
    assert_same_chunks(cython, rust)
    # the Rust parser does not run tabix, nor warns about it
    assert [w for w in cython_warnings if 'tabix' not in w] == rust_warnings


@pytest.fixture
def one_line_batches(monkeypatch):
    # the Rust parser parses each line in a batch of its own, in parallel
    monkeypatch.setattr(allel2.io.vcf_read, '_RUST_BATCH_BYTES', 1)


@pytest.mark.parametrize('fn', FILES)
@pytest.mark.parametrize('fields', FIELDS)
@pytest.mark.parametrize('chunk_length', [1, 2, 3, 2**16])
def test_test_files(fn, fields, chunk_length, monkeypatch, caplog):
    path = os.path.join(DATA, fn)
    assert_rust_matches_cython(path, monkeypatch, caplog,
                               fields=_fields(path, fields), chunk_length=chunk_length)


@pytest.mark.parametrize('fn', FILES)
@pytest.mark.parametrize('fields', [None, '*', ['POS', 'GT']])
@pytest.mark.parametrize('chunk_length', [1, 3, 2**16])
@pytest.mark.parametrize('n_threads', [1, 4])
def test_threads(fn, fields, chunk_length, n_threads, one_line_batches, monkeypatch,
                 caplog):
    # chunks are the same, in the same order, whatever the number of threads
    assert_rust_matches_cython(os.path.join(DATA, fn), monkeypatch, caplog,
                               fields=fields, chunk_length=chunk_length,
                               n_threads=n_threads)


@pytest.mark.parametrize('name', sorted(EDGE_CASES))
@pytest.mark.parametrize('n_threads', [1, 3])
def test_threads_edge_cases(name, n_threads, edge_vcf, one_line_batches, monkeypatch,
                            caplog):
    assert_rust_matches_cython(edge_vcf(name), monkeypatch, caplog,
                               fields=FIXED_FIELDS + EDGE_FILTERS + ['GT'], chunk_length=2,
                               n_threads=n_threads)


def test_n_threads_is_the_last_parameter():
    # positional arguments keep their meaning
    import inspect
    for reader in (read_vcf, iter_vcf_chunks, vcf_to_npz, vcf_to_hdf5, vcf_to_zarr,
                   vcf_to_dataframe, vcf_to_csv, vcf_to_recarray):
        names = [p.name for p in inspect.signature(reader).parameters.values()
                 if p.kind is not inspect.Parameter.VAR_KEYWORD]
        assert 'n_threads' == names[-1], reader.__name__


READERS = [
    lambda path, tmp, **kw: read_vcf(path, **kw),
    lambda path, tmp, **kw: list(iter_vcf_chunks(path, **kw)[3]),
    lambda path, tmp, **kw: vcf_to_npz(path, str(tmp / 'out.npz'), **kw),
    lambda path, tmp, **kw: vcf_to_hdf5(path, str(tmp / 'out.h5'), **kw),
    lambda path, tmp, **kw: vcf_to_zarr(path, str(tmp / 'out.zarr'), **kw),
    lambda path, tmp, **kw: vcf_to_dataframe(path, **kw),
    lambda path, tmp, **kw: vcf_to_csv(path, str(tmp / 'out.csv'), **kw),
    lambda path, tmp, **kw: vcf_to_recarray(path, **kw),
]


@pytest.mark.parametrize('reader', READERS)
@pytest.mark.parametrize('n_threads', [0, -1, 1.5, '2', True])
@pytest.mark.parametrize('backend', ['rust', 'cython'])
def test_invalid_n_threads(reader, n_threads, backend, tmp_path, monkeypatch):
    monkeypatch.setenv('ALLEL2_BACKEND', backend)
    with pytest.raises(ValueError, match='n_threads'):
        reader(os.path.join(DATA, 'sample.vcf'), tmp_path, n_threads=n_threads)


@pytest.mark.parametrize('reader', READERS)
def test_readers_take_n_threads(reader, tmp_path, monkeypatch, caplog):
    monkeypatch.delenv('ALLEL2_BACKEND', raising=False)
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        reader(os.path.join(DATA, 'sample.vcf'), tmp_path, n_threads=2)
    assert RUST + ' on 2 threads' in caplog.text


@pytest.mark.parametrize('omp, n_threads, expected', [
    ('2', None, 2),
    ('2', 3, 3),
    (None, 3, 3),
    ('1', None, 1),
])
def test_omp_num_threads(omp, n_threads, expected, monkeypatch, caplog):
    if omp is None:
        monkeypatch.delenv('OMP_NUM_THREADS', raising=False)
    else:
        monkeypatch.setenv('OMP_NUM_THREADS', omp)
    _read(os.path.join(DATA, 'sample.vcf'), monkeypatch, caplog, n_threads=n_threads)
    assert '%s on %d threads' % (RUST, expected) in caplog.text


def test_default_n_threads(monkeypatch, caplog):
    # the cores this process may run on
    monkeypatch.delenv('OMP_NUM_THREADS', raising=False)
    if hasattr(os, 'sched_getaffinity'):
        expected = len(os.sched_getaffinity(0))
    else:
        expected = os.process_cpu_count() if hasattr(os, 'process_cpu_count') \
            else os.cpu_count()
    _read(os.path.join(DATA, 'sample.vcf'), monkeypatch, caplog)
    assert '%s on %d threads' % (RUST, expected) in caplog.text


@pytest.mark.parametrize('fn', ['sample.vcf', 'test63.vcf', 'test1.vcf', 'test32.vcf'])
@pytest.mark.parametrize('options', GT_OPTIONS)
def test_gt_options(fn, options, monkeypatch, caplog):
    assert_rust_matches_cython(os.path.join(DATA, fn), monkeypatch, caplog,
                               fields=['variants/POS', 'calldata/GT'], chunk_length=2,
                               **options)


STRING_FIELDS = ['CHROM', 'ID', 'REF', 'ALT']

FIXED_OPTIONS = (
    [{'types': {f: t for f in STRING_FIELDS}}
     for t in ('S1', 'S2', 'S20', 'O', 'T', object)] +
    [{'types': {'REF': 'S4'}},
     {'types': {'ALT': 'S1'}, 'alt_number': 1},
     {'alt_number': 1},
     {'alt_number': 2},
     {'alt_number': 5},
     {'numbers': {'ALT': 2}},
     {'types': {'QUAL': 'f8'}},
     {'fills': {'QUAL': -1}}]
)


@pytest.mark.parametrize('fn', ['sample.vcf', 'altlen.vcf', 'test32.vcf', 'test16.vcf',
                                'sample.utf8.vcf'])
@pytest.mark.parametrize('options', FIXED_OPTIONS)
def test_fixed_options(fn, options, monkeypatch, caplog):
    path = os.path.join(DATA, fn)
    assert_rust_matches_cython(path, monkeypatch, caplog,
                               fields=_fields(path, 'filters'), chunk_length=2, **options)


@pytest.mark.parametrize('options', FIXED_OPTIONS)
def test_fixed_options_computed_only(options, monkeypatch, caplog):
    # numalt, altlen and is_snp without ALT nor REF
    assert_rust_matches_cython(os.path.join(DATA, 'altlen.vcf'), monkeypatch, caplog,
                               fields=['numalt', 'altlen', 'is_snp'], **options)


@pytest.mark.parametrize('fn', FILES)
@pytest.mark.parametrize('chunk_length', [1, 3, 2**16])
@pytest.mark.parametrize('samples', [None, []])
def test_all_fields(fn, chunk_length, samples, monkeypatch, caplog):
    # without samples, '*' is every variants field; none falls back
    assert_rust_matches_cython(os.path.join(DATA, fn), monkeypatch, caplog,
                               fields='*', samples=samples, chunk_length=chunk_length)
    assert FALLBACK not in caplog.text


FORMAT_INTEGERS = ['GQ', 'DP', 'HQ', 'AD', 'PL', 'MQ0']
FORMAT_STRINGS = ['ZZ', 'GTXT']

FORMAT_OPTIONS = (
    [{'types': {'calldata/' + f: t for f in FORMAT_INTEGERS}}
     for t in ('i1', 'i8', 'u1', 'u2', 'u8', 'f4', 'f8', 'S2', 'O', 'genotype/i2',
               'genotype_ac/u1')] +
    [{'types': {'GT': t}} for t in ('genotype_ac/i1', 'genotype_ac/u8', 'i2', 'u4',
                                    'f4', 'S3', 'O', 'T')] +
    [{'types': {'calldata/AB': t, 'GQ': t}} for t in ('f8', 'i4', 'S5')] +
    [{'types': {f: t for f in FORMAT_STRINGS}} for t in ('S1', 'S4', 'O', 'T')] +
    [{'numbers': {'GT': 3, 'HQ': 1, 'AD': 4, 'calldata/DP': 2, 'PL': 1, 'ZZ': 2}},
     {'numbers': {'GT': 1, 'GQ': 0}},
     {'types': {'GT': 'genotype_ac/i1'}, 'numbers': {'GT': 3}},
     {'alt_number': 1},
     {'alt_number': 4},
     {'fills': {'GT': -9, 'GQ': 9, 'HQ': -2, 'calldata/AB': 0.5, 'calldata/DP': 3}},
     {'types': {'GT': 'genotype_ac/i2'}, 'fills': {'GT': 5}},
     {'samples': ['NA00003', 'NA00001']},
     {'samples': [1]},
     {'samples': []}]
)


@pytest.mark.parametrize('fn', FILES)
@pytest.mark.parametrize('options', FORMAT_OPTIONS)
def test_format_options(fn, options, monkeypatch, caplog):
    path = os.path.join(DATA, fn)
    assert_rust_matches_cython(path, monkeypatch, caplog,
                               fields=_fields(path, 'formats'), chunk_length=2, **options)


FORMAT_HEADER = (b'##fileformat=VCFv4.2\n'
                 b'##FORMAT=<ID=GT,Number=1,Type=String,Description="">\n'
                 b'##FORMAT=<ID=DP,Number=1,Type=Integer,Description="">\n'
                 b'##FORMAT=<ID=AD,Number=R,Type=Integer,Description="">\n'
                 b'##FORMAT=<ID=GL,Number=G,Type=Float,Description="">\n'
                 b'##FORMAT=<ID=FT,Number=1,Type=String,Description="">\n'
                 b'#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\tS3\n')

# no more samples than in the header: the Cython parser crashes on them when
# the first FORMAT key is not read (see test_extra_samples_unrequested_format)
FORMAT_LINES = (FORMAT_HEADER +
                b'2L\t1\t.\tA\tC\t.\t.\t.\tGT:DP:AD:GL:FT\t0/1:5:1,4:-1,-2,-3:PASS\t'
                b'1|1:.:.:.:.\t./.:::\n'
                b'2L\t2\t.\tA\tC\t.\t.\t.\tDP:XX:GT\t3:x:0/1\t4:y\t:z:1/0/2\n'
                b'2L\t3\t.\tA\tC\t.\t.\t.\tGT:DP:DP:FT\t0:3:4:ABCDEF\t1:5\t.:6:.:X\n'
                b'2L\t4\t.\tA\tC\t.\t.\t.\t.:GT:.\t0/1\t1/1:a\t0|0\n'
                b'2L\t5\t.\tA\tC\t.\t.\t.\tGT:AD\t0/1:1,2,3,4\t-1/0:x,2\t2/3:99999999999,-129\n'
                b'2L\t6\t.\tA\tC\t.\t.\t.\tXX:GT\t1:0/1\t2:1/1\t3:0/0\n'
                b'2L\t7\t.\tA\tC\t.\t.\t.\tGT:GL\t0/1:1e3x,inf\t1/1:nan,-0x1p3\t0/0: 4,.\n'
                b'2L\t8\t.\tA\tC\t.\t.\t.\tGT:FT\t0/1:\xc3\xa9,b\t1/1:,c\t0/0\n'
                b'2L\t9\t.\tA\tC\t.\t.\t.\tGT\n'
                b'2L\t10\t.\tA\tC\t.\t.\t.\tGT\t\n'
                b'2L\t11\t.\tA\tC\t.\t.\t.\n'
                b'2L\t12\t.\tA\tC\t.\t.\t.\tGT:DP\t0/1:2\t1/1:3\n'
                b'2L\t13\t.\tA\tC\t.\t.\t.\tGT:DP\t1/2/3/4:7\t0|1|1:8\t2:9')


@pytest.mark.parametrize('fields', [
    ['GT', 'calldata/DP', 'AD', 'GL', 'FT'],
    ['calldata/DP', 'FT'],
    ['GT', 'calldata/XX'],
    ['AD'],
])
@pytest.mark.parametrize('options', [
    {},
    {'types': {'GT': 'genotype_ac/i1', 'FT': 'S2', 'calldata/DP': 'u1'}},
    {'types': {'GT': 'genotype_ac/u2', 'AD': 'genotype/i1', 'GL': 'f8'},
     'numbers': {'GT': 3}},
    {'numbers': {'GT': 3, 'calldata/DP': 2, 'AD': 1, 'FT': 2}, 'alt_number': 2},
    {'fills': {'GT': 7, 'calldata/DP': 0, 'GL': 1.5}},
    {'samples': ['S3', 'S1']},
    {'samples': ['S2']},
])
@pytest.mark.parametrize('gz', [False, True])
def test_format_edge_cases(fields, options, gz, tmp_path, monkeypatch, caplog):
    if gz:
        import gzip
        path = tmp_path / 'format.vcf.gz'
        with gzip.open(path, 'wb') as f:
            f.write(FORMAT_LINES)
    else:
        path = tmp_path / 'format.vcf'
        path.write_bytes(FORMAT_LINES)
    assert_rust_matches_cython(str(path), monkeypatch, caplog, fields=fields,
                               chunk_length=3, **options)


INFO_INTEGERS = ['NS', 'AN', 'AC', 'DP', 'BaseCounts', 'ALTLEN']
INFO_FLOATS = ['AF', 'ABHet', 'FS', 'Dels']
INFO_STRINGS = ['AA', 'EFF', 'ANN']

INFO_OPTIONS = (
    [{'types': {f: t for f in INFO_INTEGERS}}
     for t in ('i1', 'i2', 'i8', 'u1', 'u2', 'u4', 'u8', 'f4', 'f8', bool)] +
    [{'types': {f: t for f in INFO_FLOATS}} for t in ('f8', 'i4', 'u1', bool)] +
    [{'types': {f: t for f in INFO_STRINGS}} for t in ('S1', 'S3', 'S50', 'O', 'T')] +
    [{'types': {'DB': 'i4', 'H2': 'S2', 'DS': 'f4'}},
     {'numbers': {'AC': 1, 'AF': 3, 'DP': 2, 'AA': 2, 'BaseCounts': 2, 'EFF': 4, 'DB': 2}},
     {'numbers': {'AA': 0, 'DP': 0}},
     {'alt_number': 1},
     {'alt_number': 3},
     {'fills': {'DP': -9, 'AF': -1.5, 'AN': 99, 'AC': 7, 'FS': 2}},
     {'fills': {'AF': 2.5, 'ABHet': -1, 'AA': 'x', 'DB': 1}},
     {'types': {'DP': 'u1', 'AF': 'f8'}, 'fills': {'DP': 3, 'AF': 0}}]
)


@pytest.mark.parametrize('fn', FILES)
@pytest.mark.parametrize('options', INFO_OPTIONS)
def test_info_options(fn, options, monkeypatch, caplog):
    path = os.path.join(DATA, fn)
    assert_rust_matches_cython(path, monkeypatch, caplog,
                               fields=_fields(path, 'infos'), chunk_length=2, **options)


INFO_HEADER = (b'##fileformat=VCFv4.2\n'
               b'##INFO=<ID=DP,Number=1,Type=Integer,Description="">\n'
               b'##INFO=<ID=AC,Number=A,Type=Integer,Description="">\n'
               b'##INFO=<ID=AF,Number=.,Type=Float,Description="">\n'
               b'##INFO=<ID=AA,Number=1,Type=String,Description="">\n'
               b'##INFO=<ID=CS,Number=2,Type=String,Description="">\n'
               b'##INFO=<ID=DB,Number=0,Type=Flag,Description="">\n'
               b'#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n')

INFO_LINES = (INFO_HEADER +
              b'2L\t1\t.\tA\tC,G\t.\t.\tDP=5;AC=1,2;AF=0.5,0.25;AA=T;CS=ab,cd;DB\n'
              b'2L\t2\t.\tA\tC\t.\t.\t.;DP=5\n'
              b'2L\t3\t.\tA\tC\t.\t.\tDP=5;;AF=.,0.3;;\n'
              b'2L\t4\t.\tA\tC\t.\t.\t=5;DP=7;=;AA=x=y\n'
              b'2L\t5\t.\tA\tC\t.\t.\tDP;AF=;AA=;CS=,;DB=0\n'
              b'2L\t6\t.\tA\tC\t.\t.\tDP=1,2,3;AC=x,4,5;AF=1e3x,inf,nan\n'
              b'2L\t7\t.\tA\tC\t.\t.\tAA=ABCDEF;AA=X;CS=long,str;CS=,y;DP=3;DP=.\n'
              b'2L\t8\t.\tA\tC\t.\t.\tXX=1;YY;DP= 12;AF=-0x1p-2\n'
              b'2L\t9\t.\tA\tC\t.\t.\tDP=99999999999;AC=-129,300\n'
              b'2L\t10\t.\tA\tC\t.\t.\t\n'
              b'2L\t11\t.\tA\tC\t.\t.\t\t\n'
              b'2L\t12\t.\tA\tC\t.\n'
              b'2L\t13\t.\tA\tC\t.\t.\tAA=\xc3\xa9;DB\tGT\n'
              b'2L\t14\t.\tA\tC\t.\t.\tDP=4;AA=end')


@pytest.mark.parametrize('fields', [
    ['DP', 'AC', 'AF', 'AA', 'CS', 'DB'],
    ['POS', 'XX', 'YY'],
    ['AA'],
    ['DB', 'DP'],
])
@pytest.mark.parametrize('options', [
    {},
    {'types': {'AA': 'S3', 'CS': 'S2'}},
    {'types': {'DP': 'i1', 'AC': 'u1', 'AF': 'f8'}, 'fills': {'DP': 0}},
    {'numbers': {'DP': 2, 'AF': 1, 'CS': 1}, 'alt_number': 3},
])
@pytest.mark.parametrize('gz', [False, True])
def test_info_edge_cases(fields, options, gz, tmp_path, monkeypatch, caplog):
    if gz:
        import gzip
        path = tmp_path / 'info.vcf.gz'
        with gzip.open(path, 'wb') as f:
            f.write(INFO_LINES)
    else:
        path = tmp_path / 'info.vcf'
        path.write_bytes(INFO_LINES)
    assert_rust_matches_cython(str(path), monkeypatch, caplog, fields=fields,
                               chunk_length=3, **options)


@pytest.mark.parametrize('samples', [['NA00002'], ['NA00003', 'NA00001'], [], [0, 2]])
def test_samples(samples, monkeypatch, caplog):
    assert_rust_matches_cython(os.path.join(DATA, 'sample.vcf'), monkeypatch, caplog,
                               fields=['calldata/GT'], samples=samples)


@pytest.mark.parametrize('name', sorted(EDGE_CASES))
@pytest.mark.parametrize('gz', [False, True])
@pytest.mark.parametrize('options', [{}, {'numbers': {'calldata/GT': 3}},
                                     {'types': {'calldata/GT': 'u1'}}])
def test_edge_cases(name, gz, options, edge_vcf, monkeypatch, caplog):
    assert_rust_matches_cython(edge_vcf(name, gz), monkeypatch, caplog,
                               fields=['variants/POS', 'calldata/GT'], chunk_length=2,
                               **options)


EDGE_FILTERS = ['FILTER_q10', 'FILTER_s50', 'FILTER_PASSq10']


@pytest.mark.parametrize('name', sorted(EDGE_CASES))
@pytest.mark.parametrize('gz', [False, True])
@pytest.mark.parametrize('options', [{}, {'alt_number': 2},
                                     {'types': {f: 'S2' for f in STRING_FIELDS}}])
def test_edge_cases_fixed(name, gz, options, edge_vcf, monkeypatch, caplog):
    assert_rust_matches_cython(edge_vcf(name, gz), monkeypatch, caplog,
                               fields=FIXED_FIELDS + EDGE_FILTERS, chunk_length=2,
                               **options)


def test_read_vcf(monkeypatch, caplog):
    # read_vcf concatenates the chunks
    path = os.path.join(DATA, 'sample.vcf')
    monkeypatch.setenv('ALLEL2_BACKEND', 'rust')
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        callset = read_vcf(path, fields=['samples', 'POS', 'GT'], chunk_length=2)
    assert RUST in caplog.text
    assert ['NA00001', 'NA00002', 'NA00003'] == callset['samples'].tolist()
    assert [111, 112, 14370] == callset['variants/POS'][:3].tolist()
    assert [[0, -1], [0, 1], [0, 2]] == callset['calldata/GT'][-1].tolist()


# which parser reads the file

def _reason(caplog):
    messages = [r.getMessage() for r in caplog.records if FALLBACK in r.getMessage()]
    assert len(messages) == 1, caplog.text
    assert all(r.levelno == logging.DEBUG for r in caplog.records
               if FALLBACK in r.getMessage())
    return messages[0]


def _read(input, monkeypatch, caplog, backend=None, **kwargs):
    if backend is None:
        monkeypatch.delenv('ALLEL2_BACKEND', raising=False)
    else:
        monkeypatch.setenv('ALLEL2_BACKEND', backend)
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        return read_vcf(input, **kwargs)


def test_rust_is_the_default(monkeypatch, caplog):
    _read(os.path.join(DATA, 'sample.vcf.gz'), monkeypatch, caplog,
          fields=['POS', 'GT'])
    assert RUST in caplog.text
    assert FALLBACK not in caplog.text


def test_fallback_backend_variable(monkeypatch, caplog):
    callset = _read(os.path.join(DATA, 'sample.vcf'), monkeypatch, caplog,
                    backend='cython', fields=['POS', 'GT'])
    assert 'ALLEL2_BACKEND=cython' in _reason(caplog)
    assert 9 == len(callset['variants/POS'])


def test_unknown_backend(monkeypatch, caplog):
    with pytest.raises(ValueError, match='ALLEL2_BACKEND'):
        _read(os.path.join(DATA, 'sample.vcf'), monkeypatch, caplog, backend='fortran')


def test_fallback_transformers(monkeypatch, caplog):
    callset = _read(os.path.join(DATA, 'ann.vcf'), monkeypatch, caplog,
                    fields=['POS', 'ANN'], transformers=[ANNTransformer()])
    assert 'transformers' in _reason(caplog)
    assert 'variants/ANN_Allele' in callset


def test_fallback_file_object(monkeypatch, caplog):
    with open(os.path.join(DATA, 'sample.vcf'), 'rb') as f:
        data = f.read()
    callset = _read(io.BytesIO(data), monkeypatch, caplog, fields=['POS', 'GT'])
    assert 'file object' in _reason(caplog)
    assert 9 == len(callset['variants/POS'])


# regions, read from the index or by scanning

REGION_FILES = ['sample.vcf.gz', 'sample.vcf', 'test54.vcf.gz', 'region.vcf',
                'region.vcf.gz', 'region_csi.vcf.gz', 'gzip']

REGIONS = ['20', '20:14370-17330', '20:1110696-1230237', '19:111-112', 'X', 'Y',
           '2L', '2L:100-200', '2L:96-150', '2L:100-100', '2L:200-0', '2L:0-150',
           '2L:150-100', '2L:16000-80000', '2L:1000000-1000000', '2R:1-1', '3L',
           'X:15000-25000', 'chrZ']


@pytest.fixture
def region_vcf(tmp_path):
    def path(fn):
        if fn == 'gzip':
            # gzip, but not BGZF
            import gzip
            path = tmp_path / 'region.gzip.vcf.gz'
            with open(os.path.join(DATA, 'region.vcf'), 'rb') as f, \
                    gzip.open(path, 'wb') as g:
                g.write(f.read())
            return str(path)
        return os.path.join(DATA, fn)
    return path


@pytest.mark.parametrize('fn', REGION_FILES)
@pytest.mark.parametrize('region', REGIONS)
@pytest.mark.parametrize('tabix', ['tabix', None])
@pytest.mark.parametrize('n_threads', [1, 3])
def test_regions(fn, region, tabix, n_threads, region_vcf, monkeypatch, caplog):
    assert_rust_matches_cython(region_vcf(fn), monkeypatch, caplog, region=region,
                               tabix=tabix, fields='*', chunk_length=2,
                               n_threads=n_threads)


@pytest.mark.parametrize('fn, tabix, indexed', [
    ('region.vcf.gz', 'tabix', True),
    ('region_csi.vcf.gz', 'tabix', True),
    ('sample.vcf.gz', '/no/such/tabix', True),
    ('region.vcf.gz', None, False),
    ('region.vcf', 'tabix', False),
    ('gzip', 'tabix', False),
])
def test_region_from_index(fn, tabix, indexed, region_vcf, monkeypatch, caplog):
    # the index is read unless tabix is None, if the file has one
    _read(region_vcf(fn), monkeypatch, caplog, region='2L', tabix=tabix)
    assert RUST in caplog.text
    assert indexed == ('region from the index' in caplog.text)


@pytest.mark.parametrize('region', ['20:1', '20:a-b', '20:1-2-3'])
def test_bad_region(region, monkeypatch, caplog):
    for backend in 'cython', 'rust':
        with pytest.raises(ValueError, match='bad region string|invalid literal'):
            _read(os.path.join(DATA, 'sample.vcf.gz'), monkeypatch, caplog,
                  backend=backend, region=region, tabix=None)


@pytest.mark.parametrize('tabix', ['tabix', '/no/such/tabix', None])
def test_regions_without_tabix(tabix, tmp_path, monkeypatch, caplog):
    # no tabix binary needed, whatever the tabix parameter
    monkeypatch.setenv('PATH', str(tmp_path))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        callset = _read(os.path.join(DATA, 'region.vcf.gz'), monkeypatch, caplog,
                        region='2L:96-150', tabix=tabix, fields=['POS', 'ID'])
    assert RUST in caplog.text
    assert FALLBACK not in caplog.text
    assert [99, 100, 150] == callset['variants/POS'].tolist()


@pytest.mark.parametrize('n_threads', [1, 4])
def test_gzip_not_bgzf(n_threads, region_vcf, monkeypatch, caplog):
    assert_rust_matches_cython(region_vcf('gzip'), monkeypatch, caplog, fields='*',
                               chunk_length=3, n_threads=n_threads)


def test_default_fields(monkeypatch, caplog):
    # read_vcf with its default fields does not fall back
    callset = _read(os.path.join(DATA, 'sample.vcf'), monkeypatch, caplog)
    assert RUST in caplog.text
    assert FALLBACK not in caplog.text
    assert ['19', '19', '20'] == callset['variants/CHROM'][:3].tolist()
    assert np.dtypes.StringDType() == callset['variants/ALT'].dtype
    assert [True, False] == callset['variants/FILTER_PASS'][2:4].tolist()


def test_exclude_and_rename_fields(monkeypatch, caplog):
    path = os.path.join(DATA, 'sample.vcf')
    kwargs = dict(exclude_fields=['ID', 'QUAL'],
                  rename_fields={'variants/CHROM': 'variants/chromosome',
                                 'variants/FILTER_PASS': 'variants/pass'})
    expected = _read(path, monkeypatch, caplog, backend='cython', **kwargs)
    actual = _read(path, monkeypatch, caplog, backend='rust', **kwargs)
    assert RUST in caplog.text
    assert sorted(expected) == sorted(actual)
    assert 'variants/chromosome' in actual and 'variants/ID' not in actual
    for k in expected:
        assert expected[k].dtype == actual[k].dtype, k
        np.testing.assert_array_equal(expected[k], actual[k], err_msg=k)


def test_extra_samples_unrequested_format(tmp_path, monkeypatch, caplog):
    # the Cython parser crashes here, when the first FORMAT key is not read
    path = tmp_path / 'extra.vcf'
    path.write_bytes(FORMAT_HEADER +
                     b'2L\t6\t.\tA\tC\t.\t.\t.\tXX:GT\t1:0/1\t2:1/1\t3:0/0\t4:0/1\n')
    with pytest.warns(UserWarning, match='more samples than given in header') as caught:
        callset = _read(str(path), monkeypatch, caplog, fields=['GT'])
    assert RUST in caplog.text
    assert ['more samples than given in header; field: CALLDATA; variant: 0 (2L\x00:6); '
            'sample: 3:0 (unknown:unknown)'] == [str(w.message) for w in caught]
    assert [[[0, 1], [1, 1], [0, 0]]] == callset['calldata/GT'].tolist()


def test_all_fields_without_fallback(monkeypatch, caplog):
    callset = _read(os.path.join(DATA, 'sample.vcf'), monkeypatch, caplog, fields='*')
    assert RUST in caplog.text
    assert FALLBACK not in caplog.text
    assert (9, 3) == callset['calldata/GQ'].shape


@pytest.mark.parametrize('types, field', [
    ({'GT': 'genotype/f4'}, 'calldata/GT'),
    ({'GT': 'genotype_ac/f4'}, 'calldata/GT'),
    ({'GT': '>i2'}, 'calldata/GT'),
    ({'GT': bool}, 'calldata/GT'),
    ({'GT': 'f2'}, 'calldata/GT'),
    ({'GT': 'U3'}, 'calldata/GT'),
    ({'REF': 'S'}, 'variants/REF'),
])
def test_fallback_unsupported_type(types, field, monkeypatch, caplog):
    _read(os.path.join(DATA, 'sample.vcf'), monkeypatch, caplog,
          fields=['POS', 'GT', 'CHROM', 'REF'], types=types)
    assert field in _reason(caplog)


@pytest.mark.parametrize('field, kwargs', [
    ('variants/DP', {'types': {'DP': 'f2'}}),
    ('variants/DP', {'types': {'DP': 'U3'}}),
    ('variants/AA', {'types': {'AA': 'S'}}),
    ('variants/DP', {'types': {'DP': 'c8'}}),
    ('variants/DP', {'types': {'DP': '>i4'}}),
    ('variants/AF', {'types': {'AF': '>f8'}}),
    ('variants/DP', {'fills': {'DP': 1.5}}),
    ('variants/AF', {'fills': {'AF': 'x'}}),
    ('variants/DP', {'fills': {'DP': None}}),
    ('variants/DP', {'numbers': {'DP': -1}}),
    ('calldata/GQ', {'types': {'calldata/GQ': 'f2'}}),
    ('calldata/GQ', {'types': {'calldata/GQ': bool}}),
    ('calldata/GQ', {'types': {'calldata/GQ': '>u2'}}),
    ('calldata/GQ', {'fills': {'calldata/GQ': None}}),
    ('calldata/GQ', {'fills': {'calldata/GQ': 1.5}}),
    ('calldata/GQ', {'numbers': {'calldata/GQ': -1}}),
    ('calldata/GT', {'fills': {'GT': None}}),
])
def test_fallback_uncovered_type(field, kwargs, monkeypatch, caplog):
    # INFO fields that the Rust parser does not cover are read by the Cython
    # one, which skips some of them and rejects others
    path = os.path.join(DATA, 'sample.vcf')
    fields = ['POS', 'DP', 'AF', 'AA', 'GT', 'calldata/GQ']
    monkeypatch.setenv('ALLEL2_BACKEND', 'cython')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            read_vcf(path, fields=fields, **kwargs)
        expected = None
    except Exception as e:
        expected = type(e)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if expected is None:
            _read(path, monkeypatch, caplog, fields=fields, **kwargs)
        else:
            with pytest.raises(expected):
                _read(path, monkeypatch, caplog, fields=fields, **kwargs)
    assert field in _reason(caplog)


def test_flag_type_warning(monkeypatch, caplog):
    # like the Cython parser, a field of number 0 is a flag
    for backend in 'cython', 'rust':
        with pytest.warns(UserWarning, match='cannot have non-bool dtype'):
            callset = _read(os.path.join(DATA, 'sample.vcf'), monkeypatch, caplog,
                            backend=backend, fields=['DB'], types={'DB': 'i4'})
        assert np.dtype(bool) == callset['variants/DB'].dtype


def test_fallback_rejected_type(monkeypatch, caplog):
    # the Cython parser rejects strings that are neither bytes nor objects
    with pytest.raises(ValueError, match='expected string'):
        _read(os.path.join(DATA, 'sample.vcf'), monkeypatch, caplog,
              fields=['POS', 'CHROM'], types={'CHROM': 'U10'})
    assert 'variants/CHROM' in _reason(caplog)


def test_pos_type_warning(monkeypatch, caplog):
    # like the Cython parser, POS is always int32
    for backend in 'cython', 'rust':
        with pytest.warns(UserWarning, match='only int32 supported for POS'):
            callset = _read(os.path.join(DATA, 'sample.vcf'), monkeypatch, caplog,
                            backend=backend, fields=['POS'], types={'POS': 'i8'})
        assert np.dtype('int32') == callset['variants/POS'].dtype


def test_qual_type_warning(monkeypatch, caplog):
    # like the Cython parser, QUAL is always float32
    for backend in 'cython', 'rust':
        with pytest.warns(UserWarning, match='only float32 supported for QUAL'):
            callset = _read(os.path.join(DATA, 'sample.vcf'), monkeypatch, caplog,
                            backend=backend, fields=['QUAL'], types={'QUAL': 'f8'})
        assert np.dtype('float32') == callset['variants/QUAL'].dtype


@pytest.mark.parametrize('backend', ['cython', 'rust'])
def test_invalid_utf8(backend, tmp_path, monkeypatch, caplog):
    # strings that are not UTF-8 raise the same error with both parsers
    path = tmp_path / 'bad.vcf'
    path.write_bytes(HEADER + b'2L\t1\tid\xff\tA\tC\t.\t.\t.\n')
    with pytest.raises(UnicodeDecodeError):
        _read(str(path), monkeypatch, caplog, backend=backend, fields=['ID'])


@pytest.mark.parametrize('backend', ['cython', 'rust'])
@pytest.mark.parametrize('fields', [['altlen'], ['altlen', 'ALT']])
def test_altlen_shape(backend, fields, monkeypatch, caplog):
    # one ALT value per variant gives a 1-D altlen, with or without ALT
    callset = _read(os.path.join(DATA, 'altlen.vcf'), monkeypatch, caplog,
                    backend=backend, fields=fields, alt_number=1)
    assert 1 == callset['variants/altlen'].ndim


@pytest.mark.parametrize('fn', ['sample.vcf', 'test32.vcf', 'test1.vcf'])
@pytest.mark.parametrize('options', FORMAT_OPTIONS[::3] + INFO_OPTIONS[::3] + FIXED_OPTIONS[::3])
def test_threads_options(fn, options, one_line_batches, monkeypatch, caplog):
    path = os.path.join(DATA, fn)
    assert_rust_matches_cython(path, monkeypatch, caplog, fields='*', chunk_length=3,
                               n_threads=4, **options)
