Migrating from scikit-allel
===========================

scikit-allel2 continues scikit-allel for Python 3.12+, NumPy 2, zarr 3 and
pandas 3. The functions and classes are the same; this page lists what you
need to change in your code. The :ref:`release notes <release_2.0.0>` list
every change in 2.0.0.

Requirements
------------

scikit-allel2 needs Python 3.12 or later, NumPy 2.0 or later and, when you
use them, zarr 3.1 or later and pandas 3.0 or later. It runs on Linux and
macOS; Windows is not supported.

Imports
-------

The package is imported as ``allel2``. Only the import line changes::

    import allel2 as allel

scikit-allel2 does not provide an ``allel`` module, so scikit-allel and
scikit-allel2 can be installed in the same environment without conflict:
``import allel`` still loads scikit-allel.

Strings read from VCF
---------------------

:func:`allel2.read_vcf` and the functions built on it now return string
fields (``CHROM``, ``ID``, ``REF``, ``ALT``, sample names, and INFO and
FORMAT fields of type ``String`` or ``Character``) as NumPy's variable-width
``StringDType`` instead of ``object`` (or ``S1`` for ``Character``)::

    >>> callset = allel.read_vcf('example.vcf', fields=['CHROM', 'REF'])
    >>> callset['variants/CHROM'].dtype
    StringDType()

Comparisons, ``np.unique``, sorting and :class:`allel2.ChromPosIndex` work as
before. Code that checks ``dtype == object`` needs updating. To get ``object``
back for a field, ask for it through ``types``::

    >>> callset = allel.read_vcf('example.vcf', fields=['CHROM'],
    ...                          types={'CHROM': object})

:func:`allel2.vcf_to_dataframe` returns string columns with pandas' ``str``
dtype.

zarr
----

:func:`allel2.vcf_to_zarr` writes zarr format 3 by default. zarr-python 2
cannot read these stores; for tools that still depend on it, write zarr
format 2::

    >>> allel.vcf_to_zarr('example.vcf', 'example.zarr', zarr_format=2)

Appending to an existing store in the other format raises an error. Stores
written by scikit-allel (zarr format 2) can still be read; their strings are
read back as ``StringDType``.

In zarr format 3, the ``compressor`` argument accepts ``numcodecs.Blosc`` and
``numcodecs.Zstd``, which are translated to their ``zarr.codecs``
equivalents, as well as ``zarr.codecs`` codecs; any other compressor raises
an error.

Chunked arrays
--------------

:mod:`allel2.chunked` and the chunked array classes are deprecated and will
be removed in scikit-allel2 3.0. See :doc:`migrating_chunked`.
