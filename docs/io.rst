Input/output utilities
======================

Variant Call Format (VCF)
-------------------------

.. _vcf_parsers:

VCF parsers
~~~~~~~~~~~

:func:`allel2.read_vcf` and the other VCF readers below parse VCF files with a
parser written in Rust, which reads lines on several threads. The `n_threads`
parameter sets how many; by default, all the CPU cores the process may run on,
at most the value of the ``OMP_NUM_THREADS`` environment variable if it is set.
Pass ``n_threads=1`` to read on one thread, for instance in each task of a Dask
cluster.

A file compressed with bgzip is also decompressed on several threads. With
`region`, a bgzip-compressed file is read from its index (``.tbi`` or
``.csi``), without the ``tabix`` program; a file without an index, or a read
with ``tabix=None``, is scanned from its start. As before, only the variants
whose position (POS) is within the region are returned.

Some reads still go to the former parser, written in Cython, which gives the
same results on one thread:

* with `transformers`, such as :class:`allel2.ANNTransformer`;
* from a file-like object rather than a path;
* with a type, `numbers` or `fills` value that the Rust parser does not handle,
  for instance a field of type ``'f2'``, ``'U10'``, of a non-native byte order,
  or a boolean FORMAT field.

Set the ``ALLEL2_BACKEND`` environment variable to ``cython`` to use the Cython
parser for every read, for instance to compare the two parsers or to work around
a bug; ``rust``, the default, uses the Rust parser where it can. The parser
chosen for each read, and the reason for using the Cython one, are logged at the
debug level by the ``allel2.io.vcf_read`` logger::

    >>> import logging
    >>> logging.basicConfig()
    >>> logging.getLogger('allel2.io.vcf_read').setLevel(logging.DEBUG)

Readers
~~~~~~~

.. autofunction:: allel2.read_vcf
.. autofunction:: allel2.vcf_to_npz
.. autofunction:: allel2.vcf_to_hdf5
.. autofunction:: allel2.vcf_to_zarr
.. autofunction:: allel2.vcf_to_dataframe
.. autofunction:: allel2.vcf_to_csv
.. autofunction:: allel2.vcf_to_recarray
.. autofunction:: allel2.iter_vcf_chunks
.. autofunction:: allel2.read_vcf_headers
.. autoclass:: allel2.ANNTransformer
.. autofunction:: allel2.write_vcf

GFF3
----

.. autofunction:: allel2.gff3_to_dataframe
.. autofunction:: allel2.gff3_to_recarray
.. autofunction:: allel2.iter_gff3

Fasta
-----

.. autofunction:: allel2.write_fasta