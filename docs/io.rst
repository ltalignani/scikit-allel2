Input/output utilities
======================

Variant Call Format (VCF)
-------------------------

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