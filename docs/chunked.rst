Chunked storage utilities
=========================

.. deprecated:: 2.0.0
    This module will be removed in scikit-allel2 3.0. See :doc:`/migrating_chunked`.

.. automodule:: allel2.chunked

Storage
-------

Zarr
~~~~

.. autoclass:: allel2.chunked.storage_zarr.ZarrStorage
.. autoclass:: allel2.chunked.storage_zarr.ZarrMemStorage
.. autoclass:: allel2.chunked.storage_zarr.ZarrTmpStorage

.. autodata:: allel2.chunked.storage_zarr.zarr_storage
    :annotation: = 'zarr'
.. autodata:: allel2.chunked.storage_zarr.zarrmem_storage
    :annotation: = 'zarrmem'
.. autodata:: allel2.chunked.storage_zarr.zarrtmp_storage
    :annotation: = 'zarrtmp'

HDF5 (h5py)
~~~~~~~~~~~

.. autoclass:: allel2.chunked.storage_hdf5.HDF5Storage
.. autoclass:: allel2.chunked.storage_hdf5.HDF5MemStorage
.. autoclass:: allel2.chunked.storage_hdf5.HDF5TmpStorage

.. autodata:: allel2.chunked.storage_hdf5.hdf5_storage
    :annotation: = 'hdf5'
.. autodata:: allel2.chunked.storage_hdf5.hdf5mem_storage
    :annotation: = 'hdf5mem'
.. autodata:: allel2.chunked.storage_hdf5.hdf5tmp_storage
    :annotation: = 'hdf5tmp'
.. autodata:: allel2.chunked.storage_hdf5.hdf5_zlib1_storage
    :annotation: = 'hdf5_zlib1'
.. autodata:: allel2.chunked.storage_hdf5.hdf5mem_zlib1_storage
    :annotation: = 'hdf5mem_zlib1'
.. autodata:: allel2.chunked.storage_hdf5.hdf5tmp_zlib1_storage
    :annotation: = 'hdf5tmp_zlib1'
.. autodata:: allel2.chunked.storage_hdf5.hdf5_lzf_storage
    :annotation: = 'hdf5_lzf'
.. autodata:: allel2.chunked.storage_hdf5.hdf5mem_lzf_storage
    :annotation: = 'hdf5mem_lzf'
.. autodata:: allel2.chunked.storage_hdf5.hdf5tmp_lzf_storage
    :annotation: = 'hdf5tmp_lzf'

.. autofunction:: allel2.chunked.storage_hdf5.h5fmem
.. autofunction:: allel2.chunked.storage_hdf5.h5ftmp

Functions
---------

.. autofunction:: allel2.chunked.core.store
.. autofunction:: allel2.chunked.core.copy
.. autofunction:: allel2.chunked.core.map_blocks
.. autofunction:: allel2.chunked.core.reduce_axis
.. autofunction:: allel2.chunked.core.amax
.. autofunction:: allel2.chunked.core.amin
.. autofunction:: allel2.chunked.core.asum
.. autofunction:: allel2.chunked.core.count_nonzero
.. autofunction:: allel2.chunked.core.compress
.. autofunction:: allel2.chunked.core.take
.. autofunction:: allel2.chunked.core.subset
.. autofunction:: allel2.chunked.core.concatenate
.. autofunction:: allel2.chunked.core.binary_op
.. autofunction:: allel2.chunked.core.copy_table
.. autofunction:: allel2.chunked.core.compress_table
.. autofunction:: allel2.chunked.core.take_table
.. autofunction:: allel2.chunked.core.concatenate_table
.. autofunction:: allel2.chunked.core.eval_table

Classes
-------

.. autoclass:: allel2.chunked.core.ChunkedArrayWrapper
.. autoclass:: allel2.chunked.core.ChunkedTableWrapper
