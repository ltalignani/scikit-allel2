Migrating from chunked to Dask
==============================

The :mod:`allel2.chunked` module and the chunked array classes are deprecated
since scikit-allel2 2.0.0 and will be removed in scikit-allel2 3.0. Using them
emits a ``DeprecationWarning`` naming the class or function to use instead.
Python only displays such warnings by default for code run in ``__main__``
(a script or a notebook); to see them from your own modules, run Python with
``-W default::DeprecationWarning``.
The Dask classes in :mod:`allel2.model.dask` offer the same methods, and work
on the same zarr or HDF5 data.

Class by class
--------------

=====================================  ==============================================
Deprecated                             Use instead
=====================================  ==============================================
``GenotypeChunkedArray``               :class:`allel2.GenotypeDaskArray`
``HaplotypeChunkedArray``              :class:`allel2.HaplotypeDaskArray`
``AlleleCountsChunkedArray``           :class:`allel2.AlleleCountsDaskArray`
``GenotypeAlleleCountsChunkedArray``   :class:`allel2.GenotypeAlleleCountsDaskArray`
``VariantChunkedTable``                :class:`allel2.VariantTable` (in memory) or
                                       ``dask.dataframe``
``AlleleCountsChunkedTable``           a dict of :class:`allel2.AlleleCountsDaskArray`,
                                       as returned by ``count_alleles_subpops``, or
                                       ``dask.dataframe``
Functions of :mod:`allel2.chunked`     the equivalent ``dask.array`` functions
(``copy``, ``map_blocks``, ...)
Table functions (``copy_table``,       :class:`allel2.VariantTable` (in memory) or
``compress_table``, ...)               ``dask.dataframe``
=====================================  ==============================================

Wrap the same data::

    >>> import zarr
    >>> import allel2
    >>> callset = zarr.open_group('callset.zarr', mode='r')
    >>> gt = allel2.GenotypeDaskArray(callset['calldata/GT'])  # was GenotypeChunkedArray

Computations are lazy
---------------------

A chunked class computed each result straight away and stored it, by default
in memory. A Dask class builds a task graph instead: nothing runs until you ask
for the result with ``compute()``, which returns the in-memory class::

    >>> ac = gt.count_alleles(max_allele=1)  # AlleleCountsDaskArray, nothing computed yet
    >>> ac = ac.compute()                    # AlleleCountsArray
    >>> loc = ac.is_segregating()
    >>> gt_seg = gt.compress(loc, axis=0)    # still lazy

The ``storage`` and ``blen`` arguments have no Dask equivalent. Block sizes
follow the chunks of the underlying zarr or HDF5 array; to change them, call
``rechunk`` on the ``values`` attribute, which is a ``dask.array.Array``.

Writing results to zarr
-----------------------

The chunked classes could write their results to zarr or HDF5 through a
storage. With Dask, write explicitly with ``to_zarr``, on the ``values``
attribute::

    >>> ac = gt.count_alleles(max_allele=1)
    >>> ac.values.to_zarr('ac.zarr')
    >>> ac = allel2.AlleleCountsDaskArray(zarr.open_array('ac.zarr'))

Tables
------

Neither chunked table has a Dask equivalent in scikit-allel2:

* if the variant table fits in memory, load the columns you need into a
  :class:`allel2.VariantTable`::

      >>> vt = allel2.VariantTable({'POS': callset['variants/POS'][:]}, index='POS')

* otherwise, use ``dask.dataframe``, which needs ``dask[dataframe]``
  (and pyarrow) installed;

* for allele counts per population, ``GenotypeDaskArray.count_alleles_subpops``
  returns a dict of :class:`allel2.AlleleCountsDaskArray`, one per population::

      >>> acs = gt.count_alleles_subpops({'pop1': [0, 1], 'pop2': [2, 3]})
      >>> acs['pop1'].compute()
