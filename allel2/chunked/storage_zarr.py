# -*- coding: utf-8 -*-
import operator
from functools import reduce


import numbers

import numpy as np
import zarr
from zarr.storage import MemoryStore, LocalStore
import tempfile
import atexit
import os

import numcodecs


from allel2.chunked import util as _util


def default_chunks(data, expectedlen):
    return None


def _zarr_compatible(a):
    # zarr 3 cannot store object arrays, and reads strings back as StringDType
    # anyway: strings (e.g., read with h5py) are stored as StringDType; other
    # object data is left for zarr to refuse rather than silently converted
    if getattr(a, 'dtype', None) != object:
        return a
    a = np.asarray(a)
    if all(isinstance(x, (str, bytes)) for x in a.flat):
        # decode bytes explicitly: numpy 2.0 casts them to their repr
        a = np.array([x.decode() if isinstance(x, bytes) else x for x in a.flat],
                     dtype=np.dtypes.StringDType()).reshape(a.shape)
    return a


def _column_chunks(chunks, shape):
    """Chunk shape for an array of `shape`: an int is a number of rows, and a
    tuple shorter than `shape` is completed with whole dimensions; None means
    chunks chosen by zarr."""
    if chunks is None or chunks is True:
        return None
    if isinstance(chunks, str):
        # e.g. 'auto', left to zarr
        return chunks
    if len(shape) == 0:
        return ()
    if isinstance(chunks, numbers.Integral):
        chunks = (chunks,)
    return tuple(chunks[:len(shape)]) + tuple(shape[len(chunks):])


class ZarrStorage(object):
    """Storage layer using Zarr."""

    def __init__(self, **kwargs):
        self.defaults = kwargs

    def _set_defaults(self, kwargs):

        # copy in master defaults
        for k, v in self.defaults.items():
            kwargs.setdefault(k, v)

        return kwargs

    # noinspection PyUnusedLocal
    def array(self, data, expectedlen=None, **kwargs):

        # setup
        data = _util.ensure_array_like(data)
        kwargs = self._set_defaults(kwargs)

        # determine chunks
        kwargs.setdefault('chunks', default_chunks(data, expectedlen))

        # create
        data = _zarr_compatible(data)
        kwargs['chunks'] = _column_chunks(kwargs['chunks'], np.shape(data))
        z = zarr.array(data, **kwargs)

        return z

    def table(self, data, names=None, expectedlen=None, **kwargs):

        # setup
        names, columns = _util.check_table_like(data, names=names)
        kwargs = self._set_defaults(kwargs)
        chunks = kwargs.pop('chunks', None)
        g = zarr.group(**kwargs)

        # create columns
        for n, c in zip(names, columns):
            c = _zarr_compatible(c)
            column_chunks = chunks
            if column_chunks is None:
                column_chunks = default_chunks(c, expectedlen)
            column_chunks = _column_chunks(column_chunks, c.shape)
            g.create_array(name=n, data=c,
                           chunks='auto' if column_chunks is None else column_chunks)

        # create table
        ztbl = ZarrTable(g, names=names)
        return ztbl


class ZarrTable(object):

    def __init__(self, grp, names=None):
        self.grp = grp
        available_names = sorted(grp.array_keys())
        if names is None:
            names = available_names
        else:
            for n in names:
                if n not in available_names:
                    raise ValueError('name not available: %s' % n)
        self.names = names

    def __getitem__(self, item):
        return self.grp[item]

    def append(self, data):
        names, columns = _util.check_table_like(data, names=self.names)
        for n, c in zip(names, columns):
            self.grp[n].append(c)


class ZarrMemStorage(ZarrStorage):

    # noinspection PyShadowingBuiltins
    def _set_defaults(self, kwargs):
        kwargs = super(ZarrMemStorage, self)._set_defaults(kwargs)
        kwargs.setdefault('store', MemoryStore())
        return kwargs


class ZarrTmpStorage(ZarrStorage):

    def _set_defaults(self, kwargs):
        kwargs = super(ZarrTmpStorage, self)._set_defaults(kwargs)
        suffix = kwargs.pop('suffix', '.zarr')
        prefix = kwargs.pop('prefix', 'scikit_allel2_')
        # noinspection PyShadowingBuiltins
        dir = kwargs.pop('dir', None)
        # create temporary directory
        tmpdir = tempfile.TemporaryDirectory(suffix=suffix, prefix=prefix, dir=dir)
        atexit.register(tmpdir.cleanup)
        kwargs.setdefault('store', LocalStore(tmpdir.name))
        return kwargs


zarr_storage = ZarrStorage()
"""zarr storage with default parameters"""
zarrmem_storage = ZarrMemStorage()
"""zarr in-memory storage with default compression"""
zarrtmp_storage = ZarrTmpStorage()
"""zarr temporary file storage with default compression"""

_util.storage_registry['zarr'] = zarr_storage
_util.storage_registry['zarrmem'] = zarrmem_storage
_util.storage_registry['zarrtmp'] = zarrtmp_storage
