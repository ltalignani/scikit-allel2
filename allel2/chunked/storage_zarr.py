# -*- coding: utf-8 -*-
import operator
from functools import reduce


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
    # zarr 3 refuses object arrays; strings (e.g., from h5py) become StringDType
    a = np.asarray(a)
    if a.dtype == object:
        a = a.astype(np.dtypes.StringDType())
    return a


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
        z = zarr.array(_zarr_compatible(data), **kwargs)

        return z

    def table(self, data, names=None, expectedlen=None, **kwargs):

        # setup
        names, columns = _util.check_table_like(data, names=names)
        kwargs = self._set_defaults(kwargs)
        chunks = kwargs.pop('chunks', None)
        g = zarr.group(**kwargs)

        # create columns
        for n, c in zip(names, columns):
            if chunks is None:
                chunks = default_chunks(c, expectedlen)
            c = _zarr_compatible(c)
            if chunks is None:
                column_chunks = 'auto'
            elif isinstance(chunks, int):
                # rows per chunk; columns may have more dimensions than rows
                column_chunks = (chunks,) + c.shape[1:]
            else:
                column_chunks = chunks
            g.create_array(name=n, data=c, chunks=column_chunks)

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
