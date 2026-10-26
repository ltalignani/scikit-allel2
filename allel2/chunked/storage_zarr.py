# -*- coding: utf-8 -*-
import operator
from functools import reduce


import zarr
try:
    # zarr 3.0+
    from zarr.storage import MemoryStore, LocalStore
except ImportError:
    # zarr 2.x compatibility
    from zarr import MemoryStore
    from zarr.storage import DirectoryStore as LocalStore
import tempfile
import atexit
import os

import numcodecs


from allel2.chunked import util as _util


def default_chunks(data, expectedlen):
    return None


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
            if chunks is None:
                chunks = default_chunks(c, expectedlen)
            # zarr 3.0+ compatibility: array() requires shape and dtype
            try:
                # zarr 2.x style
                g.array(name=n, data=c, chunks=chunks)
            except TypeError:
                # zarr 3.0+ style - requires shape and dtype
                import numpy as np
                c_array = np.asarray(c)
                g.array(name=n, data=c, chunks=chunks, shape=c_array.shape, dtype=c_array.dtype)

        # create table
        ztbl = ZarrTable(g, names=names)
        return ztbl


class ZarrTable(object):

    def __init__(self, grp, names=None):
        self.grp = grp
        # zarr 3.0+ compatibility: array_keys() changed to keys()
        try:
            available_names = sorted(grp.array_keys())
        except AttributeError:
            # zarr 3.0+
            available_names = sorted([k for k in grp.keys() if hasattr(grp[k], 'shape')])
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
