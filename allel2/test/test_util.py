# -*- coding: utf-8 -*-
import numpy as np
import zarr

from allel2.util import get_blen_array


def test_get_blen_array_uses_chunks():
    # arrays with chunks (zarr, h5py) are iterated chunk by chunk
    z = zarr.array(np.zeros((100, 10), dtype='i1'), chunks=(7, 10))
    assert 7 == get_blen_array(z)


def test_get_blen_array_without_chunks():
    # about 1 MiB per block otherwise
    a = np.zeros((10_000, 1024), dtype='i1')
    assert 1024 == get_blen_array(a)
    assert 3 == get_blen_array(a, blen=3)


def test_get_blen_array_former_location():
    # kept importable from allel2.chunked while chunked is deprecated
    from allel2.chunked import get_blen_array as former
    assert former is get_blen_array
