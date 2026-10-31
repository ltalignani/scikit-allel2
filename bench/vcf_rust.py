"""Benchmark of the Rust VCF parser against the Cython one (scikit-allel2 2.1).

Run by hand, out of CI, on real data, for example::

    python bench/vcf_rust.py \\
        ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz
    python bench/vcf_rust.py ag1000g.phase2.ar1.pass.biallelic.3R.vcf.gz \\
        --region 3R:10000000-15000000

`read_vcf` and `vcf_to_zarr` read the file with their default fields, with the
Cython parser, then with the Rust parser on one thread and on all the available
cores. Every output is checked to be identical to the Cython one, field by
field; the script stops at the first difference. It prints a Markdown table of
the timings.
"""
import argparse
import functools
import os
import shutil
import tempfile
import time

import numpy as np
import zarr

import allel2
from allel2.io.vcf_read import _normalize_n_threads

# (backend, n_threads), the first being the reference
RUNS = [('cython', None), ('rust', 1), ('rust', None)]

# variants compared at once, to bound memory
BLOCK = 1 << 16


def assert_same_arrays(name, expected, actual):
    assert expected.dtype == actual.dtype, (name, expected.dtype, actual.dtype)
    assert expected.shape == actual.shape, (name, expected.shape, actual.shape)
    for start in range(0, max(expected.shape[0], 1), BLOCK):
        stop = start + BLOCK
        np.testing.assert_array_equal(expected[start:stop], actual[start:stop],
                                      err_msg=name)


def zarr_arrays(group, prefix=''):
    """The arrays of a zarr group and its subgroups, by path."""
    arrays = {prefix + name: array for name, array in group.arrays()}
    for name, subgroup in group.groups():
        arrays.update(zarr_arrays(subgroup, prefix + name + '/'))
    return arrays


# Each bench function reads `path` with a parser and returns the time taken and
# its output, checked to be identical to `expected`, the output of the first
# run, unless None.

def bench_read_vcf(path, region, backend, n_threads, expected):
    os.environ['ALLEL2_BACKEND'] = backend
    start = time.perf_counter()
    callset = allel2.read_vcf(path, region=region, n_threads=n_threads)
    elapsed = time.perf_counter() - start
    if expected is None:
        return elapsed, callset
    assert sorted(expected) == sorted(callset)
    for k in expected:
        assert_same_arrays(k, expected[k], callset[k])
    return elapsed, expected


def bench_vcf_to_zarr(workdir, path, region, backend, n_threads, expected):
    os.environ['ALLEL2_BACKEND'] = backend
    output = os.path.join(workdir, '%s-%s.zarr' % (backend, n_threads))
    start = time.perf_counter()
    allel2.vcf_to_zarr(path, output, region=region, n_threads=n_threads)
    elapsed = time.perf_counter() - start
    if expected is None:
        return elapsed, output
    expected_arrays = zarr_arrays(zarr.open_group(expected, mode='r'))
    actual_arrays = zarr_arrays(zarr.open_group(output, mode='r'))
    assert sorted(expected_arrays) == sorted(actual_arrays)
    for k in expected_arrays:
        assert_same_arrays(k, expected_arrays[k], actual_arrays[k])
    shutil.rmtree(output)
    return elapsed, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('path', help='VCF file')
    parser.add_argument('--region', help='region to read, e.g. 3R:10000000-15000000')
    parser.add_argument('--workdir', help='directory for the zarr outputs '
                        '(default: a temporary directory)')
    args = parser.parse_args()

    n_cores = _normalize_n_threads(None)
    name = os.path.basename(args.path) + (' ' + args.region if args.region else '')
    print('%s, scikit-allel2 %s, %d cores' % (name, allel2.__version__, n_cores),
          flush=True)

    workdir = tempfile.mkdtemp(dir=args.workdir)
    rows = []
    try:
        benches = [('read_vcf', bench_read_vcf),
                   ('vcf_to_zarr', functools.partial(bench_vcf_to_zarr, workdir))]
        for function, bench in benches:
            timings = []
            expected = None
            for backend, n_threads in RUNS:
                elapsed, expected = bench(args.path, args.region, backend, n_threads,
                                          expected)
                timings.append(elapsed)
                print('  %s %s n_threads=%s: %.1f s' % (function, backend, n_threads,
                                                        elapsed), flush=True)
            rows.append((function, timings))
            # free the reference output before the next function
            expected = None
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print()
    print('| %s | Cython | Rust, 1 thread | Rust, %d threads |' % (name, n_cores))
    print('|---|---|---|---|')
    for function, (cython, rust_1, rust_n) in rows:
        print('| `%s` | %.1f s | %.1f s (×%.1f) | %.1f s (×%.1f) |'
              % (function, cython, rust_1, cython / rust_1, rust_n, cython / rust_n))
    print()
    print('Outputs identical to the Cython ones, field by field.')


if __name__ == '__main__':
    main()
