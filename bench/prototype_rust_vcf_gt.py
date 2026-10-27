"""PROTOTYPE (throwaway): Rust GT-only VCF parser vs the Cython read_vcf.

Usage::

    python bench/prototype_rust_vcf_gt.py --out results.json

For each data set, times ``allel2.read_vcf(fields=[samples, POS, GT])``
(Cython), then ``allel2._prototype_rust_vcf.read_vcf_gt`` with 1 thread and
with all cores, and checks that the arrays are identical.
"""
import argparse
import gc
import json
import os
import platform
import time

import numpy as np

import allel2
from allel2 import _prototype_rust_vcf

FIELDS = ['samples', 'variants/POS', 'calldata/GT']


def timed(fn, *args, **kwargs):
    gc.collect()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, round(time.perf_counter() - t0, 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=os.path.expanduser('~/data/allel2-bench'))
    p.add_argument('--out', required=True)
    args = p.parse_args()
    vcf = os.path.join(args.data, 'vcf')
    datasets = [
        ('chr22 1000 Genomes (whole file)',
         os.path.join(vcf, 'ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502'
                           '.genotypes.vcf.gz'), None),
        ('Ag2 3R:10-15 Mb (tabix region)',
         os.path.join(vcf, 'ag1000g.phase2.ar1.pass.biallelic.3R.vcf.gz'),
         '3R:10000000-15000000'),
    ]
    results = []
    for name, path, region in datasets:
        expected, t_cython = timed(allel2.read_vcf, path, fields=FIELDS, region=region)
        exp_pos, exp_gt = expected['variants/POS'], expected['calldata/GT']
        exp_samples = expected['samples'].tolist()
        del expected
        rec = {'dataset': name, 'n_variants': int(exp_pos.shape[0]),
               'n_samples': len(exp_samples), 'cython_s': t_cython}
        for label, threads in ('rust_1_thread', 1), ('rust_all_threads', 0):
            (samples, pos, gt), t = timed(_prototype_rust_vcf.read_vcf_gt, path,
                                          region=region, n_threads=threads)
            rec[label + '_s'] = t
            rec[label + '_speedup'] = round(t_cython / t, 2)
            rec[label + '_identical'] = bool(
                samples == exp_samples and np.array_equal(pos, exp_pos)
                and np.array_equal(gt, exp_gt))
            del samples, pos, gt
        print(json.dumps(rec), flush=True)
        results.append(rec)
        del exp_pos, exp_gt
    meta = {'python': platform.python_version(), 'machine': platform.machine(),
            'cpu_count': os.cpu_count(), 'numpy': np.__version__,
            'allel2': allel2.__version__}
    with open(args.out, 'w') as f:
        json.dump({'meta': meta, 'results': results}, f, indent=1)


if __name__ == '__main__':
    main()
