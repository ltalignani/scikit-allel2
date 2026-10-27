"""Profile typical allel2 analyses on the benchmark data set.

Usage::

    python bench/run_profiling.py --data ~/data/allel2-bench --out results.json

Steps (see bench/README.md for the data layout):

1. read_vcf on the 1000 Genomes chr22 VCF (GT only).
2. vcf_to_zarr on an Ag2 3R region (GT only), producing the genotypes used below.
3. read_vcf of every field on a smaller Ag2 region (a realistic GATK pipeline VCF).
4. read_vcf of the accessibility VCF on the region (one record per position).
5. count_alleles, for all samples and per population.
6. PCA workflow: segregating sites, random subset, LD pruning, PCA.
7. Windowed diversity, Watterson's theta, Tajima's D and Hudson's Fst.
8. Selection scans on phased haplotypes: iHS, nSL, XP-EHH.
"""
import argparse
import csv
import json
import os
import platform
import shutil
import sys

import numpy as np
import zarr

import allel2

sys.path.insert(0, os.path.dirname(__file__))
from profiling import install_cython_timers, profile_step, timed


def region_bounds(region):
    chrom, span = region.split(':')
    start, stop = (int(x) for x in span.split('-'))
    return chrom, start, stop


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=os.path.expanduser('~/data/allel2-bench'))
    p.add_argument('--region', default='3R:10000000-15000000')
    p.add_argument('--pipeline-region', default='3R:10000000-11000000')
    p.add_argument('--pop1', default='BFgam')
    p.add_argument('--pop2', default='BFcol')
    p.add_argument('--work', default=None, help='scratch directory for zarr output')
    p.add_argument('--out', required=True)
    p.add_argument('--skip', default='', help='comma-separated step prefixes to skip')
    args = p.parse_args()
    skip = [s for s in args.skip.split(',') if s]

    kernels = install_cython_timers()
    chrom, start, stop = region_bounds(args.region)
    vcf = os.path.join(args.data, 'vcf', f'ag1000g.phase2.ar1.pass.biallelic.{chrom}.vcf.gz')
    work = args.work or os.path.join(args.data, 'work')
    os.makedirs(work, exist_ok=True)
    records = []

    def run(name, fn, *a, **k):
        if any(name.startswith(s) for s in skip):
            return None
        result, rec = profile_step(name, fn, *a, **k)
        records.append(rec)
        return result

    # 1. public VCF
    chr22 = os.path.join(args.data, 'vcf',
                         'ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502'
                         '.genotypes.vcf.gz')
    run('1 read_vcf chr22 GT', allel2.read_vcf, chr22,
        fields=['variants/POS', 'variants/REF', 'variants/ALT', 'calldata/GT'])

    # 2. Ag2 region to zarr, GT only
    gt_zarr = os.path.join(work, f'ag2.{chrom}.{start}-{stop}.gt.zarr')
    if not any('2'.startswith(s) for s in skip) and os.path.exists(gt_zarr):
        shutil.rmtree(gt_zarr)
    run('2 vcf_to_zarr Ag2 region GT', allel2.vcf_to_zarr, vcf, gt_zarr,
        region=args.region, alt_number=1,
        fields=['samples', 'variants/POS', 'variants/REF', 'variants/ALT',
                'calldata/GT'])

    # 3. realistic pipeline VCF: every field, smaller region
    run('3 read_vcf Ag2 all fields', allel2.read_vcf, vcf, region=args.pipeline_region,
        fields='*', alt_number=1)

    # 4. accessibility
    acc_vcf = os.path.join(args.data, 'accessibility', f'accessibility.{chrom}.vcf.gz')
    acc = run('4 read_vcf accessibility', allel2.read_vcf, acc_vcf, region=args.region,
              fields=['variants/POS', 'variants/Accessible'])
    is_accessible = np.zeros(stop + 1, dtype=bool)
    if acc is not None:
        is_accessible[acc['variants/POS'] - 1] = acc['variants/Accessible']

    # load genotypes and populations
    callset = zarr.open_group(gt_zarr, mode='r')
    pos = allel2.SortedIndex(callset['variants/POS'][:])
    gt = allel2.GenotypeArray(callset['calldata/GT'][:])
    samples = callset['samples'][:].tolist()
    with open(os.path.join(args.data, 'samples', 'samples.meta.txt')) as f:
        population = {r['ox_code']: r['population'] for r in csv.DictReader(f, delimiter='\t')}
    pops = np.array([population[s] for s in samples])
    subpops = {args.pop1: np.nonzero(pops == args.pop1)[0].tolist(),
               args.pop2: np.nonzero(pops == args.pop2)[0].tolist()}

    # 5. allele counts
    ac = run('5a count_alleles all samples', gt.count_alleles, max_allele=1)
    acs = run('5b count_alleles_subpops', gt.count_alleles_subpops, subpops, max_allele=1)

    # 6. PCA workflow
    def pca_workflow():
        flt = ac.is_segregating() & (ac.to_frequencies()[:, 1] > 0.01)
        idx = np.nonzero(flt)[0]
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(idx, size=min(100_000, idx.size), replace=False))
        gn = gt.take(idx, axis=0).to_n_alt()
        loc = allel2.locate_unlinked(gn, size=500, step=200, threshold=0.1)
        coords, _ = allel2.pca(gn[loc], n_components=10, scaler='patterson')
        return gn.shape[0], int(loc.sum()), coords
    # profile the parts separately as well
    flt = ac.is_segregating() & (ac.to_frequencies()[:, 1] > 0.01)
    idx = np.sort(np.random.default_rng(42).choice(
        np.nonzero(flt)[0], size=min(100_000, int(flt.sum())), replace=False))
    gn = run('6a to_n_alt (100k sites)', gt.take(idx, axis=0).to_n_alt)
    loc = run('6b locate_unlinked', allel2.locate_unlinked, gn, size=500, step=200,
              threshold=0.1)
    if loc is not None:
        run('6c pca', allel2.pca, gn[loc], n_components=10, scaler='patterson')
    run('6 PCA workflow total', pca_workflow)

    # 7. diversity
    a1, a2 = acs[args.pop1], acs[args.pop2]
    win = {'size': 20_000, 'start': start, 'stop': stop}
    run('7a windowed_diversity', allel2.windowed_diversity, pos, a1,
        is_accessible=is_accessible, **win)
    run('7b windowed_watterson_theta', allel2.windowed_watterson_theta, pos, a1,
        is_accessible=is_accessible, **win)
    run('7c windowed_tajima_d', allel2.windowed_tajima_d, pos, a1, **win)
    run('7d windowed_hudson_fst', allel2.windowed_hudson_fst, pos, a1, a2, **win)
    gt = None  # free memory before loading haplotypes

    # 8. selection scans on phased haplotypes
    hap_root = os.path.join(args.data, 'haplotypes', 'zarr', 'ag1000g.phase2.ar1.haplotypes',
                            chrom)
    hpos = zarr.open_array(os.path.join(hap_root, 'variants', 'POS'), mode='r')[:]
    i0, i1 = np.searchsorted(hpos, [start, stop + 1])
    hgt = zarr.open_array(os.path.join(hap_root, 'calldata', 'GT'), mode='r')[i0:i1]
    h_all = allel2.HaplotypeArray(hgt.reshape(hgt.shape[0], -1))
    with open(os.path.join(args.data, 'haplotypes', 'haplotypes.autosomes.meta.txt')) as f:
        hpops = np.array([r['population'] for r in csv.DictReader(f, delimiter='\t')])
    h1 = h_all.take(np.nonzero(hpops == args.pop1)[0], axis=1)
    h2 = h_all.take(np.nonzero(hpops == args.pop2)[0], axis=1)
    hp = hpos[i0:i1]
    # keep sites segregating in either population (as for XP-EHH)
    seg = h1.count_alleles(max_allele=1).is_segregating() | \
        h2.count_alleles(max_allele=1).is_segregating()
    h1, h2, hp = h1.compress(seg, axis=0), h2.compress(seg, axis=0), hp[seg]
    seg1 = h1.count_alleles(max_allele=1).is_segregating()
    h1s, hp1 = h1.compress(seg1, axis=0), hp[seg1]
    # attribution needs a single thread (cProfile only sees the main thread)
    run('8a ihs (1 thread)', allel2.ihs, h1s, hp1, include_edges=True, use_threads=False)
    run('8b nsl (1 thread)', allel2.nsl, h1s, use_threads=False)
    run('8c xpehh (1 thread)', allel2.xpehh, h1, h2, hp, include_edges=True,
        use_threads=False)
    for name, fn, a, k in [
            ('8a ihs (threads)', allel2.ihs, (h1s, hp1), {'include_edges': True}),
            ('8b nsl (threads)', allel2.nsl, (h1s,), {}),
            ('8c xpehh (threads)', allel2.xpehh, (h1, h2, hp), {'include_edges': True})]:
        if not any(name.startswith(s) for s in skip):
            records.append(timed(name, fn, *a, **k)[1])

    meta = {
        'python': platform.python_version(), 'machine': platform.machine(),
        'system': platform.system(), 'cpu_count': os.cpu_count(),
        'numpy': np.__version__, 'zarr': zarr.__version__, 'allel2': allel2.__version__,
        'region': args.region, 'pipeline_region': args.pipeline_region,
        'pops': [args.pop1, args.pop2], 'n_variants': int(pos.shape[0]),
        'n_samples': len(samples), 'n_haplotypes': [int(h1.shape[1]), int(h2.shape[1])],
        'n_hap_variants': [int(h1s.shape[0]), int(h1.shape[0])],
        'wrapped_cython_kernels': kernels,
    }
    with open(args.out, 'w') as f:
        json.dump({'meta': meta, 'steps': records}, f, indent=1)


if __name__ == '__main__':
    main()
