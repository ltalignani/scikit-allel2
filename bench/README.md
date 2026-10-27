# Benchmarks

Scripts to profile allel2 on real data and attribute run time to the Cython
kernels (`allel2.opt.*`), NumPy and I/O.

## Data

The scripts expect the benchmark data set under `~/data/allel2-bench/`
(override with `--data`); it is not part of the repository.

```text
~/data/allel2-bench/
├── vcf/
│   ├── ag1000g.phase2.ar1.pass.biallelic.{3L,3R}.vcf.gz (+ .tbi)
│   └── ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz (+ .tbi)
├── accessibility/accessibility.{3L,3R}.vcf.gz (+ .tbi)
├── haplotypes/
│   ├── zarr/ag1000g.phase2.ar1.haplotypes/{3L,3R}/   # unzipped MalariaGEN archives
│   └── haplotypes.autosomes.meta.txt
└── samples/samples.meta.txt
```

- Ag1000G phase 2 (Ag2) VCFs, accessibility, haplotypes and sample metadata come
  from MalariaGEN (copied from the IFB cluster).
- The chr22 VCF is 1000 Genomes phase 3:
  `https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/`.
- The haplotype archives (`ag1000g.phase2.ar1.haplotypes.{3L,3R}.{variants,calldata.GT}.zip`)
  unzip into a zarr format 2 hierarchy without a root `.zgroup`: open arrays by path.

## Profiling

```bash
python bench/run_profiling.py --out results.json
```

Steps: `read_vcf` (chr22, and every field of a GATK VCF), `vcf_to_zarr`,
accessibility, `count_alleles`, PCA workflow (`locate_unlinked`, `pca`), windowed
diversity and Fst, and selection scans (iHS, nSL, XP-EHH). Use `--region`,
`--pipeline-region`, `--pop1`, `--pop2` and `--skip` to change the workload.

For each step, `results.json` records the wall time, the share of profiled time
per category (`cython`, `numpy`, `zarr/numcodecs`, `allel2 (python)`, ...) and the
time per Cython kernel. cProfile only sees the main thread, so the selection
scans are profiled with `use_threads=False` and timed separately with threads.

Profiling adds overhead; compare wall times between runs of the same script only.
