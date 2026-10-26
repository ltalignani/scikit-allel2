# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **scikit-allel2**, a Python package for exploratory analysis of large scale genetic variation data. This project aims to make scikit-allel compatible with Python 3.10+, NumPy 2.0+, and zarr 3.0+ while maintaining functionality and performance.

### Migration Goals
- **MILESTONE 1**: Update Cython code for NumPy 2.0, zarr 3.0, and Python 3.10+ compatibility
- **MILESTONE 2**: Ensure installation/runtime on Python 3.10-3.12+, update CI
- **MILESTONE 3**: Prepare for PyPI publishing as scikit-allel2

### Key Compatibility Issues
- Cython extensions (allel2/opt/*.pyx) need updating for NumPy 2.0 C API changes
- Zarr-related code requires migration to zarr 3.0+ API
- CI configuration currently only tests Python 3.6-3.9

## Development Commands

### Installation and Setup
```bash
# Install package in development mode with all dependencies
pip install -e .

# Install development dependencies
pip install -r requirements_dev_base.txt -r requirements_test.txt -r requirements_dev_optional.txt -r requirements_rtfd.txt
```

### Testing
```bash
# Run all tests
pytest -v allel2

# Run tests with coverage (if enabled)
pytest -v --cov=allel2 allel2

# Run doctests (if enabled)
pytest -v --doctest-modules allel2
```

### Linting
```bash
# Check for syntax errors and undefined names
flake8 allel2 --count --select=E9,F63,F7,F82 --show-source --statistics

# Full linting with warnings
flake8 allel2 --count --exit-zero --max-line-length=127 --statistics
```

### Documentation
```bash
# Build HTML documentation
cd docs && make html

# The documentation uses Sphinx - see docs/Makefile for all available targets
```

## Architecture Overview

### Package Structure
- **`allel2/model/`**: Core data structures and array implementations
  - `ndarray.py`: NumPy-based array classes for genetic data
  - `chunked.py`: Chunked array implementations for memory efficiency
  - `dask.py`: Dask-based distributed array support (optional)
  - `generic.py`: Generic array interface definitions

- **`allel2/stats/`**: Statistical analysis functions
  - `diversity.py`: Genetic diversity metrics (π, θ, Tajima's D)
  - `fst.py`: Population differentiation statistics
  - `ld.py`: Linkage disequilibrium analysis
  - `selection.py`: Selection tests (EHH, iHS, XP-EHH)
  - `admixture.py`: Admixture and population structure (f-statistics)
  - `distance.py`: Genetic distance and PCoA
  - `window.py`: Windowing functions for genome-wide analysis

- **`allel2/io/`**: Data input/output
  - `vcf_read.py`: VCF file reading (optimized with Cython)
  - `vcf_write.py`: VCF file writing
  - `fasta.py`: FASTA file handling
  - `gff.py`: GFF/GTF file reading

- **`allel2/opt/`**: Performance-optimized Cython modules
  - `*.pyx` files: Cython implementations for computational bottlenecks

### Build System
- Uses setuptools with Cython extensions defined in `setup.py`
- Cython files (`*.pyx`) are automatically discovered and compiled
- Build requires NumPy headers for C extension compilation
- Package metadata is in `pyproject.toml`, extensions config in `setup.py`

### Key Dependencies
- **Core**: NumPy 2.0+, Dask (optional)
- **Optional**: SciPy, matplotlib, pandas, zarr, h5py, scikit-learn
- **Development**: pytest, flake8, Sphinx (docs)

### Testing Structure
- Tests are in `allel2/test/` directory, mirroring the main package structure
- Uses pytest with doctest support
- Test data files in `allel2/test/data/`
- CI runs on Python 3.10-3.12 across Linux/Windows/macOS with NumPy 1.x and 2.x

### Important Notes
- This is a scientific computing package focused on population genetics
- Many functions operate on large genomic datasets and are optimized for performance
- The package supports both in-memory (NumPy) and out-of-core (Dask) computation
- Cython extensions provide performance-critical implementations