"""Smoke test run by cibuildwheel on every wheel, in a clean environment.

Checks that the wheel imports, carries a real version, ships its test data
and its compiled extensions, by reading a VCF and counting alleles.
"""
import importlib.resources

import allel2

version = allel2.__version__
# setuptools-scm falls back to 0.1.devN when it cannot see any tag
assert version and not version.startswith('0.1.dev'), version

vcf = importlib.resources.files('allel2.test') / 'data' / 'sample.vcf'
callset = allel2.read_vcf(str(vcf), fields=['variants/POS', 'calldata/GT'])
gt = allel2.GenotypeArray(callset['calldata/GT'])
assert gt.shape == (9, 3, 2), gt.shape
assert callset['variants/POS'][:3].tolist() == [111, 112, 14370]
assert gt.count_alleles()[4].tolist() == [0, 2, 4]

print('scikit-allel2 %s: smoke test passed' % version)
