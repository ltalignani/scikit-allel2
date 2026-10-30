"""Smoke test run by cibuildwheel on every wheel, in a clean environment.

Checks that the wheel imports, carries a real version, ships its test data
and its compiled extensions (Cython and Rust), by reading a VCF with the Rust
parser and counting alleles.
"""
import importlib.resources
import logging

import allel2
import allel2.opt.io_vcf_read_rs  # noqa: F401

version = allel2.__version__
# setuptools-scm falls back to 0.1.devN when it cannot see any tag
assert version and not version.startswith('0.1.dev'), version

# the parser choice is logged at debug level
messages = []
handler = logging.Handler(logging.DEBUG)
handler.emit = lambda record: messages.append(record.getMessage())
logger = logging.getLogger('allel2.io.vcf_read')
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

vcf = importlib.resources.files('allel2.test') / 'data' / 'sample.vcf'
callset = allel2.read_vcf(str(vcf), fields=['variants/POS', 'calldata/GT'])
assert any('with the Rust VCF parser' in m for m in messages), messages
gt = allel2.GenotypeArray(callset['calldata/GT'])
assert gt.shape == (9, 3, 2), gt.shape
assert callset['variants/POS'][:3].tolist() == [111, 112, 14370]
assert gt.count_alleles()[4].tolist() == [0, 2, 4]

print('scikit-allel2 %s: smoke test passed' % version)
