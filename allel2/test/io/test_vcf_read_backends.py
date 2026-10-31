# -*- coding: utf-8 -*-
"""The Rust and Cython VCF parsers give the same output.

The Cython parser is the reference for as long as it exists: each read is done
once with ``ALLEL2_BACKEND=cython`` and once with the default backend, and the
chunks must match. Fields not yet handled by the Rust parser fall back to the
Cython parser, with a debug message giving the reason.
"""
import io
import logging
import os
import warnings

import numpy as np
import pytest

from allel2.io.vcf_read import iter_vcf_chunks, read_vcf, ANNTransformer


DATA = os.path.join(os.path.dirname(__file__), os.pardir, 'data')
LOGGER = 'allel2.io.vcf_read'
FALLBACK = 'falling back to the Cython VCF parser'
RUST = 'with the Rust VCF parser'

# every test VCF that has a header line
FILES = ['altlen.vcf', 'ann.vcf', 'eff.vcf', 'empty.vcf', 'sample.utf8.vcf',
         'sample.vcf', 'sample.vcf.gz', 'test1.vcf', 'test14.vcf', 'test16.vcf',
         'test176.vcf', 'test32.vcf', 'test4.vcf', 'test48b.vcf', 'test54.vcf.gz',
         'test63.vcf', 'unsorted.vcf']

HEADER = (b'##fileformat=VCFv4.2\n'
          b'##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
          b'#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\tS3\n')

# lines that exercise the corners of POS and GT parsing
EDGE_CASES = {
    'crlf': HEADER.replace(b'\n', b'\r\n') +
    b'2L\t12\t.\tA\tC\t.\t.\t.\tGT\t0/1\t1|1\t./.\r\n'
    b'2L\t15\t.\tA\tC\t.\t.\t.\tGT\t0/0\t0|1\t1/.\r\n',
    'no_final_newline': HEADER +
    b'2L\t12\t.\tA\tC\t.\t.\t.\tGT\t0/1\t1|1\t./.\n'
    b'2L\t15\t.\tA\tC\t.\t.\t.\tGT\t0/0\t0|1\t1/.',
    'gt_not_first': HEADER +
    b'2L\t12\t.\tA\tC\t.\t.\t.\tDP:GT\t3:0/1\t4:1|1\t.:./.\n'
    b'2L\t15\t.\tA\tC\t.\t.\t.\tDP:.:GT\t3:x:0/1\t4\t\n',
    'missing_and_short': HEADER +
    b'2L\t12\t.\tA\tC\t.\t.\t.\tGT\t.\t0\t0/1/2/3\n'
    b'2L\t13\t.\tA\tC\t.\t.\t.\t.\t0/1\t1/1\t1/1\n'
    b'2L\t14\t.\tA\tC\t.\t.\t.\tGT\t0/1\n'
    b'2L\t15\t.\tA\tC\t.\t.\t.\n'
    b'2L\t16\n',
    'large_alleles': HEADER +
    b'2L\t12\t.\tA\tC\t.\t.\t.\tGT\t127/128\t255/256\t-1/-2\n'
    b'2L\t13\t.\tA\tC\t.\t.\t.\tGT\t+3/ 4\t99999999999999999999/0\t1/1\n',
    'dodgy_values': HEADER +
    b'2L\taaa\t.\tA\tC\t.\t.\t.\tGT\taa/bb\t12aa/22\t1/1\n'
    b'2L\t12aaa\t.\tA\tC\t.\t.\t.\tGT\t0/1\t1/1\t1/1\n'
    b'\t\t.\tA\tC\t.\t.\t.\tGT\t0/1\t1/1\t1/1\n'
    b'2L\t99999999999\t.\tA\tC\t.\t.\t.\tGT\t0/1\t1/1\t1/1\n',
    'extra_samples': HEADER +
    b'2L\t12\t.\tA\tC\t.\t.\t.\tGT\t0/1\t1/1\t1/1\t0/0\t1/0\n',
    'empty_line': HEADER +
    b'2L\t12\t.\tA\tC\t.\t.\t.\tGT\t0/1\t1/1\t1/1\n'
    b'\n'
    b'2L\t14\t.\tA\tC\t.\t.\t.\tGT\t0/1\t1/1\t1/1\n',
}

FIELDS = [
    ['variants/POS'],
    ['calldata/GT'],
    ['variants/POS', 'calldata/GT'],
    ['calldata/GT', 'POS'],
]

GT_OPTIONS = (
    [{'types': {'calldata/GT': t}}
     for t in ('i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8')] +
    [{},
     {'types': {'calldata/GT': 'genotype/u8'}},
     {'numbers': {'calldata/GT': 1}},
     {'numbers': {'calldata/GT': 3}},
     {'fills': {'calldata/GT': -9}},
     {'types': {'calldata/GT': 'u2'}, 'fills': {'calldata/GT': 7}},
     {'alt_number': 1}]
)


@pytest.fixture
def edge_vcf(tmp_path):
    def write(name, gz=False):
        data = EDGE_CASES[name]
        if gz:
            import gzip
            path = tmp_path / (name + '.vcf.gz')
            with gzip.open(path, 'wb') as f:
                f.write(data)
        else:
            path = tmp_path / (name + '.vcf')
            path.write_bytes(data)
        return str(path)
    return write


def _chunks(path, backend, monkeypatch, caplog, **kwargs):
    """Read all chunks with the given backend, returning chunks, warnings and
    whether the read fell back to the Cython parser."""
    monkeypatch.setenv('ALLEL2_BACKEND', backend)
    caplog.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            fields, samples, headers, it = iter_vcf_chunks(path, **kwargs)
            chunks = list(it)
    used_rust = any(RUST in r.getMessage() for r in caplog.records)
    return (fields, samples, chunks), [str(w.message) for w in caught], used_rust


def assert_same_chunks(expected, actual):
    e_fields, e_samples, e_chunks = expected
    a_fields, a_samples, a_chunks = actual
    assert e_fields == a_fields
    np.testing.assert_array_equal(e_samples, a_samples)
    assert len(e_chunks) == len(a_chunks)
    for (e, e_len, e_chrom, e_pos), (a, a_len, a_chrom, a_pos) in zip(e_chunks, a_chunks):
        assert (e_len, e_chrom, e_pos) == (a_len, a_chrom, a_pos)
        assert sorted(e) == sorted(a)
        for k in e:
            assert e[k].dtype == a[k].dtype, k
            assert e[k].shape == a[k].shape, k
            np.testing.assert_array_equal(e[k], a[k], err_msg=k)


def assert_rust_matches_cython(path, monkeypatch, caplog, **kwargs):
    cython, cython_warnings, _ = _chunks(path, 'cython', monkeypatch, caplog, **kwargs)
    rust, rust_warnings, used_rust = _chunks(path, 'rust', monkeypatch, caplog, **kwargs)
    assert used_rust, caplog.text
    assert_same_chunks(cython, rust)
    assert cython_warnings == rust_warnings


@pytest.mark.parametrize('fn', FILES)
@pytest.mark.parametrize('fields', FIELDS)
@pytest.mark.parametrize('chunk_length', [1, 2, 3, 2**16])
def test_test_files(fn, fields, chunk_length, monkeypatch, caplog):
    assert_rust_matches_cython(os.path.join(DATA, fn), monkeypatch, caplog,
                               fields=fields, chunk_length=chunk_length)


@pytest.mark.parametrize('fn', ['sample.vcf', 'test63.vcf', 'test1.vcf', 'test32.vcf'])
@pytest.mark.parametrize('options', GT_OPTIONS)
def test_gt_options(fn, options, monkeypatch, caplog):
    assert_rust_matches_cython(os.path.join(DATA, fn), monkeypatch, caplog,
                               fields=['variants/POS', 'calldata/GT'], chunk_length=2,
                               **options)


@pytest.mark.parametrize('samples', [['NA00002'], ['NA00003', 'NA00001'], [], [0, 2]])
def test_samples(samples, monkeypatch, caplog):
    assert_rust_matches_cython(os.path.join(DATA, 'sample.vcf'), monkeypatch, caplog,
                               fields=['calldata/GT'], samples=samples)


@pytest.mark.parametrize('name', sorted(EDGE_CASES))
@pytest.mark.parametrize('gz', [False, True])
@pytest.mark.parametrize('options', [{}, {'numbers': {'calldata/GT': 3}},
                                     {'types': {'calldata/GT': 'u1'}}])
def test_edge_cases(name, gz, options, edge_vcf, monkeypatch, caplog):
    assert_rust_matches_cython(edge_vcf(name, gz), monkeypatch, caplog,
                               fields=['variants/POS', 'calldata/GT'], chunk_length=2,
                               **options)


def test_read_vcf(monkeypatch, caplog):
    # read_vcf concatenates the chunks
    path = os.path.join(DATA, 'sample.vcf')
    monkeypatch.setenv('ALLEL2_BACKEND', 'rust')
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        callset = read_vcf(path, fields=['samples', 'POS', 'GT'], chunk_length=2)
    assert RUST in caplog.text
    assert ['NA00001', 'NA00002', 'NA00003'] == callset['samples'].tolist()
    assert [111, 112, 14370] == callset['variants/POS'][:3].tolist()
    assert [[0, -1], [0, 1], [0, 2]] == callset['calldata/GT'][-1].tolist()


# which parser reads the file

def _reason(caplog):
    messages = [r.getMessage() for r in caplog.records if FALLBACK in r.getMessage()]
    assert len(messages) == 1, caplog.text
    assert all(r.levelno == logging.DEBUG for r in caplog.records
               if FALLBACK in r.getMessage())
    return messages[0]


def _read(input, monkeypatch, caplog, backend=None, **kwargs):
    if backend is None:
        monkeypatch.delenv('ALLEL2_BACKEND', raising=False)
    else:
        monkeypatch.setenv('ALLEL2_BACKEND', backend)
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        return read_vcf(input, **kwargs)


def test_rust_is_the_default(monkeypatch, caplog):
    _read(os.path.join(DATA, 'sample.vcf.gz'), monkeypatch, caplog,
          fields=['POS', 'GT'])
    assert RUST in caplog.text
    assert FALLBACK not in caplog.text


def test_fallback_backend_variable(monkeypatch, caplog):
    callset = _read(os.path.join(DATA, 'sample.vcf'), monkeypatch, caplog,
                    backend='cython', fields=['POS', 'GT'])
    assert 'ALLEL2_BACKEND=cython' in _reason(caplog)
    assert 9 == len(callset['variants/POS'])


def test_unknown_backend(monkeypatch, caplog):
    with pytest.raises(ValueError, match='ALLEL2_BACKEND'):
        _read(os.path.join(DATA, 'sample.vcf'), monkeypatch, caplog, backend='fortran')


def test_fallback_transformers(monkeypatch, caplog):
    callset = _read(os.path.join(DATA, 'ann.vcf'), monkeypatch, caplog,
                    fields=['POS', 'ANN'], transformers=[ANNTransformer()])
    assert 'transformers' in _reason(caplog)
    assert 'variants/ANN_Allele' in callset


def test_fallback_file_object(monkeypatch, caplog):
    with open(os.path.join(DATA, 'sample.vcf'), 'rb') as f:
        data = f.read()
    callset = _read(io.BytesIO(data), monkeypatch, caplog, fields=['POS', 'GT'])
    assert 'file object' in _reason(caplog)
    assert 9 == len(callset['variants/POS'])


def test_fallback_region(monkeypatch, caplog):
    callset = _read(os.path.join(DATA, 'sample.vcf.gz'), monkeypatch, caplog,
                    fields=['POS', 'GT'], region='20')
    assert 'region' in _reason(caplog)
    assert 6 == len(callset['variants/POS'])


@pytest.mark.parametrize('fields, field', [
    (['POS', 'CHROM'], 'variants/CHROM'),
    (['POS', 'GT', 'GQ'], 'calldata/GQ'),
    (['POS', 'DP'], 'variants/DP'),
    (None, 'variants/'),
])
def test_fallback_unsupported_field(fields, field, monkeypatch, caplog):
    callset = _read(os.path.join(DATA, 'sample.vcf'), monkeypatch, caplog,
                    fields=fields)
    assert field in _reason(caplog)
    assert 9 == len(callset['variants/POS'])


@pytest.mark.parametrize('types', [{'GT': 'f4'}, {'GT': 'genotype_ac/i1'}])
def test_fallback_unsupported_type(types, monkeypatch, caplog):
    _read(os.path.join(DATA, 'sample.vcf'), monkeypatch, caplog,
          fields=['POS', 'GT'], types=types)
    assert 'calldata/GT' in _reason(caplog)


def test_pos_type_warning(monkeypatch, caplog):
    # like the Cython parser, POS is always int32
    for backend in 'cython', 'rust':
        with pytest.warns(UserWarning, match='only int32 supported for POS'):
            callset = _read(os.path.join(DATA, 'sample.vcf'), monkeypatch, caplog,
                            backend=backend, fields=['POS'], types={'POS': 'i8'})
        assert np.dtype('int32') == callset['variants/POS'].dtype
