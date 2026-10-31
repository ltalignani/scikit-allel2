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

from allel2.io.vcf_read import (iter_vcf_chunks, read_vcf, read_vcf_headers,
                                ANNTransformer)


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
    # line terminators and null bytes, as the Cython parser reads them
    'lone_cr': HEADER +
    b'2L\t12\t.\tA\tC\t1\tPASS\t.\tGT\t0/1\t1|1\t./.\r'
    b'2L\t15\t.\tA\tC\t2\tPASS\t.\tGT\t0/0\t0|1\t1/.\r\r\n'
    b'2L\t16\t.\tA\tC\t3\tPASS\t.\tGT\t0/0\t0|1\t1/.\n',
    'final_cr': HEADER.replace(b'\n', b'\r\n') +
    b'2L\t12\t.\tA\tC\t1\tPASS\t.\tGT\t0/1\t1|1\t./.\r\n'
    b'2L\t15\t.\tA\tC\t2\tPASS\r',
    'null_byte': HEADER +
    b'2L\t12\t.\tA\tC\t1\tPASS\t.\tGT\t0/1\t1|1\t./.\n'
    b'2L\t1\x005\t.\tA\tC\t2\tPASS\t.\tGT\t0/0\t0|1\t1/.\n'
    b'2L\t16\t.\tA\tC\t3\tPASS\t.\tGT\t0/0\t0|1\t1/.\n',
    'null_line': HEADER +
    b'2L\t12\t.\tA\tC\t1\tPASS\t.\tGT\t0/1\t1|1\t./.\n'
    b'\x002L\t16\t.\tA\tC\t3\tPASS\t.\tGT\t0/0\t0|1\t1/.\n',
    # lines that exercise the fixed columns
    'qual': HEADER +
    b'2L\t1\t.\tA\tC\t12.5\t.\t.\tGT\t0/1\t1/1\t1/1\n'
    b'2L\t2\t.\tA\tC\tabc\t.\t.\tGT\t0/1\t1/1\t1/1\n'
    b'2L\t3\t.\tA\tC\t12x\t.\t.\tGT\t0/1\t1/1\t1/1\n'
    b'2L\t4\t.\tA\tC\t 3e2\t.\t.\tGT\t0/1\t1/1\t1/1\n'
    b'2L\t5\t.\tA\tC\tinf\t.\n'
    b'2L\t6\t.\tA\tC\t-NaN\n'
    b'2L\t7\t.\tA\tC\t1e400\t\t.\n'
    b'2L\t8\t.\tA\tC\t.5\n'
    b'2L\t9\t.\tA\tC\t0x1.8p3\t.\n'
    b'2L\t10\t.\tA\tC\t0xg\t.\n'
    b'2L\t11\t.\tA\tC\tInfinity\t.\n'
    b'2L\t12\t.\tA\tC\tnan(123)x\t.\n'
    b'2L\t13\t.\tA\tC\t1e\t.\n'
    b'2L\t14\t.\tA\tC\t-.\t.\n'
    b'2L\t15\t.\tA\tC\t1.e-3x',
    'filter': HEADER +
    b'2L\t1\t.\tA\tC\t.\tPASS\t.\n'
    b'2L\t2\t.\tA\tC\t.\tq10;s50\t.\n'
    b'2L\t3\t.\tA\tC\t.\ts50,q10:PASS\t.\n'
    b'2L\t4\t.\tA\tC\t.\t.PASS\t.\n'
    b'2L\t5\t.\tA\tC\t.\t;;q10;\t.\n'
    b'2L\t6\t.\tA\tC\t.\tPASSq10\t.\n'
    b'2L\t7\t.\tA\tC\t.\t\t.\n'
    b'2L\t8\t.\tA\tC\t.\tq10\n'
    b'2L\t9\t.\tA\tC\t.\n',
    'alleles': HEADER +
    b'2L\t1\tid1\tA\tC,G,T,AC\t.\t.\t.\n'
    b'2L\t2\tid2;id3\tAT\tA,*\t.\t.\t.\n'
    b'2L\t3\t.\tA\t.\t.\t.\t.\n'
    b'2L\t4\t.\tA\t.,C\t.\t.\t.\n'
    b'2L\t5\t.\tA\t\t.\t.\t.\n'
    b'2L\t6\t.\tA.\tC\t.\t.\t.\n'
    b'2L\t7\t.\t.\tC\t.\t.\t.\n'
    b'2L\t8\t.\tA\tC,,G\t.\t.\t.\n'
    b'2L\t9\t.\tACGTACGT\tACGTACGTAC,<DEL>\t.\t.\t.\n'
    b'2L\t10\t.\tA\tC,G\n'
    b'2L\t11\t.\tA\n'
    b'2L\t12\tid4\n'
    b'\xc3\xa9\t13\t\xc3\xa9\t\xc3\xa9\t\xc3\xa9,T\t.\t.\t.\n'
    b'2L\t14\t.\t\xc3\xa9\tT\t.\t.\t.\n',
}

FIXED_FIELDS = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER_PASS',
                'numalt', 'altlen', 'is_snp']

FIELDS = [
    ['variants/POS'],
    ['calldata/GT'],
    ['variants/POS', 'calldata/GT'],
    ['calldata/GT', 'POS'],
    ['CHROM'],
    ['ID', 'REF'],
    ['ALT', 'numalt', 'altlen', 'is_snp'],
    ['numalt', 'altlen', 'is_snp'],
    ['QUAL'],
    ['FILTER_PASS'],
    None,
    'filters',
    'infos',
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


def _fields(path, fields):
    # 'filters': the fixed and computed fields, and every declared filter;
    # 'infos': POS and every declared INFO field
    if fields == 'filters':
        filters = ['FILTER_' + f for f in read_vcf_headers(path).filters]
        return FIXED_FIELDS + filters + ['calldata/GT']
    if fields == 'infos':
        return ['POS'] + ['variants/' + k for k in read_vcf_headers(path).infos]
    return fields


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
    path = os.path.join(DATA, fn)
    assert_rust_matches_cython(path, monkeypatch, caplog,
                               fields=_fields(path, fields), chunk_length=chunk_length)


@pytest.mark.parametrize('fn', ['sample.vcf', 'test63.vcf', 'test1.vcf', 'test32.vcf'])
@pytest.mark.parametrize('options', GT_OPTIONS)
def test_gt_options(fn, options, monkeypatch, caplog):
    assert_rust_matches_cython(os.path.join(DATA, fn), monkeypatch, caplog,
                               fields=['variants/POS', 'calldata/GT'], chunk_length=2,
                               **options)


STRING_FIELDS = ['CHROM', 'ID', 'REF', 'ALT']

FIXED_OPTIONS = (
    [{'types': {f: t for f in STRING_FIELDS}}
     for t in ('S1', 'S2', 'S20', 'O', 'T', object)] +
    [{'types': {'REF': 'S4'}},
     {'types': {'ALT': 'S1'}, 'alt_number': 1},
     {'alt_number': 1},
     {'alt_number': 2},
     {'alt_number': 5},
     {'numbers': {'ALT': 2}},
     {'types': {'QUAL': 'f8'}},
     {'fills': {'QUAL': -1}}]
)


@pytest.mark.parametrize('fn', ['sample.vcf', 'altlen.vcf', 'test32.vcf', 'test16.vcf',
                                'sample.utf8.vcf'])
@pytest.mark.parametrize('options', FIXED_OPTIONS)
def test_fixed_options(fn, options, monkeypatch, caplog):
    path = os.path.join(DATA, fn)
    assert_rust_matches_cython(path, monkeypatch, caplog,
                               fields=_fields(path, 'filters'), chunk_length=2, **options)


@pytest.mark.parametrize('options', FIXED_OPTIONS)
def test_fixed_options_computed_only(options, monkeypatch, caplog):
    # numalt, altlen and is_snp without ALT nor REF
    assert_rust_matches_cython(os.path.join(DATA, 'altlen.vcf'), monkeypatch, caplog,
                               fields=['numalt', 'altlen', 'is_snp'], **options)


@pytest.mark.parametrize('fn', FILES)
@pytest.mark.parametrize('chunk_length', [1, 3, 2**16])
def test_all_variants_fields(fn, chunk_length, monkeypatch, caplog):
    # without samples, '*' is every variants field
    assert_rust_matches_cython(os.path.join(DATA, fn), monkeypatch, caplog,
                               fields='*', samples=[], chunk_length=chunk_length)


INFO_INTEGERS = ['NS', 'AN', 'AC', 'DP', 'BaseCounts', 'ALTLEN']
INFO_FLOATS = ['AF', 'ABHet', 'FS', 'Dels']
INFO_STRINGS = ['AA', 'EFF', 'ANN']

INFO_OPTIONS = (
    [{'types': {f: t for f in INFO_INTEGERS}}
     for t in ('i1', 'i2', 'i8', 'u1', 'u2', 'u4', 'u8', 'f4', 'f8', bool)] +
    [{'types': {f: t for f in INFO_FLOATS}} for t in ('f8', 'i4', 'u1', bool)] +
    [{'types': {f: t for f in INFO_STRINGS}} for t in ('S1', 'S3', 'S50', 'O', 'T')] +
    [{'types': {'DB': 'i4', 'H2': 'S2', 'DS': 'f4'}},
     {'numbers': {'AC': 1, 'AF': 3, 'DP': 2, 'AA': 2, 'BaseCounts': 2, 'EFF': 4, 'DB': 2}},
     {'numbers': {'AA': 0, 'DP': 0}},
     {'alt_number': 1},
     {'alt_number': 3},
     {'fills': {'DP': -9, 'AF': -1.5, 'AN': 99, 'AC': 7, 'FS': 2}},
     {'fills': {'AF': 2.5, 'ABHet': -1, 'AA': 'x', 'DB': 1}},
     {'types': {'DP': 'u1', 'AF': 'f8'}, 'fills': {'DP': 3, 'AF': 0}}]
)


@pytest.mark.parametrize('fn', FILES)
@pytest.mark.parametrize('options', INFO_OPTIONS)
def test_info_options(fn, options, monkeypatch, caplog):
    path = os.path.join(DATA, fn)
    assert_rust_matches_cython(path, monkeypatch, caplog,
                               fields=_fields(path, 'infos'), chunk_length=2, **options)


INFO_HEADER = (b'##fileformat=VCFv4.2\n'
               b'##INFO=<ID=DP,Number=1,Type=Integer,Description="">\n'
               b'##INFO=<ID=AC,Number=A,Type=Integer,Description="">\n'
               b'##INFO=<ID=AF,Number=.,Type=Float,Description="">\n'
               b'##INFO=<ID=AA,Number=1,Type=String,Description="">\n'
               b'##INFO=<ID=CS,Number=2,Type=String,Description="">\n'
               b'##INFO=<ID=DB,Number=0,Type=Flag,Description="">\n'
               b'#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n')

INFO_LINES = (INFO_HEADER +
              b'2L\t1\t.\tA\tC,G\t.\t.\tDP=5;AC=1,2;AF=0.5,0.25;AA=T;CS=ab,cd;DB\n'
              b'2L\t2\t.\tA\tC\t.\t.\t.;DP=5\n'
              b'2L\t3\t.\tA\tC\t.\t.\tDP=5;;AF=.,0.3;;\n'
              b'2L\t4\t.\tA\tC\t.\t.\t=5;DP=7;=;AA=x=y\n'
              b'2L\t5\t.\tA\tC\t.\t.\tDP;AF=;AA=;CS=,;DB=0\n'
              b'2L\t6\t.\tA\tC\t.\t.\tDP=1,2,3;AC=x,4,5;AF=1e3x,inf,nan\n'
              b'2L\t7\t.\tA\tC\t.\t.\tAA=ABCDEF;AA=X;CS=long,str;CS=,y;DP=3;DP=.\n'
              b'2L\t8\t.\tA\tC\t.\t.\tXX=1;YY;DP= 12;AF=-0x1p-2\n'
              b'2L\t9\t.\tA\tC\t.\t.\tDP=99999999999;AC=-129,300\n'
              b'2L\t10\t.\tA\tC\t.\t.\t\n'
              b'2L\t11\t.\tA\tC\t.\t.\t\t\n'
              b'2L\t12\t.\tA\tC\t.\n'
              b'2L\t13\t.\tA\tC\t.\t.\tAA=\xc3\xa9;DB\tGT\n'
              b'2L\t14\t.\tA\tC\t.\t.\tDP=4;AA=end')


@pytest.mark.parametrize('fields', [
    ['DP', 'AC', 'AF', 'AA', 'CS', 'DB'],
    ['POS', 'XX', 'YY'],
    ['AA'],
    ['DB', 'DP'],
])
@pytest.mark.parametrize('options', [
    {},
    {'types': {'AA': 'S3', 'CS': 'S2'}},
    {'types': {'DP': 'i1', 'AC': 'u1', 'AF': 'f8'}, 'fills': {'DP': 0}},
    {'numbers': {'DP': 2, 'AF': 1, 'CS': 1}, 'alt_number': 3},
])
@pytest.mark.parametrize('gz', [False, True])
def test_info_edge_cases(fields, options, gz, tmp_path, monkeypatch, caplog):
    if gz:
        import gzip
        path = tmp_path / 'info.vcf.gz'
        with gzip.open(path, 'wb') as f:
            f.write(INFO_LINES)
    else:
        path = tmp_path / 'info.vcf'
        path.write_bytes(INFO_LINES)
    assert_rust_matches_cython(str(path), monkeypatch, caplog, fields=fields,
                               chunk_length=3, **options)


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


EDGE_FILTERS = ['FILTER_q10', 'FILTER_s50', 'FILTER_PASSq10']


@pytest.mark.parametrize('name', sorted(EDGE_CASES))
@pytest.mark.parametrize('gz', [False, True])
@pytest.mark.parametrize('options', [{}, {'alt_number': 2},
                                     {'types': {f: 'S2' for f in STRING_FIELDS}}])
def test_edge_cases_fixed(name, gz, options, edge_vcf, monkeypatch, caplog):
    assert_rust_matches_cython(edge_vcf(name, gz), monkeypatch, caplog,
                               fields=FIXED_FIELDS + EDGE_FILTERS, chunk_length=2,
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


def test_default_fields(monkeypatch, caplog):
    # read_vcf with its default fields does not fall back
    callset = _read(os.path.join(DATA, 'sample.vcf'), monkeypatch, caplog)
    assert RUST in caplog.text
    assert FALLBACK not in caplog.text
    assert ['19', '19', '20'] == callset['variants/CHROM'][:3].tolist()
    assert np.dtypes.StringDType() == callset['variants/ALT'].dtype
    assert [True, False] == callset['variants/FILTER_PASS'][2:4].tolist()


def test_exclude_and_rename_fields(monkeypatch, caplog):
    path = os.path.join(DATA, 'sample.vcf')
    kwargs = dict(exclude_fields=['ID', 'QUAL'],
                  rename_fields={'variants/CHROM': 'variants/chromosome',
                                 'variants/FILTER_PASS': 'variants/pass'})
    expected = _read(path, monkeypatch, caplog, backend='cython', **kwargs)
    actual = _read(path, monkeypatch, caplog, backend='rust', **kwargs)
    assert RUST in caplog.text
    assert sorted(expected) == sorted(actual)
    assert 'variants/chromosome' in actual and 'variants/ID' not in actual
    for k in expected:
        assert expected[k].dtype == actual[k].dtype, k
        np.testing.assert_array_equal(expected[k], actual[k], err_msg=k)


@pytest.mark.parametrize('fields, field', [
    (['POS', 'GT', 'GQ'], 'calldata/GQ'),
    # all fields include FORMAT fields
    ('*', 'calldata/'),
])
def test_fallback_unsupported_field(fields, field, monkeypatch, caplog):
    callset = _read(os.path.join(DATA, 'sample.vcf'), monkeypatch, caplog,
                    fields=fields)
    assert field in _reason(caplog)
    assert 9 == len(callset['variants/POS'])


@pytest.mark.parametrize('types, field', [
    ({'GT': 'f4'}, 'calldata/GT'),
    ({'GT': 'genotype_ac/i1'}, 'calldata/GT'),
    ({'REF': 'S'}, 'variants/REF'),
])
def test_fallback_unsupported_type(types, field, monkeypatch, caplog):
    _read(os.path.join(DATA, 'sample.vcf'), monkeypatch, caplog,
          fields=['POS', 'GT', 'CHROM', 'REF'], types=types)
    assert field in _reason(caplog)


@pytest.mark.parametrize('field, kwargs', [
    ('DP', {'types': {'DP': 'f2'}}),
    ('DP', {'types': {'DP': 'U3'}}),
    ('AA', {'types': {'AA': 'S'}}),
    ('DP', {'types': {'DP': 'c8'}}),
    ('DP', {'types': {'DP': '>i4'}}),
    ('AF', {'types': {'AF': '>f8'}}),
    ('DP', {'fills': {'DP': 1.5}}),
    ('AF', {'fills': {'AF': 'x'}}),
    ('DP', {'fills': {'DP': None}}),
    ('DP', {'numbers': {'DP': -1}}),
])
def test_fallback_info(field, kwargs, monkeypatch, caplog):
    # INFO fields that the Rust parser does not cover are read by the Cython
    # one, which skips some of them and rejects others
    path = os.path.join(DATA, 'sample.vcf')
    fields = ['POS', 'DP', 'AF', 'AA']
    monkeypatch.setenv('ALLEL2_BACKEND', 'cython')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            read_vcf(path, fields=fields, **kwargs)
        expected = None
    except Exception as e:
        expected = type(e)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if expected is None:
            _read(path, monkeypatch, caplog, fields=fields, **kwargs)
        else:
            with pytest.raises(expected):
                _read(path, monkeypatch, caplog, fields=fields, **kwargs)
    assert 'variants/' + field in _reason(caplog)


def test_flag_type_warning(monkeypatch, caplog):
    # like the Cython parser, a field of number 0 is a flag
    for backend in 'cython', 'rust':
        with pytest.warns(UserWarning, match='cannot have non-bool dtype'):
            callset = _read(os.path.join(DATA, 'sample.vcf'), monkeypatch, caplog,
                            backend=backend, fields=['DB'], types={'DB': 'i4'})
        assert np.dtype(bool) == callset['variants/DB'].dtype


def test_fallback_rejected_type(monkeypatch, caplog):
    # the Cython parser rejects strings that are neither bytes nor objects
    with pytest.raises(ValueError, match='expected string'):
        _read(os.path.join(DATA, 'sample.vcf'), monkeypatch, caplog,
              fields=['POS', 'CHROM'], types={'CHROM': 'U10'})
    assert 'variants/CHROM' in _reason(caplog)


def test_pos_type_warning(monkeypatch, caplog):
    # like the Cython parser, POS is always int32
    for backend in 'cython', 'rust':
        with pytest.warns(UserWarning, match='only int32 supported for POS'):
            callset = _read(os.path.join(DATA, 'sample.vcf'), monkeypatch, caplog,
                            backend=backend, fields=['POS'], types={'POS': 'i8'})
        assert np.dtype('int32') == callset['variants/POS'].dtype


def test_qual_type_warning(monkeypatch, caplog):
    # like the Cython parser, QUAL is always float32
    for backend in 'cython', 'rust':
        with pytest.warns(UserWarning, match='only float32 supported for QUAL'):
            callset = _read(os.path.join(DATA, 'sample.vcf'), monkeypatch, caplog,
                            backend=backend, fields=['QUAL'], types={'QUAL': 'f8'})
        assert np.dtype('float32') == callset['variants/QUAL'].dtype


@pytest.mark.parametrize('backend', ['cython', 'rust'])
def test_invalid_utf8(backend, tmp_path, monkeypatch, caplog):
    # strings that are not UTF-8 raise the same error with both parsers
    path = tmp_path / 'bad.vcf'
    path.write_bytes(HEADER + b'2L\t1\tid\xff\tA\tC\t.\t.\t.\n')
    with pytest.raises(UnicodeDecodeError):
        _read(str(path), monkeypatch, caplog, backend=backend, fields=['ID'])


@pytest.mark.parametrize('backend', ['cython', 'rust'])
@pytest.mark.parametrize('fields', [['altlen'], ['altlen', 'ALT']])
def test_altlen_shape(backend, fields, monkeypatch, caplog):
    # one ALT value per variant gives a 1-D altlen, with or without ALT
    callset = _read(os.path.join(DATA, 'altlen.vcf'), monkeypatch, caplog,
                    backend=backend, fields=fields, alt_number=1)
    assert 1 == callset['variants/altlen'].ndim
