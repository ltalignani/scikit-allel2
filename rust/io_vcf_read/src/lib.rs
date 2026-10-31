//! Rust VCF parser behind `allel2.io.vcf_read.iter_vcf_chunks`.
//!
//! Python reads and normalises the headers and the parameters; this module
//! parses the data lines into chunks shaped exactly like those of the Cython
//! `VCFChunkIterator`: `(chunk, chunk_length, chrom, pos)`, where `chunk` maps
//! field names to NumPy arrays. The Cython parser is the reference: values,
//! dtypes, shapes and warning messages match it, quirks included.
//!
//! Fields handled so far: the fixed columns (CHROM, POS, ID, REF, ALT, QUAL,
//! the FILTER flags and the INFO fields), the computed fields `numalt`,
//! `altlen` and `is_snp`, and `calldata/GT`.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::sync::Mutex;

use flate2::read::MultiGzDecoder;
use numpy::{Element, IntoPyArray, PyArrayMethods};
use pyo3::exceptions::{PyOSError, PyOverflowError, PyUserWarning, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyString, PyTuple};

const BUFFER_SIZE: usize = 1 << 20;

/// Outcome of parsing an integer the way C `strtol` does in the Cython parser.
enum Int {
    /// empty or '.', no warning
    Missing,
    Parsed(i64),
    /// leading digits parsed, trailing characters ignored, with a warning
    Partial(i64),
    /// no digits, with a warning
    Error,
}

fn strtol(s: &[u8]) -> Int {
    if s.is_empty() || s == b"." {
        return Int::Missing;
    }
    let mut i = 0;
    while i < s.len() && matches!(s[i], b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c) {
        i += 1;
    }
    let mut negative = false;
    if i < s.len() && (s[i] == b'+' || s[i] == b'-') {
        negative = s[i] == b'-';
        i += 1;
    }
    let start = i;
    let mut value: i64 = 0;
    let mut overflow = false;
    while i < s.len() && s[i].is_ascii_digit() {
        if !overflow {
            let digit = (s[i] - b'0') as i64;
            let next = value.checked_mul(10).and_then(|v| {
                if negative { v.checked_sub(digit) } else { v.checked_add(digit) }
            });
            match next {
                Some(v) => value = v,
                None => overflow = true,
            }
        }
        i += 1;
    }
    if i == start {
        return Int::Error;
    }
    if overflow {
        // strtol saturates
        value = if negative { i64::MIN } else { i64::MAX };
    }
    if i == s.len() { Int::Parsed(value) } else { Int::Partial(value) }
}

/// Outcome of parsing a floating point value the way C `strtod` does in the
/// Cython parser, with the same cases as `Int`.
enum Float {
    Missing,
    Parsed(f64),
    Partial(f64),
    Error,
}

fn strtod(s: &[u8]) -> Float {
    if s.is_empty() || s == b"." {
        return Float::Missing;
    }
    match float_prefix(s) {
        None => Float::Error,
        Some((n, v)) if n == s.len() => Float::Parsed(v),
        Some((_, v)) => Float::Partial(v),
    }
}

fn is_c_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

/// The longest prefix of `s` that C `strtod` parses: its length and value.
fn float_prefix(s: &[u8]) -> Option<(usize, f64)> {
    let mut i = 0;
    while i < s.len() && is_c_space(s[i]) {
        i += 1;
    }
    let number_start = i;
    let mut sign = 1.0;
    if i < s.len() && (s[i] == b'+' || s[i] == b'-') {
        if s[i] == b'-' {
            sign = -1.0;
        }
        i += 1;
    }
    let rest = &s[i..];
    let starts_with = |word: &[u8]| rest.len() >= word.len() && rest[..word.len()].eq_ignore_ascii_case(word);
    if starts_with(b"infinity") {
        return Some((i + 8, sign * f64::INFINITY));
    }
    if starts_with(b"inf") {
        return Some((i + 3, sign * f64::INFINITY));
    }
    if starts_with(b"nan") {
        let mut end = i + 3;
        // optional "(n-char-sequence)"
        if s.get(end) == Some(&b'(') {
            let mut k = end + 1;
            while k < s.len() && (s[k].is_ascii_alphanumeric() || s[k] == b'_') {
                k += 1;
            }
            if s.get(k) == Some(&b')') {
                end = k + 1;
            }
        }
        return Some((end, f64::NAN));
    }
    if rest.len() > 2 && rest[0] == b'0' && rest[1] | 0x20 == b'x' {
        if let Some((n, v)) = hex_float(&rest[2..]) {
            return Some((i + 2 + n, sign * v));
        }
    }
    // decimal: digits, optionally a point and digits, at least one digit,
    // then an exponent if it has digits
    let digits = |from: usize| s[from..].iter().take_while(|c| c.is_ascii_digit()).count();
    let integer = digits(i);
    let mut end = i + integer;
    let mut fraction = 0;
    if s.get(end) == Some(&b'.') {
        fraction = digits(end + 1);
        end += 1 + fraction;
    }
    if integer + fraction == 0 {
        return None;
    }
    if matches!(s.get(end), Some(b'e' | b'E')) {
        let mut k = end + 1;
        if matches!(s.get(k), Some(b'+' | b'-')) {
            k += 1;
        }
        let exponent = digits(k);
        if exponent > 0 {
            end = k + exponent;
        }
    }
    // the prefix is ASCII, and correctly rounded like strtod
    let text = std::str::from_utf8(&s[number_start..end]).ok()?;
    Some((end, text.parse::<f64>().ok()?))
}

/// A hexadecimal float after its "0x": length and value, if it has digits.
fn hex_float(s: &[u8]) -> Option<(usize, f64)> {
    let mut i = 0;
    let mut mantissa = 0.0;
    let mut exponent: i64 = 0;
    let mut n_digits = 0;
    let mut point = false;
    while i < s.len() {
        let c = s[i];
        if c == b'.' && !point {
            point = true;
        } else if let Some(d) = (c as char).to_digit(16) {
            mantissa = mantissa * 16.0 + d as f64;
            n_digits += 1;
            if point {
                exponent -= 4;
            }
        } else {
            break;
        }
        i += 1;
    }
    if n_digits == 0 {
        return None;
    }
    if matches!(s.get(i), Some(b'p' | b'P')) {
        let mut k = i + 1;
        let negative = s.get(k) == Some(&b'-');
        if matches!(s.get(k), Some(b'+' | b'-')) {
            k += 1;
        }
        let start = k;
        let mut e: i64 = 0;
        while k < s.len() && s[k].is_ascii_digit() {
            e = (e * 10 + (s[k] - b'0') as i64).min(1 << 20);
            k += 1;
        }
        if k > start {
            exponent += if negative { -e } else { e };
            i = k;
        }
    }
    Some((i, mantissa * 2f64.powi(exponent.clamp(-1 << 20, 1 << 20) as i32)))
}

/// Integer dtypes of genotype and INFO arrays, with C casts from `long`.
trait Integer: Element + Copy + Send + Sync + TryFrom<i128> {
    fn from_long(v: i64) -> Self;
    fn default_fill() -> Self;
}

macro_rules! impl_integer {
    ($($t:ty => $fill:expr),*) => {$(
        impl Integer for $t {
            #[inline]
            fn from_long(v: i64) -> Self { v as $t }
            fn default_fill() -> Self { $fill }
        }
    )*};
}

impl_integer!(i8 => -1, i16 => -1, i32 => -1, i64 => -1,
             u8 => u8::MAX, u16 => u16::MAX, u32 => u32::MAX, u64 => u64::MAX);

/// Parse an integer into `slot`, leaving it unchanged when missing or
/// unparsable. Returns the warning to give, if any.
fn parse_integer<T: Integer>(s: &[u8], slot: &mut T) -> Option<&'static str> {
    match strtol(s) {
        Int::Missing => None,
        Int::Parsed(v) => {
            *slot = T::from_long(v);
            None
        }
        Int::Partial(v) => {
            *slot = T::from_long(v);
            Some("not all characters parsed for integer value")
        }
        Int::Error => Some("error parsing integer value"),
    }
}

/// Floating point dtypes of QUAL and INFO arrays, with C casts from `double`.
trait Floating: Element + Copy + Send + Sync {
    fn from_double(v: f64) -> Self;
}

impl Floating for f32 {
    fn from_double(v: f64) -> Self { v as f32 }
}

impl Floating for f64 {
    fn from_double(v: f64) -> Self { v }
}

/// Parse a floating point value into `slot`, as `parse_integer` does.
fn parse_floating<T: Floating>(s: &[u8], slot: &mut T) -> Option<&'static str> {
    match strtod(s) {
        Float::Missing => None,
        Float::Parsed(v) => {
            *slot = T::from_double(v);
            None
        }
        Float::Partial(v) => {
            *slot = T::from_double(v);
            Some("not all characters parsed for floating point value")
        }
        Float::Error => Some("error parsing floating point value"),
    }
}

/// Parser state labels used in warning messages, as in the Cython parser.
#[derive(Clone, Copy, PartialEq)]
enum State {
    Chrom,
    Pos,
    Qual,
    Info,
    Calldata,
    Eol,
    Eof,
}

impl State {
    fn label(self) -> &'static str {
        match self {
            State::Chrom => "CHROM",
            State::Pos => "POS",
            State::Qual => "QUAL",
            State::Info => "INFO",
            State::Calldata => "CALLDATA",
            State::Eol => "EOL",
            State::Eof => "EOF",
        }
    }
}

#[derive(Clone)]
struct GtSpec {
    dtype: String,
    number: usize,
    fill: Option<i128>,
}

/// How a string field is stored: as Python strings in an object array, or as
/// bytes truncated to a fixed width.
#[derive(Clone, Copy, PartialEq)]
enum StrType {
    Object,
    Bytes(usize),
}

impl StrType {
    /// From a dtype code, 'O' or 'S<width>'.
    fn parse(field: &str, dtype: &str) -> PyResult<Self> {
        let width = dtype.strip_prefix('S').and_then(|n| n.parse::<usize>().ok());
        match (dtype, width) {
            ("O", _) => Ok(StrType::Object),
            (_, Some(n)) if n > 0 => Ok(StrType::Bytes(n)),
            _ => Err(PyValueError::new_err(format!("unsupported dtype {dtype:?} for field {field:?}"))),
        }
    }
}

/// The values of a string field, before conversion to a NumPy array.
struct StrColumn {
    kind: StrType,
    data: Vec<u8>,
    // end of each value in `data`, for objects only
    ends: Vec<usize>,
}

impl StrColumn {
    fn new(kind: StrType) -> Self {
        StrColumn { kind, data: Vec::new(), ends: Vec::new() }
    }

    fn push(&mut self, value: &[u8]) {
        match self.kind {
            StrType::Object => {
                self.data.extend_from_slice(value);
                self.ends.push(self.data.len());
            }
            StrType::Bytes(width) => {
                let n = value.len().min(width);
                self.data.extend_from_slice(&value[..n]);
                self.data.resize(self.data.len() + width - n, 0);
            }
        }
    }

    fn into_numpy<'py>(self, py: Python<'py>, shape: &[usize]) -> PyResult<Bound<'py, PyAny>> {
        match self.kind {
            StrType::Object => {
                let mut start = 0;
                let values = self.ends.iter().map(|&end| {
                    let value = &self.data[start..end];
                    start = end;
                    Some(value)
                });
                object_array(py, values, shape)
            }
            StrType::Bytes(width) => bytes_array(py, self.data, width, shape),
        }
    }
}

/// A NumPy object array of Python strings decoded from UTF-8, '' for `None`.
fn object_array<'a, 'py>(py: Python<'py>, values: impl Iterator<Item = Option<&'a [u8]>>,
                         shape: &[usize]) -> PyResult<Bound<'py, PyAny>> {
    let mut objects = Vec::new();
    for value in values {
        let bytes = value.unwrap_or(b"");
        let object = match std::str::from_utf8(bytes) {
            Ok(s) => PyString::new(py, s).into_any(),
            // raises the same UnicodeDecodeError as the Cython parser
            Err(_) => PyBytes::new(py, bytes).call_method1("decode", ("utf8",))?,
        };
        objects.push(object.unbind());
    }
    objects.into_pyarray(py).call_method1("reshape", (PyTuple::new(py, shape)?,))
}

/// A NumPy array of fixed-width bytes `S<width>` over `data`.
fn bytes_array<'py>(py: Python<'py>, data: Vec<u8>, width: usize, shape: &[usize])
                    -> PyResult<Bound<'py, PyAny>> {
    data.into_pyarray(py)
        .call_method1("view", (format!("S{width}"),))?
        .call_method1("reshape", (PyTuple::new(py, shape)?,))
}

/// The values of an INFO field in a chunk, `number` per variant.
trait InfoColumn: Send + Sync {
    /// A column of the same type with no values, for a new chunk.
    fn empty(&self) -> Box<dyn InfoColumn>;
    /// Append the fill values of a new variant.
    fn push_fill(&mut self, number: usize);
    /// Parse `value` into the values of the last variant, as the Cython
    /// parser does: values beyond `number` are ignored, and a repeated key
    /// overwrites what its values give.
    fn store(&mut self, value: &[u8], number: usize, warn: &mut dyn FnMut(&'static str));
    fn into_numpy<'py>(self: Box<Self>, py: Python<'py>, shape: &[usize])
                       -> PyResult<Bound<'py, PyAny>>;
}

fn info_values(value: &[u8], number: usize) -> impl Iterator<Item = &[u8]> {
    value.split(|&c| c == b',').take(number)
}

#[derive(Default)]
struct FlagColumn(Vec<bool>);

impl InfoColumn for FlagColumn {
    fn empty(&self) -> Box<dyn InfoColumn> {
        Box::new(FlagColumn::default())
    }
    fn push_fill(&mut self, _number: usize) {
        self.0.push(false);
    }
    fn store(&mut self, _value: &[u8], _number: usize, _warn: &mut dyn FnMut(&'static str)) {
        if let Some(flag) = self.0.last_mut() {
            *flag = true;
        }
    }
    fn into_numpy<'py>(self: Box<Self>, py: Python<'py>, _shape: &[usize])
                       -> PyResult<Bound<'py, PyAny>> {
        Ok(self.0.into_pyarray(py).into_any())
    }
}

struct NumberColumn<T> {
    data: Vec<T>,
    fill: T,
    parse: fn(&[u8], &mut T) -> Option<&'static str>,
}

impl<T: Element + Copy + Send + Sync + 'static> InfoColumn for NumberColumn<T> {
    fn empty(&self) -> Box<dyn InfoColumn> {
        Box::new(NumberColumn { data: Vec::new(), fill: self.fill, parse: self.parse })
    }
    fn push_fill(&mut self, number: usize) {
        self.data.resize(self.data.len() + number, self.fill);
    }
    fn store(&mut self, value: &[u8], number: usize, warn: &mut dyn FnMut(&'static str)) {
        let offset = self.data.len() - number;
        for (slot, value) in self.data[offset..].iter_mut().zip(info_values(value, number)) {
            if let Some(message) = (self.parse)(value, slot) {
                warn(message);
            }
        }
    }
    fn into_numpy<'py>(self: Box<Self>, py: Python<'py>, shape: &[usize])
                       -> PyResult<Bound<'py, PyAny>> {
        self.data.into_pyarray(py).reshape(shape).map(|a| a.into_any())
    }
}

struct BytesColumn {
    width: usize,
    data: Vec<u8>,
}

impl InfoColumn for BytesColumn {
    fn empty(&self) -> Box<dyn InfoColumn> {
        Box::new(BytesColumn { width: self.width, data: Vec::new() })
    }
    fn push_fill(&mut self, number: usize) {
        self.data.resize(self.data.len() + number * self.width, 0);
    }
    fn store(&mut self, value: &[u8], number: usize, _warn: &mut dyn FnMut(&'static str)) {
        let width = self.width;
        let offset = self.data.len() - number * width;
        for (slot, value) in self.data[offset..].chunks_mut(width).zip(info_values(value, number)) {
            // overwrites the first bytes only, as the Cython parser does
            let n = value.len().min(width);
            slot[..n].copy_from_slice(&value[..n]);
        }
    }
    fn into_numpy<'py>(self: Box<Self>, py: Python<'py>, shape: &[usize])
                       -> PyResult<Bound<'py, PyAny>> {
        bytes_array(py, self.data, self.width, shape)
    }
}

#[derive(Default)]
struct ObjectColumn {
    data: Vec<u8>,
    // range of each value in `data`, `None` for ''
    values: Vec<Option<(usize, usize)>>,
}

impl InfoColumn for ObjectColumn {
    fn empty(&self) -> Box<dyn InfoColumn> {
        Box::new(ObjectColumn::default())
    }
    fn push_fill(&mut self, number: usize) {
        self.values.resize(self.values.len() + number, None);
    }
    fn store(&mut self, value: &[u8], number: usize, _warn: &mut dyn FnMut(&'static str)) {
        let offset = self.values.len() - number;
        for (slot, value) in self.values[offset..].iter_mut().zip(info_values(value, number)) {
            // an empty value leaves the slot unchanged
            if !value.is_empty() {
                let start = self.data.len();
                self.data.extend_from_slice(value);
                *slot = Some((start, self.data.len()));
            }
        }
    }
    fn into_numpy<'py>(self: Box<Self>, py: Python<'py>, shape: &[usize])
                       -> PyResult<Bound<'py, PyAny>> {
        let values = self.values.iter().map(|v| v.map(|(start, end)| &self.data[start..end]));
        object_array(py, values, shape)
    }
}

fn integer_column<T: Integer + 'static>(fill: Option<i128>, dtype: &str) -> PyResult<Box<dyn InfoColumn>> {
    let fill = fill_value::<T>(fill, dtype)?;
    Ok(Box::new(NumberColumn { data: Vec::new(), fill, parse: parse_integer::<T> }))
}

fn floating_column<T: Floating + 'static>(fill: T) -> Box<dyn InfoColumn> {
    Box::new(NumberColumn { data: Vec::new(), fill, parse: parse_floating::<T> })
}

/// A column for an INFO field of dtype code `dtype`: 'b1' for flags, 'i1' to
/// 'u8', 'f4', 'f8', 'S<width>' or 'O'.
fn info_column(name: &str, dtype: &str, fill: Option<&Bound<'_, PyAny>>) -> PyResult<Box<dyn InfoColumn>> {
    let integer_fill = || fill.map(|f| f.extract::<i128>()).transpose();
    let floating_fill = || fill.map_or(Ok(f64::NAN), |f| f.extract::<f64>());
    Ok(match dtype {
        "b1" => Box::new(FlagColumn::default()),
        "i1" => integer_column::<i8>(integer_fill()?, "int8")?,
        "i2" => integer_column::<i16>(integer_fill()?, "int16")?,
        "i4" => integer_column::<i32>(integer_fill()?, "int32")?,
        "i8" => integer_column::<i64>(integer_fill()?, "int64")?,
        "u1" => integer_column::<u8>(integer_fill()?, "uint8")?,
        "u2" => integer_column::<u16>(integer_fill()?, "uint16")?,
        "u4" => integer_column::<u32>(integer_fill()?, "uint32")?,
        "u8" => integer_column::<u64>(integer_fill()?, "uint64")?,
        "f4" => floating_column(floating_fill()? as f32),
        "f8" => floating_column(floating_fill()?),
        _ => match StrType::parse(name, dtype)? {
            StrType::Object => Box::new(ObjectColumn::default()),
            StrType::Bytes(width) => Box::new(BytesColumn { width, data: Vec::new() }),
        },
    })
}

/// An INFO field to store.
struct InfoSpec {
    key: Vec<u8>,
    number: usize,
    // with no values, the prototype of each chunk's column
    column: Box<dyn InfoColumn>,
}

/// The fixed columns and computed fields to store.
struct FixedSpec {
    chrom: Option<StrType>,
    id: Option<StrType>,
    ref_: Option<StrType>,
    alt: Option<StrType>,
    // number of ALT values stored per variant
    alt_number: usize,
    numalt: bool,
    altlen: bool,
    is_snp: bool,
    qual: bool,
    // sorted, as the Cython parser outputs them
    filters: Vec<Vec<u8>>,
    // sorted by key, for searching
    infos: Vec<InfoSpec>,
}

/// What to parse, fixed for the whole file.
struct Spec {
    samples: Vec<String>,
    loc_samples: Vec<bool>,
    n_samples_out: usize,
    store_pos: bool,
    fixed: FixedSpec,
    gt: Option<GtSpec>,
}

/// One chunk of parsed data, before conversion to NumPy arrays.
struct Chunk<T> {
    length: usize,
    chrom_values: Option<StrColumn>,
    pos: Vec<i32>,
    id: Option<StrColumn>,
    ref_: Option<StrColumn>,
    alt: Option<StrColumn>,
    numalt: Vec<i32>,
    altlen: Vec<i32>,
    is_snp: Vec<bool>,
    qual: Vec<f32>,
    // one flag per filter, variant after variant
    filters: Vec<bool>,
    // in the order of `FixedSpec::infos`
    infos: Vec<Box<dyn InfoColumn>>,
    gt: Vec<T>,
    // CHROM of the last variant, null-terminated
    chrom: Vec<u8>,
    last_pos: i64,
    warnings: Vec<String>,
}

impl<T> Chunk<T> {
    fn new(fixed: &FixedSpec) -> Self {
        Chunk {
            length: 0,
            chrom_values: fixed.chrom.map(StrColumn::new),
            pos: Vec::new(),
            id: fixed.id.map(StrColumn::new),
            ref_: fixed.ref_.map(StrColumn::new),
            alt: fixed.alt.map(StrColumn::new),
            numalt: Vec::new(),
            altlen: Vec::new(),
            is_snp: Vec::new(),
            qual: Vec::new(),
            filters: Vec::new(),
            infos: fixed.infos.iter().map(|i| i.column.empty()).collect(),
            gt: Vec::new(),
            chrom: Vec::new(),
            last_pos: 0,
            warnings: Vec::new(),
        }
    }
}

#[pyclass(module = "allel2.opt.io_vcf_read_rs")]
struct VCFChunkIterator {
    // PyO3 classes must be Sync; the lock is never contended
    reader: Mutex<Box<dyn BufRead + Send>>,
    chunk_length: usize,
    spec: Spec,
    // index of the last variant read, -1 before the first
    variant_index: i64,
    eof: bool,
}

fn open(path: &str) -> PyResult<Box<dyn BufRead + Send>> {
    let file = File::open(path).map_err(|e| PyOSError::new_err(format!("{path}: {e}")))?;
    // same rule as the Python input stream
    let reader: Box<dyn Read + Send> = if path.ends_with("gz") {
        Box::new(MultiGzDecoder::new(BufReader::with_capacity(BUFFER_SIZE, file)))
    } else {
        Box::new(file)
    };
    Ok(Box::new(BufReader::with_capacity(BUFFER_SIZE, reader)))
}

/// Read one line into `line` without its terminator. Returns `None` at end of
/// file, otherwise whether the line had a terminator.
///
/// As in the Cython parser, a line ends at LF, CR LF or a lone CR, and a null
/// byte is the end of the file.
fn read_line(reader: &mut dyn BufRead, line: &mut Vec<u8>) -> std::io::Result<Option<bool>> {
    line.clear();
    loop {
        let buf = reader.fill_buf()?;
        if buf.is_empty() {
            return Ok(if line.is_empty() { None } else { Some(false) });
        }
        let Some(i) = memchr::memchr3(b'\n', b'\r', 0, buf) else {
            line.extend_from_slice(buf);
            let n = buf.len();
            reader.consume(n);
            continue;
        };
        let c = buf[i];
        line.extend_from_slice(&buf[..i]);
        if c == 0 {
            // left unread, so that reading stops here
            reader.consume(i);
            return Ok(if line.is_empty() { None } else { Some(false) });
        }
        reader.consume(i + 1);
        if c == b'\r' && reader.fill_buf()?.first() == Some(&b'\n') {
            reader.consume(1);
        }
        return Ok(Some(true));
    }
}

/// Context needed to format a warning like the Cython parser does.
struct Context<'a> {
    variant_index: i64,
    chrom: &'a [u8],
    pos: i64,
}

impl Context<'_> {
    fn message(&self, message: &str, state: State) -> String {
        let mut m = format!("{message}; field: {}; variant: {}", state.label(), self.variant_index);
        if !matches!(state, State::Chrom | State::Pos) {
            m.push_str(&format!(" ({}:{})", String::from_utf8_lossy(self.chrom), self.pos));
        }
        m
    }
}

impl VCFChunkIterator {
    /// Parse up to `chunk_length` lines. Runs without the GIL.
    fn parse_chunk<T: Integer>(&mut self, fill: T) -> std::io::Result<Chunk<T>> {
        let mut chunk = Chunk::new(&self.spec.fixed);
        let reader = self.reader.get_mut().unwrap_or_else(|e| e.into_inner());
        let mut line = Vec::new();
        while chunk.length < self.chunk_length {
            let terminated = match read_line(reader.as_mut(), &mut line)? {
                None => {
                    // end of file at the start of a line
                    self.eof = true;
                    chunk.chrom.clear();
                    chunk.last_pos = 0;
                    break;
                }
                Some(t) => t,
            };
            let line_end = if terminated { State::Eol } else { State::Eof };
            let pos = parse_line(&line, line_end, self.variant_index, &self.spec, &mut chunk, fill);
            self.variant_index += 1;
            chunk.length += 1;
            chunk.last_pos = pos;
            if !terminated {
                self.eof = true;
                break;
            }
        }
        Ok(chunk)
    }

    fn next_chunk<'py, T: Integer>(&mut self, py: Python<'py>, fill: T)
                                  -> PyResult<Option<Bound<'py, PyAny>>> {
        let chunk = py
            .detach(|| self.parse_chunk(fill))
            .map_err(|e| PyOSError::new_err(e.to_string()))?;
        if !chunk.warnings.is_empty() {
            // through Python, as messages may contain null bytes (see Context)
            let warn = py.import("warnings")?.getattr("warn")?;
            let category = py.get_type::<PyUserWarning>();
            for w in &chunk.warnings {
                warn.call1((w.as_str(), &category))?;
            }
        }
        if chunk.length == 0 {
            return Ok(None);
        }
        let spec = &self.spec;
        let n = chunk.length;
        // arrays of one value per variant are 1-D, as in the Cython parser
        let alt_shape: &[usize] = if spec.fixed.alt_number == 1 { &[n] } else { &[n, spec.fixed.alt_number] };
        let dict = PyDict::new(py);
        if let Some(c) = chunk.chrom_values {
            dict.set_item("variants/CHROM", c.into_numpy(py, &[n])?)?;
        }
        if spec.store_pos {
            dict.set_item("variants/POS", chunk.pos.into_pyarray(py))?;
        }
        if let Some(c) = chunk.id {
            dict.set_item("variants/ID", c.into_numpy(py, &[n])?)?;
        }
        if let Some(c) = chunk.ref_ {
            dict.set_item("variants/REF", c.into_numpy(py, &[n])?)?;
        }
        if let Some(c) = chunk.alt {
            dict.set_item("variants/ALT", c.into_numpy(py, alt_shape)?)?;
        }
        if spec.fixed.numalt {
            dict.set_item("variants/numalt", chunk.numalt.into_pyarray(py))?;
        }
        if spec.fixed.altlen {
            dict.set_item("variants/altlen", chunk.altlen.into_pyarray(py).reshape(alt_shape)?)?;
        }
        if spec.fixed.is_snp {
            dict.set_item("variants/is_snp", chunk.is_snp.into_pyarray(py))?;
        }
        if spec.fixed.qual {
            dict.set_item("variants/QUAL", chunk.qual.into_pyarray(py))?;
        }
        let n_filters = spec.fixed.filters.len();
        for (i, key) in spec.fixed.filters.iter().enumerate() {
            let flags: Vec<bool> = chunk.filters.iter().skip(i).step_by(n_filters).copied().collect();
            let field = format!("variants/FILTER_{}", String::from_utf8_lossy(key));
            dict.set_item(field, flags.into_pyarray(py))?;
        }
        for (info, column) in spec.fixed.infos.iter().zip(chunk.infos) {
            let shape: &[usize] = if info.number == 1 { &[n] } else { &[n, info.number] };
            let field = format!("variants/{}", String::from_utf8_lossy(&info.key));
            dict.set_item(field, column.into_numpy(py, shape)?)?;
        }
        if let Some(g) = &spec.gt {
            let array = chunk.gt.into_pyarray(py);
            let array = if g.number == 1 {
                array.reshape([n, spec.n_samples_out])?.into_any()
            } else {
                array.reshape([n, spec.n_samples_out, g.number])?.into_any()
            };
            dict.set_item("calldata/GT", array)?;
        }
        let tuple = (dict, chunk.length, PyBytes::new(py, &chunk.chrom), chunk.last_pos);
        Ok(Some(tuple.into_pyobject(py)?.into_any()))
    }
}

/// Parse one data line into the next variant of `chunk`, and return its POS
/// (0 when missing or unparsable). Every field gets a value for the variant,
/// the fill value when the line is too short for it.
fn parse_line<T: Integer>(
    line: &[u8],
    line_end: State,
    previous_variant: i64,
    spec: &Spec,
    chunk: &mut Chunk<T>,
    fill: T,
) -> i64 {
    let mut columns = line.split(|&c| c == b'\t');
    // CHROM, POS, ID, REF, ALT, QUAL, FILTER, INFO and FORMAT
    let fixed: [Option<&[u8]>; 9] = std::array::from_fn(|_| columns.next());
    // state of a warning about column `i`, which ends the line if no tab follows
    let state = |i: usize, own: State| if fixed[i + 1].is_some() { own } else { line_end };
    let warnings = &mut chunk.warnings;

    // CHROM
    let chrom_field = fixed[0].unwrap_or(b"");
    let chrom = &mut chunk.chrom;
    chrom.clear();
    chrom.extend_from_slice(chrom_field);
    let mut ctx = Context { variant_index: previous_variant, chrom: b"", pos: 0 };
    if chrom_field.is_empty() {
        warnings.push(ctx.message("empty CHROM", state(0, State::Chrom)));
    }
    chrom.push(0);
    ctx.chrom = chrom;

    // POS
    let mut pos: i64 = 0;
    if let Some(pos_field) = fixed[1] {
        let pos_state = state(1, State::Pos);
        if pos_field.is_empty() {
            warnings.push(ctx.message("empty POS", pos_state));
        } else {
            match strtol(pos_field) {
                Int::Missing => {}
                Int::Parsed(v) => pos = v,
                Int::Partial(v) => {
                    pos = v;
                    ctx.pos = v;
                    warnings.push(ctx.message("not all characters parsed for integer value", pos_state));
                }
                Int::Error => warnings.push(ctx.message("error parsing integer value", pos_state)),
            }
        }
    }
    ctx.pos = pos;
    ctx.variant_index = previous_variant + 1;

    let f = &spec.fixed;
    if let Some(c) = &mut chunk.chrom_values {
        c.push(chrom_field);
    }
    if spec.store_pos {
        // C cast from long, as in the Cython parser
        chunk.pos.push(pos as i32);
    }
    if let Some(c) = &mut chunk.id {
        c.push(fixed[2].unwrap_or(b""));
    }
    if let Some(c) = &mut chunk.ref_ {
        c.push(fixed[3].unwrap_or(b""));
    }

    // ALT and the fields computed from it
    if chunk.alt.is_some() || f.numalt || f.altlen || f.is_snp {
        let ref_len = fixed[3].map_or(0, |r| ref_length(r, f.ref_));
        let alt = parse_alt(fixed[4], fixed[5].is_some(), ref_len, f.alt_number);
        if let Some(c) = &mut chunk.alt {
            for value in &alt.values {
                c.push(value);
            }
        }
        if f.numalt {
            chunk.numalt.push(alt.numalt);
        }
        if f.altlen {
            chunk.altlen.extend_from_slice(&alt.altlen);
        }
        if f.is_snp {
            chunk.is_snp.push(alt.is_snp);
        }
    }

    // QUAL
    if f.qual {
        let mut value = f32::NAN;
        if let Some(q) = fixed[5] {
            if let Some(message) = parse_floating(q, &mut value) {
                warnings.push(ctx.message(message, state(5, State::Qual)));
            }
        }
        chunk.qual.push(value);
    }

    // FILTER
    if !f.filters.is_empty() {
        let offset = chunk.filters.len();
        chunk.filters.resize(offset + f.filters.len(), false);
        // a leading '.' is missing, whatever follows
        if let Some(column) = fixed[6].filter(|c| !c.starts_with(b".")) {
            // some of these delimiters are not strictly kosher, but are seen
            for key in column.split(|&c| matches!(c, b',' | b':' | b';')) {
                if let Ok(i) = f.filters.binary_search_by(|k| k.as_slice().cmp(key)) {
                    chunk.filters[offset + i] = true;
                }
            }
        }
    }

    // INFO
    if !f.infos.is_empty() {
        for (info, column) in f.infos.iter().zip(chunk.infos.iter_mut()) {
            column.push_fill(info.number);
        }
        // a leading '.' is missing, whatever follows
        if let Some(column) = fixed[7].filter(|c| !c.starts_with(b".")) {
            let mut warn = |message: &str| warnings.push(ctx.message(message, State::Info));
            for item in column.split(|&c| c == b';') {
                // a key alone, as for a flag, has an empty value
                let (key, value) = match memchr::memchr(b'=', item) {
                    Some(i) => (&item[..i], &item[i + 1..]),
                    None => (item, &b""[..]),
                };
                if key.is_empty() {
                    if value.len() < item.len() {
                        warn("missing INFO key");
                    }
                    continue;
                }
                if let Ok(i) = f.infos.binary_search_by(|info| info.key.as_slice().cmp(key)) {
                    chunk.infos[i].store(value, f.infos[i].number, &mut warn);
                }
            }
        }
    }

    let Some(g) = &spec.gt else { return pos };
    let number = g.number;
    let offset = chunk.gt.len();
    chunk.gt.resize(offset + spec.n_samples_out * number, fill);
    let gt = &mut chunk.gt[offset..];
    let samples = &spec.samples;
    let loc_samples = &spec.loc_samples;

    // FORMAT, only parsed when followed by call data
    let Some(format) = fixed[8] else { return pos };
    let mut columns = columns.peekable();
    if columns.peek().is_none() {
        return pos;
    }
    let formats = parse_format(format);

    let n_samples = samples.len();
    let mut sample_output_index: usize = 0;
    for (sample_index, column) in columns.enumerate() {
        if sample_index >= n_samples {
            let fmt = match formats.first() {
                Some(Some(_)) => "GT",
                _ => "unknown",
            };
            warnings.push(format!(
                "{}; sample: {sample_index}:0 (unknown:{fmt})",
                ctx.message("more samples than given in header", State::Calldata)
            ));
            break;
        }
        if !loc_samples[sample_index] {
            continue;
        }
        let out = &mut gt[sample_output_index * number..(sample_output_index + 1) * number];
        sample_output_index += 1;
        for (field_index, value) in column.split(|&c| c == b':').enumerate() {
            if field_index >= formats.len() || formats[field_index].is_none() {
                continue;
            }
            for (value_index, allele) in value.split(|&c| c == b'/' || c == b'|').enumerate() {
                if value_index >= number {
                    // more alleles than room for, ignored
                    break;
                }
                if let Some(message) = parse_integer(allele, &mut out[value_index]) {
                    warnings.push(format!(
                        "{}; sample: {sample_index}:{field_index} ({}:GT)",
                        ctx.message(message, State::Calldata),
                        samples[sample_index]
                    ));
                }
            }
        }
    }
    pos
}

/// Keys of the FORMAT column: `Some(())` for GT, `None` for any other key.
///
/// Like the Cython parser, empty keys are dropped, and a '.' key is dropped
/// but stays in the buffer, so that it prefixes the next key.
fn parse_format(format: &[u8]) -> Vec<Option<()>> {
    let mut keys = Vec::new();
    let mut temp: Vec<u8> = Vec::new();
    let mut store = |temp: &mut Vec<u8>| {
        if temp.is_empty() || temp.as_slice() == b"." {
            return;
        }
        keys.push(if temp.as_slice() == b"GT" { Some(()) } else { None });
        temp.clear();
    };
    for &c in format {
        if c == b':' {
            store(&mut temp);
        } else {
            temp.push(c);
        }
    }
    store(&mut temp);
    keys
}

/// Length of REF, as used by `altlen` and `is_snp`: bytes but '.' when the
/// Cython parser stores REF as bytes, otherwise characters, 0 for '.'.
fn ref_length(value: &[u8], kind: Option<StrType>) -> usize {
    match kind {
        Some(StrType::Bytes(_)) => value.iter().filter(|&&c| c != b'.').count(),
        _ if value == b"." => 0,
        _ => std::str::from_utf8(value).map_or(value.len(), |s| s.chars().count()),
    }
}

/// ALT values of one variant and the fields computed from them.
struct Alt<'a> {
    // `number` values, empty when missing
    values: Vec<&'a [u8]>,
    numalt: i32,
    altlen: Vec<i32>,
    is_snp: bool,
}

/// Parse the ALT column. `followed` tells whether another column follows: an
/// empty ALT is missing then, but one empty allele at the end of the line, as
/// in the Cython parser.
fn parse_alt(column: Option<&[u8]>, followed: bool, ref_len: usize, number: usize) -> Alt<'_> {
    let mut alt = Alt { values: vec![b"" as &[u8]; number], numalt: 0, altlen: vec![0; number], is_snp: false };
    let Some(column) = column else { return alt };
    if column.starts_with(b".") || (column.is_empty() && followed) {
        // missing, whatever follows a '.'
        return alt;
    }
    let mut is_snp = ref_len == 1;
    let mut n = 0;
    for (i, value) in column.split(|&c| c == b',').enumerate() {
        let len = value.iter().filter(|&&c| c != b'.' && c != b'*').count();
        is_snp = is_snp && len == 1;
        if i < number {
            alt.values[i] = value;
            // C cast to int32
            alt.altlen[i] = (len as i64 - ref_len as i64) as i32;
        }
        n = i + 1;
    }
    alt.numalt = n as i32;
    alt.is_snp = is_snp;
    alt
}

fn fill_value<T: Integer>(fill: Option<i128>, dtype: &str) -> PyResult<T> {
    match fill {
        None => Ok(T::default_fill()),
        Some(v) => T::try_from(v).map_err(|_| {
            PyOverflowError::new_err(format!("fill value {v} out of range for {dtype}"))
        }),
    }
}

/// Call `$f::<T>($args.., fill)` with `T` the genotype dtype of `$gt`.
macro_rules! with_gt_dtype {
    ($gt:expr, $f:ident($($arg:expr),*)) => {{
        let g: &GtSpec = $gt;
        match g.dtype.as_str() {
            "int8" => $f::<i8>($($arg,)* fill_value(g.fill, &g.dtype)?),
            "int16" => $f::<i16>($($arg,)* fill_value(g.fill, &g.dtype)?),
            "int32" => $f::<i32>($($arg,)* fill_value(g.fill, &g.dtype)?),
            "int64" => $f::<i64>($($arg,)* fill_value(g.fill, &g.dtype)?),
            "uint8" => $f::<u8>($($arg,)* fill_value(g.fill, &g.dtype)?),
            "uint16" => $f::<u16>($($arg,)* fill_value(g.fill, &g.dtype)?),
            "uint32" => $f::<u32>($($arg,)* fill_value(g.fill, &g.dtype)?),
            "uint64" => $f::<u64>($($arg,)* fill_value(g.fill, &g.dtype)?),
            other => Err(PyValueError::new_err(format!("unsupported genotype dtype: {other}"))),
        }
    }};
}

fn accept_fill<T: Integer>(_fill: T) -> PyResult<()> {
    Ok(())
}

fn check_gt(g: &GtSpec) -> PyResult<()> {
    with_gt_dtype!(g, accept_fill())
}

fn next_chunk_for<'py, T: Integer>(it: &mut VCFChunkIterator, py: Python<'py>, fill: T)
                                  -> PyResult<Option<Bound<'py, PyAny>>> {
    it.next_chunk(py, fill)
}

#[pymethods]
impl VCFChunkIterator {
    /// Iterate over the data lines of the VCF file at `path`, after its
    /// header. `samples` are all the samples of the header, `loc_samples`
    /// selects those to output. `fields` are the normalised fields to store;
    /// `string_types` maps the string fields among them to their NumPy dtype
    /// string ('|O' or '|S<width>'). `alt_number` is the number of ALT values
    /// per variant. `gt` is `(dtype name, ploidy, fill or None)` when
    /// `calldata/GT` is stored.
    #[new]
    ///
    /// `infos` are the INFO fields among `fields`, as `(name, dtype code,
    /// number, fill or None)`, the dtype code being 'b1' for flags, 'i1' to
    /// 'u8', 'f4', 'f8', 'S<width>' or 'O'.
    #[pyo3(signature = (path, chunk_length, samples, loc_samples, fields, string_types,
                        alt_number, infos=Vec::new(), gt=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(path: &str, chunk_length: usize, samples: Vec<String>, loc_samples: Vec<bool>,
           fields: Vec<String>, string_types: HashMap<String, String>, alt_number: usize,
           infos: Vec<(String, String, usize, Option<Bound<'_, PyAny>>)>,
           gt: Option<(String, usize, Option<i128>)>) -> PyResult<Self> {
        if chunk_length == 0 {
            return Err(PyValueError::new_err("chunk_length must be positive"));
        }
        if loc_samples.len() != samples.len() {
            return Err(PyValueError::new_err("loc_samples and samples differ in length"));
        }
        if alt_number == 0 {
            return Err(PyValueError::new_err("alt_number must be positive"));
        }
        let mut fixed = FixedSpec {
            chrom: None, id: None, ref_: None, alt: None, alt_number,
            numalt: false, altlen: false, is_snp: false, qual: false, filters: Vec::new(),
            infos: Vec::new(),
        };
        let mut info_fields = Vec::new();
        for (name, dtype, number, fill) in &infos {
            let flag = dtype == "b1";
            fixed.infos.push(InfoSpec {
                key: name.as_bytes().to_vec(),
                // flags are one value per variant, whatever their number
                number: if flag { 1 } else { *number },
                column: info_column(name, dtype, fill.as_ref())?,
            });
            info_fields.push(format!("variants/{name}"));
            if fixed.infos.last().is_some_and(|i| i.number == 0) {
                return Err(PyValueError::new_err(format!("INFO field {name:?} of number 0 must be a flag")));
            }
        }
        fixed.infos.sort_by(|a, b| a.key.cmp(&b.key));
        let mut store_pos = false;
        let mut store_gt = false;
        for field in &fields {
            let string_type = || match string_types.get(field) {
                Some(dtype) => StrType::parse(field, dtype).map(Some),
                None => Err(PyValueError::new_err(format!("no dtype for field {field:?}"))),
            };
            match field.as_str() {
                "variants/CHROM" => fixed.chrom = string_type()?,
                "variants/POS" => store_pos = true,
                "variants/ID" => fixed.id = string_type()?,
                "variants/REF" => fixed.ref_ = string_type()?,
                "variants/ALT" => fixed.alt = string_type()?,
                "variants/QUAL" => fixed.qual = true,
                "variants/numalt" => fixed.numalt = true,
                "variants/altlen" => fixed.altlen = true,
                "variants/is_snp" => fixed.is_snp = true,
                "calldata/GT" => store_gt = true,
                f if info_fields.iter().any(|i| i == f) => {}
                f => match f.strip_prefix("variants/FILTER_") {
                    Some(key) => fixed.filters.push(key.as_bytes().to_vec()),
                    None => return Err(PyValueError::new_err(format!(
                        "field {f:?} is not handled by the Rust parser"))),
                },
            }
        }
        fixed.filters.sort();
        fixed.filters.dedup();
        if store_gt != gt.is_some() {
            return Err(PyValueError::new_err("gt must be given if and only if calldata/GT is a field"));
        }
        let gt = gt.map(|(dtype, number, fill)| GtSpec { dtype, number, fill });
        if let Some(g) = &gt {
            // fail early on an unsupported dtype or a fill out of range
            check_gt(g)?;
            if g.number == 0 {
                return Err(PyValueError::new_err("genotype number must be positive"));
            }
        }

        // skip the header, already read by Python
        let mut reader = open(path)?;
        let mut line = Vec::new();
        loop {
            match read_line(reader.as_mut(), &mut line)? {
                None => return Err(PyValueError::new_err(format!(
                    "{path}: VCF file is missing mandatory header line (\"#CHROM...\")"))),
                Some(_) if line.starts_with(b"#CHROM") => break,
                Some(_) => {}
            }
        }

        let n_samples_out = loc_samples.iter().filter(|&&b| b).count();
        Ok(VCFChunkIterator {
            reader: Mutex::new(reader),
            chunk_length,
            spec: Spec { samples, loc_samples, n_samples_out, store_pos, fixed, gt },
            variant_index: -1,
            eof: false,
        })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        if self.eof {
            return Ok(None);
        }
        match self.spec.gt.clone() {
            None => self.next_chunk::<i8>(py, -1),
            Some(g) => with_gt_dtype!(&g, next_chunk_for(self, py)),
        }
    }
}

#[pymodule]
fn io_vcf_read_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VCFChunkIterator>()?;
    Ok(())
}
