//! Rust VCF parser behind `allel2.io.vcf_read.iter_vcf_chunks`.
//!
//! Python reads and normalises the headers and the parameters; this module
//! parses the data lines into chunks shaped exactly like those of the Cython
//! `VCFChunkIterator`: `(chunk, chunk_length, chrom, pos)`, where `chunk` maps
//! field names to NumPy arrays. The Cython parser is the reference: values,
//! dtypes, shapes and warning messages match it, quirks included.
//!
//! Fields handled: the fixed columns (CHROM, POS, ID, REF, ALT, QUAL, the
//! FILTER flags and the INFO fields), the computed fields `numalt`, `altlen`
//! and `is_snp`, and the FORMAT fields, as call data. Python sends the others
//! to the Cython parser.

use std::any::Any;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::sync::{mpsc, Mutex};

use flate2::read::MultiGzDecoder;
use numpy::{Element, IntoPyArray, PyArrayMethods};
use pyo3::exceptions::{PyOSError, PyOverflowError, PyRuntimeError, PyUserWarning, PyValueError};
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
    /// Add one, wrapping around as in C.
    fn increment(self) -> Self;
}

macro_rules! impl_integer {
    ($($t:ty => $fill:expr),*) => {$(
        impl Integer for $t {
            #[inline]
            fn from_long(v: i64) -> Self { v as $t }
            fn default_fill() -> Self { $fill }
            fn increment(self) -> Self { self.wrapping_add(1) }
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

    fn append(&mut self, other: StrColumn) {
        let offset = self.data.len();
        self.ends.extend(other.ends.iter().map(|end| end + offset));
        self.data.extend(other.data);
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

/// The values of an INFO or FORMAT field in a chunk: `number` per variant,
/// and per sample for FORMAT fields.
trait Column: Send + Sync {
    /// A column of the same type with no values, for a new chunk.
    fn empty(&self) -> Box<dyn Column>;
    /// Append `n` fill values.
    fn push_fill(&mut self, n: usize);
    /// Parse `value` into the `number` values from index `at`, as the Cython
    /// parser does: values beyond `number` are ignored, and a repeated key
    /// overwrites what its values give.
    fn store(&mut self, at: usize, value: &[u8], number: usize, warn: &mut dyn FnMut(&'static str));
    /// Append the values of `other`, a column of the same type; false if it
    /// is not.
    fn append(&mut self, other: Box<dyn Column>) -> bool;
    fn into_any(self: Box<Self>) -> Box<dyn Any>;
    fn into_numpy<'py>(self: Box<Self>, py: Python<'py>, shape: &[usize])
                       -> PyResult<Bound<'py, PyAny>>;
}

/// `other` as a column of type `C`, if it is one.
fn downcast<C: Column + 'static>(other: Box<dyn Column>) -> Option<Box<C>> {
    other.into_any().downcast::<C>().ok()
}

fn is_comma(c: &u8) -> bool {
    *c == b','
}

fn is_allele_separator(c: &u8) -> bool {
    matches!(c, b'/' | b'|')
}

#[derive(Default)]
struct FlagColumn(Vec<bool>);

impl Column for FlagColumn {
    fn empty(&self) -> Box<dyn Column> {
        Box::new(FlagColumn::default())
    }
    fn push_fill(&mut self, n: usize) {
        self.0.resize(self.0.len() + n, false);
    }
    fn store(&mut self, at: usize, _value: &[u8], _number: usize, _warn: &mut dyn FnMut(&'static str)) {
        self.0[at] = true;
    }
    fn append(&mut self, other: Box<dyn Column>) -> bool {
        let Some(other) = downcast::<Self>(other) else { return false };
        self.0.extend(other.0);
        true
    }
    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
    fn into_numpy<'py>(self: Box<Self>, py: Python<'py>, shape: &[usize])
                       -> PyResult<Bound<'py, PyAny>> {
        self.0.into_pyarray(py).reshape(shape).map(|a| a.into_any())
    }
}

/// Integers or floating point values, split on commas, or on '/' and '|'
/// for genotypes.
struct NumberColumn<T> {
    data: Vec<T>,
    fill: T,
    parse: fn(&[u8], &mut T) -> Option<&'static str>,
    separator: fn(&u8) -> bool,
}

impl<T: Element + Copy + Send + Sync + 'static> Column for NumberColumn<T> {
    fn empty(&self) -> Box<dyn Column> {
        Box::new(NumberColumn { data: Vec::new(), ..*self })
    }
    fn push_fill(&mut self, n: usize) {
        self.data.resize(self.data.len() + n, self.fill);
    }
    fn store(&mut self, at: usize, value: &[u8], number: usize, warn: &mut dyn FnMut(&'static str)) {
        let values = value.split(self.separator).take(number);
        for (slot, value) in self.data[at..at + number].iter_mut().zip(values) {
            if let Some(message) = (self.parse)(value, slot) {
                warn(message);
            }
        }
    }
    fn append(&mut self, other: Box<dyn Column>) -> bool {
        let Some(other) = downcast::<Self>(other) else { return false };
        self.data.extend(other.data);
        true
    }
    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
    fn into_numpy<'py>(self: Box<Self>, py: Python<'py>, shape: &[usize])
                       -> PyResult<Bound<'py, PyAny>> {
        self.data.into_pyarray(py).reshape(shape).map(|a| a.into_any())
    }
}

/// Allele counts of genotypes, `genotype_ac/<dtype>`: one count per allele
/// below `number`, whatever the ploidy.
struct AlleleCountColumn<T>(Vec<T>);

impl<T: Integer + 'static> Column for AlleleCountColumn<T> {
    fn empty(&self) -> Box<dyn Column> {
        Box::new(AlleleCountColumn::<T>(Vec::new()))
    }
    fn push_fill(&mut self, n: usize) {
        self.0.resize(self.0.len() + n, T::from_long(0));
    }
    fn store(&mut self, at: usize, value: &[u8], number: usize, warn: &mut dyn FnMut(&'static str)) {
        for allele in value.split(is_allele_separator) {
            let (allele, message) = match strtol(allele) {
                Int::Missing => continue,
                Int::Parsed(v) => (v, None),
                Int::Partial(v) => (v, Some("not all characters parsed for integer value")),
                Int::Error => {
                    warn("error parsing integer value");
                    continue;
                }
            };
            if let Some(message) = message {
                warn(message);
            }
            if allele < number as i64 {
                // the Cython parser does not check for negative alleles: they
                // count for the values before, of earlier samples or variants
                // of the chunk; those before the chunk, which the Cython parser
                // writes out of its array, are dropped
                if let Some(count) = usize::try_from(at as i64 + allele).ok().and_then(|i| self.0.get_mut(i)) {
                    *count = count.increment();
                }
            }
        }
    }
    fn append(&mut self, other: Box<dyn Column>) -> bool {
        let Some(other) = downcast::<Self>(other) else { return false };
        self.0.extend(other.0);
        true
    }
    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
    fn into_numpy<'py>(self: Box<Self>, py: Python<'py>, shape: &[usize])
                       -> PyResult<Bound<'py, PyAny>> {
        self.0.into_pyarray(py).reshape(shape).map(|a| a.into_any())
    }
}

struct BytesColumn {
    width: usize,
    data: Vec<u8>,
}

impl Column for BytesColumn {
    fn empty(&self) -> Box<dyn Column> {
        Box::new(BytesColumn { width: self.width, data: Vec::new() })
    }
    fn push_fill(&mut self, n: usize) {
        self.data.resize(self.data.len() + n * self.width, 0);
    }
    fn store(&mut self, at: usize, value: &[u8], number: usize, _warn: &mut dyn FnMut(&'static str)) {
        let width = self.width;
        let slots = self.data[at * width..(at + number) * width].chunks_mut(width);
        for (slot, value) in slots.zip(value.split(is_comma).take(number)) {
            // overwrites the first bytes only, as the Cython parser does
            let n = value.len().min(width);
            slot[..n].copy_from_slice(&value[..n]);
        }
    }
    fn append(&mut self, other: Box<dyn Column>) -> bool {
        let Some(other) = downcast::<Self>(other) else { return false };
        self.data.extend(other.data);
        true
    }
    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
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

impl Column for ObjectColumn {
    fn empty(&self) -> Box<dyn Column> {
        Box::new(ObjectColumn::default())
    }
    fn push_fill(&mut self, n: usize) {
        self.values.resize(self.values.len() + n, None);
    }
    fn store(&mut self, at: usize, value: &[u8], number: usize, _warn: &mut dyn FnMut(&'static str)) {
        let slots = self.values[at..at + number].iter_mut();
        for (slot, value) in slots.zip(value.split(is_comma).take(number)) {
            // an empty value leaves the slot unchanged
            if !value.is_empty() {
                let start = self.data.len();
                self.data.extend_from_slice(value);
                *slot = Some((start, self.data.len()));
            }
        }
    }
    fn append(&mut self, other: Box<dyn Column>) -> bool {
        let Some(other) = downcast::<Self>(other) else { return false };
        let offset = self.data.len();
        self.values.extend(other.values.iter().map(|v| v.map(|(start, end)| (start + offset, end + offset))));
        self.data.extend(other.data);
        true
    }
    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
    fn into_numpy<'py>(self: Box<Self>, py: Python<'py>, shape: &[usize])
                       -> PyResult<Bound<'py, PyAny>> {
        let values = self.values.iter().map(|v| v.map(|(start, end)| &self.data[start..end]));
        object_array(py, values, shape)
    }
}

/// How the values of an integer column are read.
#[derive(Clone, Copy)]
enum IntegerKind {
    Values,
    Genotype,
    AlleleCount,
}

fn integer_column<T: Integer + 'static>(kind: IntegerKind, fill: Option<i128>, dtype: &str)
                                        -> PyResult<Box<dyn Column>> {
    Ok(match kind {
        IntegerKind::AlleleCount => Box::new(AlleleCountColumn::<T>(Vec::new())),
        IntegerKind::Values | IntegerKind::Genotype => Box::new(NumberColumn {
            data: Vec::new(),
            fill: fill_value::<T>(fill, dtype)?,
            parse: parse_integer::<T>,
            separator: if matches!(kind, IntegerKind::Genotype) { is_allele_separator } else { is_comma },
        }),
    })
}

fn floating_column<T: Floating + 'static>(fill: T) -> Box<dyn Column> {
    Box::new(NumberColumn { data: Vec::new(), fill, parse: parse_floating::<T>, separator: is_comma })
}

/// A column for a field of dtype code `code`: 'b1' for flags, 'i1' to 'u8',
/// 'f4', 'f8', 'S<width>' or 'O'; for FORMAT fields, 'gt:' or 'ac:' before an
/// integer code for genotypes or their allele counts.
fn column(name: &str, code: &str, fill: Option<&Bound<'_, PyAny>>) -> PyResult<Box<dyn Column>> {
    let (kind, dtype) = match code.split_once(':') {
        Some(("gt", dtype)) => (IntegerKind::Genotype, dtype),
        Some(("ac", dtype)) => (IntegerKind::AlleleCount, dtype),
        _ => (IntegerKind::Values, code),
    };
    let integer_fill = || fill.map(|f| f.extract::<i128>()).transpose();
    let floating_fill = || fill.map_or(Ok(f64::NAN), |f| f.extract::<f64>());
    let column: Box<dyn Column> = match dtype {
        "i1" => integer_column::<i8>(kind, integer_fill()?, "int8")?,
        "i2" => integer_column::<i16>(kind, integer_fill()?, "int16")?,
        "i4" => integer_column::<i32>(kind, integer_fill()?, "int32")?,
        "i8" => integer_column::<i64>(kind, integer_fill()?, "int64")?,
        "u1" => integer_column::<u8>(kind, integer_fill()?, "uint8")?,
        "u2" => integer_column::<u16>(kind, integer_fill()?, "uint16")?,
        "u4" => integer_column::<u32>(kind, integer_fill()?, "uint32")?,
        "u8" => integer_column::<u64>(kind, integer_fill()?, "uint64")?,
        _ if !matches!(kind, IntegerKind::Values) => {
            return Err(PyValueError::new_err(format!("unsupported dtype {code:?} for field {name:?}")));
        }
        "b1" => Box::new(FlagColumn::default()),
        "f4" => floating_column(floating_fill()? as f32),
        "f8" => floating_column(floating_fill()?),
        _ => match StrType::parse(name, dtype)? {
            StrType::Object => Box::new(ObjectColumn::default()),
            StrType::Bytes(width) => Box::new(BytesColumn { width, data: Vec::new() }),
        },
    };
    Ok(column)
}

/// An INFO or FORMAT field to store.
struct FieldSpec {
    name: String,
    number: usize,
    // with no values, the prototype of each chunk's column
    column: Box<dyn Column>,
}

impl FieldSpec {
    /// From `(name, dtype code, number, fill or None)`, see `column`.
    fn new((name, code, number, fill): &(String, String, usize, Option<Bound<'_, PyAny>>))
           -> PyResult<Self> {
        let flag = code == "b1";
        Ok(FieldSpec {
            name: name.clone(),
            // flags are one value per variant, whatever their number
            number: if flag { 1 } else { *number },
            column: column(name, code, fill.as_ref())?,
        })
    }

    fn key(&self) -> &[u8] {
        self.name.as_bytes()
    }
}

/// The index of the field of key `key` in `fields`, sorted by key.
fn find(fields: &[FieldSpec], key: &[u8]) -> Option<usize> {
    fields.binary_search_by(|f| f.key().cmp(key)).ok()
}

/// The variants fields to store: fixed columns, computed fields and INFO.
struct VariantSpec {
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
    infos: Vec<FieldSpec>,
}

/// What to parse, fixed for the whole file.
struct Spec {
    samples: Vec<String>,
    loc_samples: Vec<bool>,
    n_samples_out: usize,
    store_pos: bool,
    fixed: VariantSpec,
    // FORMAT fields, sorted by key
    formats: Vec<FieldSpec>,
}

/// One chunk of parsed data, before conversion to NumPy arrays.
struct Chunk {
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
    // in the order of `VariantSpec::infos`
    infos: Vec<Box<dyn Column>>,
    // in the order of `Spec::formats`
    calldata: Vec<Box<dyn Column>>,
    // CHROM of the last variant, null-terminated
    chrom: Vec<u8>,
    last_pos: i64,
    warnings: Vec<String>,
}

impl Chunk {
    fn new(spec: &Spec) -> Self {
        let fixed = &spec.fixed;
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
            calldata: spec.formats.iter().map(|f| f.column.empty()).collect(),
            chrom: Vec::new(),
            last_pos: 0,
            warnings: Vec::new(),
        }
    }

    /// Append the variants of `other`, parsed after those of `self` with the
    /// same spec; false if its columns do not match.
    fn append(&mut self, other: Chunk) -> bool {
        fn append_str(a: &mut Option<StrColumn>, b: Option<StrColumn>) {
            if let (Some(a), Some(b)) = (a, b) {
                a.append(b);
            }
        }
        fn append_columns(a: &mut [Box<dyn Column>], b: Vec<Box<dyn Column>>) -> bool {
            a.iter_mut().zip(b).all(|(a, b)| a.append(b))
        }
        self.length += other.length;
        append_str(&mut self.chrom_values, other.chrom_values);
        self.pos.extend(other.pos);
        append_str(&mut self.id, other.id);
        append_str(&mut self.ref_, other.ref_);
        append_str(&mut self.alt, other.alt);
        self.numalt.extend(other.numalt);
        self.altlen.extend(other.altlen);
        self.is_snp.extend(other.is_snp);
        self.qual.extend(other.qual);
        self.filters.extend(other.filters);
        self.chrom = other.chrom;
        self.last_pos = other.last_pos;
        self.warnings.extend(other.warnings);
        append_columns(&mut self.infos, other.infos) && append_columns(&mut self.calldata, other.calldata)
    }
}

/// Lines read from the file, parsed together on one thread.
struct Batch {
    data: Vec<u8>,
    // start and end of each line in `data`, and whether it had a terminator
    lines: Vec<(usize, usize, bool)>,
    // index of the variant before the first line
    previous_variant: i64,
}

impl Batch {
    fn parse(&self, spec: &Spec) -> Chunk {
        let mut chunk = Chunk::new(spec);
        for (i, &(start, end, terminated)) in self.lines.iter().enumerate() {
            let line_end = if terminated { State::Eol } else { State::Eof };
            let previous_variant = self.previous_variant + i as i64;
            chunk.last_pos = parse_line(&self.data[start..end], line_end, previous_variant, spec, &mut chunk);
            chunk.length += 1;
        }
        chunk
    }
}

#[pyclass(module = "allel2.opt.io_vcf_read_rs")]
struct VCFChunkIterator {
    // PyO3 classes must be Sync; the lock is never contended
    reader: Mutex<Box<dyn BufRead + Send>>,
    chunk_length: usize,
    spec: Spec,
    // parses batches of lines of at least `batch_bytes`, but the last
    pool: rayon::ThreadPool,
    batch_bytes: usize,
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
fn read_line(reader: &mut dyn BufRead, line: &mut Vec<u8>) -> std::io::Result<Option<bool>> {
    line.clear();
    append_line(reader, line)
}

/// Append one line to `data` without its terminator, as `read_line` does.
///
/// As in the Cython parser, a line ends at LF, CR LF or a lone CR, and a null
/// byte is the end of the file.
fn append_line(reader: &mut dyn BufRead, data: &mut Vec<u8>) -> std::io::Result<Option<bool>> {
    let start = data.len();
    let line = data;
    loop {
        let buf = reader.fill_buf()?;
        if buf.is_empty() {
            return Ok(if line.len() == start { None } else { Some(false) });
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
            return Ok(if line.len() == start { None } else { Some(false) });
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
    ///
    /// This thread reads batches of lines, which the threads of the pool parse
    /// as they come; the chunks of the batches are then joined in order.
    fn parse_chunk(&mut self) -> std::io::Result<Chunk> {
        let reader = self.reader.get_mut().unwrap_or_else(|e| e.into_inner());
        let (spec, chunk_length, batch_bytes) = (&self.spec, self.chunk_length, self.batch_bytes);
        let mut previous_variant = self.variant_index;
        let mut eof = false;
        // whether the file ends at the start of a line
        let mut eof_at_line_start = false;
        let (sender, receiver) = mpsc::channel();
        self.pool.install(|| rayon::scope(|scope| -> std::io::Result<()> {
            let mut n_lines = 0;
            let mut n_batches = 0;
            while n_lines < chunk_length && !eof {
                let mut batch = Batch { data: Vec::new(), lines: Vec::new(), previous_variant };
                while n_lines < chunk_length && batch.data.len() < batch_bytes {
                    let start = batch.data.len();
                    match append_line(reader.as_mut(), &mut batch.data)? {
                        None => {
                            eof = true;
                            eof_at_line_start = true;
                            break;
                        }
                        Some(terminated) => {
                            batch.lines.push((start, batch.data.len(), terminated));
                            n_lines += 1;
                            if !terminated {
                                eof = true;
                                break;
                            }
                        }
                    }
                }
                if batch.lines.is_empty() {
                    break;
                }
                previous_variant += batch.lines.len() as i64;
                let (sender, index) = (sender.clone(), n_batches);
                n_batches += 1;
                scope.spawn(move |_| {
                    // the receiver outlives the scope
                    let _ = sender.send((index, batch.parse(spec)));
                });
            }
            Ok(())
        }))?;
        drop(sender);
        let mut chunks: Vec<(usize, Chunk)> = receiver.into_iter().collect();
        chunks.sort_by_key(|(index, _)| *index);
        let mut chunk = Chunk::new(spec);
        for (_, other) in chunks {
            if !chunk.append(other) {
                return Err(std::io::Error::other("internal error: chunks of different columns"));
            }
        }
        self.variant_index = previous_variant;
        self.eof = eof;
        if eof_at_line_start {
            chunk.chrom.clear();
            chunk.last_pos = 0;
        }
        Ok(chunk)
    }

    fn next_chunk<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let chunk = py
            .detach(|| self.parse_chunk())
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
            dict.set_item(format!("variants/{}", info.name), column.into_numpy(py, shape)?)?;
        }
        let n_out = spec.n_samples_out;
        for (format, column) in spec.formats.iter().zip(chunk.calldata) {
            let shape: &[usize] = if format.number == 1 { &[n, n_out] } else { &[n, n_out, format.number] };
            dict.set_item(format!("calldata/{}", format.name), column.into_numpy(py, shape)?)?;
        }
        let tuple = (dict, chunk.length, PyBytes::new(py, &chunk.chrom), chunk.last_pos);
        Ok(Some(tuple.into_pyobject(py)?.into_any()))
    }
}

/// Parse one data line into the next variant of `chunk`, and return its POS
/// (0 when missing or unparsable). Every field gets a value for the variant,
/// the fill value when the line is too short for it.
fn parse_line(line: &[u8], line_end: State, previous_variant: i64, spec: &Spec,
              chunk: &mut Chunk) -> i64 {
    let mut columns = line.split(|&c| c == b'\t');
    // CHROM, POS, ID, REF, ALT, QUAL, FILTER, INFO and FORMAT
    let fixed: [Option<&[u8]>; 9] = std::array::from_fn(|_| columns.next());
    // state of a warning about column `i`, which ends the line if no tab follows
    let state = |i: usize, own: State| if fixed[i + 1].is_some() { own } else { line_end };
    let warnings = &mut chunk.warnings;
    // index of the variant in the chunk
    let row = chunk.length;

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
                if let Some(i) = find(&f.infos, key) {
                    let number = f.infos[i].number;
                    chunk.infos[i].store(row * number, value, number, &mut warn);
                }
            }
        }
    }

    // FORMAT and call data
    let formats = &spec.formats;
    if formats.is_empty() {
        return pos;
    }
    let n_out = spec.n_samples_out;
    for (format, column) in formats.iter().zip(chunk.calldata.iter_mut()) {
        column.push_fill(n_out * format.number);
    }
    // FORMAT, only parsed when followed by call data
    let Some(format) = fixed[8] else { return pos };
    let mut columns = columns.peekable();
    if columns.peek().is_none() {
        return pos;
    }
    let keys = parse_format(format, formats);
    let samples = &spec.samples;
    let mut sample_output_index = 0;
    for (sample_index, column) in columns.enumerate() {
        if sample_index >= samples.len() {
            // the Cython parser names the format at random, or crashes, when
            // the first key is not stored
            let fmt = match keys.first() {
                Some(Some(i)) => formats[*i].name.as_str(),
                _ => "unknown",
            };
            warnings.push(format!(
                "{}; sample: {sample_index}:0 (unknown:{fmt})",
                ctx.message("more samples than given in header", State::Calldata)
            ));
            break;
        }
        if !spec.loc_samples[sample_index] {
            continue;
        }
        for (field_index, value) in column.split(|&c| c == b':').enumerate() {
            // keys beyond FORMAT, or not stored, are skipped
            let Some(&Some(i)) = keys.get(field_index) else { continue };
            let number = formats[i].number;
            let at = (row * n_out + sample_output_index) * number;
            let mut warn = |message: &str| warnings.push(format!(
                "{}; sample: {sample_index}:{field_index} ({}:{})",
                ctx.message(message, State::Calldata), samples[sample_index], formats[i].name
            ));
            chunk.calldata[i].store(at, value, number, &mut warn);
        }
        sample_output_index += 1;
    }
    pos
}

/// For each key of the FORMAT column, the index of its field in `formats`,
/// `None` if not stored.
///
/// Like the Cython parser, empty keys are dropped, and a '.' key is dropped
/// but stays in the buffer, so that it prefixes the next key.
fn parse_format(format: &[u8], formats: &[FieldSpec]) -> Vec<Option<usize>> {
    let mut keys = Vec::new();
    let mut temp: Vec<u8> = Vec::new();
    let mut store = |temp: &mut Vec<u8>| {
        if temp.is_empty() || temp.as_slice() == b"." {
            return;
        }
        keys.push(find(formats, temp));
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

#[pymethods]
impl VCFChunkIterator {
    /// Iterate over the data lines of the VCF file at `path`, after its
    /// header. `samples` are all the samples of the header, `loc_samples`
    /// selects those to output. `fields` are the normalised fields to store;
    /// `string_types` maps the fixed string fields among them to their dtype
    /// code ('O' or 'S<width>'). `alt_number` is the number of ALT values per
    /// variant. `infos` and `formats` are the INFO and FORMAT fields among
    /// `fields`, as `(name, dtype code, number, fill or None)`; see `column`
    /// for the dtype codes. Lines are parsed on `n_threads` threads, in
    /// batches of at least `batch_bytes`.
    #[new]
    #[pyo3(signature = (path, chunk_length, samples, loc_samples, fields, string_types,
                        alt_number, infos=Vec::new(), formats=Vec::new(), n_threads=1,
                        batch_bytes=1 << 22))]
    #[allow(clippy::too_many_arguments)]
    fn new(path: &str, chunk_length: usize, samples: Vec<String>, loc_samples: Vec<bool>,
           fields: Vec<String>, string_types: HashMap<String, String>, alt_number: usize,
           infos: Vec<(String, String, usize, Option<Bound<'_, PyAny>>)>,
           formats: Vec<(String, String, usize, Option<Bound<'_, PyAny>>)>,
           n_threads: usize, batch_bytes: usize) -> PyResult<Self> {
        if chunk_length == 0 {
            return Err(PyValueError::new_err("chunk_length must be positive"));
        }
        if n_threads == 0 {
            return Err(PyValueError::new_err("n_threads must be positive"));
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        if loc_samples.len() != samples.len() {
            return Err(PyValueError::new_err("loc_samples and samples differ in length"));
        }
        if alt_number == 0 {
            return Err(PyValueError::new_err("alt_number must be positive"));
        }
        let mut fixed = VariantSpec {
            chrom: None, id: None, ref_: None, alt: None, alt_number,
            numalt: false, altlen: false, is_snp: false, qual: false, filters: Vec::new(),
            infos: Vec::new(),
        };
        fixed.infos = infos.iter().map(FieldSpec::new).collect::<PyResult<_>>()?;
        fixed.infos.sort_by(|a, b| a.name.cmp(&b.name));
        let mut formats = formats.iter().map(FieldSpec::new).collect::<PyResult<Vec<_>>>()?;
        formats.sort_by(|a, b| a.name.cmp(&b.name));
        let mut store_pos = false;
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
                f if f.strip_prefix("variants/").is_some_and(|n| find(&fixed.infos, n.as_bytes()).is_some()) => {}
                f if f.strip_prefix("calldata/").is_some_and(|n| find(&formats, n.as_bytes()).is_some()) => {}
                f => match f.strip_prefix("variants/FILTER_") {
                    Some(key) => fixed.filters.push(key.as_bytes().to_vec()),
                    None => return Err(PyValueError::new_err(format!(
                        "field {f:?} is not handled by the Rust parser"))),
                },
            }
        }
        fixed.filters.sort();
        fixed.filters.dedup();

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
            spec: Spec { samples, loc_samples, n_samples_out, store_pos, fixed, formats },
            pool,
            batch_bytes,
            variant_index: -1,
            eof: false,
        })
    }

    /// The number of threads parsing lines.
    #[getter]
    fn n_threads(&self) -> usize {
        self.pool.current_num_threads()
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        if self.eof {
            return Ok(None);
        }
        self.next_chunk(py)
    }
}

#[pymodule]
fn io_vcf_read_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VCFChunkIterator>()?;
    Ok(())
}
