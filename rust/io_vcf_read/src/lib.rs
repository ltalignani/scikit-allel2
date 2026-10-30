//! Rust VCF parser behind `allel2.io.vcf_read.iter_vcf_chunks`.
//!
//! Python reads and normalises the headers and the parameters; this module
//! parses the data lines into chunks shaped exactly like those of the Cython
//! `VCFChunkIterator`: `(chunk, chunk_length, chrom, pos)`, where `chunk` maps
//! field names to NumPy arrays. The Cython parser is the reference: values,
//! dtypes, shapes and warning messages match it, quirks included.
//!
//! Fields handled so far: `variants/POS` and `calldata/GT`.

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::sync::Mutex;

use flate2::read::MultiGzDecoder;
use numpy::{Element, IntoPyArray, PyArrayMethods};
use pyo3::exceptions::{PyOSError, PyOverflowError, PyUserWarning, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

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

/// Integer dtypes of genotype arrays, with C casts from `long`.
trait Allele: Element + Copy + Send + TryFrom<i128> {
    fn from_long(v: i64) -> Self;
    fn default_fill() -> Self;
}

macro_rules! impl_allele {
    ($($t:ty => $fill:expr),*) => {$(
        impl Allele for $t {
            #[inline]
            fn from_long(v: i64) -> Self { v as $t }
            fn default_fill() -> Self { $fill }
        }
    )*};
}

impl_allele!(i8 => -1, i16 => -1, i32 => -1, i64 => -1,
             u8 => u8::MAX, u16 => u16::MAX, u32 => u32::MAX, u64 => u64::MAX);

/// Parser state labels used in warning messages, as in the Cython parser.
#[derive(Clone, Copy, PartialEq)]
enum State {
    Chrom,
    Pos,
    Calldata,
    Eol,
    Eof,
}

impl State {
    fn label(self) -> &'static str {
        match self {
            State::Chrom => "CHROM",
            State::Pos => "POS",
            State::Calldata => "CALLDATA",
            State::Eol => "EOL",
            State::Eof => "EOF",
        }
    }
}

struct GtSpec {
    dtype: String,
    number: usize,
    fill: Option<i128>,
}

/// One chunk of parsed data, before conversion to NumPy arrays.
struct Chunk<T> {
    length: usize,
    pos: Vec<i32>,
    gt: Vec<T>,
    chrom: Vec<u8>,
    last_pos: i64,
    warnings: Vec<String>,
}

#[pyclass(module = "allel2.opt.io_vcf_read_rs")]
struct VCFChunkIterator {
    reader: Mutex<Box<dyn BufRead + Send>>,
    chunk_length: usize,
    samples: Vec<String>,
    loc_samples: Vec<bool>,
    n_samples_out: usize,
    store_pos: bool,
    gt: Option<GtSpec>,
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
    if reader.read_until(b'\n', line)? == 0 {
        return Ok(None);
    }
    let terminated = line.last() == Some(&b'\n');
    if terminated {
        line.pop();
        if line.last() == Some(&b'\r') {
            line.pop();
        }
    }
    Ok(Some(terminated))
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
    fn parse_chunk<T: Allele>(&mut self, fill: T) -> std::io::Result<Chunk<T>> {
        let width = self.gt.as_ref().map_or(0, |g| self.n_samples_out * g.number);
        let mut chunk = Chunk {
            length: 0,
            pos: Vec::new(),
            gt: Vec::new(),
            chrom: Vec::new(),
            last_pos: 0,
            warnings: Vec::new(),
        };
        let reader = self.reader.get_mut().unwrap();
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
            let end = if terminated { State::Eol } else { State::Eof };
            if self.gt.is_some() {
                chunk.gt.resize(chunk.gt.len() + width, fill);
            }
            let gt_offset = chunk.gt.len() - width;
            let pos = parse_line(
                &line,
                end,
                self.variant_index,
                &self.samples,
                &self.loc_samples,
                self.gt.as_ref().map(|g| g.number),
                &mut chunk.gt[gt_offset..],
                &mut chunk.chrom,
                &mut chunk.warnings,
            );
            self.variant_index += 1;
            chunk.length += 1;
            chunk.last_pos = pos;
            if self.store_pos {
                // C cast from long, as in the Cython parser
                chunk.pos.push(pos as i32);
            }
            if !terminated {
                self.eof = true;
                break;
            }
        }
        Ok(chunk)
    }

    fn next_chunk<'py, T: Allele>(&mut self, py: Python<'py>, fill: T)
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
        let dict = PyDict::new(py);
        if self.store_pos {
            dict.set_item("variants/POS", chunk.pos.into_pyarray(py))?;
        }
        if let Some(g) = &self.gt {
            let array = chunk.gt.into_pyarray(py);
            let array = if g.number == 1 {
                array.reshape([chunk.length, self.n_samples_out])?.into_any()
            } else {
                array.reshape([chunk.length, self.n_samples_out, g.number])?.into_any()
            };
            dict.set_item("calldata/GT", array)?;
        }
        let tuple = (dict, chunk.length, PyBytes::new(py, &chunk.chrom), chunk.last_pos);
        Ok(Some(tuple.into_pyobject(py)?.into_any()))
    }
}

/// Parse one data line: returns POS (0 when missing or unparsable), stores the
/// genotypes into `gt` (already filled) and the chrom, null-terminated as the
/// Cython parser returns it, into `chrom`.
#[allow(clippy::too_many_arguments)]
fn parse_line<T: Allele>(
    line: &[u8],
    end: State,
    previous_variant: i64,
    samples: &[String],
    loc_samples: &[bool],
    gt_number: Option<usize>,
    gt: &mut [T],
    chrom: &mut Vec<u8>,
    warnings: &mut Vec<String>,
) -> i64 {
    let mut columns = line.split(|&c| c == b'\t');
    let has_tab = memchr(b'\t', line).is_some();

    // CHROM
    let chrom_field = columns.next().unwrap_or(b"");
    chrom.clear();
    chrom.extend_from_slice(chrom_field);
    let mut ctx = Context { variant_index: previous_variant, chrom: b"", pos: 0 };
    if chrom_field.is_empty() {
        let state = if has_tab { State::Chrom } else { end };
        warnings.push(ctx.message("empty CHROM", state));
    }
    chrom.push(0);
    ctx.chrom = chrom;

    // POS
    let mut pos: i64 = 0;
    if has_tab {
        let rest = &line[chrom_field.len() + 1..];
        let pos_field = columns.next().unwrap_or(b"");
        let state = if memchr(b'\t', rest).is_some() { State::Pos } else { end };
        if pos_field.is_empty() {
            warnings.push(ctx.message("empty POS", state));
        } else {
            match strtol(pos_field) {
                Int::Missing => {}
                Int::Parsed(v) => pos = v,
                Int::Partial(v) => {
                    pos = v;
                    ctx.pos = v;
                    warnings.push(ctx.message("not all characters parsed for integer value", state));
                }
                Int::Error => warnings.push(ctx.message("error parsing integer value", state)),
            }
        }
    }
    ctx.pos = pos;
    ctx.variant_index = previous_variant + 1;

    let Some(number) = gt_number else { return pos };

    // skip ID, REF, ALT, QUAL, FILTER and INFO
    for _ in 0..6 {
        if columns.next().is_none() {
            return pos;
        }
    }
    // FORMAT, only parsed when followed by call data
    let Some(format) = columns.next() else { return pos };
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
                let message = match strtol(allele) {
                    Int::Missing => None,
                    Int::Parsed(v) => {
                        out[value_index] = T::from_long(v);
                        None
                    }
                    Int::Partial(v) => {
                        out[value_index] = T::from_long(v);
                        Some("not all characters parsed for integer value")
                    }
                    Int::Error => Some("error parsing integer value"),
                };
                if let Some(message) = message {
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

#[inline]
fn memchr(needle: u8, haystack: &[u8]) -> Option<usize> {
    haystack.iter().position(|&c| c == needle)
}

fn fill_value<T: Allele>(fill: Option<i128>, dtype: &str) -> PyResult<T> {
    match fill {
        None => Ok(T::default_fill()),
        Some(v) => T::try_from(v).map_err(|_| {
            PyOverflowError::new_err(format!("fill value {v} out of range for {dtype}"))
        }),
    }
}

macro_rules! dispatch_dtype {
    ($self:ident, $py:ident, $dtype:expr, $fill:expr) => {
        match $dtype {
            "int8" => $self.next_chunk::<i8>($py, fill_value($fill, $dtype)?),
            "int16" => $self.next_chunk::<i16>($py, fill_value($fill, $dtype)?),
            "int32" => $self.next_chunk::<i32>($py, fill_value($fill, $dtype)?),
            "int64" => $self.next_chunk::<i64>($py, fill_value($fill, $dtype)?),
            "uint8" => $self.next_chunk::<u8>($py, fill_value($fill, $dtype)?),
            "uint16" => $self.next_chunk::<u16>($py, fill_value($fill, $dtype)?),
            "uint32" => $self.next_chunk::<u32>($py, fill_value($fill, $dtype)?),
            "uint64" => $self.next_chunk::<u64>($py, fill_value($fill, $dtype)?),
            other => Err(PyValueError::new_err(format!("unsupported genotype dtype: {other}"))),
        }
    };
}

#[pymethods]
impl VCFChunkIterator {
    /// Iterate over the data lines of the VCF file at `path`, after its
    /// header. `samples` are all the samples of the header, `loc_samples`
    /// selects those to output. `gt` is `(dtype name, ploidy, fill or None)`,
    /// or `None` to skip genotypes.
    #[new]
    #[pyo3(signature = (path, chunk_length, samples, loc_samples, store_pos, gt=None))]
    fn new(path: &str, chunk_length: usize, samples: Vec<String>, loc_samples: Vec<bool>,
           store_pos: bool, gt: Option<(String, usize, Option<i128>)>) -> PyResult<Self> {
        if chunk_length == 0 {
            return Err(PyValueError::new_err("chunk_length must be positive"));
        }
        if loc_samples.len() != samples.len() {
            return Err(PyValueError::new_err("loc_samples and samples differ in length"));
        }
        let gt = gt.map(|(dtype, number, fill)| GtSpec { dtype, number, fill });
        if let Some(g) = &gt {
            // fail early on an unsupported dtype or a fill out of range
            match g.dtype.as_str() {
                "int8" => fill_value::<i8>(g.fill, &g.dtype).map(|_| ()),
                "int16" => fill_value::<i16>(g.fill, &g.dtype).map(|_| ()),
                "int32" => fill_value::<i32>(g.fill, &g.dtype).map(|_| ()),
                "int64" => fill_value::<i64>(g.fill, &g.dtype).map(|_| ()),
                "uint8" => fill_value::<u8>(g.fill, &g.dtype).map(|_| ()),
                "uint16" => fill_value::<u16>(g.fill, &g.dtype).map(|_| ()),
                "uint32" => fill_value::<u32>(g.fill, &g.dtype).map(|_| ()),
                "uint64" => fill_value::<u64>(g.fill, &g.dtype).map(|_| ()),
                other => Err(PyValueError::new_err(format!("unsupported genotype dtype: {other}"))),
            }?;
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
            samples,
            loc_samples,
            n_samples_out,
            store_pos,
            gt,
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
        match &self.gt {
            None => self.next_chunk::<i8>(py, -1),
            Some(g) => {
                let dtype = g.dtype.clone();
                let fill = g.fill;
                dispatch_dtype!(self, py, dtype.as_str(), fill)
            }
        }
    }
}

#[pymodule]
fn io_vcf_read_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VCFChunkIterator>()?;
    Ok(())
}
