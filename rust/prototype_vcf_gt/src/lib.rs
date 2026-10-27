//! PROTOTYPE (throwaway): Rust VCF parser for the GT-only path of
//! `allel2.read_vcf(fields=['variants/POS', 'calldata/GT'])`.
//!
//! Answers one question: is a Rust parser more than twice as fast as the
//! Cython `VCFChunkIterator` on real data? Not production code: no INFO/FORMAT
//! fields other than GT, diploid output only, minimal error handling.

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::process::{Command, Stdio};

use flate2::read::MultiGzDecoder;
use numpy::{IntoPyArray, PyArray1, PyArray3, PyArrayMethods};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

const PLOIDY: usize = 2;

struct Region {
    chrom: Vec<u8>,
    start: i64,
    stop: i64,
}

fn parse_region(region: &str) -> PyResult<Region> {
    let (chrom, span) = match region.split_once(':') {
        Some((c, s)) => (c, Some(s)),
        None => (region, None),
    };
    let (start, stop) = match span {
        None => (1, i64::MAX),
        Some(s) => {
            let (a, b) = s
                .split_once('-')
                .ok_or_else(|| PyValueError::new_err("region must be chrom:start-stop"))?;
            let p = |x: &str| x.replace(',', "").parse::<i64>();
            (
                p(a).map_err(|e| PyValueError::new_err(e.to_string()))?,
                p(b).map_err(|e| PyValueError::new_err(e.to_string()))?,
            )
        }
    };
    Ok(Region { chrom: chrom.as_bytes().to_vec(), start, stop })
}

/// Parse one integer allele index; '.' or anything unparsable is missing (-1).
#[inline]
fn parse_allele(s: &[u8]) -> i8 {
    if s.is_empty() || s[0] == b'.' {
        return -1;
    }
    let mut v: i32 = 0;
    for &c in s {
        if !c.is_ascii_digit() {
            return -1;
        }
        v = v * 10 + (c - b'0') as i32;
    }
    v as i8
}

/// Parse a data line into `pos` and the `n_samples * PLOIDY` genotype slots.
/// Returns false when the line is outside the region (to be dropped).
fn parse_line(line: &[u8], region: Option<&Region>, n_samples: usize, pos: &mut i32,
              gt: &mut [i8]) -> bool {
    gt.fill(-1);
    let mut fields = line.split(|&c| c == b'\t');
    let chrom = fields.next().unwrap_or(b"");
    let p: i64 = std::str::from_utf8(fields.next().unwrap_or(b"0"))
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(-1);
    if let Some(r) = region {
        if chrom != r.chrom.as_slice() || p < r.start || p > r.stop {
            return false;
        }
    }
    *pos = p as i32;
    // skip ID REF ALT QUAL FILTER INFO
    for _ in 0..6 {
        fields.next();
    }
    let format = match fields.next() {
        Some(f) => f,
        None => return true,
    };
    let gt_index = match format.split(|&c| c == b':').position(|k| k == b"GT") {
        Some(i) => i,
        None => return true,
    };
    for (i, sample) in fields.take(n_samples).enumerate() {
        let value = if gt_index == 0 {
            match memchr::memchr(b':', sample) {
                Some(j) => &sample[..j],
                None => sample,
            }
        } else {
            match sample.split(|&c| c == b':').nth(gt_index) {
                Some(v) => v,
                None => continue,
            }
        };
        let slots = &mut gt[i * PLOIDY..(i + 1) * PLOIDY];
        for (k, allele) in value.split(|&c| c == b'/' || c == b'|').take(PLOIDY).enumerate() {
            slots[k] = parse_allele(allele);
        }
    }
    true
}

fn open_input(path: &str, region: Option<&str>, tabix: &str)
              -> PyResult<(Box<dyn BufRead + Send>, Option<std::process::Child>)> {
    const BUF: usize = 1 << 22;
    match region {
        Some(r) => {
            // stderr is kept separate from the VCF stream
            let mut child = Command::new(tabix)
                .args(["-h", path, r])
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()
                .map_err(|e| PyIOError::new_err(format!("tabix: {e}")))?;
            let out = child.stdout.take().unwrap();
            Ok((Box::new(BufReader::with_capacity(BUF, out)), Some(child)))
        }
        None => {
            let f = File::open(path).map_err(|e| PyIOError::new_err(e.to_string()))?;
            let r: Box<dyn Read + Send> = if path.ends_with("gz") {
                Box::new(MultiGzDecoder::new(BufReader::with_capacity(BUF, f)))
            } else {
                Box::new(f)
            };
            Ok((Box::new(BufReader::with_capacity(BUF, r)), None))
        }
    }
}

/// Read samples, POS and GT. Returns (samples, pos[int32], gt[int8, (n, s, 2)]).
#[pyfunction]
#[pyo3(signature = (path, region=None, tabix="tabix", n_threads=0, batch_lines=16384))]
fn read_vcf_gt<'py>(py: Python<'py>, path: &str, region: Option<&str>, tabix: &str,
                    n_threads: usize, batch_lines: usize)
                    -> PyResult<(Vec<String>, Bound<'py, PyArray1<i32>>, Bound<'py, PyArray3<i8>>)> {
    let parsed_region = region.map(parse_region).transpose()?;
    let (mut reader, child) = open_input(path, region, tabix)?;

    let result = py.detach(|| -> PyResult<(Vec<String>, Vec<i32>, Vec<i8>)> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        // headers
        let mut samples = Vec::new();
        let mut line = Vec::new();
        loop {
            line.clear();
            if reader.read_until(b'\n', &mut line)? == 0 {
                return Err(PyValueError::new_err("missing #CHROM header line"));
            }
            if line.starts_with(b"#CHROM") {
                let text = String::from_utf8_lossy(&line);
                samples = text.trim_end().split('\t').skip(9).map(String::from).collect();
                break;
            }
        }
        let n_samples = samples.len();
        let width = n_samples * PLOIDY;

        let mut all_pos: Vec<i32> = Vec::new();
        let mut all_gt: Vec<i8> = Vec::new();
        let mut lines: Vec<Vec<u8>> = (0..batch_lines).map(|_| Vec::new()).collect();
        loop {
            // fill a batch of lines (single-threaded read)
            let mut n = 0;
            while n < batch_lines {
                let l = &mut lines[n];
                l.clear();
                if reader.read_until(b'\n', l)? == 0 {
                    break;
                }
                if l.last() == Some(&b'\n') {
                    l.pop();
                }
                n += 1;
            }
            if n == 0 {
                break;
            }
            // parse the batch in parallel
            let mut pos = vec![0i32; n];
            // at least one slot per line so that sample-less VCFs still yield POS
            let slot = width.max(1);
            let mut gt = vec![-1i8; n * slot];
            let mut keep = vec![false; n];
            pool.install(|| {
                pos.par_iter_mut()
                    .zip(gt.par_chunks_mut(slot))
                    .zip(keep.par_iter_mut())
                    .zip(lines[..n].par_iter())
                    .for_each(|(((p, g), k), l)| {
                        *k = parse_line(l, parsed_region.as_ref(), n_samples, p, g);
                    });
            });
            for i in 0..n {
                if keep[i] {
                    all_pos.push(pos[i]);
                    all_gt.extend_from_slice(&gt[i * slot..i * slot + width]);
                }
            }
            if n < batch_lines {
                break;
            }
        }
        Ok((samples, all_pos, all_gt))
    });
    if let Some(mut c) = child {
        let _ = c.wait();
    }
    let (samples, pos, gt) = result?;
    let n = pos.len();
    let n_samples = samples.len();
    let gt = gt
        .into_pyarray(py)
        .reshape([n, n_samples, PLOIDY])?;
    Ok((samples, pos.into_pyarray(py), gt))
}

#[pymodule]
fn _prototype_rust_vcf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_vcf_gt, m)?)?;
    Ok(())
}
