//! Tests for the streaming bulk slice readers `read_typed_slice_from_reader` and
//! `read_complex_slice_from_reader`: the `Read`-based counterparts of
//! `read_typed_slice` / `read_complex_slice`.
//!
//! Coverage: byte-identity with the in-memory bulk path; correctness over
//! adversarial readers (one byte per `read`, awkward chunk sizes); array lengths
//! that cross the decoder's internal allocation-step boundary; element-type
//! mismatch and truncation errors; and back-to-back arrays in one stream (the
//! reader must be left positioned exactly after each array).

use beve::Complex;
use std::io::{self, Cursor, Read};

/// Yields at most one byte per `read`, to stress the bulk reader's `read_exact`
/// against maximally fragmented input.
struct OneByteReader<'a> {
    data: &'a [u8],
    pos: usize,
}
impl Read for OneByteReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.pos >= self.data.len() || buf.is_empty() {
            return Ok(0);
        }
        buf[0] = self.data[self.pos];
        self.pos += 1;
        Ok(1)
    }
}

/// Yields a fixed awkward chunk per `read` so reads rarely align with the
/// decoder's step boundary.
struct OddChunkReader<'a> {
    data: &'a [u8],
    pos: usize,
    chunk: usize,
}
impl Read for OddChunkReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.chunk.min(buf.len()).min(self.data.len() - self.pos);
        buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

/// All lengths, including ones that cross the decoder's 8 MiB allocation step
/// (1 Mi elems per step for an 8-byte element). Exercised by the *cheap* readers
/// (in-memory bulk + `Cursor`), which read in bulk.
const LENS: &[usize] = &[0, 1, 2, 1000, 65537, 1_048_577, 3_000_000];

/// Lengths for the *fragmenting* readers (one byte at a time, tiny odd chunks).
/// Kept small on purpose: they exist to stress the block-refill boundaries, which
/// a few-KiB array covers fully. Running a one-byte reader over millions of
/// elements adds no coverage and is pointlessly slow (especially under the
/// big-endian CI's emulation).
const FRAG_LENS: &[usize] = &[0, 1, 2, 7, 255, 4099];

/// Lengths at or above this cross the 8 MiB allocation step; for those we also
/// run one *block-sized* fragmenting reader, which crosses the step boundary with
/// multiple reads without the per-byte cost.
const STEP_THRESHOLD: usize = 1_000_000;

fn check_typed<T>(make: impl Fn(usize) -> T)
where
    T: beve::BeveTypedSlice + Copy + PartialEq + std::fmt::Debug,
{
    // Correctness at every size via the cheap readers.
    for &n in LENS {
        let v: Vec<T> = (0..n).map(&make).collect();
        let bytes = beve::to_vec_typed_slice(&v);

        let in_mem = beve::read_typed_slice::<T>(&bytes).expect("read_typed_slice");
        assert_eq!(in_mem, v, "in-memory bulk mismatch (n={n})");

        let cur: Vec<T> = beve::read_typed_slice_from_reader(Cursor::new(&bytes)).expect("cursor");
        assert_eq!(cur, v, "streaming cursor mismatch (n={n})");

        // Cross the allocation step under a block-sized (not per-byte) reader.
        if n >= STEP_THRESHOLD {
            let odd: Vec<T> = beve::read_typed_slice_from_reader(OddChunkReader {
                data: &bytes,
                pos: 0,
                chunk: 65537,
            })
            .unwrap_or_else(|e| panic!("step odd-chunk n={n}: {e:?}"));
            assert_eq!(odd, v, "streaming step odd-chunk mismatch (n={n})");
        }
    }

    // Extreme fragmentation on bounded sizes.
    for &n in FRAG_LENS {
        let v: Vec<T> = (0..n).map(&make).collect();
        let bytes = beve::to_vec_typed_slice(&v);

        let ob: Vec<T> = beve::read_typed_slice_from_reader(OneByteReader {
            data: &bytes,
            pos: 0,
        })
        .expect("one-byte");
        assert_eq!(ob, v, "streaming one-byte mismatch (n={n})");

        for chunk in [1usize, 7, 4096] {
            let odd: Vec<T> = beve::read_typed_slice_from_reader(OddChunkReader {
                data: &bytes,
                pos: 0,
                chunk,
            })
            .unwrap_or_else(|e| panic!("odd-chunk({chunk}) n={n}: {e:?}"));
            assert_eq!(odd, v, "streaming odd-chunk={chunk} mismatch (n={n})");
        }
    }
}

fn check_complex<T>(make: impl Fn(usize) -> Complex<T>)
where
    T: beve::BeveTypedSlice + Copy + PartialEq + std::fmt::Debug,
{
    for &n in LENS {
        let v: Vec<Complex<T>> = (0..n).map(&make).collect();
        let bytes = beve::to_vec_complex_slice(&v);

        let in_mem = beve::read_complex_slice::<T>(&bytes).expect("read_complex_slice");
        assert_eq!(in_mem, v, "in-memory complex bulk mismatch (n={n})");

        let cur: Vec<Complex<T>> =
            beve::read_complex_slice_from_reader(Cursor::new(&bytes)).expect("cursor");
        assert_eq!(cur, v, "streaming complex cursor mismatch (n={n})");

        if n >= STEP_THRESHOLD {
            let odd: Vec<Complex<T>> = beve::read_complex_slice_from_reader(OddChunkReader {
                data: &bytes,
                pos: 0,
                chunk: 65537,
            })
            .unwrap_or_else(|e| panic!("complex step odd-chunk n={n}: {e:?}"));
            assert_eq!(odd, v, "streaming complex step odd-chunk mismatch (n={n})");
        }
    }

    for &n in FRAG_LENS {
        let v: Vec<Complex<T>> = (0..n).map(&make).collect();
        let bytes = beve::to_vec_complex_slice(&v);

        let ob: Vec<Complex<T>> = beve::read_complex_slice_from_reader(OneByteReader {
            data: &bytes,
            pos: 0,
        })
        .expect("one-byte");
        assert_eq!(ob, v, "streaming complex one-byte mismatch (n={n})");

        for chunk in [3usize, 13, 4096] {
            let odd: Vec<Complex<T>> = beve::read_complex_slice_from_reader(OddChunkReader {
                data: &bytes,
                pos: 0,
                chunk,
            })
            .unwrap_or_else(|e| panic!("complex odd-chunk({chunk}) n={n}: {e:?}"));
            assert_eq!(
                odd, v,
                "streaming complex odd-chunk={chunk} mismatch (n={n})"
            );
        }
    }
}

#[test]
fn typed_u8() {
    check_typed(|i| (i % 251) as u8);
}
#[test]
fn typed_i16() {
    check_typed(|i| (i as i64 % 60000 - 30000) as i16);
}
#[test]
fn typed_u32() {
    check_typed(|i| (i as u32).wrapping_mul(2654435761));
}
#[test]
fn typed_i64() {
    check_typed(|i| (i as i64).wrapping_mul(-0x61C8864680B583EB));
}
#[test]
fn typed_u128() {
    check_typed(|i| (i as u128) << 64 | i as u128);
}
#[test]
fn typed_f32() {
    check_typed(|i| i as f32 * 0.25 - 3.5);
}
#[test]
fn typed_f64() {
    check_typed(|i| i as f64 * 1e-3 - 1234.5);
}
#[test]
fn typed_f16() {
    // Half-floats are 2-byte elements; this also exercises the smallest element
    // width's allocation-step count and confirms all-zero is a valid bit pattern.
    check_typed(|i| half::f16::from_f32(i as f32 * 0.5 - 3.0));
}
#[test]
fn typed_bf16() {
    check_typed(|i| half::bf16::from_f32(i as f32 * 0.25 - 1.0));
}

#[test]
fn complex_f32() {
    check_complex(|i| Complex {
        re: i as f32 * 0.5,
        im: -(i as f32) * 0.25,
    });
}
#[test]
fn complex_f64() {
    check_complex(|i| Complex {
        re: (i as f64).sin(),
        im: (i as f64).cos(),
    });
}
#[test]
fn complex_i16() {
    check_complex(|i| Complex {
        re: (i as i64 % 30000 - 15000) as i16,
        im: (i as i64 % 20000 - 10000) as i16,
    });
}

#[test]
fn element_type_mismatch_errors() {
    let v: Vec<f64> = vec![1.0, 2.0, 3.0];
    let bytes = beve::to_vec_typed_slice(&v);
    // Decoding f64 bytes as f32 must be rejected (class/byte_code mismatch), not
    // silently misread.
    assert!(beve::read_typed_slice_from_reader::<f32, _>(Cursor::new(&bytes)).is_err());
    // And a complex reader must reject a plain typed array.
    assert!(beve::read_complex_slice_from_reader::<f64, _>(Cursor::new(&bytes)).is_err());
}

#[test]
fn truncated_payload_errors_without_committing() {
    let v: Vec<f64> = (0..1000)
        .collect::<Vec<u64>>()
        .iter()
        .map(|&x| x as f64)
        .collect();
    let mut bytes = beve::to_vec_typed_slice(&v);
    bytes.truncate(bytes.len() - 100); // chop the tail
    let r = beve::read_typed_slice_from_reader::<f64, _>(Cursor::new(&bytes));
    assert!(r.is_err(), "truncated payload must error");
}

#[test]
fn two_arrays_in_one_stream_read_sequentially() {
    // The reader must be left positioned exactly after the first array so the
    // second decodes cleanly from the same stream.
    let a: Vec<f32> = (0..1000).map(|i| i as f32).collect();
    let b: Vec<Complex<f32>> = (0..500)
        .map(|i| Complex {
            re: i as f32,
            im: -(i as f32),
        })
        .collect();

    let mut stream = beve::to_vec_typed_slice(&a);
    stream.extend_from_slice(&beve::to_vec_complex_slice(&b));

    let mut cursor = Cursor::new(&stream);
    let a2: Vec<f32> = beve::read_typed_slice_from_reader(&mut cursor).expect("first array");
    let b2: Vec<Complex<f32>> =
        beve::read_complex_slice_from_reader(&mut cursor).expect("second array");
    assert_eq!(a2, a);
    assert_eq!(b2, b);
}
