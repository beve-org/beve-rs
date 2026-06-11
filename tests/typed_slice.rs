//! Tests for the bulk-slice fast path: `to_writer_typed_slice` / `typed_slice_size`
//! (Layers 1–2) and the `TypedSlice` serde wrapper (Layer 3).

use beve::{BeveTypedSlice, TypedSlice, to_writer_typed_slice, typed_slice_size};
use half::{bf16, f16};
use serde::{Deserialize, Serialize};

const TYPE_TYPED_ARRAY: u8 = 4; // mirrors src/header.rs (private)

/// Minimal SIZE encoder mirroring `src/size.rs`, used to build expected bytes.
fn write_size_for_test(mut n: u64, out: &mut Vec<u8>) {
    if n < (1 << 6) {
        out.push((n as u8) << 2);
    } else if n < (1 << 14) {
        out.push((((n & 0x3f) as u8) << 2) | 0b01);
        n >>= 6;
        out.push(n as u8);
    } else if n < (1 << 30) {
        out.push((((n & 0x3f) as u8) << 2) | 0b10);
        n >>= 6;
        out.push(n as u8);
        out.push((n >> 8) as u8);
        out.push((n >> 16) as u8);
    } else {
        out.push((((n & 0x3f) as u8) << 2) | 0b11);
        n >>= 6;
        for i in 0..7 {
            out.push((n >> (i * 8)) as u8);
        }
    }
}

/// Element counts spanning empty, small, and the 1/2/4-byte SIZE-prefix boundaries.
const LENGTHS: &[usize] = &[0, 1, 2, 3, 8, 63, 64, 100, 16383, 16384];

/// Exercise Layers 1–2 for one slice of any `BeveTypedSlice` type.
fn check_layer12<T>(slice: &[T])
where
    T: BeveTypedSlice
        + Serialize
        + serde::de::DeserializeOwned
        + PartialEq
        + core::fmt::Debug
        + Clone,
{
    let mut bulk = Vec::new();
    to_writer_typed_slice(&mut bulk, slice).unwrap();

    // (Test 6) The `W: Write` primitive agrees byte-for-byte with the `Vec<u8>` one.
    assert_eq!(
        bulk,
        beve::to_vec_typed_slice(slice),
        "to_writer_typed_slice must match to_vec_typed_slice (len={})",
        slice.len()
    );

    // (Test 2) The analytic size is exact for every length, including SIZE boundaries.
    assert_eq!(
        typed_slice_size(slice),
        bulk.len() as u64,
        "typed_slice_size must equal the written length (len={})",
        slice.len()
    );

    // (Test 1) Round-trips through the normal deserializer.
    let back: Vec<T> = beve::from_slice(&bulk).unwrap();
    assert_eq!(&back, slice, "round-trip mismatch (len={})", slice.len());

    // (Test 1) For a non-empty slice the bulk path equals the per-element streaming
    // path. (An empty `Vec<T>` cannot have its element type detected and encodes as
    // a generic array, so it is deliberately excluded here.)
    if !slice.is_empty() {
        let owned = slice.to_vec();
        let mut stream = Vec::new();
        beve::to_writer_streaming(&mut stream, &owned).unwrap();
        assert_eq!(
            bulk,
            stream,
            "bulk path must equal element-by-element streaming (len={})",
            slice.len()
        );
    }

    // (Test 5, simulated) Build the expected bytes element-by-element in
    // little-endian via `write_one_le`. That construction is endianness-independent
    // in its *result*, so matching it pins the big-endian fallback branch — which is
    // defined to emit exactly these per-element LE bytes — even on an LE host.
    let mut manual = Vec::new();
    manual.push(
        ((T::BYTE_CODE & 0b111) << 5) | ((T::CLASS & 0b11) << 3) | (TYPE_TYPED_ARRAY & 0b111),
    );
    write_size_for_test(slice.len() as u64, &mut manual);
    for v in slice {
        T::write_one_le(v, &mut manual);
    }
    assert_eq!(
        bulk,
        manual,
        "bulk LE output must equal per-element LE construction (BE-fallback parity, len={})",
        slice.len()
    );
}

macro_rules! check_type {
    ($t:ty, $f:expr) => {{
        for &n in LENGTHS {
            let v: Vec<$t> = (0..n).map($f).collect();
            check_layer12::<$t>(&v);
        }
    }};
}

#[test]
fn layer12_all_beve_typed_slice_types() {
    check_type!(i8, |i: usize| (i as i8).wrapping_mul(3).wrapping_sub(7));
    check_type!(i16, |i: usize| (i as i16)
        .wrapping_mul(1234)
        .wrapping_add(5));
    check_type!(i32, |i: usize| (i as i32).wrapping_mul(-3).wrapping_add(1));
    check_type!(i64, |i: usize| (i as i64)
        .wrapping_mul(1_000_003)
        .wrapping_sub(5));
    check_type!(i128, |i: usize| (i as i128) * 1_000_000_007 - 11);
    check_type!(u8, |i: usize| (i % 251) as u8);
    check_type!(u16, |i: usize| (i as u16).wrapping_mul(40001));
    check_type!(u32, |i: usize| (i as u32).wrapping_mul(2_654_435_761));
    check_type!(u64, |i: usize| (i as u64)
        .wrapping_mul(11_400_714_819_323_198_485));
    check_type!(u128, |i: usize| (i as u128) * 340_282_366_920_938_463 + 1);
    check_type!(f32, |i: usize| (i as f32) * 1.5 - 3.25);
    check_type!(f64, |i: usize| (i as f64) * 0.1 - 7.0);
    check_type!(f16, |i: usize| f16::from_f32((i as f32) * 0.5 - 1.0));
    check_type!(bf16, |i: usize| bf16::from_f32((i as f32) * 0.25 + 2.0));
}

#[test]
fn analytic_size_size_prefix_boundaries() {
    for &n in &[0usize, 63, 64, 16383, 16384] {
        let data = vec![0u8; n];
        let mut buf = Vec::new();
        to_writer_typed_slice(&mut buf, &data).unwrap();
        assert_eq!(typed_slice_size(&data), buf.len() as u64, "n={n}");
    }
    // header byte (1) + SIZE-prefix width + payload (n * 1 for u8).
    let expect = |n: usize, prefix: u64| 1 + prefix + n as u64;
    assert_eq!(typed_slice_size(&[0u8; 63]), expect(63, 1));
    assert_eq!(typed_slice_size(&[0u8; 64]), expect(64, 2));
    assert_eq!(typed_slice_size(&[0u8; 16383]), expect(16383, 2));
    assert_eq!(typed_slice_size(&[0u8; 16384]), expect(16384, 4));
}

// ---------------------------------------------------------------------------
// Layer 3 — `TypedSlice` through serde
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct FrameTyped<'a> {
    id: u32,
    samples: TypedSlice<'a, f64>,
    tail: u8,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct FrameVec {
    id: u32,
    samples: Vec<f64>,
    tail: u8,
}

#[test]
fn layer3_struct_field_equals_plain_vec_field() {
    for &n in &[1usize, 2, 8, 64, 1000] {
        let data: Vec<f64> = (0..n).map(|i| i as f64 * 0.5 - 1.0).collect();
        let typed = FrameTyped {
            id: 7,
            samples: TypedSlice(&data),
            tail: 9,
        };
        let plain = FrameVec {
            id: 7,
            samples: data.clone(),
            tail: 9,
        };

        // Streaming: identical bytes.
        let mut a = Vec::new();
        beve::to_writer_streaming(&mut a, &typed).unwrap();
        let mut b = Vec::new();
        beve::to_writer_streaming(&mut b, &plain).unwrap();
        assert_eq!(
            a, b,
            "streaming: TypedSlice field must equal Vec field (n={n})"
        );

        // Buffered: identical bytes, and identical to the streaming output.
        let ta = beve::to_vec(&typed).unwrap();
        let tb = beve::to_vec(&plain).unwrap();
        assert_eq!(
            ta, tb,
            "buffered: TypedSlice field must equal Vec field (n={n})"
        );
        assert_eq!(
            a, ta,
            "streaming and buffered TypedSlice outputs must agree (n={n})"
        );

        // Round-trips back into the plain form.
        let back: FrameVec = beve::from_slice(&a).unwrap();
        assert_eq!(back, plain, "round-trip mismatch (n={n})");
    }
}

/// A writer that counts both bytes and the number of write calls, with no buffer.
#[derive(Default)]
struct CountingWriter {
    bytes: u64,
    calls: u64,
}

impl std::io::Write for CountingWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.bytes += buf.len() as u64;
        self.calls += 1;
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// The defining property of the bulk path: encoding a `TypedSlice` body issues a
/// constant number of writer calls regardless of element count (one `write_all`
/// for the body), whereas a bare `Vec<T>` field issues O(N). This pins the
/// "N writer calls → 1" win and, with a counting sink, the O(1) measure.
///
/// Little-endian only: the single bulk `write_all` is the LE reinterpret-as-bytes
/// path. On big-endian `TypedSlice` correctly falls back to per-element writes
/// (the wire is LE; native bytes must be converted), so the write count is O(N)
/// there by design.
#[cfg(target_endian = "little")]
#[test]
fn layer3_bulk_path_collapses_to_constant_write_count() {
    fn count_typed(n: usize) -> CountingWriter {
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut w = CountingWriter::default();
        beve::to_writer_streaming(
            &mut w,
            &FrameTyped {
                id: 1,
                samples: TypedSlice(&data),
                tail: 2,
            },
        )
        .unwrap();
        w
    }
    fn count_vec(n: usize) -> CountingWriter {
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut w = CountingWriter::default();
        beve::to_writer_streaming(
            &mut w,
            &FrameVec {
                id: 1,
                samples: data,
                tail: 2,
            },
        )
        .unwrap();
        w
    }

    let small = count_typed(8);
    let big = count_typed(100_000);
    assert_eq!(
        small.calls, big.calls,
        "TypedSlice must collapse N element writes to a constant number of write calls"
    );
    // The struct framing is constant, so the byte delta is exactly the typed-array
    // size delta (payload growth plus the wider SIZE prefix) — a single bulk write.
    let (small_n, big_n) = (8usize, 100_000usize);
    let expected_delta =
        typed_slice_size(&vec![0.0f64; big_n]) - typed_slice_size(&vec![0.0f64; small_n]);
    assert_eq!(big.bytes - small.bytes, expected_delta);

    let vsmall = count_vec(8);
    let vbig = count_vec(100_000);
    assert!(
        vbig.calls > vsmall.calls,
        "a bare Vec<f64> field is expected to scale write calls with N"
    );
    assert!(
        vbig.calls > big.calls,
        "TypedSlice must issue far fewer writes than the per-element Vec path"
    );
}

/// An empty `TypedSlice` still encodes a typed array of the element type (matching
/// the bulk primitive) on every target, whereas an empty `Vec<T>` encodes a
/// generic array. Both decode back to an empty `Vec<T>`. This documents and pins
/// the one length at which `TypedSlice` and a bare `Vec<T>` diverge.
#[test]
fn empty_typed_slice_is_typed_array_not_generic() {
    #[derive(Serialize)]
    struct STyped<'a> {
        v: TypedSlice<'a, f64>,
    }
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct SVec {
        v: Vec<f64>,
    }

    let empty: [f64; 0] = [];

    let mut bulk = Vec::new();
    to_writer_typed_slice(&mut bulk, &empty).unwrap();
    assert_eq!(bulk, beve::to_vec_typed_slice(&empty));
    assert_eq!(typed_slice_size(&empty), bulk.len() as u64);

    // The serde wrapper reaches the same typed-array bytes as the bulk primitive
    // for an empty slice on every target (an empty payload needs no endianness
    // handling), so it never silently degrades to a generic empty array.
    let mut wrapper = Vec::new();
    beve::to_writer_streaming(&mut wrapper, &TypedSlice(&empty)).unwrap();
    assert_eq!(
        wrapper, bulk,
        "empty TypedSlice (streaming) must equal the bulk primitive"
    );
    assert_eq!(
        beve::to_vec(&TypedSlice(&empty)).unwrap(),
        bulk,
        "empty TypedSlice (buffered) must equal the bulk primitive"
    );

    let mut a = Vec::new();
    beve::to_writer_streaming(
        &mut a,
        &STyped {
            v: TypedSlice(&empty),
        },
    )
    .unwrap();
    let back: SVec = beve::from_slice(&a).unwrap();
    assert_eq!(back, SVec { v: vec![] });

    let mut b = Vec::new();
    beve::to_writer_streaming(&mut b, &SVec { v: vec![] }).unwrap();
    assert_ne!(
        a, b,
        "empty TypedSlice (typed array) must differ from empty Vec (generic array)"
    );
}

/// `TypedSlice` nested as a sequence element (a generic array of typed arrays)
/// must route through the element-path dispatch and round-trip — and match the
/// equivalent `Vec`-of-`Vec` form.
#[test]
fn typed_slice_nested_as_sequence_element() {
    let a = vec![1.0f64, 2.0, 3.0];
    let b = vec![4.0f64, 5.0];

    let typed = (TypedSlice(&a), TypedSlice(&b));
    let plain = (a.clone(), b.clone());

    let buffered = beve::to_vec(&typed).unwrap();
    let mut streamed = Vec::new();
    beve::to_writer_streaming(&mut streamed, &typed).unwrap();

    assert_eq!(buffered, streamed, "buffered vs streaming TypedSlice tuple");
    assert_eq!(
        buffered,
        beve::to_vec(&plain).unwrap(),
        "TypedSlice tuple must equal the plain Vec-of-Vec tuple"
    );

    for bytes in [buffered, streamed] {
        let back: Vec<Vec<f64>> = beve::from_slice(&bytes).unwrap();
        assert_eq!(back, vec![a.clone(), b.clone()]);
    }
}
