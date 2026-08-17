//! `ComplexSlice` bulk serialization.
//!
//! The wrapper writes a complex array as one bulk copy of the borrowed
//! `(re, im)` payload instead of a `serialize_element` per value. That is a
//! throughput change only, so the whole suite is one claim in several
//! positions: **the bytes must be exactly what the element-by-element path
//! produced**. The reference for "element-by-element" is a plain
//! `Vec<Complex<T>>`, whose derived `Serialize` walks elements one at a time —
//! which is precisely what `ComplexSlice` itself used to do.

use beve::{BeveTypedSlice, Complex, ComplexSlice, to_vec_complex_slice};
use half::{bf16, f16};
use serde::{Deserialize, Serialize};

/// Element counts spanning empty, small, and the 1/2/4-byte SIZE-prefix
/// boundaries — the widths at which a miscomputed element count would show up.
const LENGTHS: &[usize] = &[0, 1, 2, 3, 8, 63, 64, 100, 16383, 16384];

/// Every claim that must hold for one slice of one scalar type.
fn check_scalar<T>(values: &[Complex<T>])
where
    T: BeveTypedSlice + Copy + core::fmt::Debug + PartialEq + serde::de::DeserializeOwned,
    Complex<T>: Serialize + core::fmt::Debug + PartialEq,
    for<'a> ComplexSlice<'a, T>: Serialize,
{
    let n = values.len();
    let owned = values.to_vec();

    let bulk = beve::to_vec(&ComplexSlice(values)).expect("buffered encode");
    let element_wise = beve::to_vec(&owned).expect("buffered encode of Vec");
    assert_eq!(
        bulk, element_wise,
        "buffered: ComplexSlice must equal the element-wise Vec (len={n})"
    );

    let mut bulk_stream = Vec::new();
    beve::to_writer_streaming(&mut bulk_stream, &ComplexSlice(values)).expect("streaming encode");
    let mut element_wise_stream = Vec::new();
    beve::to_writer_streaming(&mut element_wise_stream, &owned).expect("streaming encode of Vec");
    assert_eq!(
        bulk_stream, element_wise_stream,
        "streaming: ComplexSlice must equal the element-wise Vec (len={n})"
    );
    assert_eq!(
        bulk_stream, bulk,
        "streaming and buffered must agree (len={n})"
    );

    // `serialized_size` measures through the same streaming dispatch, so a
    // bulk arm that skipped the size accounting would show up here.
    assert_eq!(
        beve::serialized_size(&ComplexSlice(values)).expect("size"),
        bulk.len() as u64,
        "serialized_size must equal the written length (len={n})"
    );

    let back: Vec<Complex<T>> = beve::from_slice(&bulk).expect("decode");
    assert_eq!(&back, &owned, "round-trip mismatch (len={n})");

    // Parity with the non-serde primitive, which writes the same framing
    // directly. Empty is excluded: see `an_empty_slice_keeps_the_generic_array`.
    if n > 0 {
        assert_eq!(
            bulk,
            to_vec_complex_slice(values),
            "ComplexSlice must equal to_vec_complex_slice (len={n})"
        );
    }
}

macro_rules! check_type {
    ($t:ty, $f:expr) => {{
        for &n in LENGTHS {
            let v: Vec<Complex<$t>> = (0..n).map($f).collect();
            check_scalar::<$t>(&v);
        }
    }};
}

#[test]
fn complex_slice_bulk_matches_element_wise_for_every_scalar() {
    check_type!(i8, |i: usize| Complex {
        re: (i as i8).wrapping_mul(3),
        im: (i as i8).wrapping_sub(7),
    });
    check_type!(i16, |i: usize| Complex {
        re: (i as i16).wrapping_mul(1234),
        im: (i as i16).wrapping_add(5),
    });
    check_type!(i32, |i: usize| Complex {
        re: (i as i32).wrapping_mul(-3),
        im: (i as i32).wrapping_add(1),
    });
    check_type!(i64, |i: usize| Complex {
        re: (i as i64).wrapping_mul(1_000_003),
        im: (i as i64).wrapping_sub(5),
    });
    check_type!(i128, |i: usize| Complex {
        re: (i as i128) * 1_000_000_007,
        im: (i as i128) - 11,
    });
    check_type!(u8, |i: usize| Complex {
        re: (i % 251) as u8,
        im: (i % 97) as u8,
    });
    check_type!(u16, |i: usize| Complex {
        re: (i as u16).wrapping_mul(40001),
        im: (i as u16).wrapping_add(3),
    });
    check_type!(u32, |i: usize| Complex {
        re: (i as u32).wrapping_mul(2_654_435_761),
        im: i as u32,
    });
    check_type!(u64, |i: usize| Complex {
        re: (i as u64).wrapping_mul(11_400_714_819_323_198_485),
        im: i as u64,
    });
    check_type!(u128, |i: usize| Complex {
        re: (i as u128) * 340_282_366_920_938_463,
        im: (i as u128) + 1,
    });
    check_type!(f32, |i: usize| Complex {
        re: (i as f32) * 1.5 - 3.25,
        im: (i as f32) * -0.5,
    });
    check_type!(f64, |i: usize| Complex {
        re: (i as f64) * 0.1 - 7.0,
        im: (i as f64) * 2.5,
    });
    // f16 and bf16 are the types whose complex width is NOT `1 << byte_code`
    // (bf16 in particular reports byte_code 0 while being 2 bytes), so they are
    // where an element count derived from the byte code would go wrong.
    check_type!(f16, |i: usize| Complex {
        re: f16::from_f32((i as f32) * 0.5 - 1.0),
        im: f16::from_f32((i as f32) * 0.25),
    });
    check_type!(bf16, |i: usize| Complex {
        re: bf16::from_f32((i as f32) * 0.25 + 2.0),
        im: bf16::from_f32(-(i as f32)),
    });
}

/// An empty slice has no element from which to detect a type, and has always
/// encoded as a generic empty array here. The bulk path must not quietly
/// promote it to a zero-length complex array: consumers that read the element
/// type off the header (the MATLAB export among them) can tell the two apart.
#[test]
fn an_empty_slice_keeps_the_generic_array() {
    let empty: [Complex<f32>; 0] = [];

    let wrapper = beve::to_vec(&ComplexSlice(&empty)).unwrap();
    let plain = beve::to_vec(&Vec::<Complex<f32>>::new()).unwrap();
    assert_eq!(wrapper, plain, "empty must still encode as the Vec does");

    assert_ne!(
        wrapper,
        to_vec_complex_slice(&empty),
        "the non-serde primitive writes a typed empty array; the divergence is \
         deliberate and documented on ComplexSlice"
    );

    let back: Vec<Complex<f32>> = beve::from_slice(&wrapper).unwrap();
    assert!(back.is_empty());
}

// ---------------------------------------------------------------------------
// Positions: struct field, nested in a sequence, and through the
// `serialize_with` helpers
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct FrameSlice<'a> {
    id: u32,
    iq: ComplexSlice<'a, i16>,
    tail: u8,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct FrameVec {
    id: u32,
    iq: Vec<Complex<i16>>,
    tail: u8,
}

fn sample_iq(n: usize) -> Vec<Complex<i16>> {
    (0..n)
        .map(|i| Complex {
            re: (i as i16).wrapping_mul(31),
            im: (i as i16).wrapping_mul(-17),
        })
        .collect()
}

#[test]
fn a_struct_field_matches_the_plain_vec_field() {
    for &n in &[0usize, 1, 8, 64, 1000] {
        let data = sample_iq(n);
        let slice = FrameSlice {
            id: 7,
            iq: ComplexSlice(&data),
            tail: 9,
        };
        let plain = FrameVec {
            id: 7,
            iq: data.clone(),
            tail: 9,
        };

        let a = beve::to_vec(&slice).unwrap();
        let b = beve::to_vec(&plain).unwrap();
        assert_eq!(a, b, "buffered struct field (n={n})");

        let mut sa = Vec::new();
        beve::to_writer_streaming(&mut sa, &slice).unwrap();
        let mut sb = Vec::new();
        beve::to_writer_streaming(&mut sb, &plain).unwrap();
        assert_eq!(sa, sb, "streaming struct field (n={n})");
        assert_eq!(sa, a, "streaming and buffered struct field (n={n})");

        let back: FrameVec = beve::from_slice(&a).unwrap();
        assert_eq!(back, plain, "struct field round-trip (n={n})");
    }
}

/// A complex array nested in a sequence is one *value*, so it has to land as a
/// single generic-array element — not as a run of complex singles spliced into
/// the parent. This is the seq-element dispatch, a separate code path from the
/// top-level one above.
#[test]
fn nested_in_a_sequence_it_stays_one_element() {
    let a = sample_iq(5);
    let b = sample_iq(9);

    let slices = (ComplexSlice(&a), ComplexSlice(&b));
    let vecs = (a.clone(), b.clone());

    let bulk = beve::to_vec(&slices).unwrap();
    assert_eq!(
        bulk,
        beve::to_vec(&vecs).unwrap(),
        "buffered: nested ComplexSlice must equal nested Vec"
    );

    let mut bulk_stream = Vec::new();
    beve::to_writer_streaming(&mut bulk_stream, &slices).unwrap();
    let mut plain_stream = Vec::new();
    beve::to_writer_streaming(&mut plain_stream, &vecs).unwrap();
    assert_eq!(
        bulk_stream, plain_stream,
        "streaming: nested ComplexSlice must equal nested Vec"
    );

    let back: (Vec<Complex<i16>>, Vec<Complex<i16>>) = beve::from_slice(&bulk).unwrap();
    assert_eq!(back, (a, b), "nested round-trip");
}

// A foreign complex element type, layout-compatible with `Complex<i16>`
// (`#[repr(C)]`, two `i16` fields re/im) — the `num_complex::Complex<i16>` case
// that the `serialize_with` helpers exist for.
#[derive(Serialize, Debug, Clone, Copy, PartialEq)]
#[repr(C)]
struct ForeignIq {
    re: i16,
    im: i16,
}
// SAFETY: `#[repr(C)]` over two `i16`, real first — no padding, every bit
// pattern initialized.
unsafe impl beve::ComplexElement for ForeignIq {
    type Component = i16;
}

#[derive(Serialize)]
struct ForeignFrame {
    id: u32,
    #[serde(serialize_with = "beve::complex::i16_array")]
    iq: Vec<ForeignIq>,
    tail: u8,
}

#[test]
fn the_serialize_with_helper_reaches_the_bulk_path() {
    for &n in &[0usize, 1, 8, 1000] {
        let data = sample_iq(n);
        let foreign = ForeignFrame {
            id: 7,
            iq: data
                .iter()
                .map(|c| ForeignIq { re: c.re, im: c.im })
                .collect(),
            tail: 9,
        };
        let plain = FrameVec {
            id: 7,
            iq: data,
            tail: 9,
        };

        assert_eq!(
            beve::to_vec(&foreign).unwrap(),
            beve::to_vec(&plain).unwrap(),
            "buffered: complex::i16_array must equal the plain Vec field (n={n})"
        );

        let mut a = Vec::new();
        beve::to_writer_streaming(&mut a, &foreign).unwrap();
        let mut b = Vec::new();
        beve::to_writer_streaming(&mut b, &plain).unwrap();
        assert_eq!(
            a, b,
            "streaming: complex::i16_array must equal the plain Vec field (n={n})"
        );
    }
}

/// Human-readable formats must never see the bulk tag: the marker is beve's,
/// and a JSON consumer needs the portable sequence.
#[test]
fn a_human_readable_format_still_gets_a_sequence() {
    let data = sample_iq(4);
    assert_eq!(
        serde_json::to_string(&ComplexSlice(&data)).unwrap(),
        serde_json::to_string(&data).unwrap(),
        "JSON must render ComplexSlice exactly as the Vec"
    );
}
