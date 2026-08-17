//! `ComplexElement` for `num_complex::Complex<T>`, behind the `num-complex`
//! feature.
//!
//! Routing a `num_complex` slice through the bulk helpers must change throughput
//! and nothing else: the bytes must equal what the same values produce as
//! `beve::Complex<T>`, and a round trip must return them unchanged.

#![cfg(feature = "num-complex")]

use beve::Complex;
use num_complex::Complex as NumComplex;
use serde::{Deserialize, Serialize};

/// Spans empty, small, and the 1/2/4-byte SIZE-prefix boundaries.
const LENGTHS: &[usize] = &[0, 1, 2, 3, 64, 16383, 16384];

#[derive(Serialize)]
struct ForeignF32<'a> {
    #[serde(serialize_with = "beve::complex::f32_array")]
    iq: &'a [NumComplex<f32>],
}

#[derive(Serialize)]
struct NativeF32<'a> {
    iq: beve::ComplexSlice<'a, f32>,
}

/// A `num_complex` slice must emit exactly what the native `ComplexSlice` emits.
#[test]
fn num_complex_encodes_as_the_native_complex_slice_does() {
    for &n in LENGTHS {
        let foreign: Vec<NumComplex<f32>> = (0..n)
            .map(|i| NumComplex::new(i as f32 * 0.5, -(i as f32)))
            .collect();
        let native: Vec<Complex<f32>> = foreign
            .iter()
            .map(|c| Complex { re: c.re, im: c.im })
            .collect();

        let via_foreign = beve::to_vec(&ForeignF32 { iq: &foreign }).expect("foreign encode");
        let via_native = beve::to_vec(&NativeF32 {
            iq: beve::ComplexSlice(&native),
        })
        .expect("native encode");

        assert_eq!(via_foreign, via_native, "len {n}: bytes must be identical");
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct RoundTrip {
    #[serde(with = "beve::complex_array::i16")]
    iq: Vec<NumComplex<i16>>,
}

/// The decode direction additionally needs `AnyBitPattern`, which is why the
/// feature pulls `num-complex/bytemuck`.
#[test]
fn num_complex_round_trips_through_the_bulk_codec() {
    for &n in LENGTHS {
        let value = RoundTrip {
            iq: (0..n)
                .map(|i| NumComplex::new(i as i16, -(i as i16)))
                .collect(),
        };
        let bytes = beve::to_vec(&value).expect("encode");
        let back: RoundTrip = beve::from_slice(&bytes).expect("decode");
        assert_eq!(back, value, "len {n}");
    }
}

/// Every scalar's impl must round-trip. A `Component` naming the wrong scalar
/// only trips the size assert where the widths differ, so a same-width mistake
/// (`u32` naming `i32`) would otherwise reach the wire as the wrong class.
macro_rules! round_trip_each_scalar {
    ($($name:ident => $with:literal, $scalar:ty),* $(,)?) => {
        $(
            #[test]
            fn $name() {
                #[derive(Serialize, Deserialize, Debug, PartialEq)]
                struct Frame {
                    #[serde(with = $with)]
                    iq: Vec<NumComplex<$scalar>>,
                }
                let value = Frame {
                    iq: (0..17u8)
                        .map(|i| NumComplex::new(i as $scalar, (i / 2) as $scalar))
                        .collect(),
                };
                let bytes = beve::to_vec(&value).expect("encode");
                let back: Frame = beve::from_slice(&bytes).expect("decode");
                assert_eq!(back, value);
            }
        )*
    };
}

round_trip_each_scalar! {
    round_trip_f32 => "beve::complex_array::f32", f32,
    round_trip_f64 => "beve::complex_array::f64", f64,
    round_trip_i8 => "beve::complex_array::i8", i8,
    round_trip_i16 => "beve::complex_array::i16", i16,
    round_trip_i32 => "beve::complex_array::i32", i32,
    round_trip_i64 => "beve::complex_array::i64", i64,
    round_trip_i128 => "beve::complex_array::i128", i128,
    round_trip_u8 => "beve::complex_array::u8", u8,
    round_trip_u16 => "beve::complex_array::u16", u16,
    round_trip_u32 => "beve::complex_array::u32", u32,
    round_trip_u64 => "beve::complex_array::u64", u64,
    round_trip_u128 => "beve::complex_array::u128", u128,
}

#[derive(Serialize)]
struct WriteI16 {
    #[serde(serialize_with = "beve::complex::i16_array")]
    iq: Vec<NumComplex<i16>>,
}

#[derive(Deserialize, Debug, PartialEq)]
struct ReadF32 {
    #[serde(with = "beve::complex_array::f32")]
    iq: Vec<NumComplex<f32>>,
}

/// The widening decode reaches `num_complex` fields too: an `i16`-class array
/// on the wire read into an `f32` field converts in one pass.
#[test]
fn num_complex_takes_the_widening_decode() {
    let written = WriteI16 {
        iq: (0..64)
            .map(|i| NumComplex::new(i as i16, -(i as i16)))
            .collect(),
    };
    let bytes = beve::to_vec(&written).expect("encode i16");
    let read: ReadF32 = beve::from_slice(&bytes).expect("decode as f32");

    let expected: Vec<NumComplex<f32>> = written
        .iq
        .iter()
        .map(|c| NumComplex::new(c.re as f32, c.im as f32))
        .collect();
    assert_eq!(read.iq, expected);
}
