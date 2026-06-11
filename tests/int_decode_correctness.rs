//! Correctness of integer typed-array decoding across every width, through both
//! `from_slice` and `from_reader_streaming`.
//!
//! These pin the native-width / sign-extension conversion used by the per-element
//! integer decode path: every width round-trips exactly, including the sign
//! boundaries (MIN/MAX and small negatives that must sign-extend correctly) and
//! lengths that mix positive and negative values.

use std::io::Cursor;

fn roundtrip<T>(values: &[T])
where
    T: beve::BeveTypedSlice + serde::Serialize + serde::de::DeserializeOwned + Copy + PartialEq + std::fmt::Debug,
{
    let v: Vec<T> = values.to_vec();
    let bytes = beve::to_vec(&v).expect("encode");

    let from_slice: Vec<T> = beve::from_slice(&bytes).expect("from_slice");
    assert_eq!(from_slice, v, "from_slice mismatch");

    let streamed: Vec<T> = beve::from_reader_streaming(Cursor::new(&bytes)).expect("stream");
    assert_eq!(streamed, v, "from_reader_streaming mismatch");
}

#[test]
fn unsigned_edges_all_widths() {
    roundtrip::<u8>(&[0, 1, 127, 128, 254, u8::MAX]);
    roundtrip::<u16>(&[0, 1, 255, 256, 32767, 32768, u16::MAX]);
    roundtrip::<u32>(&[0, 1, 65535, 65536, 0x7FFF_FFFF, 0x8000_0000, u32::MAX]);
    roundtrip::<u64>(&[0, 1, u32::MAX as u64, 1 << 32, u64::MAX]);
    roundtrip::<u128>(&[0, 1, u64::MAX as u128, 1u128 << 64, u128::MAX]);
}

#[test]
fn signed_edges_all_widths() {
    // Small negatives are the sign-extension stress: the high bytes of the
    // widened value must be 0xFF, not 0.
    roundtrip::<i8>(&[0, -1, 1, i8::MIN, i8::MAX, -128, 127]);
    roundtrip::<i16>(&[0, -1, 1, i16::MIN, i16::MAX, -256, 255]);
    roundtrip::<i32>(&[0, -1, 1, i32::MIN, i32::MAX, -70000, 70000]);
    roundtrip::<i64>(&[0, -1, 1, i64::MIN, i64::MAX, -(1 << 40), 1 << 40]);
    roundtrip::<i128>(&[0, -1, 1, i128::MIN, i128::MAX, -(1i128 << 80), 1i128 << 80]);
}

#[test]
fn mixed_sign_runs() {
    // Alternating sign over a longer run, to confirm per-element extension is
    // independent (no state leaking between elements).
    let v: Vec<i32> = (0..1000i32)
        .map(|i| if i % 2 == 0 { -i - 1 } else { i })
        .collect();
    roundtrip(&v);

    let v: Vec<i16> = (0..1000i16)
        .map(|i| if i % 3 == 0 { i16::MIN + i } else { i })
        .collect();
    roundtrip(&v);
}

#[test]
fn empty_and_single() {
    roundtrip::<u8>(&[]);
    roundtrip::<i64>(&[]);
    roundtrip::<u32>(&[42]);
    roundtrip::<i16>(&[-7]);
}
