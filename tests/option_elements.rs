//! An array of `Option`s that are all `Some` is written as the typed, bool,
//! string or complex array its values make, and every decoder reads each
//! element back as `Some`, as serde's `Option` visitor expects.

use serde::{Deserialize, Serialize};
use std::io::Cursor;

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Copy)]
struct Meters(f64);

/// The low three bits of a BEVE header are its type; 5 is a generic array.
const GENERIC_ARRAY: u8 = 5;

/// Through the slice decoder, the streaming decoder and `Value`, checking that
/// the bytes are not a generic array (so the element decoders are the ones
/// exercised).
fn roundtrip<T>(value: &T)
where
    T: Serialize + for<'de> Deserialize<'de> + PartialEq + std::fmt::Debug,
{
    let bytes = beve::to_vec(value).unwrap();
    assert_ne!(
        bytes[0] & 0x07,
        GENERIC_ARRAY,
        "{value:?} wrote a generic array"
    );
    assert_eq!(&beve::from_slice::<T>(&bytes).unwrap(), value, "from_slice");
    assert_eq!(
        &beve::from_reader_streaming::<_, T>(Cursor::new(&bytes)).unwrap(),
        value,
        "from_reader_streaming"
    );
    let v: beve::Value = beve::from_slice(&bytes).unwrap();
    assert_eq!(&beve::from_value::<T>(v).unwrap(), value, "from_value");
}

#[test]
fn all_some_arrays_roundtrip() {
    roundtrip(&vec![Some(1u8), Some(u8::MAX)]);
    roundtrip(&vec![Some(-1i32), Some(i32::MAX)]);
    roundtrip(&vec![Some(u64::MAX)]);
    roundtrip(&vec![Some(1.5f32), Some(-0.0)]);
    roundtrip(&vec![Some(2.5f64), Some(f64::INFINITY)]);
    roundtrip(&vec![Some(true), Some(false), Some(true)]);
    roundtrip(&vec![Some("a".to_string()), Some(String::new())]);
    roundtrip(&vec![Some(beve::Complex {
        re: 1.0f64,
        im: -2.0,
    })]);
    roundtrip(&vec![Some(Meters(3.0)), Some(Meters(-4.0))]);
}

#[test]
fn arrays_with_none_still_roundtrip() {
    for value in [vec![Some(1.0f64), None], vec![None, None]] {
        let bytes = beve::to_vec(&value).unwrap();
        assert_eq!(beve::from_slice::<Vec<Option<f64>>>(&bytes).unwrap(), value);
        assert_eq!(
            beve::from_reader_streaming::<_, Vec<Option<f64>>>(Cursor::new(&bytes)).unwrap(),
            value
        );
    }
}

/// A reader asking for `Option`s accepts a typed array written without them.
#[test]
fn typed_arrays_read_as_options() {
    let bytes = beve::to_vec(&vec![1u16, 2, 3]).unwrap();
    assert_eq!(
        beve::from_slice::<Vec<Option<u16>>>(&bytes).unwrap(),
        [Some(1), Some(2), Some(3)]
    );
    let strings = beve::to_vec_str_slice(&["x", "y"]);
    assert_eq!(
        beve::from_reader_streaming::<_, Vec<Option<String>>>(Cursor::new(&strings)).unwrap(),
        [Some("x".to_string()), Some("y".to_string())]
    );
}
