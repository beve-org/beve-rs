#![deny(warnings)]

use std::collections::BTreeMap;
use std::io::Cursor;

use beve::{
    Complex, DecodedMatrix, MatrixDecodeMode, MatrixLayout, MatrixOwned, Value, from_reader,
};
use half::{bf16, f16};
use serde::ser::{SerializeMap, SerializeSeq};
use serde::{Deserialize, Serialize};

fn write_size_for_test(mut n: u64, out: &mut Vec<u8>) {
    if n < (1 << 6) {
        out.push((n as u8) << 2);
        return;
    }
    if n < (1 << 14) {
        out.push((((n & 0x3f) as u8) << 2) | 0b01);
        n >>= 6;
        out.push(n as u8);
        return;
    }
    if n < (1 << 30) {
        out.push((((n & 0x3f) as u8) << 2) | 0b10);
        n >>= 6;
        out.push(n as u8);
        out.push((n >> 8) as u8);
        out.push((n >> 16) as u8);
        return;
    }
    out.push((((n & 0x3f) as u8) << 2) | 0b11);
    n >>= 6;
    for i in 0..7 {
        out.push((n >> (i * 8)) as u8);
    }
}

fn size_prefix_for_test(n: u64) -> Vec<u8> {
    let mut out = Vec::new();
    write_size_for_test(n, &mut out);
    out
}

struct UnknownLenSeq<'a, T>(&'a [T]);

impl<T: Serialize> Serialize for UnknownLenSeq<'_, T> {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> core::result::Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(None)?;
        for value in self.0 {
            seq.serialize_element(value)?;
        }
        seq.end()
    }
}

struct MixedWidthNumericSeq;

impl Serialize for MixedWidthNumericSeq {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> core::result::Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(Some(3))?;
        seq.serialize_element(&1u8)?;
        seq.serialize_element(&2u16)?;
        seq.serialize_element(&3u32)?;
        seq.end()
    }
}

struct EmptyUnknownLenMap;

impl Serialize for EmptyUnknownLenMap {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> core::result::Result<S::Ok, S::Error> {
        let map = serializer.serialize_map(None)?;
        map.end()
    }
}

struct UnknownLenStringMap;

impl Serialize for UnknownLenStringMap {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> core::result::Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("alpha", &7u8)?;
        map.serialize_entry("beta", &true)?;
        map.end()
    }
}

struct MixedWidthKeyMap;

impl Serialize for MixedWidthKeyMap {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> core::result::Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry(&1i8, &10u8)?;
        map.serialize_entry(&2i16, &11u8)?;
        map.end()
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
enum UnitEnum {
    Alpha,
    Beta,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
enum TaggedEnum {
    Scalar(u8),
    Pair(u8, bool),
    Named { count: u16 },
}

#[test]
fn size_prefix_thresholds_match_spec() {
    for (len, prefix) in [
        (0usize, vec![0x00]),
        (63usize, vec![0xfc]),
        (64usize, vec![0x01, 0x01]),
        (16_383usize, vec![0xfd, 0xff]),
        (16_384usize, vec![0x02, 0x00, 0x01, 0x00]),
    ] {
        let value = "a".repeat(len);
        let bytes = beve::to_vec(&value).unwrap();
        let mut expected = vec![0x02];
        expected.extend_from_slice(&prefix);
        assert_eq!(&bytes[..expected.len()], expected.as_slice(), "len={len}");

        let back: String = beve::from_slice(&bytes).unwrap();
        assert_eq!(back, value);
    }
}

#[test]
fn scalar_wire_format_covers_all_scalar_kinds() {
    assert_eq!(beve::to_vec(&Option::<()>::None).unwrap(), vec![0x00]);
    assert_eq!(beve::to_vec(&false).unwrap(), vec![0x08]);
    assert_eq!(beve::to_vec(&true).unwrap(), vec![0x18]);

    let i16_bytes = beve::to_vec(&-2i16).unwrap();
    assert_eq!(i16_bytes, vec![0x29, 0xfe, 0xff]);

    let i128_value = -0x0102_0304_0506_0708_090a_0b0c_0d0e_0f10i128;
    let i128_bytes = beve::to_vec(&i128_value).unwrap();
    assert_eq!(i128_bytes[0], 0x89);
    assert_eq!(&i128_bytes[1..], &i128_value.to_le_bytes());

    let u16_bytes = beve::to_vec(&0x1234u16).unwrap();
    assert_eq!(u16_bytes, vec![0x31, 0x34, 0x12]);

    let u128_value = 0x0011_2233_4455_6677_8899_aabb_ccdd_eeffu128;
    let u128_bytes = beve::to_vec(&u128_value).unwrap();
    assert_eq!(u128_bytes[0], 0x91);
    assert_eq!(&u128_bytes[1..], &u128_value.to_le_bytes());

    let bf16_value = bf16::from_f32(-3.5);
    let bf16_bytes = beve::to_vec(&bf16_value).unwrap();
    assert_eq!(
        bf16_bytes,
        [
            0x01,
            bf16_value.to_bits() as u8,
            (bf16_value.to_bits() >> 8) as u8
        ]
    );

    let f16_value = f16::from_f32(1.5);
    let f16_bytes = beve::to_vec(&f16_value).unwrap();
    assert_eq!(
        f16_bytes,
        [
            0x21,
            f16_value.to_bits() as u8,
            (f16_value.to_bits() >> 8) as u8
        ]
    );

    let f32_value = -12.25f32;
    let f32_bytes = beve::to_vec(&f32_value).unwrap();
    assert_eq!(f32_bytes[0], 0x41);
    assert_eq!(&f32_bytes[1..], &f32_value.to_le_bytes());

    let f64_value = std::f64::consts::PI;
    let f64_bytes = beve::to_vec(&f64_value).unwrap();
    assert_eq!(f64_bytes[0], 0x61);
    assert_eq!(&f64_bytes[1..], &f64_value.to_le_bytes());

    let string_bytes = beve::to_vec(&"A").unwrap();
    assert_eq!(string_bytes, vec![0x02, 0x04, b'A']);
}

#[test]
fn typed_arrays_cover_widths_and_unknown_length_backpatching() {
    let i128_bytes = beve::to_vec_typed_slice(&[1i128, -2i128]);
    assert_eq!(i128_bytes[0], 0x8c);
    assert_eq!(i128_bytes[1], 0x08);
    let i128_back: Vec<i128> = beve::from_slice(&i128_bytes).unwrap();
    assert_eq!(i128_back, vec![1, -2]);

    let u128_bytes = beve::to_vec_typed_slice(&[1u128, 2u128]);
    assert_eq!(u128_bytes[0], 0x94);
    assert_eq!(u128_bytes[1], 0x08);
    let u128_back: Vec<u128> = beve::from_slice(&u128_bytes).unwrap();
    assert_eq!(u128_back, vec![1, 2]);

    let bf16_bytes = beve::to_vec_typed_slice(&[bf16::from_f32(1.0), bf16::from_f32(-2.0)]);
    assert_eq!(bf16_bytes[0], 0x04);

    let f16_bytes = beve::to_vec_typed_slice(&[f16::from_f32(1.0), f16::from_f32(-2.0)]);
    assert_eq!(f16_bytes[0], 0x24);

    let string_array = vec!["a".to_string(), "bc".to_string()];
    let string_bytes = beve::to_vec(&string_array).unwrap();
    assert_eq!(string_bytes, vec![0x3c, 0x08, 0x04, b'a', 0x08, b'b', b'c']);

    let known_numeric = beve::to_vec(&vec![1u32, 2, 3]).unwrap();
    let unknown_numeric = beve::to_vec(&UnknownLenSeq(&[1u32, 2, 3])).unwrap();
    assert_eq!(unknown_numeric, known_numeric);

    let known_bools = beve::to_vec(&vec![true, false, true, false, true]).unwrap();
    let unknown_bools = beve::to_vec(&UnknownLenSeq(&[true, false, true, false, true])).unwrap();
    assert_eq!(unknown_bools, known_bools);

    let labels = ["left", "right"];
    let known_strings = beve::to_vec(&labels.to_vec()).unwrap();
    let unknown_strings = beve::to_vec(&UnknownLenSeq(&labels)).unwrap();
    assert_eq!(unknown_strings, known_strings);

    let samples = [
        Complex {
            re: 1.0f64,
            im: 2.0,
        },
        Complex {
            re: -3.5f64,
            im: 4.25,
        },
    ];
    let known_complex = beve::to_vec(&samples.to_vec()).unwrap();
    let unknown_complex = beve::to_vec(&UnknownLenSeq(&samples)).unwrap();
    assert_eq!(unknown_complex, known_complex);

    let empty: [u32; 0] = [];
    assert_eq!(
        beve::to_vec(&UnknownLenSeq(&empty)).unwrap(),
        vec![0x05, 0x00]
    );
}

#[test]
fn heterogeneous_numeric_sequences_fall_back_to_generic_arrays() {
    let bytes = beve::to_vec(&MixedWidthNumericSeq).unwrap();
    assert_eq!(bytes[0], 0x05);

    let value: Value = beve::from_slice(&bytes).unwrap();
    let arr = value.as_array().unwrap();
    assert_eq!(arr.len(), 3);
    assert_eq!(arr[0].as_u64(), Some(1));
    assert_eq!(arr[1].as_u64(), Some(2));
    assert_eq!(arr[2].as_u64(), Some(3));
}

#[test]
fn variants_are_ordinary_values_per_spec_v2() {
    // A unit variant is its name as a plain string: `"Beta"`.
    let unit = beve::to_vec(&UnitEnum::Beta).unwrap();
    assert_eq!(unit, vec![0x02, 0x10, b'B', b'e', b't', b'a']);
    let back: UnitEnum = beve::from_slice(&unit).unwrap();
    assert_eq!(back, UnitEnum::Beta);

    // A newtype variant is a single-key object: `{ "Scalar": 7 }`. Object
    // header, count 1, key length 6, the key, then the value.
    let newtype = beve::to_vec(&TaggedEnum::Scalar(7)).unwrap();
    assert_eq!(
        newtype,
        vec![
            0x03, 0x04, 0x18, b'S', b'c', b'a', b'l', b'a', b'r', 0x11, 0x07
        ]
    );
    let back: TaggedEnum = beve::from_slice(&newtype).unwrap();
    assert_eq!(back, TaggedEnum::Scalar(7));

    // A tuple variant is the same object with an array payload, and a struct
    // variant the same with an object payload. Neither carries an extension.
    let tuple = beve::to_vec(&TaggedEnum::Pair(3, true)).unwrap();
    assert_eq!(tuple[0], 0x03);
    let tuple_back: TaggedEnum = beve::from_slice(&tuple).unwrap();
    assert_eq!(tuple_back, TaggedEnum::Pair(3, true));

    let named = beve::to_vec(&TaggedEnum::Named { count: 9 }).unwrap();
    assert_eq!(named[0], 0x03);
    let named_back: TaggedEnum = beve::from_slice(&named).unwrap();
    assert_eq!(named_back, TaggedEnum::Named { count: 9 });
}

/// The type tag extension (id 1) is reserved and deprecated in Version 2. No
/// encoder path may emit its header byte, `0x0E`.
#[test]
fn no_encoder_path_emits_the_deprecated_type_tag() {
    let mut streamed = Vec::new();
    beve::to_writer_streaming(&mut streamed, &TaggedEnum::Named { count: 9 }).unwrap();

    for bytes in [
        beve::to_vec(&UnitEnum::Beta).unwrap(),
        beve::to_vec(&TaggedEnum::Scalar(7)).unwrap(),
        beve::to_vec(&TaggedEnum::Pair(3, true)).unwrap(),
        beve::to_vec(&TaggedEnum::Named { count: 9 }).unwrap(),
        // Also as an element inside a generic array, which has its own paths.
        beve::to_vec(&vec![TaggedEnum::Scalar(1), TaggedEnum::Pair(2, false)]).unwrap(),
        streamed,
    ] {
        assert!(
            !bytes.contains(&0x0E),
            "type tag extension byte found in {bytes:02x?}"
        );
    }
}

/// Version 1 documents still decode: both the type tag extension and the bare
/// positional index the old numeric enum encoding wrote.
#[test]
fn version_1_variants_still_decode() {
    // `0x0e` type tag, index 0 as u32, then the value.
    let v1_newtype = vec![0x0e, 0x51, 0x00, 0x00, 0x00, 0x00, 0x11, 0x07];
    let back: TaggedEnum = beve::from_slice(&v1_newtype).unwrap();
    assert_eq!(back, TaggedEnum::Scalar(7));

    // `0x0e` type tag with a string tag rather than an index.
    let v1_named_tag = vec![
        0x0e, 0x02, 0x18, b'S', b'c', b'a', b'l', b'a', b'r', 0x11, 0x07,
    ];
    let back: TaggedEnum = beve::from_slice(&v1_named_tag).unwrap();
    assert_eq!(back, TaggedEnum::Scalar(7));

    // A bare positional index, which is what the removed numeric encoding
    // wrote for a unit variant.
    let v1_unit = vec![0x51, 0x01, 0x00, 0x00, 0x00];
    let back: UnitEnum = beve::from_slice(&v1_unit).unwrap();
    assert_eq!(back, UnitEnum::Beta);
}

#[test]
fn object_key_modes_match_spec() {
    let named = BTreeMap::from([(String::from("a"), 1u8)]);
    assert_eq!(
        beve::to_vec(&named).unwrap(),
        vec![0x03, 0x04, 0x04, b'a', 0x11, 0x01]
    );

    let signed = BTreeMap::from([(-2i16, 7u8), (5i16, 9u8)]);
    assert_eq!(
        beve::to_vec(&signed).unwrap(),
        vec![0x2b, 0x08, 0xfe, 0xff, 0x11, 0x07, 0x05, 0x00, 0x11, 0x09]
    );
    let signed_back: BTreeMap<i16, u8> = beve::from_slice(&beve::to_vec(&signed).unwrap()).unwrap();
    assert_eq!(signed_back, signed);

    let key = 0x0011_2233_4455_6677_8899_aabb_ccdd_eeffu128;
    let unsigned = BTreeMap::from([(key, 3u8)]);
    let mut expected = vec![0x93, 0x04];
    expected.extend_from_slice(&key.to_le_bytes());
    expected.extend_from_slice(&[0x11, 0x03]);
    assert_eq!(beve::to_vec(&unsigned).unwrap(), expected);

    let unsigned_back: BTreeMap<u128, u8> =
        beve::from_slice(&beve::to_vec(&unsigned).unwrap()).unwrap();
    assert_eq!(unsigned_back, unsigned);
}

#[test]
fn object_key_constraints_are_enforced() {
    let bool_keys = BTreeMap::from([(true, 1u8)]);
    let err = beve::to_vec(&bool_keys).unwrap_err().to_string();
    assert!(err.contains("boolean not allowed as object key"));

    let err = beve::to_vec(&MixedWidthKeyMap).unwrap_err().to_string();
    assert!(err.contains("same width"));
}

#[test]
fn unknown_length_maps_patch_sizes_and_empty_default_to_string_keys() {
    assert_eq!(beve::to_vec(&EmptyUnknownLenMap).unwrap(), vec![0x03, 0x00]);

    let bytes = beve::to_vec(&UnknownLenStringMap).unwrap();
    let mut expected = vec![0x03];
    expected.extend_from_slice(&size_prefix_for_test(2));
    expected.extend_from_slice(&size_prefix_for_test(5));
    expected.extend_from_slice(b"alpha");
    expected.extend_from_slice(&[0x11, 0x07]);
    expected.extend_from_slice(&size_prefix_for_test(4));
    expected.extend_from_slice(b"beta");
    expected.push(0x18);
    assert_eq!(bytes, expected);

    let value: Value = beve::from_slice(&bytes).unwrap();
    assert_eq!(value["alpha"].as_u64(), Some(7));
    assert_eq!(value["beta"].as_bool(), Some(true));
}

#[test]
fn matrix_extensions_cover_integer_bool_and_complex_payloads() {
    let ints = MatrixOwned {
        layout: MatrixLayout::Left,
        extents: vec![2, 2],
        data: vec![1u16, 2, 3, 4],
    };
    let int_bytes = beve::to_vec(&ints).unwrap();
    assert_eq!(int_bytes[0], 0x16);
    let int_back: MatrixOwned<u16> = beve::from_slice(&int_bytes).unwrap();
    assert_eq!(int_back, ints);

    let flags = MatrixOwned {
        layout: MatrixLayout::Right,
        extents: vec![2, 2],
        data: vec![true, false, true, true],
    };
    let flag_bytes = beve::to_vec(&flags).unwrap();
    assert_eq!(flag_bytes[0], 0x16);
    let flag_back: MatrixOwned<bool> = beve::from_slice(&flag_bytes).unwrap();
    assert_eq!(flag_back, flags);
    let raw = beve::decode_matrix_slice::<bool>(&flag_bytes, MatrixDecodeMode::Raw).unwrap();
    match raw {
        DecodedMatrix::Raw(matrix) => {
            assert_eq!(matrix.layout, MatrixLayout::Right);
            assert_eq!(matrix.extents, vec![2, 2]);
            let values = matrix.value.as_array().unwrap();
            assert_eq!(values.len(), 4);
            assert_eq!(values[0].as_bool(), Some(true));
            assert_eq!(values[1].as_bool(), Some(false));
        }
        DecodedMatrix::Materialized(_) => panic!("expected raw matrix"),
    }

    let complex32 = MatrixOwned {
        layout: MatrixLayout::Left,
        extents: vec![2],
        data: vec![
            Complex {
                re: 1.0f32,
                im: -1.0,
            },
            Complex {
                re: 2.5f32,
                im: 0.75,
            },
        ],
    };
    let complex32_bytes = beve::to_vec(&complex32).unwrap();
    assert_eq!(complex32_bytes[0], 0x16);
    let complex32_back: MatrixOwned<Complex<f32>> = beve::from_slice(&complex32_bytes).unwrap();
    assert_eq!(complex32_back, complex32);

    let complex64 = MatrixOwned {
        layout: MatrixLayout::Right,
        extents: vec![2],
        data: vec![
            Complex {
                re: 3.0f64,
                im: 4.0,
            },
            Complex {
                re: -5.0f64,
                im: 6.5,
            },
        ],
    };
    let complex64_bytes = beve::to_vec(&complex64).unwrap();
    assert_eq!(complex64_bytes[0], 0x16);
    let complex64_back: MatrixOwned<Complex<f64>> = beve::from_slice(&complex64_bytes).unwrap();
    assert_eq!(complex64_back, complex64);
}

#[test]
fn reader_and_writer_apis_preserve_wire_format() {
    let value = TaggedEnum::Scalar(7);
    let mut out = Vec::new();
    beve::to_writer(&mut out, &value).unwrap();
    assert_eq!(out, beve::to_vec(&value).unwrap());

    let decoded: TaggedEnum = from_reader(Cursor::new(out)).unwrap();
    assert_eq!(decoded, value);
}

#[test]
fn malformed_inputs_are_rejected() {
    let err = beve::from_slice::<String>(&[0x02, 0x04, 0xff])
        .unwrap_err()
        .to_string();
    assert!(err.contains("invalid utf-8"));

    let err = beve::from_slice::<Vec<String>>(&[0x3c, 0x04, 0x04, 0xff])
        .unwrap_err()
        .to_string();
    assert!(err.contains("invalid utf-8"));

    let err = beve::from_slice::<Value>(&[0x03, 0x04, 0x04, 0xff, 0x11, 0x01])
        .unwrap_err()
        .to_string();
    assert!(err.contains("invalid utf-8 in key"));

    // 0x1e=EXT_COMPLEX, 0x28=signed i16 single complex, 4 data bytes → valid complex<i16>
    assert!(beve::validate_slice(&[0x1e, 0x28, 0x00, 0x00, 0x00, 0x00]).is_ok());
    // class=3 is invalid (only 0=float, 1=signed, 2=unsigned are valid)
    assert!(beve::validate_slice(&[0x1e, 0x18, 0x00, 0x00]).is_err());

    let mut bad_matrix =
        beve::fast::to_vec_matrix_f64(beve::fast::MatrixLayoutFast::Left, &[2], &[1.0, 2.0]);
    bad_matrix[2] = 0x1c;
    assert!(beve::validate_slice(&bad_matrix).is_err());
}

// ---------------------------------------------------------------------------
// BEVE Version 2: variants are ordinary values, so BEVE equals JSON
// ---------------------------------------------------------------------------

/// The property that makes the Version 2 variant encoding correct: converting a
/// BEVE document to JSON yields exactly what `serde_json` produces for the same
/// value. Version 1's type tag extension could not satisfy this, because a
/// positional index has no JSON equivalent beyond an opaque
/// `{"index":_,"value":_}`.
///
/// This holds across all four of serde's enum representations, and it holds
/// without any variant-specific support: serde's derive lowers the internally
/// tagged, adjacently tagged and untagged forms to ordinary maps and bare
/// values before the serializer ever sees them, and the externally tagged
/// default is written as the single-key object `serde_json` also writes.
mod json_equivalence {
    use serde::{Deserialize, Serialize};

    /// Note the fixtures below avoid whole-number floats: the JSON converter
    /// renders `5.0` as `5`, a number-formatting difference of its own that
    /// would otherwise mask the structural property under test.
    fn assert_matches_serde_json<T: Serialize>(value: &T) {
        let beve_bytes = beve::to_vec(value).unwrap();
        let via_beve: serde_json::Value =
            serde_json::from_str(&beve::beve_slice_to_json_string(&beve_bytes).unwrap()).unwrap();
        let direct = serde_json::to_value(value).unwrap();
        assert_eq!(via_beve, direct);
    }

    /// Serde's default. `{"Circle":{"radius":5.0}}`.
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    enum External {
        Empty,
        Circle { radius: f64 },
        Pair(u8, bool),
        Wrapped(String),
    }

    /// `#[serde(tag)]`. `{"kind":"circle","radius":5.0}`.
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    #[serde(tag = "kind")]
    enum Internal {
        Circle { radius: f64 },
        Square { side: f64 },
    }

    /// `#[serde(tag, content)]`. `{"t":"Circle","c":5.0}`.
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    #[serde(tag = "t", content = "c")]
    enum Adjacent {
        Circle(f64),
        Pair(u8, bool),
    }

    /// `#[serde(untagged)]`. The bare value, with no discriminator at all.
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    #[serde(untagged)]
    enum Untagged {
        Num(u32),
        Text(String),
    }

    #[test]
    fn externally_tagged_matches_serde_json() {
        for v in [
            External::Empty,
            External::Circle { radius: 5.25 },
            External::Pair(3, true),
            External::Wrapped("hi".into()),
        ] {
            assert_matches_serde_json(&v);
            let back: External = beve::from_slice(&beve::to_vec(&v).unwrap()).unwrap();
            assert_eq!(back, v);
        }
    }

    #[test]
    fn internally_tagged_matches_serde_json() {
        for v in [
            Internal::Circle { radius: 5.25 },
            Internal::Square { side: 2.75 },
        ] {
            assert_matches_serde_json(&v);
            let back: Internal = beve::from_slice(&beve::to_vec(&v).unwrap()).unwrap();
            assert_eq!(back, v);
        }
    }

    #[test]
    fn adjacently_tagged_matches_serde_json() {
        for v in [Adjacent::Circle(5.25), Adjacent::Pair(3, true)] {
            assert_matches_serde_json(&v);
            let back: Adjacent = beve::from_slice(&beve::to_vec(&v).unwrap()).unwrap();
            assert_eq!(back, v);
        }
    }

    #[test]
    fn untagged_matches_serde_json() {
        for v in [Untagged::Num(7), Untagged::Text("hi".into())] {
            assert_matches_serde_json(&v);
            let back: Untagged = beve::from_slice(&beve::to_vec(&v).unwrap()).unwrap();
            assert_eq!(back, v);
        }
    }

    /// Glaze's Version 2 shape for a `std::variant` declaring `tag`/`ids` is a
    /// tagged object with the discriminator merged in as a member. That is
    /// serde's internally tagged representation, so it is expressible with a
    /// plain attribute and needs no support from this crate.
    #[test]
    fn the_glaze_v2_tagged_shape_is_reachable_with_a_serde_attribute() {
        let json = beve::beve_slice_to_json_string(
            &beve::to_vec(&Internal::Circle { radius: 5.25 }).unwrap(),
        )
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["kind"], "Circle");
        assert_eq!(v["radius"], 5.25);
    }
}

/// The buffered and streaming readers must agree on every document, and the
/// buffered and streaming writers on every value. Both halves of the crate
/// grew a variant path in Version 2, and a divergence between them is invisible
/// to any test that exercises only one side.
mod reader_writer_parity {
    use super::*;

    /// `Scalar` is a UNIT variant here, so a `{"Scalar": <payload>}` document
    /// carries a value this schema does not want. That is ordinary schema drift:
    /// it is what a peer writes after a variant's payload is dropped locally.
    #[derive(Debug, PartialEq, Deserialize)]
    enum DriftedEnum {
        Scalar,
        #[allow(dead_code)]
        Other(u32),
    }

    #[derive(Debug, PartialEq, Deserialize)]
    struct DriftedOuter {
        a: DriftedEnum,
        b: u8,
    }

    /// Assert both readers reach the same `Ok` value for the same bytes.
    fn assert_readers_agree<T>(bytes: &[u8], expected: T)
    where
        T: serde::de::DeserializeOwned + PartialEq + core::fmt::Debug,
    {
        let buffered: T = beve::from_slice(bytes).expect("buffered reader rejected the document");
        let streamed: T =
            beve::from_reader_streaming(bytes).expect("streaming reader rejected the document");
        assert_eq!(buffered, expected, "buffered reader");
        assert_eq!(streamed, expected, "streaming reader");
    }

    /// Assert both readers reject the same bytes. Which error is not pinned;
    /// that both refuse is the property.
    fn assert_readers_both_reject<T>(bytes: &[u8])
    where
        T: serde::de::DeserializeOwned + core::fmt::Debug,
    {
        let buffered = beve::from_slice::<T>(bytes);
        let streamed = beve::from_reader_streaming::<_, T>(bytes);
        assert!(
            buffered.is_err(),
            "buffered reader accepted malformed input: {buffered:?}"
        );
        assert!(
            streamed.is_err(),
            "streaming reader accepted malformed input: {streamed:?}"
        );
    }

    /// A unit-variant target must skip a payload the wire declares, not read it
    /// as the next sibling. The streaming reader used to return `(Scalar, 7)`
    /// here, silently substituting the discarded payload for the real element.
    #[test]
    fn a_dropped_payload_is_discarded_not_read_as_the_next_element() {
        // `[{"Scalar": 7u32}, 9u8]`
        let mut bytes = vec![0x05, 0x08, 0x03, 0x04, 0x18];
        bytes.extend_from_slice(b"Scalar");
        bytes.extend_from_slice(&[0x51, 0x07, 0x00, 0x00, 0x00, 0x11, 0x09]);

        assert_readers_agree(&bytes, (DriftedEnum::Scalar, 9u8));
    }

    /// The same drift as a struct field. The streaming reader used to fail
    /// outright while the buffered reader succeeded.
    #[test]
    fn a_dropped_payload_is_discarded_inside_a_struct() {
        // `{"a": {"Scalar": 7u32}, "b": 5u8}`
        let mut bytes = vec![0x03, 0x08, 0x04, b'a', 0x03, 0x04, 0x18];
        bytes.extend_from_slice(b"Scalar");
        bytes.extend_from_slice(&[0x51, 0x07, 0x00, 0x00, 0x00, 0x04, b'b', 0x11, 0x05]);

        assert_readers_agree(
            &bytes,
            DriftedOuter {
                a: DriftedEnum::Scalar,
                b: 5,
            },
        );
    }

    /// The object header promises one value. If it is missing, that is a
    /// truncated document and both readers must say so rather than returning a
    /// variant built from bytes that were never there.
    #[test]
    fn a_truncated_variant_object_is_rejected() {
        let mut bytes = vec![0x03, 0x04, 0x18];
        bytes.extend_from_slice(b"Scalar");
        assert_readers_both_reject::<DriftedEnum>(&bytes);
    }

    /// A variant object must have exactly one key, in both readers.
    #[test]
    fn a_multi_key_variant_object_is_rejected() {
        let mut bytes = vec![0x03, 0x08, 0x18];
        bytes.extend_from_slice(b"Scalar");
        bytes.extend_from_slice(&[0x11, 0x07, 0x04, b'x', 0x11, 0x01]);
        assert_readers_both_reject::<TaggedEnum>(&bytes);
    }

    /// Every variant kind, through both readers and both writers. This is the
    /// matrix that the single-sided tests leave uncovered.
    #[test]
    fn every_variant_kind_round_trips_through_both_readers_and_writers() {
        macro_rules! check {
            ($value:expr, $ty:ty) => {{
                let value: $ty = $value;
                let buffered = beve::to_vec(&value).unwrap();

                let mut streamed = Vec::new();
                beve::to_writer_streaming(&mut streamed, &value).unwrap();
                assert_eq!(
                    buffered, streamed,
                    "writers disagree for {:?}: {buffered:02x?} vs {streamed:02x?}",
                    value
                );

                assert_readers_agree::<$ty>(&buffered, $value);
            }};
        }

        check!(UnitEnum::Alpha, UnitEnum);
        check!(UnitEnum::Beta, UnitEnum);
        check!(TaggedEnum::Scalar(7), TaggedEnum);
        check!(TaggedEnum::Pair(3, true), TaggedEnum);
        check!(TaggedEnum::Named { count: 9 }, TaggedEnum);
    }

    /// Each variant kind as the FIRST element of an unknown-length sequence.
    /// That is the one position routed through the element serializer, which
    /// owns the array's element count; a path that forgot to count itself wrote
    /// a header claiming fewer elements than it had emitted, and the tail
    /// decoded as trailing garbage.
    #[test]
    fn variants_leading_an_unknown_length_sequence_keep_their_element_count() {
        for values in [
            vec![TaggedEnum::Scalar(1), TaggedEnum::Scalar(2)],
            vec![TaggedEnum::Pair(1, true), TaggedEnum::Scalar(2)],
            vec![TaggedEnum::Named { count: 1 }, TaggedEnum::Scalar(2)],
            // A single element makes an off-by-one produce a count of zero.
            vec![TaggedEnum::Scalar(1)],
        ] {
            let bytes = beve::to_vec(&UnknownLenSeq(&values)).unwrap();
            beve::validate_slice(&bytes).unwrap_or_else(|e| {
                panic!("not a well-formed document for {values:?}: {e:?} in {bytes:02x?}")
            });
            let back: Vec<TaggedEnum> = beve::from_slice(&bytes).unwrap();
            assert_eq!(back, values, "element lost in {bytes:02x?}");
        }

        // The unit-variant path for the same position.
        let units = vec![UnitEnum::Beta, UnitEnum::Alpha];
        let bytes = beve::to_vec(&UnknownLenSeq(&units)).unwrap();
        beve::validate_slice(&bytes).unwrap();
        let back: Vec<UnitEnum> = beve::from_slice(&bytes).unwrap();
        assert_eq!(back, units);
    }

    /// Where the two writers agree, and the one documented place they do not.
    ///
    /// Sequences agree, because both coalesce a homogeneous one into a typed
    /// array. Tuples do not, and cannot: serde routes Rust arrays through
    /// `serialize_tuple`, so a homogeneous `[u8; 4]` is indistinguishable from a
    /// mixed `(u8, bool)`, and the two want opposite encodings. The buffered
    /// writer detects and can rewrite its own header when a later element
    /// disagrees; the streaming writer would be committed the moment it wrote
    /// one, so it emits generic unconditionally rather than failing on every
    /// mixed tuple.
    ///
    /// Both encodings are valid and decode to the same value through either
    /// reader, which is what this pins. Anyone tempted to "fix" the divergence
    /// should read the streaming `serialize_tuple` comment first.
    #[test]
    fn the_two_writers_agree_except_on_tuples_where_they_provably_cannot() {
        macro_rules! bytes_both_ways {
            ($value:expr) => {{
                let buffered = beve::to_vec(&$value).unwrap();
                let mut streamed = Vec::new();
                beve::to_writer_streaming(&mut streamed, &$value).unwrap();
                // Whichever they produce must be a well-formed document.
                for (which, b) in [("buffered", &buffered), ("streaming", &streamed)] {
                    beve::validate_slice(b).unwrap_or_else(|e| {
                        panic!(
                            "{which} produced a malformed document for {}: {e:?}",
                            stringify!($value)
                        )
                    });
                }
                (buffered, streamed)
            }};
        }
        macro_rules! agree {
            ($value:expr) => {{
                let (buffered, streamed) = bytes_both_ways!($value);
                assert_eq!(
                    buffered,
                    streamed,
                    "writers must agree for {}: {buffered:02x?} vs {streamed:02x?}",
                    stringify!($value)
                );
            }};
        }
        /// Asserts the divergence exists AND that both forms decode to the same
        /// value through both readers, which is the property that makes it benign.
        macro_rules! differ_but_round_trip {
            ($value:expr, $ty:ty) => {{
                let (buffered, streamed) = bytes_both_ways!($value);
                assert_ne!(
                    buffered,
                    streamed,
                    "expected the documented tuple divergence for {}",
                    stringify!($value)
                );
                for (which, b) in [("buffered", &buffered), ("streaming", &streamed)] {
                    let via_slice: $ty = beve::from_slice(b)
                        .unwrap_or_else(|e| panic!("{which} bytes failed from_slice: {e:?}"));
                    let via_reader: $ty = beve::from_reader_streaming(&b[..])
                        .unwrap_or_else(|e| panic!("{which} bytes failed streaming read: {e:?}"));
                    assert_eq!(via_slice, $value, "{which} bytes, buffered reader");
                    assert_eq!(via_reader, $value, "{which} bytes, streaming reader");
                }
            }};
        }

        // Sequences agree, homogeneous or not.
        agree!(vec![1u8, 2, 3]);
        agree!(vec![1.5f64, 2.5]);
        agree!(vec!["a".to_string(), "b".to_string()]);
        agree!(Vec::<u8>::new());
        // A mixed tuple agrees too: the buffered writer also lands on generic.
        agree!((1u8, true));
        agree!((1u8, "two".to_string(), 3.5f64));
        // Nested sequences agree.
        agree!(vec![vec![1u8, 2], vec![3, 4]]);

        // Homogeneous arrays and tuples are the documented exception: buffered
        // coalesces to a typed array, streaming stays generic.
        differ_but_round_trip!([1u8, 2, 3, 4], [u8; 4]);
        differ_but_round_trip!([1.5f64, 2.5], [f64; 2]);
        differ_but_round_trip!((1u8, 2u8), (u8, u8));

        // A mixed-width tuple is not homogeneous, so both stay generic.
        agree!((1u8, 2u16));
    }

    /// Version 1 fixtures must decode identically in both readers too.
    #[test]
    fn version_1_variants_decode_the_same_in_both_readers() {
        assert_readers_agree(
            &[0x0e, 0x51, 0x00, 0x00, 0x00, 0x00, 0x11, 0x07],
            TaggedEnum::Scalar(7),
        );
        assert_readers_agree(
            &[
                0x0e, 0x02, 0x18, b'S', b'c', b'a', b'l', b'a', b'r', 0x11, 0x07,
            ],
            TaggedEnum::Scalar(7),
        );
        assert_readers_agree(&[0x51, 0x01, 0x00, 0x00, 0x00], UnitEnum::Beta);
    }

    /// Byte-for-byte output captured from beve 4.0.0, covering the three places
    /// it put a unit variant. Each is a distinct wire form, and the middle one
    /// regressed once already:
    ///
    /// - top level and struct field: a bare positional index, no extension
    /// - leading a sequence: the type-tag extension, the index, and an explicit
    ///   `null` payload (`write_null` after `write_enum_tag`)
    ///
    /// That third form is why a unit-variant target must consume a value after
    /// the extension. Reading the tag and stopping leaves the `null` to be taken
    /// as the next element, which made a 4.x-written `Vec<SomeUnitEnum>`
    /// undecodable.
    #[test]
    fn unit_variants_written_by_version_4_still_decode() {
        // `UnitEnum::Beta` at the root.
        assert_readers_agree(&[0x51, 0x01, 0x00, 0x00, 0x00], UnitEnum::Beta);

        // `vec![UnitEnum::Beta, UnitEnum::Alpha]`: the first element carries the
        // extension plus a null, the second is a bare index.
        assert_readers_agree(
            &[
                0x05, 0x08, 0x0e, 0x51, 0x01, 0x00, 0x00, 0x00, 0x00, 0x51, 0x00, 0x00, 0x00, 0x00,
            ],
            vec![UnitEnum::Beta, UnitEnum::Alpha],
        );

        // A unit variant as a struct field, with a sibling field after it. The
        // bare index must not swallow the next key.
        #[derive(Debug, PartialEq, Deserialize)]
        struct V4Outer {
            a: UnitEnum,
            b: u8,
        }
        assert_readers_agree(
            &[
                0x03, 0x08, 0x04, b'a', 0x51, 0x01, 0x00, 0x00, 0x00, 0x04, b'b', 0x11, 0x05,
            ],
            V4Outer {
                a: UnitEnum::Beta,
                b: 5,
            },
        );
    }
}
