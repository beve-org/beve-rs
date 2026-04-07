#![deny(warnings)]

use std::collections::BTreeMap;
use std::io::Cursor;

use beve::{
    Complex, DecodedMatrix, EnumEncoding, MatrixDecodeMode, MatrixLayout, MatrixOwned,
    SerializerOptions, Value, from_reader,
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
fn enum_encoding_matches_number_and_string_spec_modes() {
    let number_unit = beve::to_vec(&UnitEnum::Beta).unwrap();
    assert_eq!(number_unit, vec![0x51, 0x01, 0x00, 0x00, 0x00]);

    let string_opts = SerializerOptions {
        enum_encoding: EnumEncoding::String,
    };
    let string_unit = beve::to_vec_with_options(&UnitEnum::Beta, string_opts).unwrap();
    assert_eq!(string_unit, vec![0x02, 0x10, b'B', b'e', b't', b'a']);

    let number_newtype = beve::to_vec(&TaggedEnum::Scalar(7)).unwrap();
    assert_eq!(
        number_newtype,
        vec![0x0e, 0x51, 0x00, 0x00, 0x00, 0x00, 0x11, 0x07]
    );
    let back: TaggedEnum = beve::from_slice(&number_newtype).unwrap();
    assert_eq!(back, TaggedEnum::Scalar(7));

    let string_newtype = beve::to_vec_with_options(&TaggedEnum::Scalar(7), string_opts).unwrap();
    assert_eq!(
        string_newtype,
        vec![
            0x0e, 0x02, 0x18, b'S', b'c', b'a', b'l', b'a', b'r', 0x11, 0x07
        ]
    );
    let back: TaggedEnum = beve::from_slice(&string_newtype).unwrap();
    assert_eq!(back, TaggedEnum::Scalar(7));

    let tuple_bytes = beve::to_vec(&TaggedEnum::Pair(3, true)).unwrap();
    assert_eq!(tuple_bytes[0], 0x0e);
    let tuple_back: TaggedEnum = beve::from_slice(&tuple_bytes).unwrap();
    assert_eq!(tuple_back, TaggedEnum::Pair(3, true));

    let struct_bytes =
        beve::to_vec_with_options(&TaggedEnum::Named { count: 9 }, string_opts).unwrap();
    assert_eq!(struct_bytes[0], 0x0e);
    let struct_back: TaggedEnum = beve::from_slice(&struct_bytes).unwrap();
    assert_eq!(struct_back, TaggedEnum::Named { count: 9 });
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

    let opts = SerializerOptions {
        enum_encoding: EnumEncoding::String,
    };
    let mut out_with_opts = Vec::new();
    beve::to_writer_with_options(&mut out_with_opts, &value, opts).unwrap();
    assert_eq!(
        out_with_opts,
        beve::to_vec_with_options(&value, opts).unwrap()
    );

    let decoded: TaggedEnum = from_reader(Cursor::new(out_with_opts)).unwrap();
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
