#![deny(warnings)]

use half::{bf16, f16};
use serde::{Deserialize, Serialize};

#[test]
fn roundtrip_primitives() {
    fn rt<T: Serialize + for<'de> Deserialize<'de> + PartialEq + core::fmt::Debug>(v: T) {
        let bytes = beve::to_vec(&v).unwrap();
        let back: T = beve::from_slice(&bytes).unwrap();
        assert_eq!(v, back);
    }
    rt::<Option<()>>(None);
    rt(true);
    rt(false);
    rt(0i8);
    rt(-5i32);
    rt(123i64);
    rt(255u8);
    rt(65535u16);
    rt(1_000_000u32);
    rt(1_000_000_000_000u64);
    rt(std::f32::consts::PI);
    rt(-std::f64::consts::E);
    rt(String::from("hello"));
}

#[test]
fn roundtrip_arrays() {
    let v: Vec<u32> = vec![1, 2, 3, 4, 5];
    let bytes = beve::to_vec(&v).unwrap();
    let back: Vec<u32> = beve::from_slice(&bytes).unwrap();
    assert_eq!(v, back);

    let vb: Vec<bool> = vec![true, false, true, true, false, false, true, false, true];
    let bytes = beve::to_vec(&vb).unwrap();
    let back: Vec<bool> = beve::from_slice(&bytes).unwrap();
    assert_eq!(vb, back);

    let vs: Vec<String> = vec!["a".into(), "bb".into(), "ccc".into()];
    let bytes = beve::to_vec(&vs).unwrap();
    let back: Vec<String> = beve::from_slice(&bytes).unwrap();
    assert_eq!(vs, back);
}

#[test]
fn roundtrip_half_scalars() {
    let f16_values = [
        f16::from_f32(0.0),
        f16::from_f32(-3.5),
        f16::from_bits(0x7bff), // max finite
    ];
    for &value in &f16_values {
        let bytes = beve::to_vec(&value).unwrap();
        let back: f16 = beve::from_slice(&bytes).unwrap();
        assert_eq!(value, back);
    }

    let bf16_values = [
        bf16::from_f32(1.0),
        bf16::from_f32(-2.25),
        bf16::from_bits(0x7f80), // infinity
    ];
    for &value in &bf16_values {
        let bytes = beve::to_vec(&value).unwrap();
        let back: bf16 = beve::from_slice(&bytes).unwrap();
        assert_eq!(value, back);
    }
}

#[test]
fn roundtrip_half_arrays() {
    let f16_vec: Vec<f16> = vec![f16::from_f32(-1.0), f16::from_f32(0.5), f16::from_f32(3.75)];
    let bytes = beve::to_vec(&f16_vec).unwrap();
    let back: Vec<f16> = beve::from_slice(&bytes).unwrap();
    assert_eq!(f16_vec, back);

    let bf16_vec: Vec<bf16> = vec![
        bf16::from_f32(-10.0),
        bf16::from_f32(0.0),
        bf16::from_f32(42.0),
    ];
    let bytes = beve::to_vec(&bf16_vec).unwrap();
    let back: Vec<bf16> = beve::from_slice(&bytes).unwrap();
    assert_eq!(bf16_vec, back);
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Point {
    x: f64,
    y: f64,
}

#[test]
fn roundtrip_struct() {
    let p = Point { x: 1.0, y: -2.0 };
    let bytes = beve::to_vec(&p).unwrap();
    let back: Point = beve::from_slice(&bytes).unwrap();
    assert_eq!(p, back);
}

#[test]
fn roundtrip_map_keys() {
    use std::collections::BTreeMap;
    let mut m: BTreeMap<u32, i32> = BTreeMap::new();
    m.insert(1, -1);
    m.insert(2, 2);
    let bytes = beve::to_vec(&m).unwrap();
    let back: BTreeMap<u32, i32> = beve::from_slice(&bytes).unwrap();
    assert_eq!(m, back);
}

#[test]
fn roundtrip_empty_hashmap() {
    use std::collections::HashMap;

    let m: HashMap<String, i32> = HashMap::new();
    let bytes = beve::to_vec(&m).unwrap();
    let back: HashMap<String, i32> = beve::from_slice(&bytes).unwrap();
    assert_eq!(m, back);

    let m: HashMap<u64, u64> = HashMap::new();
    let bytes = beve::to_vec(&m).unwrap();
    let back: HashMap<u64, u64> = beve::from_slice(&bytes).unwrap();
    assert_eq!(m, back);
}

#[test]
fn roundtrip_hashmap_in_struct() {
    use std::collections::HashMap;
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Wrapper {
        name: String,
        rates: HashMap<u64, u64>,
        flags: HashMap<String, bool>,
    }

    let w = Wrapper {
        name: "test".into(),
        rates: [(1, 100), (2, 200)].into_iter().collect(),
        flags: [("a".into(), true), ("b".into(), false)].into_iter().collect(),
    };
    let bytes = beve::to_vec(&w).unwrap();
    let back: Wrapper = beve::from_slice(&bytes).unwrap();
    assert_eq!(w, back);

    let w = Wrapper {
        name: "test".into(),
        rates: HashMap::new(),
        flags: HashMap::new(),
    };
    let bytes = beve::to_vec(&w).unwrap();
    let back: Wrapper = beve::from_slice(&bytes).unwrap();
    assert_eq!(w, back);
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
enum MyEnum {
    Unit,
    Newtype(i32),
    Tuple(i32, u32),
    Struct { a: i32, b: u32 },
}

#[test]
fn roundtrip_enum() {
    let v = MyEnum::Unit;
    let bytes = beve::to_vec(&v).unwrap();
    let back: MyEnum = beve::from_slice(&bytes).unwrap();
    assert_eq!(v, back);

    let v = MyEnum::Newtype(5);
    let bytes = beve::to_vec(&v).unwrap();
    let back: MyEnum = beve::from_slice(&bytes).unwrap();
    assert_eq!(v, back);

    let v = MyEnum::Tuple(-1, 2);
    let bytes = beve::to_vec(&v).unwrap();
    let back: MyEnum = beve::from_slice(&bytes).unwrap();
    assert_eq!(v, back);

    let v = MyEnum::Struct { a: -7, b: 8 };
    let bytes = beve::to_vec(&v).unwrap();
    let back: MyEnum = beve::from_slice(&bytes).unwrap();
    assert_eq!(v, back);
}

#[test]
fn complex_serde_emits_extension() {
    let c32 = beve::Complex {
        re: 1.25f32,
        im: -2.5f32,
    };
    let bytes = beve::to_vec(&c32).unwrap();
    assert_eq!(bytes[0], 0x1e);
    assert_eq!(bytes[1], 0x40);
    let back: beve::Complex<f32> = beve::from_slice(&bytes).unwrap();
    assert_eq!(back, c32);

    let c64 = beve::Complex {
        re: 1.5f64,
        im: -2.25f64,
    };
    let bytes = beve::to_vec(&c64).unwrap();
    assert_eq!(bytes[0], 0x1e);
    assert_eq!(bytes[1], 0x60);
    let back: beve::Complex<f64> = beve::from_slice(&bytes).unwrap();
    assert_eq!(back, c64);
}

#[test]
fn complex_vec_serde_emits_array_extension() {
    let values = vec![
        beve::Complex {
            re: 1.0f64,
            im: 2.0,
        },
        beve::Complex { re: -3.0, im: 4.5 },
    ];
    let bytes = beve::to_vec(&values).unwrap();
    assert_eq!(bytes[0], 0x1e);
    assert_eq!(bytes[1], 0x61);
    let back: Vec<beve::Complex<f64>> = beve::from_slice(&bytes).unwrap();
    assert_eq!(back, values);
}

#[test]
fn complex_vec_f32_serde_emits_array_extension() {
    let values = vec![
        beve::Complex {
            re: 1.0f32,
            im: 2.0,
        },
        beve::Complex { re: -3.0, im: 4.5 },
    ];
    let bytes = beve::to_vec(&values).unwrap();
    assert_eq!(bytes[0], 0x1e);
    assert_eq!(bytes[1], 0x41);
    let back: Vec<beve::Complex<f32>> = beve::from_slice(&bytes).unwrap();
    assert_eq!(back, values);
}

#[test]
fn matrix_extension_api_roundtrip_and_decode_modes() {
    let owned = beve::MatrixOwned {
        layout: beve::MatrixLayout::Right,
        extents: vec![2, 3],
        data: vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    };

    let bytes = beve::to_vec(&owned).unwrap();
    assert_eq!(bytes[0], 0x16);

    let back: beve::MatrixOwned<f64> = beve::from_slice(&bytes).unwrap();
    assert_eq!(back, owned);

    let decoded =
        beve::decode_matrix_slice::<f64>(&bytes, beve::MatrixDecodeMode::Materialized).unwrap();
    match decoded {
        beve::DecodedMatrix::Materialized(m) => assert_eq!(m, owned),
        beve::DecodedMatrix::Raw(_) => panic!("expected materialized matrix"),
    }

    let decoded = beve::decode_matrix_slice::<f64>(&bytes, beve::MatrixDecodeMode::Raw).unwrap();
    match decoded {
        beve::DecodedMatrix::Raw(m) => {
            assert_eq!(m.layout, beve::MatrixLayout::Right);
            assert_eq!(m.extents, vec![2, 3]);
            let values = m.value.as_array().unwrap();
            assert_eq!(values.len(), 6);
            assert_eq!(values[0].as_f64(), Some(1.0));
            assert_eq!(values[5].as_f64(), Some(6.0));
        }
        beve::DecodedMatrix::Materialized(_) => panic!("expected raw matrix"),
    }
}

#[test]
fn matrix_falls_back_to_map_for_unsupported_elements() {
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    struct Cell {
        id: u32,
        ok: bool,
    }

    let owned = beve::MatrixOwned {
        layout: beve::MatrixLayout::Left,
        extents: vec![2],
        data: vec![Cell { id: 10, ok: true }, Cell { id: 11, ok: false }],
    };

    let bytes = beve::to_vec(&owned).unwrap();
    assert_ne!(bytes[0], 0x16);

    let back: beve::MatrixOwned<Cell> = beve::from_slice(&bytes).unwrap();
    assert_eq!(back, owned);

    let decoded = beve::decode_matrix_slice::<Cell>(&bytes, beve::MatrixDecodeMode::Raw).unwrap();
    match decoded {
        beve::DecodedMatrix::Raw(raw) => {
            assert_eq!(raw.layout, beve::MatrixLayout::Left);
            assert_eq!(raw.extents, vec![2]);
            let arr = raw.value.as_array().unwrap();
            assert_eq!(arr.len(), 2);
            assert_eq!(arr[0]["id"].as_u64(), Some(10));
            assert_eq!(arr[1]["ok"].as_bool(), Some(false));
        }
        beve::DecodedMatrix::Materialized(_) => panic!("expected raw matrix"),
    }
}

#[test]
fn matrix_shape_validation_errors() {
    #[derive(Serialize)]
    struct MatrixRepr<T> {
        layout: beve::MatrixLayout,
        extents: Vec<usize>,
        value: Vec<T>,
    }

    let bad_empty = MatrixRepr::<f64> {
        layout: beve::MatrixLayout::Right,
        extents: vec![],
        value: vec![],
    };
    let err = beve::from_slice::<beve::MatrixOwned<f64>>(&beve::to_vec(&bad_empty).unwrap())
        .unwrap_err()
        .to_string();
    assert!(err.contains("matrix extents cannot be empty"));

    let bad_zero = MatrixRepr::<f64> {
        layout: beve::MatrixLayout::Right,
        extents: vec![2, 0],
        value: vec![],
    };
    let err = beve::from_slice::<beve::MatrixOwned<f64>>(&beve::to_vec(&bad_zero).unwrap())
        .unwrap_err()
        .to_string();
    assert!(err.contains("matrix dimensions cannot be zero"));

    let bad_len = MatrixRepr {
        layout: beve::MatrixLayout::Right,
        extents: vec![2, 2],
        value: vec![1.0f64, 2.0, 3.0],
    };
    let err = beve::from_slice::<beve::MatrixOwned<f64>>(&beve::to_vec(&bad_len).unwrap())
        .unwrap_err()
        .to_string();
    assert!(err.contains("does not match product of extents"));

    let bad_overflow = MatrixRepr::<f64> {
        layout: beve::MatrixLayout::Right,
        extents: vec![usize::MAX, 2],
        value: vec![],
    };
    let err = beve::from_slice::<beve::MatrixOwned<f64>>(&beve::to_vec(&bad_overflow).unwrap())
        .unwrap_err()
        .to_string();
    assert!(err.contains("matrix extents overflow"));
}

#[test]
fn matrix_shape_validation_on_serialize() {
    let invalid = beve::MatrixOwned {
        layout: beve::MatrixLayout::Right,
        extents: vec![2, 2],
        data: vec![1.0f64, 2.0, 3.0],
    };
    let err = beve::to_vec(&invalid).unwrap_err().to_string();
    assert!(err.contains("does not match product of extents"));
}

#[test]
fn matrix_extents_use_compact_unsigned_widths() {
    let cases = [
        (255usize, 0x14u8),    // u8 extents
        (256usize, 0x34u8),    // u16 extents
        (65_535usize, 0x34u8), // u16 extents
        (65_536usize, 0x54u8), // u32 extents
    ];

    for (extent, header) in cases {
        let owned = beve::MatrixOwned {
            layout: beve::MatrixLayout::Right,
            extents: vec![extent],
            data: vec![0u8; extent],
        };
        let bytes = beve::to_vec(&owned).unwrap();
        assert_eq!(bytes[0], 0x16);
        assert_eq!(bytes[2], header);
    }
}

#[test]
fn matrix_decode_accepts_u64_extents_header() {
    let bytes =
        beve::fast::to_vec_matrix_f64(beve::fast::MatrixLayoutFast::Left, &[2], &[10.0, 20.0]);
    assert_eq!(bytes[0], 0x16);
    assert_eq!(bytes[2], 0x74);

    let matrix: beve::MatrixOwned<f64> = beve::from_slice(&bytes).unwrap();
    assert_eq!(matrix.layout, beve::MatrixLayout::Left);
    assert_eq!(matrix.extents, vec![2]);
    assert_eq!(matrix.data, vec![10.0, 20.0]);
}

#[test]
fn to_vec_into_reuses_buffer() {
    let mut out = Vec::with_capacity(256);
    out.extend_from_slice(&[0xaa; 32]);
    let ptr_before = out.as_ptr();
    let cap_before = out.capacity();

    let payload = vec![1u32, 2, 3, 4];
    beve::to_vec_into(&mut out, &payload).unwrap();
    assert_eq!(out, beve::to_vec(&payload).unwrap());
    assert_eq!(out.as_ptr(), ptr_before);
    assert_eq!(out.capacity(), cap_before);

    beve::to_vec_into(&mut out, &true).unwrap();
    assert_eq!(out, beve::to_vec(&true).unwrap());
    assert_eq!(out.as_ptr(), ptr_before);
    assert_eq!(out.capacity(), cap_before);
}

#[test]
fn serializer_buffer_can_be_reused() {
    let mut ser = beve::Serializer::with_capacity(128);

    ser.serialize_to_buffer(&123u32).unwrap();
    assert_eq!(ser.as_slice(), beve::to_vec(&123u32).unwrap());

    ser.serialize_to_buffer(&"hello").unwrap();
    assert_eq!(ser.as_slice(), beve::to_vec(&"hello").unwrap());
}

#[test]
fn to_vec_into_with_options_respects_enum_encoding() {
    #[derive(Serialize)]
    enum E {
        Unit,
    }

    let opts = beve::SerializerOptions {
        enum_encoding: beve::EnumEncoding::String,
    };

    let mut out = Vec::with_capacity(64);
    out.extend_from_slice(&[0xaa; 12]);
    let ptr_before = out.as_ptr();
    let cap_before = out.capacity();

    beve::to_vec_into_with_options(&mut out, &E::Unit, opts).unwrap();
    assert_eq!(out, beve::to_vec_with_options(&E::Unit, opts).unwrap());
    assert_ne!(out, beve::to_vec(&E::Unit).unwrap());
    assert_eq!(out.as_ptr(), ptr_before);
    assert_eq!(out.capacity(), cap_before);
}

#[test]
fn reusable_encode_buffers_clear_on_error() {
    struct AlwaysErr;
    impl Serialize for AlwaysErr {
        fn serialize<S: serde::Serializer>(&self, _serializer: S) -> Result<S::Ok, S::Error> {
            Err(serde::ser::Error::custom("boom"))
        }
    }

    let mut out = vec![1, 2, 3, 4];
    let ptr_before = out.as_ptr();
    let cap_before = out.capacity();
    let err = beve::to_vec_into(&mut out, &AlwaysErr).unwrap_err();
    assert!(err.to_string().contains("boom"));
    assert!(out.is_empty());
    assert_eq!(out.as_ptr(), ptr_before);
    assert_eq!(out.capacity(), cap_before);

    let mut ser = beve::Serializer::with_capacity(32);
    ser.serialize_to_buffer(&123u32).unwrap();
    assert!(!ser.as_slice().is_empty());
    let err = ser.serialize_to_buffer(&AlwaysErr).unwrap_err();
    assert!(err.to_string().contains("boom"));
    assert!(ser.as_slice().is_empty());
}

#[test]
fn json_bytes_to_beve_and_back() {
    let json_input = br#"
        {
            "name": "widget",
            "count": 5,
            "flags": [true, false, null],
            "nested": { "value": -12.5 }
        }
    "#;
    let expected: serde_json::Value = serde_json::from_slice(json_input).unwrap();

    let beve_bytes = beve::json_slice_to_beve(json_input).unwrap();
    let json_bytes = beve::beve_slice_to_json(&beve_bytes).unwrap();
    let roundtrip: serde_json::Value = serde_json::from_slice(&json_bytes).unwrap();

    assert_eq!(expected, roundtrip);
}

#[test]
fn json_string_to_beve_and_back() {
    let json_input = r#"{"message":"hello","scale":1.5,"items":[1,2,3,null]}"#;
    let expected: serde_json::Value = serde_json::from_str(json_input).unwrap();

    let beve_bytes = beve::json_str_to_beve(json_input).unwrap();
    let json_text = beve::beve_slice_to_json_string(&beve_bytes).unwrap();
    let roundtrip: serde_json::Value = serde_json::from_str(&json_text).unwrap();

    assert_eq!(expected, roundtrip);
}

#[test]
fn beve_typed_arrays_to_json() {
    let numbers = beve::fast::to_vec_typed_slice(&[1u32, 2, 3, 10]);
    let json = beve::beve_slice_to_json_string(&numbers).unwrap();
    assert_eq!(json, "[1,2,3,10]");

    let floats = beve::fast::to_vec_typed_slice(&[1.5f64, -2.25, 3.0]);
    let json = beve::beve_slice_to_json_string(&floats).unwrap();
    assert_eq!(json, "[1.5,-2.25,3]");

    let mut bools = Vec::new();
    beve::fast::write_bool_slice(&mut bools, &[true, false, true, true, false]);
    let json = beve::beve_slice_to_json_string(&bools).unwrap();
    assert_eq!(json, "[true,false,true,true,false]");

    let strings = beve::fast::to_vec_str_slice(&["a", "bb", "ccc"]);
    let json = beve::beve_slice_to_json_string(&strings).unwrap();
    assert_eq!(json, "[\"a\",\"bb\",\"ccc\"]");
}

#[test]
fn beve_matrix_extension_to_json() {
    let bytes = beve::fast::to_vec_matrix_f64(
        beve::fast::MatrixLayoutFast::Right,
        &[2, 3],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
    let json = beve::beve_slice_to_json_string(&bytes).unwrap();
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(
        value["layout"],
        serde_json::Value::String("layout_right".into())
    );
    assert_eq!(value["extents"], serde_json::json!([2, 3]));
    assert_eq!(value["value"], serde_json::json!([1, 2, 3, 4, 5, 6]));
}

#[test]
fn beve_complex_integer_extensions_to_json() {
    // Signed 16-bit complex value: (-2, 5)
    let signed = vec![0x1e, 0x28, 0xfe, 0xff, 0x05, 0x00];
    let json = beve::beve_slice_to_json_string(&signed).unwrap();
    assert_eq!(json, "[-2,5]");

    // Unsigned 8-bit complex array: [(1,2), (3,4)]
    let unsigned = vec![0x1e, 0x11, 0x08, 0x01, 0x02, 0x03, 0x04];
    let json = beve::beve_slice_to_json_string(&unsigned).unwrap();
    assert_eq!(json, "[[1,2],[3,4]]");
}

#[test]
fn validate_slice_accepts_valid_beve() {
    let bytes = beve::to_vec(&MyEnum::Struct { a: -7, b: 8 }).unwrap();
    beve::validate_slice(&bytes).unwrap();
}

#[test]
fn validate_slice_rejects_truncated_beve() {
    let mut bytes = beve::to_vec(&Point { x: 1.0, y: -2.0 }).unwrap();
    bytes.pop();
    assert!(beve::validate_slice(&bytes).is_err());
}

#[test]
fn validate_slice_rejects_trailing_data() {
    let mut bytes = beve::to_vec(&123u32).unwrap();
    bytes.push(0);
    assert!(beve::validate_slice(&bytes).is_err());
}

#[test]
fn validate_reader_accepts_valid_beve() {
    let bytes = beve::to_vec(&vec![1u32, 2, 3, 4]).unwrap();
    let cursor = std::io::Cursor::new(bytes);
    beve::validate_reader(cursor).unwrap();
}

#[test]
fn validate_slice_rejects_empty_input() {
    assert!(beve::validate_slice(&[]).is_err());
}

#[test]
fn validate_slice_rejects_concatenated_values() {
    let mut bytes = beve::to_vec(&1u8).unwrap();
    bytes.extend_from_slice(&beve::to_vec(&2u8).unwrap());
    assert!(beve::validate_slice(&bytes).is_err());
}

#[test]
fn validate_reader_rejects_truncated_beve() {
    let mut bytes = beve::to_vec(&Point { x: 1.0, y: -2.0 }).unwrap();
    bytes.pop();
    let cursor = std::io::Cursor::new(bytes);
    assert!(beve::validate_reader(cursor).is_err());
}

#[test]
fn validate_reader_rejects_trailing_data() {
    let mut bytes = beve::to_vec(&123u32).unwrap();
    bytes.push(0);
    let cursor = std::io::Cursor::new(bytes);
    assert!(beve::validate_reader(cursor).is_err());
}

#[test]
fn validate_slice_accepts_complex_extension() {
    let bytes = beve::to_vec_complex64_slice(&[
        beve::Complex { re: 1.0, im: 2.0 },
        beve::Complex { re: -3.0, im: 4.5 },
    ]);
    beve::validate_slice(&bytes).unwrap();
}

#[test]
fn validate_slice_accepts_matrix_extension() {
    let bytes = beve::fast::to_vec_matrix_f64(
        beve::fast::MatrixLayoutFast::Right,
        &[2, 3],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
    beve::validate_slice(&bytes).unwrap();
}
