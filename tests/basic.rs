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
        flags: [("a".into(), true), ("b".into(), false)]
            .into_iter()
            .collect(),
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
fn complex_integer_single_roundtrip() {
    macro_rules! test_complex_single {
        ($t:ty, $re:expr, $im:expr, $expected_header:expr) => {{
            let c = beve::Complex::<$t> { re: $re, im: $im };
            let bytes = beve::to_vec(&c).unwrap();
            assert_eq!(bytes[0], 0x1e); // EXT_COMPLEX
            assert_eq!(bytes[1], $expected_header);
            let back: beve::Complex<$t> = beve::from_slice(&bytes).unwrap();
            assert_eq!(back, c);
        }};
    }

    // Complex header: (byte_code << 5) | (class << 3) | is_array
    // class: 0=float, 1=signed, 2=unsigned
    // byte_code: 0=1B, 1=2B, 2=4B, 3=8B, 4=16B
    test_complex_single!(i8, -1, 2, 0x08); // bc=0, class=1
    test_complex_single!(i16, -300, 400, 0x28); // bc=1, class=1
    test_complex_single!(i32, -70000, 80000, 0x48); // bc=2, class=1
    test_complex_single!(i64, -1_000_000, 2_000_000, 0x68); // bc=3, class=1
    test_complex_single!(u8, 1, 2, 0x10); // bc=0, class=2
    test_complex_single!(u16, 1000, 2000, 0x30); // bc=1, class=2
    test_complex_single!(u32, 70000, 80000, 0x50); // bc=2, class=2
    test_complex_single!(u64, 1_000_000, 2_000_000, 0x70); // bc=3, class=2
}

#[test]
fn complex_integer_vec_roundtrip() {
    macro_rules! test_complex_vec {
        ($t:ty, $vals:expr, $expected_header:expr) => {{
            let values: Vec<beve::Complex<$t>> = $vals;
            let bytes = beve::to_vec(&values).unwrap();
            assert_eq!(bytes[0], 0x1e); // EXT_COMPLEX
            assert_eq!(bytes[1], $expected_header | 0x01); // is_array bit set
            let back: Vec<beve::Complex<$t>> = beve::from_slice(&bytes).unwrap();
            assert_eq!(back, values);
        }};
    }

    test_complex_vec!(
        i8,
        vec![
            beve::Complex { re: -1, im: 2 },
            beve::Complex { re: 3, im: -4 }
        ],
        0x08
    );
    test_complex_vec!(
        i32,
        vec![
            beve::Complex {
                re: -100_000,
                im: 200_000,
            },
            beve::Complex { re: 0, im: -1 }
        ],
        0x48
    );
    test_complex_vec!(
        u16,
        vec![
            beve::Complex { re: 1000, im: 2000 },
            beve::Complex { re: 0, im: 65535 }
        ],
        0x30
    );
    test_complex_vec!(
        u64,
        vec![beve::Complex {
            re: u64::MAX,
            im: 0,
        }],
        0x70
    );
}

#[test]
fn complex_generic_slice_encoding() {
    // Test the generic to_vec_complex_slice function
    let values = vec![
        beve::Complex { re: 1i32, im: -2 },
        beve::Complex { re: 3, im: 4 },
    ];
    let bytes = beve::to_vec_complex_slice(&values);
    let back: Vec<beve::Complex<i32>> = beve::from_slice(&bytes).unwrap();
    assert_eq!(back, values);

    let values = vec![
        beve::Complex { re: 10u8, im: 20 },
        beve::Complex { re: 30, im: 40 },
    ];
    let bytes = beve::to_vec_complex_slice(&values);
    let back: Vec<beve::Complex<u8>> = beve::from_slice(&bytes).unwrap();
    assert_eq!(back, values);
}

#[test]
fn complex_fields_in_struct_roundtrip() {
    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct Measurement {
        label: String,
        sample: beve::Complex<f32>,
        count: u32,
        buffer: Vec<beve::Complex<i32>>,
        scale: f64,
    }

    let m = Measurement {
        label: "test".to_string(),
        sample: beve::Complex { re: 1.5, im: -2.5 },
        count: 42,
        buffer: vec![
            beve::Complex { re: 10, im: -20 },
            beve::Complex { re: 30, im: 40 },
            beve::Complex { re: -1, im: 0 },
        ],
        scale: 3.125,
    };
    let bytes = beve::to_vec(&m).unwrap();
    let back: Measurement = beve::from_slice(&bytes).unwrap();
    assert_eq!(back, m);
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

// =============== Zero-copy deserialization tests ===============

#[test]
fn zero_copy_borrowed_str() {
    let bytes = beve::to_vec(&"hello world").unwrap();
    let borrowed: &str = beve::from_slice(&bytes).unwrap();
    assert_eq!(borrowed, "hello world");
}

#[test]
fn zero_copy_struct_with_borrowed_fields() {
    #[derive(Serialize)]
    struct InfoOwned {
        name: String,
        tag: String,
    }

    #[derive(Deserialize, Debug, PartialEq)]
    struct Info<'a> {
        name: &'a str,
        tag: &'a str,
    }

    let original = InfoOwned {
        name: "alice".to_string(),
        tag: "admin".to_string(),
    };
    let bytes = beve::to_vec(&original).unwrap();
    let info: Info = beve::from_slice(&bytes).unwrap();
    assert_eq!(info.name, "alice");
    assert_eq!(info.tag, "admin");
}

#[test]
fn zero_copy_cow_str() {
    use std::borrow::Cow;

    // Serde's generic Cow<T> impl always produces Cow::Owned via T::Owned::deserialize.
    // Zero-copy borrowing is available through &str fields directly.
    let bytes = beve::to_vec(&"hello").unwrap();
    let cow: Cow<str> = beve::from_slice(&bytes).unwrap();
    assert_eq!(&*cow, "hello");
}

#[test]
fn zero_copy_vec_of_borrowed_str() {
    let vs: Vec<String> = vec!["alpha".into(), "beta".into(), "gamma".into()];
    let bytes = beve::to_vec(&vs).unwrap();
    let borrowed: Vec<&str> = beve::from_slice(&bytes).unwrap();
    assert_eq!(borrowed, vec!["alpha", "beta", "gamma"]);
}

#[test]
fn zero_copy_mixed_struct() {
    #[derive(Serialize)]
    struct DataOwned {
        label: String,
        values: Vec<f64>,
        count: u32,
    }

    #[derive(Deserialize, Debug, PartialEq)]
    struct Data<'a> {
        label: &'a str,
        values: Vec<f64>,
        count: u32,
    }

    let original = DataOwned {
        label: "sensor_1".to_string(),
        values: vec![1.0, 2.5, 3.75],
        count: 42,
    };
    let bytes = beve::to_vec(&original).unwrap();
    let data: Data = beve::from_slice(&bytes).unwrap();
    assert_eq!(data.label, "sensor_1");
    assert_eq!(data.values, vec![1.0, 2.5, 3.75]);
    assert_eq!(data.count, 42);
}

#[test]
fn zero_copy_map_with_borrowed_keys() {
    use std::collections::BTreeMap;

    let mut m: BTreeMap<String, i32> = BTreeMap::new();
    m.insert("x".into(), 1);
    m.insert("y".into(), 2);
    let bytes = beve::to_vec(&m).unwrap();

    let borrowed: BTreeMap<&str, i32> = beve::from_slice(&bytes).unwrap();
    assert_eq!(borrowed["x"], 1);
    assert_eq!(borrowed["y"], 2);
}

#[test]
fn zero_copy_enum_string_encoding() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    enum Status<'a> {
        Active,
        Error(&'a str),
    }

    let opts = beve::SerializerOptions {
        enum_encoding: beve::EnumEncoding::String,
    };

    // Unit variant
    let bytes = beve::to_vec_with_options(&Status::Active, opts).unwrap();
    let back: Status = beve::from_slice(&bytes).unwrap();
    assert_eq!(back, Status::Active);

    // Newtype variant with borrowed str
    let bytes = beve::to_vec_with_options(&Status::Error("something broke"), opts).unwrap();
    let back: Status = beve::from_slice(&bytes).unwrap();
    assert_eq!(back, Status::Error("something broke"));
}

#[test]
fn zero_copy_borrowed_bytes_in_struct() {
    #[derive(Serialize)]
    struct DataOwned {
        payload: Vec<u8>,
        tag: String,
    }

    #[derive(Deserialize, Debug, PartialEq)]
    struct Data<'a> {
        #[serde(borrow)]
        payload: &'a [u8],
        tag: &'a str,
    }

    let original = DataOwned {
        payload: vec![0xDE, 0xAD, 0xBE, 0xEF],
        tag: "test".to_string(),
    };
    let bytes = beve::to_vec(&original).unwrap();
    let data: Data = beve::from_slice(&bytes).unwrap();
    assert_eq!(data.payload, &[0xDE, 0xAD, 0xBE, 0xEF]);
    assert_eq!(data.tag, "test");
}

#[test]
fn zero_copy_borrowed_bytes_standalone() {
    // Serialize a Vec<u8> and deserialize as borrowed &[u8]
    let data: Vec<u8> = vec![10, 20, 30];
    let bytes = beve::to_vec(&data).unwrap();

    // Vec<u8> serializes element-by-element via serialize_seq, but our
    // serializer detects homogeneous u8 and writes a typed u8 array.
    // deserialize_bytes detects this and returns borrowed bytes.
    let borrowed: &[u8] = beve::from_slice(&bytes).unwrap();
    assert_eq!(borrowed, &[10, 20, 30]);
}

#[test]
fn zero_copy_borrowed_bytes_empty() {
    let data: Vec<u8> = vec![];
    let bytes = beve::to_vec(&data).unwrap();
    let borrowed: &[u8] = beve::from_slice(&bytes).unwrap();
    assert!(borrowed.is_empty());
}

#[test]
fn zero_copy_borrowed_bytes_in_struct_via_bytes_nonempty() {
    #[derive(Serialize, Deserialize)]
    struct Wrap<'a> {
        #[serde(borrow)]
        data: &'a [u8],
    }

    let wrapper_bytes = beve::to_vec(&Wrap { data: &[1, 2, 3] }).unwrap();
    let w: Wrap = beve::from_slice(&wrapper_bytes).unwrap();
    assert_eq!(w.data, &[1, 2, 3]);
}

#[test]
fn zero_copy_borrowed_bytes_in_struct_via_bytes_empty() {
    #[derive(Serialize, Deserialize)]
    struct Wrap<'a> {
        #[serde(borrow)]
        data: &'a [u8],
    }

    let empty_bytes = beve::to_vec(&Wrap { data: &[] }).unwrap();
    let w: Wrap = beve::from_slice(&empty_bytes).unwrap();
    assert!(w.data.is_empty());
}
