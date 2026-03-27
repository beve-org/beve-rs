use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Cursor;

// ---------------------------------------------------------------------------
// Test types
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Point {
    x: f64,
    y: f64,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct ScalarFields {
    a: bool,
    b: u32,
    c: i64,
    d: f32,
    e: String,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Nested {
    name: String,
    point: Point,
    tags: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
enum Color {
    Red,
    Green,
    Blue,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
enum Shape {
    Circle(f64),
    Rect { w: f64, h: f64 },
    Triangle(f64, f64, f64),
    Point,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct WithOptional {
    a: Option<u32>,
    b: Option<u32>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Empty {}

// ---------------------------------------------------------------------------
// Basic scalar struct — byte-for-byte match with to_vec
// ---------------------------------------------------------------------------

#[test]
fn streaming_scalar_struct_matches_to_vec() {
    let p = Point { x: 1.0, y: -2.5 };
    let expected = beve::to_vec(&p).unwrap();

    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &p).unwrap();

    assert_eq!(buf, expected);
}

#[test]
fn streaming_scalar_fields_match_to_vec() {
    let v = ScalarFields {
        a: true,
        b: 42,
        c: -100,
        d: 1.5,
        e: "hello".into(),
    };
    let expected = beve::to_vec(&v).unwrap();

    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &v).unwrap();

    assert_eq!(buf, expected);
}

// ---------------------------------------------------------------------------
// Round-trip through streaming → from_slice
// ---------------------------------------------------------------------------

#[test]
fn streaming_round_trip_point() {
    let p = Point { x: 1.0, y: -2.5 };
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &p).unwrap();
    let p2: Point = beve::from_slice(&buf).unwrap();
    assert_eq!(p, p2);
}

#[test]
fn streaming_round_trip_nested() {
    let n = Nested {
        name: "test".into(),
        point: Point { x: 0.0, y: 1.0 },
        tags: vec!["a".into(), "b".into(), "c".into()],
    };
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &n).unwrap();
    let n2: Nested = beve::from_slice(&buf).unwrap();
    assert_eq!(n, n2);
}

#[test]
fn streaming_round_trip_hashmap_string_keys() {
    let mut m = HashMap::new();
    m.insert("x".to_string(), 1.0f64);
    m.insert("y".to_string(), 2.0);

    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &m).unwrap();
    let m2: HashMap<String, f64> = beve::from_slice(&buf).unwrap();
    assert_eq!(m, m2);
}

#[test]
fn streaming_round_trip_vec_string() {
    let v = vec!["hello".to_string(), "world".to_string()];
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &v).unwrap();
    let v2: Vec<String> = beve::from_slice(&buf).unwrap();
    assert_eq!(v, v2);
}

#[test]
fn streaming_round_trip_vec_f64() {
    let v = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &v).unwrap();
    let v2: Vec<f64> = beve::from_slice(&buf).unwrap();
    assert_eq!(v, v2);
}

#[test]
fn streaming_round_trip_vec_i32() {
    let v = vec![10i32, -20, 30, -40];
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &v).unwrap();
    let v2: Vec<i32> = beve::from_slice(&buf).unwrap();
    assert_eq!(v, v2);
}

#[test]
fn streaming_round_trip_vec_bool() {
    let v = vec![true, false, true, true, false];
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &v).unwrap();
    let v2: Vec<bool> = beve::from_slice(&buf).unwrap();
    assert_eq!(v, v2);
}

// ---------------------------------------------------------------------------
// Enum support
// ---------------------------------------------------------------------------

#[test]
fn streaming_round_trip_unit_variant() {
    let c = Color::Green;
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &c).unwrap();
    let c2: Color = beve::from_slice(&buf).unwrap();
    assert_eq!(c, c2);

    // Byte-for-byte match for standalone enum
    assert_eq!(buf, beve::to_vec(&c).unwrap());
}

#[test]
fn streaming_round_trip_newtype_variant() {
    let s = Shape::Circle(2.5);
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &s).unwrap();
    let s2: Shape = beve::from_slice(&buf).unwrap();
    assert_eq!(s, s2);
}

#[test]
fn streaming_round_trip_struct_variant() {
    let s = Shape::Rect { w: 3.0, h: 4.0 };
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &s).unwrap();
    let s2: Shape = beve::from_slice(&buf).unwrap();
    assert_eq!(s, s2);
}

#[test]
fn streaming_round_trip_tuple_variant() {
    let s = Shape::Triangle(3.0, 4.0, 5.0);
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &s).unwrap();
    let s2: Shape = beve::from_slice(&buf).unwrap();
    assert_eq!(s, s2);
}

#[test]
fn streaming_round_trip_unit_variant_point() {
    let s = Shape::Point;
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &s).unwrap();
    let s2: Shape = beve::from_slice(&buf).unwrap();
    assert_eq!(s, s2);
}

// ---------------------------------------------------------------------------
// Optional fields
// ---------------------------------------------------------------------------

#[test]
fn streaming_round_trip_optional() {
    let v = WithOptional {
        a: Some(42),
        b: None,
    };
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &v).unwrap();
    let v2: WithOptional = beve::from_slice(&buf).unwrap();
    assert_eq!(v, v2);

    // Byte-for-byte match — Options serialize the same way
    assert_eq!(buf, beve::to_vec(&v).unwrap());
}

// ---------------------------------------------------------------------------
// Empty containers
// ---------------------------------------------------------------------------

#[test]
fn streaming_round_trip_empty_struct() {
    let e = Empty {};
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &e).unwrap();
    let e2: Empty = beve::from_slice(&buf).unwrap();
    assert_eq!(e, e2);
    assert_eq!(buf, beve::to_vec(&e).unwrap());
}

#[test]
fn streaming_round_trip_empty_vec() {
    let v: Vec<u32> = vec![];
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &v).unwrap();
    let v2: Vec<u32> = beve::from_slice(&buf).unwrap();
    assert_eq!(v, v2);
}

#[test]
fn streaming_round_trip_empty_map() {
    let m: HashMap<String, u32> = HashMap::new();
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &m).unwrap();
    let m2: HashMap<String, u32> = beve::from_slice(&buf).unwrap();
    assert_eq!(m, m2);
}

// ---------------------------------------------------------------------------
// Scalars
// ---------------------------------------------------------------------------

#[test]
fn streaming_scalar_bool() {
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &true).unwrap();
    let v: bool = beve::from_slice(&buf).unwrap();
    assert!(v);
    assert_eq!(buf, beve::to_vec(&true).unwrap());
}

#[test]
fn streaming_scalar_u64() {
    let val = 12345u64;
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &val).unwrap();
    let v: u64 = beve::from_slice(&buf).unwrap();
    assert_eq!(v, val);
    assert_eq!(buf, beve::to_vec(&val).unwrap());
}

#[test]
fn streaming_scalar_string() {
    let val = "hello world".to_string();
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &val).unwrap();
    let v: String = beve::from_slice(&buf).unwrap();
    assert_eq!(v, val);
    assert_eq!(buf, beve::to_vec(&val).unwrap());
}

#[test]
fn streaming_scalar_none() {
    let val: Option<u32> = None;
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &val).unwrap();
    let v: Option<u32> = beve::from_slice(&buf).unwrap();
    assert_eq!(v, val);
    assert_eq!(buf, beve::to_vec(&val).unwrap());
}

#[test]
fn streaming_scalar_unit() {
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &()).unwrap();
    assert_eq!(buf, beve::to_vec(&()).unwrap());
}

// ---------------------------------------------------------------------------
// Unknown-length error
// ---------------------------------------------------------------------------

#[test]
fn streaming_rejects_unknown_length_seq() {
    // Create a type with unknown-length serialization
    struct UnknownLen;
    impl Serialize for UnknownLen {
        fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
            use serde::ser::SerializeSeq;
            let mut seq = s.serialize_seq(None)?;
            seq.serialize_element(&1u32)?;
            seq.end()
        }
    }

    let mut buf = Vec::new();
    let result = beve::to_writer_streaming(&mut buf, &UnknownLen);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("known-length"), "got: {err_msg}");
}

#[test]
fn streaming_rejects_unknown_length_map() {
    struct UnknownLenMap;
    impl Serialize for UnknownLenMap {
        fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
            use serde::ser::SerializeMap;
            let mut map = s.serialize_map(None)?;
            map.serialize_entry("k", &1u32)?;
            map.end()
        }
    }

    let mut buf = Vec::new();
    let result = beve::to_writer_streaming(&mut buf, &UnknownLenMap);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("known-length"), "got: {err_msg}");
}

// ---------------------------------------------------------------------------
// Write to Cursor (simulated file I/O)
// ---------------------------------------------------------------------------

#[test]
fn streaming_write_to_cursor() {
    let p = Point { x: 1.0, y: -2.5 };
    let cursor = Cursor::new(Vec::new());
    beve::to_writer_streaming(cursor, &p).unwrap();
}

// ---------------------------------------------------------------------------
// Custom options (enum encoding)
// ---------------------------------------------------------------------------

#[test]
fn streaming_string_enum_encoding() {
    let c = Color::Blue;
    let mut buf = Vec::new();
    let opts = beve::SerializerOptions {
        enum_encoding: beve::EnumEncoding::String,
    };
    beve::to_writer_streaming_with_options(&mut buf, &c, opts).unwrap();

    // Should match to_vec_with_options
    let expected = beve::to_vec_with_options(&c, opts).unwrap();
    assert_eq!(buf, expected);
}

#[test]
fn streaming_string_enum_round_trip() {
    let opts = beve::SerializerOptions {
        enum_encoding: beve::EnumEncoding::String,
    };

    // Unit variant
    let c = Color::Green;
    let mut buf = Vec::new();
    beve::to_writer_streaming_with_options(&mut buf, &c, opts).unwrap();
    let c2: Color = beve::from_reader_streaming(std::io::Cursor::new(&buf)).unwrap();
    assert_eq!(c, c2);

    // Tuple variant
    let s = Shape::Circle(2.5);
    buf.clear();
    beve::to_writer_streaming_with_options(&mut buf, &s, opts).unwrap();
    let s2: Shape = beve::from_reader_streaming(std::io::Cursor::new(&buf)).unwrap();
    assert_eq!(s, s2);

    // Struct variant
    let s = Shape::Rect { w: 3.0, h: 4.0 };
    buf.clear();
    beve::to_writer_streaming_with_options(&mut buf, &s, opts).unwrap();
    let s2: Shape = beve::from_reader_streaming(std::io::Cursor::new(&buf)).unwrap();
    assert_eq!(s, s2);

    // Newtype-like tuple variant
    let s = Shape::Triangle(3.0, 4.0, 5.0);
    buf.clear();
    beve::to_writer_streaming_with_options(&mut buf, &s, opts).unwrap();
    let s2: Shape = beve::from_reader_streaming(std::io::Cursor::new(&buf)).unwrap();
    assert_eq!(s, s2);

    // Unit variant on Shape
    let s = Shape::Point;
    buf.clear();
    beve::to_writer_streaming_with_options(&mut buf, &s, opts).unwrap();
    let s2: Shape = beve::from_reader_streaming(std::io::Cursor::new(&buf)).unwrap();
    assert_eq!(s, s2);
}

// ---------------------------------------------------------------------------
// Integer-keyed maps
// ---------------------------------------------------------------------------

#[test]
fn streaming_round_trip_u32_key_map() {
    let mut m = HashMap::new();
    m.insert(1u32, "one".to_string());
    m.insert(2u32, "two".to_string());

    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &m).unwrap();
    let m2: HashMap<u32, String> = beve::from_slice(&buf).unwrap();
    assert_eq!(m, m2);
}

#[test]
fn streaming_round_trip_i64_key_map() {
    let mut m = HashMap::new();
    m.insert(-1i64, 100u32);
    m.insert(42i64, 200u32);

    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &m).unwrap();
    let m2: HashMap<i64, u32> = beve::from_slice(&buf).unwrap();
    assert_eq!(m, m2);
}

// ---------------------------------------------------------------------------
// Vec of structs (each element is individually serialized)
// ---------------------------------------------------------------------------

#[test]
fn streaming_round_trip_vec_of_structs() {
    let v = vec![
        Point { x: 1.0, y: 2.0 },
        Point { x: 3.0, y: 4.0 },
        Point { x: 5.0, y: 6.0 },
    ];
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &v).unwrap();
    let v2: Vec<Point> = beve::from_slice(&buf).unwrap();
    assert_eq!(v, v2);
}

// ---------------------------------------------------------------------------
// Nested maps/vecs
// ---------------------------------------------------------------------------

#[test]
fn streaming_round_trip_map_of_vecs() {
    let mut m = HashMap::new();
    m.insert("a".to_string(), vec![1.0f64, 2.0, 3.0]);
    m.insert("b".to_string(), vec![4.0, 5.0]);

    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &m).unwrap();
    let m2: HashMap<String, Vec<f64>> = beve::from_slice(&buf).unwrap();
    assert_eq!(m, m2);
}

// ---------------------------------------------------------------------------
// Tuple types
// ---------------------------------------------------------------------------

#[test]
fn streaming_round_trip_tuple() {
    let t = (42u32, "hello".to_string(), true);
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &t).unwrap();
    let t2: (u32, String, bool) = beve::from_slice(&buf).unwrap();
    assert_eq!(t, t2);
}

// ---------------------------------------------------------------------------
// Vec<u8> round-trip
// ---------------------------------------------------------------------------

#[test]
fn streaming_round_trip_vec_u8() {
    let data = vec![1u8, 2, 3, 4, 5];
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &data).unwrap();
    let data2: Vec<u8> = beve::from_slice(&buf).unwrap();
    assert_eq!(data, data2);
}

// ---------------------------------------------------------------------------
// Validation: streaming output is valid BEVE
// ---------------------------------------------------------------------------

#[test]
fn streaming_output_validates() {
    let n = Nested {
        name: "test".into(),
        point: Point { x: 1.0, y: 2.0 },
        tags: vec!["a".into(), "b".into()],
    };
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &n).unwrap();
    beve::validate_slice(&buf).unwrap();
}

#[test]
fn streaming_vec_output_validates() {
    let v = vec![1u32, 2, 3, 4, 5];
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &v).unwrap();
    beve::validate_slice(&buf).unwrap();
}

// ---------------------------------------------------------------------------
// Large data (ensure no panics, validates)
// ---------------------------------------------------------------------------

#[test]
fn streaming_large_vec() {
    let v: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &v).unwrap();
    beve::validate_slice(&buf).unwrap();
    let v2: Vec<f64> = beve::from_slice(&buf).unwrap();
    assert_eq!(v, v2);
}

#[test]
fn streaming_large_struct_vec() {
    let v: Vec<Point> = (0..1_000)
        .map(|i| Point {
            x: i as f64,
            y: -(i as f64),
        })
        .collect();
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &v).unwrap();
    beve::validate_slice(&buf).unwrap();
    let v2: Vec<Point> = beve::from_slice(&buf).unwrap();
    assert_eq!(v, v2);
}

// ---------------------------------------------------------------------------
// Writer error propagation
// ---------------------------------------------------------------------------

#[test]
fn streaming_writer_error_propagation() {
    struct FailWriter;
    impl std::io::Write for FailWriter {
        fn write(&mut self, _buf: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("disk full"))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    let p = Point { x: 1.0, y: 2.0 };
    let result = beve::to_writer_streaming(FailWriter, &p);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("disk full"), "got: {msg}");
}

// ---------------------------------------------------------------------------
// StreamingSerializer direct usage
// ---------------------------------------------------------------------------

#[test]
fn streaming_serializer_direct_usage() {
    use beve::StreamingSerializer;
    use serde::Serialize;

    let mut buf = Vec::new();
    let mut ser = StreamingSerializer::new(&mut buf);
    let p = Point { x: 1.0, y: 2.0 };
    p.serialize(&mut ser).unwrap();

    let p2: Point = beve::from_slice(&buf).unwrap();
    assert_eq!(p, p2);
}

// ---------------------------------------------------------------------------
// Typed array: byte-for-byte match with to_vec for homogeneous sequences
// ---------------------------------------------------------------------------

#[test]
fn streaming_typed_array_vec_f64_matches_to_vec() {
    let v = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
    let expected = beve::to_vec(&v).unwrap();
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &v).unwrap();
    assert_eq!(buf, expected);
}

#[test]
fn streaming_typed_array_vec_f32_matches_to_vec() {
    let v = vec![1.0f32, 2.0, 3.0];
    let expected = beve::to_vec(&v).unwrap();
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &v).unwrap();
    assert_eq!(buf, expected);
}

#[test]
fn streaming_typed_array_vec_u32_matches_to_vec() {
    let v = vec![10u32, 20, 30, 40, 50];
    let expected = beve::to_vec(&v).unwrap();
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &v).unwrap();
    assert_eq!(buf, expected);
}

#[test]
fn streaming_typed_array_vec_u8_matches_to_vec() {
    let v = vec![1u8, 2, 3, 4, 5];
    let expected = beve::to_vec(&v).unwrap();
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &v).unwrap();
    assert_eq!(buf, expected);
}

#[test]
fn streaming_typed_array_vec_i64_matches_to_vec() {
    let v = vec![-100i64, 0, 100, 200];
    let expected = beve::to_vec(&v).unwrap();
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &v).unwrap();
    assert_eq!(buf, expected);
}

#[test]
fn streaming_typed_array_vec_i8_matches_to_vec() {
    let v = vec![-1i8, 0, 1, 127, -128];
    let expected = beve::to_vec(&v).unwrap();
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &v).unwrap();
    assert_eq!(buf, expected);
}

#[test]
fn streaming_typed_array_vec_bool_matches_to_vec() {
    let v = vec![true, false, true, true, false, false, true, false, true];
    let expected = beve::to_vec(&v).unwrap();
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &v).unwrap();
    assert_eq!(buf, expected);
}

#[test]
fn streaming_typed_array_vec_string_matches_to_vec() {
    let v = vec!["hello".to_string(), "world".to_string(), "foo".to_string()];
    let expected = beve::to_vec(&v).unwrap();
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &v).unwrap();
    assert_eq!(buf, expected);
}

#[test]
fn streaming_typed_array_empty_vec_matches_to_vec() {
    let v: Vec<f64> = vec![];
    let expected = beve::to_vec(&v).unwrap();
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &v).unwrap();
    assert_eq!(buf, expected);
}

// ---------------------------------------------------------------------------
// Typed arrays inside structs match to_vec
// ---------------------------------------------------------------------------

#[test]
fn streaming_struct_with_typed_arrays_matches_to_vec() {
    let n = Nested {
        name: "test".into(),
        point: Point { x: 1.0, y: 2.0 },
        tags: vec!["a".into(), "b".into(), "c".into()],
    };
    let expected = beve::to_vec(&n).unwrap();
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &n).unwrap();
    assert_eq!(buf, expected);
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct SciData {
    label: String,
    values: Vec<f64>,
    flags: Vec<bool>,
    ids: Vec<u32>,
}

#[test]
fn streaming_struct_multi_typed_arrays_matches_to_vec() {
    let d = SciData {
        label: "experiment".into(),
        values: vec![1.5, 2.5, 3.5],
        flags: vec![true, false, true],
        ids: vec![100, 200, 300],
    };
    let expected = beve::to_vec(&d).unwrap();
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &d).unwrap();
    assert_eq!(buf, expected);
}

// ---------------------------------------------------------------------------
// Vec of structs uses generic array (not typed)
// ---------------------------------------------------------------------------

#[test]
fn streaming_vec_of_structs_is_generic() {
    let v = vec![Point { x: 1.0, y: 2.0 }, Point { x: 3.0, y: 4.0 }];
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &v).unwrap();
    // Should round-trip correctly
    let v2: Vec<Point> = beve::from_slice(&buf).unwrap();
    assert_eq!(v, v2);
    // Validates
    beve::validate_slice(&buf).unwrap();
}

// ---------------------------------------------------------------------------
// Nested Vec<Vec<T>> — outer generic, inner typed
// ---------------------------------------------------------------------------

#[test]
fn streaming_nested_vec_vec_f64() {
    let v = vec![vec![1.0f64, 2.0], vec![3.0, 4.0, 5.0]];
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &v).unwrap();
    let v2: Vec<Vec<f64>> = beve::from_slice(&buf).unwrap();
    assert_eq!(v, v2);
    // Outer is generic (elements are arrays), inner should be typed
    let expected = beve::to_vec(&v).unwrap();
    assert_eq!(buf, expected);
}

// ===========================================================================
// Streaming DESERIALIZATION tests
// ===========================================================================

/// Helper: serialize with to_vec, then deserialize via from_reader_streaming.
fn streaming_de_round_trip<T>(value: &T) -> T
where
    T: Serialize + serde::de::DeserializeOwned + std::fmt::Debug + PartialEq,
{
    let bytes = beve::to_vec(value).unwrap();
    beve::from_reader_streaming(Cursor::new(bytes)).unwrap()
}

// ---------------------------------------------------------------------------
// Scalars
// ---------------------------------------------------------------------------

#[test]
fn streaming_de_bool() {
    assert!(streaming_de_round_trip(&true));
    assert!(!streaming_de_round_trip(&false));
}

#[test]
fn streaming_de_u32() {
    assert_eq!(streaming_de_round_trip(&42u32), 42u32);
}

#[test]
fn streaming_de_i64() {
    assert_eq!(streaming_de_round_trip(&-100i64), -100i64);
}

#[test]
fn streaming_de_f64() {
    assert_eq!(streaming_de_round_trip(&1.5f64), 1.5f64);
}

#[test]
fn streaming_de_string() {
    assert_eq!(
        streaming_de_round_trip(&"hello world".to_string()),
        "hello world"
    );
}

#[test]
fn streaming_de_none() {
    assert_eq!(streaming_de_round_trip(&None::<u32>), None::<u32>);
}

#[test]
fn streaming_de_some() {
    assert_eq!(streaming_de_round_trip(&Some(42u32)), Some(42u32));
}

#[test]
fn streaming_de_unit() {
    streaming_de_round_trip(&());
}

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

#[test]
fn streaming_de_point() {
    let p = Point { x: 1.0, y: -2.5 };
    assert_eq!(streaming_de_round_trip(&p), p);
}

#[test]
fn streaming_de_scalar_fields() {
    let v = ScalarFields {
        a: true,
        b: 42,
        c: -100,
        d: 1.5,
        e: "hello".into(),
    };
    assert_eq!(streaming_de_round_trip(&v), v);
}

#[test]
fn streaming_de_nested() {
    let n = Nested {
        name: "test".into(),
        point: Point { x: 0.0, y: 1.0 },
        tags: vec!["a".into(), "b".into(), "c".into()],
    };
    assert_eq!(streaming_de_round_trip(&n), n);
}

// ---------------------------------------------------------------------------
// Typed arrays
// ---------------------------------------------------------------------------

#[test]
fn streaming_de_vec_f64() {
    let v = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
    assert_eq!(streaming_de_round_trip(&v), v);
}

#[test]
fn streaming_de_vec_u32() {
    let v = vec![10u32, 20, 30, 40];
    assert_eq!(streaming_de_round_trip(&v), v);
}

#[test]
fn streaming_de_vec_i8() {
    let v = vec![-1i8, 0, 1, 127, -128];
    assert_eq!(streaming_de_round_trip(&v), v);
}

#[test]
fn streaming_de_vec_bool() {
    let v = vec![true, false, true, true, false, false, true, false, true];
    assert_eq!(streaming_de_round_trip(&v), v);
}

#[test]
fn streaming_de_vec_string() {
    let v = vec!["hello".to_string(), "world".to_string()];
    assert_eq!(streaming_de_round_trip(&v), v);
}

#[test]
fn streaming_de_vec_u8() {
    let v = vec![1u8, 2, 3, 4, 5];
    assert_eq!(streaming_de_round_trip(&v), v);
}

#[test]
fn streaming_de_empty_vec() {
    let v: Vec<f64> = vec![];
    assert_eq!(streaming_de_round_trip(&v), v);
}

// ---------------------------------------------------------------------------
// Generic arrays (Vec of structs)
// ---------------------------------------------------------------------------

#[test]
fn streaming_de_vec_of_structs() {
    let v = vec![Point { x: 1.0, y: 2.0 }, Point { x: 3.0, y: 4.0 }];
    assert_eq!(streaming_de_round_trip(&v), v);
}

// ---------------------------------------------------------------------------
// Maps
// ---------------------------------------------------------------------------

#[test]
fn streaming_de_hashmap_string_keys() {
    let mut m = HashMap::new();
    m.insert("x".to_string(), 1.0f64);
    m.insert("y".to_string(), 2.0);
    assert_eq!(streaming_de_round_trip(&m), m);
}

#[test]
fn streaming_de_u32_key_map() {
    let mut m = HashMap::new();
    m.insert(1u32, "one".to_string());
    m.insert(2u32, "two".to_string());
    assert_eq!(streaming_de_round_trip(&m), m);
}

#[test]
fn streaming_de_i64_key_map() {
    let mut m = HashMap::new();
    m.insert(-1i64, 100u32);
    m.insert(42i64, 200u32);
    assert_eq!(streaming_de_round_trip(&m), m);
}

#[test]
fn streaming_de_empty_map() {
    let m: HashMap<String, u32> = HashMap::new();
    assert_eq!(streaming_de_round_trip(&m), m);
}

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

#[test]
fn streaming_de_unit_variant() {
    assert_eq!(streaming_de_round_trip(&Color::Green), Color::Green);
}

#[test]
fn streaming_de_newtype_variant() {
    assert_eq!(
        streaming_de_round_trip(&Shape::Circle(2.5)),
        Shape::Circle(2.5)
    );
}

#[test]
fn streaming_de_struct_variant() {
    let s = Shape::Rect { w: 3.0, h: 4.0 };
    assert_eq!(streaming_de_round_trip(&s), s);
}

#[test]
fn streaming_de_tuple_variant() {
    let s = Shape::Triangle(3.0, 4.0, 5.0);
    assert_eq!(streaming_de_round_trip(&s), s);
}

// ---------------------------------------------------------------------------
// Tuples
// ---------------------------------------------------------------------------

#[test]
fn streaming_de_tuple() {
    let t = (42u32, "hello".to_string(), true);
    assert_eq!(streaming_de_round_trip(&t), t);
}

// ---------------------------------------------------------------------------
// Optional fields in struct
// ---------------------------------------------------------------------------

#[test]
fn streaming_de_optional() {
    let v = WithOptional {
        a: Some(42),
        b: None,
    };
    assert_eq!(streaming_de_round_trip(&v), v);
}

// ---------------------------------------------------------------------------
// Large data
// ---------------------------------------------------------------------------

#[test]
fn streaming_de_large_vec_f64() {
    let v: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
    assert_eq!(streaming_de_round_trip(&v), v);
}

#[test]
fn streaming_de_large_struct() {
    let d = SciData {
        label: "experiment".into(),
        values: (0..10_000).map(|i| i as f64 * 0.1).collect(),
        flags: (0..10_000).map(|i| i % 3 == 0).collect(),
        ids: (0..10_000).map(|i| i as u32).collect(),
    };
    assert_eq!(streaming_de_round_trip(&d), d);
}

// ---------------------------------------------------------------------------
// Nested containers
// ---------------------------------------------------------------------------

#[test]
fn streaming_de_map_of_vecs() {
    let mut m = HashMap::new();
    m.insert("a".to_string(), vec![1.0f64, 2.0, 3.0]);
    m.insert("b".to_string(), vec![4.0, 5.0]);
    assert_eq!(streaming_de_round_trip(&m), m);
}

#[test]
fn streaming_de_nested_vec_vec() {
    let v = vec![vec![1.0f64, 2.0], vec![3.0, 4.0, 5.0]];
    assert_eq!(streaming_de_round_trip(&v), v);
}

// ---------------------------------------------------------------------------
// Full streaming round-trip: to_writer_streaming → from_reader_streaming
// ---------------------------------------------------------------------------

#[test]
fn streaming_full_round_trip() {
    let d = SciData {
        label: "full_streaming".into(),
        values: vec![1.0, 2.0, 3.0],
        flags: vec![true, false],
        ids: vec![100, 200],
    };
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, &d).unwrap();
    let d2: SciData = beve::from_reader_streaming(Cursor::new(buf)).unwrap();
    assert_eq!(d, d2);
}

// ---------------------------------------------------------------------------
// EOF handling
// ---------------------------------------------------------------------------

#[test]
fn streaming_de_truncated_input() {
    let bytes = beve::to_vec(&42u32).unwrap();
    // Truncate to just the header byte
    let result: beve::Result<u32> = beve::from_reader_streaming(Cursor::new(&bytes[..1]));
    assert!(result.is_err());
}

#[test]
fn streaming_de_empty_input() {
    let result: beve::Result<u32> = beve::from_reader_streaming(Cursor::new(&[]));
    assert!(result.is_err());
}
