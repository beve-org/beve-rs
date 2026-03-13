#![deny(warnings)]

use beve::{Key, Number, Object, Value, from_value, from_value_ref};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// ============ Roundtrip Tests ============

#[test]
fn roundtrip_null() {
    let v = Value::Null;
    let bytes = beve::to_vec(&v).unwrap();
    let back: Value = beve::from_slice(&bytes).unwrap();
    assert!(back.is_null());
}

#[test]
fn roundtrip_bool() {
    for b in [true, false] {
        let v = Value::Bool(b);
        let bytes = beve::to_vec(&v).unwrap();
        let back: Value = beve::from_slice(&bytes).unwrap();
        assert_eq!(back.as_bool(), Some(b));
    }
}

#[test]
fn roundtrip_signed_integers() {
    let values: Vec<i64> = vec![
        0,
        1,
        -1,
        i8::MIN as i64,
        i8::MAX as i64,
        i16::MIN as i64,
        i16::MAX as i64,
        i32::MIN as i64,
        i32::MAX as i64,
        i64::MIN,
        i64::MAX,
    ];

    for n in values {
        let v = Value::Number(Number::I64(n));
        let bytes = beve::to_vec(&v).unwrap();
        let back: Value = beve::from_slice(&bytes).unwrap();
        // Note: small positive values may come back as unsigned
        match back.as_number() {
            Some(Number::I64(m)) => assert_eq!(*m, n),
            Some(Number::U64(m)) => assert_eq!(*m as i64, n),
            _ => panic!("Expected number, got {:?}", back),
        }
    }
}

#[test]
fn roundtrip_unsigned_integers() {
    let values: Vec<u64> = vec![
        0,
        1,
        u8::MAX as u64,
        u16::MAX as u64,
        u32::MAX as u64,
        u64::MAX,
    ];

    for n in values {
        let v = Value::Number(Number::U64(n));
        let bytes = beve::to_vec(&v).unwrap();
        let back: Value = beve::from_slice(&bytes).unwrap();
        assert_eq!(back.as_number().unwrap().as_u64(), Some(n));
    }
}

#[test]
fn roundtrip_floats() {
    let values: Vec<f64> = vec![
        0.0,
        1.0,
        -1.0,
        std::f64::consts::PI,
        std::f64::consts::E,
        f64::MIN,
        f64::MAX,
    ];

    for n in values {
        let v = Value::Number(Number::F64(n));
        let bytes = beve::to_vec(&v).unwrap();
        let back: Value = beve::from_slice(&bytes).unwrap();
        assert_eq!(back.as_f64(), Some(n));
    }
}

#[test]
fn roundtrip_string() {
    let values = vec!["", "hello", "world", "hello world", "unicode: 日本語 🎉"];

    for s in values {
        let v = Value::String(s.to_string());
        let bytes = beve::to_vec(&v).unwrap();
        let back: Value = beve::from_slice(&bytes).unwrap();
        assert_eq!(back.as_str(), Some(s));
    }
}

#[test]
fn roundtrip_array() {
    let v = Value::Array(vec![
        Value::Number(Number::I64(1)),
        Value::Number(Number::I64(2)),
        Value::Number(Number::I64(3)),
    ]);
    let bytes = beve::to_vec(&v).unwrap();
    let back: Value = beve::from_slice(&bytes).unwrap();
    let arr = back.as_array().unwrap();
    assert_eq!(arr.len(), 3);
}

#[test]
fn roundtrip_mixed_array() {
    let v = Value::Array(vec![
        Value::Null,
        Value::Bool(true),
        Value::Number(Number::I64(42)),
        Value::String("hello".to_string()),
    ]);
    let bytes = beve::to_vec(&v).unwrap();
    let back: Value = beve::from_slice(&bytes).unwrap();
    let arr = back.as_array().unwrap();
    assert_eq!(arr.len(), 4);
    assert!(arr[0].is_null());
    assert_eq!(arr[1].as_bool(), Some(true));
    assert_eq!(arr[3].as_str(), Some("hello"));
}

#[test]
fn roundtrip_object_string_keys() {
    let mut obj = Object::new();
    obj.insert(
        Key::String("name".to_string()),
        Value::String("Alice".to_string()),
    );
    obj.insert(
        Key::String("age".to_string()),
        Value::Number(Number::U64(30)),
    );

    let v = Value::Object(obj);
    let bytes = beve::to_vec(&v).unwrap();
    let back: Value = beve::from_slice(&bytes).unwrap();

    let obj = back.as_object().unwrap();
    assert_eq!(obj.len(), 2);
    assert_eq!(
        obj.get(&Key::String("name".to_string()))
            .and_then(|v| v.as_str()),
        Some("Alice")
    );
}

#[test]
fn roundtrip_nested_structure() {
    let mut inner = Object::new();
    inner.insert(
        Key::String("x".to_string()),
        Value::Number(Number::F64(1.5)),
    );
    inner.insert(
        Key::String("y".to_string()),
        Value::Number(Number::F64(2.5)),
    );

    let mut outer = Object::new();
    outer.insert(Key::String("point".to_string()), Value::Object(inner));
    outer.insert(
        Key::String("values".to_string()),
        Value::Array(vec![
            Value::Number(Number::I64(1)),
            Value::Number(Number::I64(2)),
        ]),
    );

    let v = Value::Object(outer);
    let bytes = beve::to_vec(&v).unwrap();
    let back: Value = beve::from_slice(&bytes).unwrap();

    assert!(back.is_object());
    let point = back.get_key("point").unwrap();
    assert!(point.is_object());
}

// ============ Bidirectional Struct <-> Value Roundtrips ============

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Point {
    x: f64,
    y: f64,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Person {
    name: String,
    age: u32,
    active: bool,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Container {
    label: String,
    values: Vec<i32>,
    metadata: Option<String>,
}

#[test]
fn struct_to_value_to_struct() {
    // Struct -> BEVE -> Value -> BEVE -> Struct
    let original = Point { x: 1.5, y: -2.5 };

    // Serialize struct to BEVE
    let bytes1 = beve::to_vec(&original).unwrap();

    // Deserialize to Value
    let value: Value = beve::from_slice(&bytes1).unwrap();
    assert!(value.is_object());

    // Serialize Value back to BEVE
    let bytes2 = beve::to_vec(&value).unwrap();

    // Deserialize back to struct
    let recovered: Point = beve::from_slice(&bytes2).unwrap();
    assert_eq!(original, recovered);
}

#[test]
fn value_to_struct() {
    // Construct a Value that matches Person struct
    let mut obj = Object::new();
    obj.insert(
        Key::String("name".to_string()),
        Value::String("Alice".to_string()),
    );
    obj.insert(
        Key::String("age".to_string()),
        Value::Number(Number::U64(30)),
    );
    obj.insert(Key::String("active".to_string()), Value::Bool(true));
    let value = Value::Object(obj);

    // Serialize Value to BEVE
    let bytes = beve::to_vec(&value).unwrap();

    // Deserialize to concrete struct
    let person: Person = beve::from_slice(&bytes).unwrap();
    assert_eq!(person.name, "Alice");
    assert_eq!(person.age, 30);
    assert!(person.active);
}

#[test]
fn nested_struct_to_value_roundtrip() {
    let original = Container {
        label: "test".to_string(),
        values: vec![1, 2, 3, 4, 5],
        metadata: Some("extra info".to_string()),
    };

    // Struct -> BEVE -> Value
    let bytes1 = beve::to_vec(&original).unwrap();
    let value: Value = beve::from_slice(&bytes1).unwrap();

    // Verify structure
    assert!(value.is_object());
    assert_eq!(
        value.get_key("label").and_then(|v| v.as_str()),
        Some("test")
    );
    let values = value.get_key("values").and_then(|v| v.as_array()).unwrap();
    assert_eq!(values.len(), 5);

    // Value -> BEVE -> Struct
    let bytes2 = beve::to_vec(&value).unwrap();
    let recovered: Container = beve::from_slice(&bytes2).unwrap();
    assert_eq!(original, recovered);
}

#[test]
fn vec_to_value_to_vec() {
    let original: Vec<i32> = vec![10, 20, 30, 40, 50];

    // Vec -> BEVE -> Value -> BEVE -> Vec
    let bytes1 = beve::to_vec(&original).unwrap();
    let value: Value = beve::from_slice(&bytes1).unwrap();
    let bytes2 = beve::to_vec(&value).unwrap();
    let recovered: Vec<i32> = beve::from_slice(&bytes2).unwrap();

    assert_eq!(original, recovered);
}

#[test]
fn btreemap_to_value_to_btreemap() {
    let mut original: BTreeMap<String, i32> = BTreeMap::new();
    original.insert("a".to_string(), 1);
    original.insert("b".to_string(), 2);
    original.insert("c".to_string(), 3);

    // BTreeMap -> BEVE -> Value -> BEVE -> BTreeMap
    let bytes1 = beve::to_vec(&original).unwrap();
    let value: Value = beve::from_slice(&bytes1).unwrap();
    let bytes2 = beve::to_vec(&value).unwrap();
    let recovered: BTreeMap<String, i32> = beve::from_slice(&bytes2).unwrap();

    assert_eq!(original, recovered);
}

#[test]
fn integer_keyed_map_to_value_roundtrip() {
    let mut original: BTreeMap<u32, String> = BTreeMap::new();
    original.insert(1, "one".to_string());
    original.insert(2, "two".to_string());
    original.insert(3, "three".to_string());

    // BTreeMap<u32, String> -> BEVE -> Value -> BEVE -> BTreeMap<u32, String>
    let bytes1 = beve::to_vec(&original).unwrap();
    let value: Value = beve::from_slice(&bytes1).unwrap();

    // Verify we can access by integer keys
    assert_eq!(value.get_uint_key(1).and_then(|v| v.as_str()), Some("one"));

    let bytes2 = beve::to_vec(&value).unwrap();
    let recovered: BTreeMap<u32, String> = beve::from_slice(&bytes2).unwrap();

    assert_eq!(original, recovered);
}

// ============ Deserializing from typed data ============

#[test]
fn deserialize_struct_as_value() {
    let p = Point { x: 1.0, y: -2.0 };
    let bytes = beve::to_vec(&p).unwrap();
    let value: Value = beve::from_slice(&bytes).unwrap();

    assert!(value.is_object());
    let x = value.get_key("x").unwrap();
    let y = value.get_key("y").unwrap();
    assert_eq!(x.as_f64(), Some(1.0));
    assert_eq!(y.as_f64(), Some(-2.0));
}

#[test]
fn deserialize_vec_as_value() {
    let v: Vec<i32> = vec![1, 2, 3, 4, 5];
    let bytes = beve::to_vec(&v).unwrap();
    let value: Value = beve::from_slice(&bytes).unwrap();

    assert!(value.is_array());
    let arr = value.as_array().unwrap();
    assert_eq!(arr.len(), 5);
    assert_eq!(arr[0].as_i64(), Some(1));
    assert_eq!(arr[4].as_i64(), Some(5));
}

#[test]
fn deserialize_btreemap_as_value() {
    let mut m: BTreeMap<String, i32> = BTreeMap::new();
    m.insert("a".to_string(), 1);
    m.insert("b".to_string(), 2);

    let bytes = beve::to_vec(&m).unwrap();
    let value: Value = beve::from_slice(&bytes).unwrap();

    assert!(value.is_object());
    assert_eq!(value.get_key("a").and_then(|v| v.as_i64()), Some(1));
    assert_eq!(value.get_key("b").and_then(|v| v.as_i64()), Some(2));
}

#[test]
fn deserialize_integer_keyed_map_as_value() {
    let mut m: BTreeMap<u32, String> = BTreeMap::new();
    m.insert(1, "one".to_string());
    m.insert(2, "two".to_string());

    let bytes = beve::to_vec(&m).unwrap();
    let value: Value = beve::from_slice(&bytes).unwrap();

    assert!(value.is_object());
    assert_eq!(value.get_uint_key(1).and_then(|v| v.as_str()), Some("one"));
    assert_eq!(value.get_uint_key(2).and_then(|v| v.as_str()), Some("two"));
}

// ============ Index operator tests ============

#[test]
fn test_index_operators() {
    let v = Value::Array(vec![
        Value::Number(Number::I64(10)),
        Value::Number(Number::I64(20)),
        Value::Number(Number::I64(30)),
    ]);

    assert_eq!(v[0].as_i64(), Some(10));
    assert_eq!(v[1].as_i64(), Some(20));
    assert!(v[100].is_null()); // Out of bounds returns null

    let mut obj = Object::new();
    obj.insert(
        Key::String("foo".to_string()),
        Value::String("bar".to_string()),
    );
    let v = Value::Object(obj);

    assert_eq!(v["foo"].as_str(), Some("bar"));
    assert!(v["missing"].is_null()); // Missing key returns null
}

// ============ From impl tests ============

#[test]
fn test_from_impls() {
    let v: Value = 42i32.into();
    assert!(v.is_number());

    let v: Value = "hello".into();
    assert!(v.is_string());

    let v: Value = true.into();
    assert!(v.is_bool());

    let v: Value = vec![1, 2, 3].into();
    assert!(v.is_array());

    let v: Value = Option::<i32>::None.into();
    assert!(v.is_null());

    let v: Value = Some(42).into();
    assert!(v.is_number());
}

// ============ Number method tests ============

#[test]
fn test_number_methods() {
    let n = Number::I64(-42);
    assert!(n.is_signed());
    assert!(!n.is_unsigned());
    assert!(!n.is_float());
    assert_eq!(n.as_i64(), Some(-42));
    assert_eq!(n.as_u64(), None); // Negative can't be u64

    let n = Number::U64(100);
    assert!(!n.is_signed());
    assert!(n.is_unsigned());
    assert_eq!(n.as_i64(), Some(100));
    assert_eq!(n.as_u64(), Some(100));

    let n = Number::F64(2.5);
    assert!(n.is_float());
    assert_eq!(n.as_i64(), None); // Has fractional part

    let n = Number::F64(42.0);
    assert_eq!(n.as_i64(), Some(42)); // Whole number float
}

// ============ Display tests ============

#[test]
fn test_display() {
    assert_eq!(Value::Null.to_string(), "null");
    assert_eq!(Value::Bool(true).to_string(), "true");
    assert_eq!(Value::Bool(false).to_string(), "false");
    assert_eq!(Value::Number(Number::I64(42)).to_string(), "42");
    assert_eq!(Value::Number(Number::F64(2.5)).to_string(), "2.5");
    assert_eq!(Value::String("hello".to_string()).to_string(), "\"hello\"");

    let arr = Value::Array(vec![
        Value::Number(Number::I64(1)),
        Value::Number(Number::I64(2)),
    ]);
    assert_eq!(arr.to_string(), "[1, 2]");
}

// ============ Key tests ============

#[test]
fn test_key_types() {
    let k = Key::from("hello");
    assert_eq!(k.as_str(), Some("hello"));
    assert_eq!(k.as_i128(), None);

    let k = Key::from(42i128);
    assert_eq!(k.as_str(), None);
    assert_eq!(k.as_i128(), Some(42));

    let k = Key::from(100u128);
    assert_eq!(k.as_u128(), Some(100));
}

// ============ Complex structure from JSON ============

#[test]
fn value_from_json_conversion() {
    let json_input = r#"{"name":"test","count":5,"active":true,"data":[1,2,3]}"#;
    let beve_bytes = beve::json_str_to_beve(json_input).unwrap();
    let value: Value = beve::from_slice(&beve_bytes).unwrap();

    assert!(value.is_object());
    assert_eq!(value.get_key("name").and_then(|v| v.as_str()), Some("test"));
    assert_eq!(value.get_key("count").and_then(|v| v.as_i64()), Some(5));
    assert_eq!(
        value.get_key("active").and_then(|v| v.as_bool()),
        Some(true)
    );

    let data = value.get_key("data").unwrap();
    assert!(data.is_array());
    assert_eq!(data.as_array().unwrap().len(), 3);
}

// ============ Typed array preservation ============

#[test]
fn typed_array_to_value() {
    // Test that typed arrays deserialize correctly
    let numbers = beve::fast::to_vec_typed_slice(&[1i32, 2, 3, 4, 5]);
    let value: Value = beve::from_slice(&numbers).unwrap();

    assert!(value.is_array());
    let arr = value.as_array().unwrap();
    assert_eq!(arr.len(), 5);
    assert_eq!(arr[0].as_i64(), Some(1));
    assert_eq!(arr[4].as_i64(), Some(5));
}

#[test]
fn float_typed_array_to_value() {
    let floats = beve::fast::to_vec_typed_slice(&[1.5f64, 2.5, 3.5]);
    let value: Value = beve::from_slice(&floats).unwrap();

    assert!(value.is_array());
    let arr = value.as_array().unwrap();
    assert_eq!(arr.len(), 3);
    assert_eq!(arr[0].as_f64(), Some(1.5));
}

#[test]
fn bool_array_to_value() {
    let mut bools = Vec::new();
    beve::fast::write_bool_slice(&mut bools, &[true, false, true, false, true]);
    let value: Value = beve::from_slice(&bools).unwrap();

    assert!(value.is_array());
    let arr = value.as_array().unwrap();
    assert_eq!(arr.len(), 5);
    assert_eq!(arr[0].as_bool(), Some(true));
    assert_eq!(arr[1].as_bool(), Some(false));
}

#[test]
fn string_array_to_value() {
    let strings = beve::fast::to_vec_str_slice(&["hello", "world", "test"]);
    let value: Value = beve::from_slice(&strings).unwrap();

    assert!(value.is_array());
    let arr = value.as_array().unwrap();
    assert_eq!(arr.len(), 3);
    assert_eq!(arr[0].as_str(), Some("hello"));
    assert_eq!(arr[2].as_str(), Some("test"));
}

// ============ Direct from_value tests (no BEVE bytes) ============

#[test]
fn from_value_struct() {
    // Construct Value directly, convert to struct without BEVE bytes
    let mut obj = Object::new();
    obj.insert(
        Key::String("x".to_string()),
        Value::Number(Number::F64(1.5)),
    );
    obj.insert(
        Key::String("y".to_string()),
        Value::Number(Number::F64(-2.5)),
    );
    let value = Value::Object(obj);

    let point: Point = from_value(value).unwrap();
    assert_eq!(point.x, 1.5);
    assert_eq!(point.y, -2.5);
}

#[test]
fn from_value_ref_struct() {
    // Use from_value_ref to keep the Value around
    let mut obj = Object::new();
    obj.insert(
        Key::String("name".to_string()),
        Value::String("Bob".to_string()),
    );
    obj.insert(
        Key::String("age".to_string()),
        Value::Number(Number::U64(25)),
    );
    obj.insert(Key::String("active".to_string()), Value::Bool(false));
    let value = Value::Object(obj);

    let person: Person = from_value_ref(&value).unwrap();
    assert_eq!(person.name, "Bob");
    assert_eq!(person.age, 25);
    assert!(!person.active);

    // Value is still available
    assert!(value.is_object());
}

#[test]
fn from_value_vec() {
    let value = Value::Array(vec![
        Value::Number(Number::I64(10)),
        Value::Number(Number::I64(20)),
        Value::Number(Number::I64(30)),
    ]);

    let vec: Vec<i32> = from_value(value).unwrap();
    assert_eq!(vec, vec![10, 20, 30]);
}

#[test]
fn from_value_btreemap() {
    let mut obj = Object::new();
    obj.insert(Key::String("a".to_string()), Value::Number(Number::I64(1)));
    obj.insert(Key::String("b".to_string()), Value::Number(Number::I64(2)));
    let value = Value::Object(obj);

    let map: BTreeMap<String, i32> = from_value(value).unwrap();
    assert_eq!(map.get("a"), Some(&1));
    assert_eq!(map.get("b"), Some(&2));
}

#[test]
fn from_value_nested_struct() {
    let mut obj = Object::new();
    obj.insert(
        Key::String("label".to_string()),
        Value::String("test".to_string()),
    );
    obj.insert(
        Key::String("values".to_string()),
        Value::Array(vec![
            Value::Number(Number::I64(1)),
            Value::Number(Number::I64(2)),
            Value::Number(Number::I64(3)),
        ]),
    );
    obj.insert(
        Key::String("metadata".to_string()),
        Value::String("info".to_string()),
    );
    let value = Value::Object(obj);

    let container: Container = from_value(value).unwrap();
    assert_eq!(container.label, "test");
    assert_eq!(container.values, vec![1, 2, 3]);
    assert_eq!(container.metadata, Some("info".to_string()));
}

#[test]
fn from_value_option_none() {
    let value = Value::Null;
    let opt: Option<i32> = from_value(value).unwrap();
    assert_eq!(opt, None);
}

#[test]
fn from_value_option_some() {
    let value = Value::Number(Number::I64(42));
    let opt: Option<i32> = from_value(value).unwrap();
    assert_eq!(opt, Some(42));
}

#[test]
fn from_value_primitives() {
    assert!(from_value::<bool>(Value::Bool(true)).unwrap());
    assert_eq!(
        from_value::<i32>(Value::Number(Number::I64(-42))).unwrap(),
        -42
    );
    assert_eq!(
        from_value::<u64>(Value::Number(Number::U64(100))).unwrap(),
        100
    );
    assert_eq!(
        from_value::<f64>(Value::Number(Number::F64(2.5))).unwrap(),
        2.5
    );
    assert_eq!(
        from_value::<String>(Value::String("hello".to_string())).unwrap(),
        "hello"
    );
}

#[test]
fn from_value_enum() {
    #[derive(Deserialize, Debug, PartialEq)]
    enum Status {
        Active,
        Inactive,
    }

    // Unit variant
    let value = Value::String("Active".to_string());
    let status: Status = from_value(value).unwrap();
    assert_eq!(status, Status::Active);
}

#[test]
fn from_value_tuple() {
    let value = Value::Array(vec![
        Value::Number(Number::I64(1)),
        Value::String("hello".to_string()),
        Value::Bool(true),
    ]);

    let tuple: (i32, String, bool) = from_value(value).unwrap();
    assert_eq!(tuple, (1, "hello".to_string(), true));
}

#[test]
fn from_value_full_roundtrip() {
    // Struct -> BEVE bytes -> Value -> Struct (via from_value, no re-serialization)
    let original = Person {
        name: "Alice".to_string(),
        age: 30,
        active: true,
    };

    let bytes = beve::to_vec(&original).unwrap();
    let value: Value = beve::from_slice(&bytes).unwrap();

    // Now convert directly without going back through bytes
    let recovered: Person = from_value(value).unwrap();
    assert_eq!(original, recovered);
}
