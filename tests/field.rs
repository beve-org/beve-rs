#![deny(warnings)]

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

// ---- from_field: whole document ----

#[test]
fn field_empty_pointer_returns_whole_value() {
    let v = 42u32;
    let bytes = beve::to_vec(&v).unwrap();
    let back: u32 = beve::from_field(&bytes, "").unwrap();
    assert_eq!(v, back);
}

// ---- from_field: flat object ----

#[test]
fn field_flat_object_string_key() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Obj {
        alpha: i32,
        beta: String,
    }
    let data = Obj {
        alpha: 7,
        beta: "hello".into(),
    };
    let bytes = beve::to_vec(&data).unwrap();

    let a: i32 = beve::from_field(&bytes, "/alpha").unwrap();
    assert_eq!(a, 7);

    let b: String = beve::from_field(&bytes, "/beta").unwrap();
    assert_eq!(b, "hello");
}

// ---- from_field: nested objects ----

#[test]
fn field_nested_object() {
    #[derive(Serialize, Deserialize)]
    struct Inner {
        value: f64,
    }
    #[derive(Serialize, Deserialize)]
    struct Outer {
        inner: Inner,
    }
    let data = Outer {
        inner: Inner { value: 3.14 },
    };
    let bytes = beve::to_vec(&data).unwrap();
    let v: f64 = beve::from_field(&bytes, "/inner/value").unwrap();
    assert!((v - 3.14).abs() < 1e-12);
}

// ---- from_field: HashMap ----

#[test]
fn field_hashmap_lookup() {
    let mut map = HashMap::new();
    map.insert("x".to_string(), vec![1.0f64, 2.0, 3.0]);
    map.insert("y".to_string(), vec![4.0, 5.0]);
    let bytes = beve::to_vec(&map).unwrap();

    let x: Vec<f64> = beve::from_field(&bytes, "/x").unwrap();
    assert_eq!(x, vec![1.0, 2.0, 3.0]);

    let y: Vec<f64> = beve::from_field(&bytes, "/y").unwrap();
    assert_eq!(y, vec![4.0, 5.0]);
}

// ---- from_field: generic array indexing ----

#[test]
fn field_generic_array_index() {
    // A generic array of mixed types via Value
    let arr: Vec<beve::Value> = vec![
        beve::Value::String("zeroth".into()),
        beve::Value::Number(beve::Number::U64(42)),
        beve::Value::Bool(true),
    ];
    let bytes = beve::to_vec(&arr).unwrap();

    let s: String = beve::from_field(&bytes, "/0").unwrap();
    assert_eq!(s, "zeroth");

    let n: u64 = beve::from_field(&bytes, "/1").unwrap();
    assert_eq!(n, 42);

    let b: bool = beve::from_field(&bytes, "/2").unwrap();
    assert!(b);
}

// ---- from_field: nested generic array + object ----

#[test]
fn field_array_then_object() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Item {
        name: String,
        score: u32,
    }
    let items = vec![
        Item {
            name: "a".into(),
            score: 10,
        },
        Item {
            name: "b".into(),
            score: 20,
        },
        Item {
            name: "c".into(),
            score: 30,
        },
    ];
    let bytes = beve::to_vec(&items).unwrap();

    let name: String = beve::from_field(&bytes, "/1/name").unwrap();
    assert_eq!(name, "b");

    let score: u32 = beve::from_field(&bytes, "/2/score").unwrap();
    assert_eq!(score, 30);
}

// ---- from_field: key not found ----

#[test]
fn field_missing_key_is_error() {
    #[derive(Serialize)]
    struct Obj {
        x: i32,
    }
    let bytes = beve::to_vec(&Obj { x: 1 }).unwrap();
    let res: Result<i32, _> = beve::from_field(&bytes, "/missing");
    assert!(res.is_err());
}

// ---- from_field: index out of bounds ----

#[test]
fn field_index_out_of_bounds() {
    let arr: Vec<beve::Value> = vec![beve::Value::Null];
    let bytes = beve::to_vec(&arr).unwrap();
    let res: Result<(), _> = beve::from_field(&bytes, "/5");
    assert!(res.is_err());
}

// ---- from_field: invalid pointer (no leading /) ----

#[test]
fn field_invalid_pointer_is_error() {
    let bytes = beve::to_vec(&42u32).unwrap();
    let res: Result<u32, _> = beve::from_field(&bytes, "bad");
    assert!(res.is_err());
}

// ---- from_field: tilde escapes ----

#[test]
fn field_tilde_escape() {
    let mut map = HashMap::new();
    map.insert("a/b".to_string(), 1u32);
    map.insert("c~d".to_string(), 2u32);
    let bytes = beve::to_vec(&map).unwrap();

    let v1: u32 = beve::from_field(&bytes, "/a~1b").unwrap();
    assert_eq!(v1, 1);

    let v2: u32 = beve::from_field(&bytes, "/c~0d").unwrap();
    assert_eq!(v2, 2);
}

// ---- from_field_slice: basic numeric slice ----

#[test]
fn field_slice_u32() {
    let data: Vec<u32> = (0..100).collect();
    let bytes = beve::to_vec(&data).unwrap();

    let slice: Vec<u32> = beve::from_field_slice(&bytes, "", 10, 5).unwrap();
    assert_eq!(slice, vec![10, 11, 12, 13, 14]);
}

#[test]
fn field_slice_f64() {
    let data: Vec<f64> = (0..50).map(|i| i as f64 * 0.5).collect();
    let bytes = beve::to_vec(&data).unwrap();

    let slice: Vec<f64> = beve::from_field_slice(&bytes, "", 0, 3).unwrap();
    assert_eq!(slice, vec![0.0, 0.5, 1.0]);
}

#[test]
fn field_slice_nested() {
    #[derive(Serialize)]
    struct Data {
        samples: Vec<f32>,
    }
    let data = Data {
        samples: (0..1000).map(|i| i as f32).collect(),
    };
    let bytes = beve::to_vec(&data).unwrap();

    let slice: Vec<f32> = beve::from_field_slice(&bytes, "/samples", 500, 3).unwrap();
    assert_eq!(slice, vec![500.0, 501.0, 502.0]);
}

// ---- from_field_slice: out of bounds ----

#[test]
fn field_slice_out_of_bounds() {
    let data: Vec<u32> = vec![1, 2, 3];
    let bytes = beve::to_vec(&data).unwrap();
    let res: Result<Vec<u32>, _> = beve::from_field_slice(&bytes, "", 2, 5);
    assert!(res.is_err());
}

// ---- from_field_slice: non-typed-array is error ----

#[test]
fn field_slice_on_string_is_error() {
    let bytes = beve::to_vec(&"hello").unwrap();
    let res: Result<Vec<u8>, _> = beve::from_field_slice(&bytes, "", 0, 3);
    assert!(res.is_err());
}

// ---- skip_value covers all types ----

#[test]
fn skip_covers_all_types() {
    // Build a generic array containing one of each major type
    let arr: Vec<beve::Value> = vec![
        beve::Value::Null,
        beve::Value::Bool(true),
        beve::Value::Number(beve::Number::U64(99)),
        beve::Value::Number(beve::Number::I64(-7)),
        beve::Value::Number(beve::Number::F64(2.5)),
        beve::Value::String("hello".into()),
        beve::Value::Array(vec![
            beve::Value::Number(beve::Number::U64(1)),
            beve::Value::Number(beve::Number::U64(2)),
        ]),
    ];
    let bytes = beve::to_vec(&arr).unwrap();

    // Navigate to the last element by skipping all previous ones
    let last: beve::Value = beve::from_field(&bytes, "/6").unwrap();
    match &last {
        beve::Value::Array(inner) => assert_eq!(inner.len(), 2),
        _ => panic!("expected array"),
    }
}

// ---- from_field: deeply nested ----

#[test]
fn field_three_levels_deep() {
    #[derive(Serialize, Deserialize)]
    struct A {
        b: B,
    }
    #[derive(Serialize, Deserialize)]
    struct B {
        c: C,
    }
    #[derive(Serialize, Deserialize)]
    struct C {
        val: u64,
    }
    let data = A {
        b: B {
            c: C { val: 999 },
        },
    };
    let bytes = beve::to_vec(&data).unwrap();
    let v: u64 = beve::from_field(&bytes, "/b/c/val").unwrap();
    assert_eq!(v, 999);
}

// ---- from_field: typed array as leaf ----

#[test]
fn field_typed_array_leaf() {
    #[derive(Serialize, Deserialize)]
    struct Data {
        nums: Vec<i64>,
    }
    let data = Data {
        nums: vec![-1, -2, -3],
    };
    let bytes = beve::to_vec(&data).unwrap();
    let v: Vec<i64> = beve::from_field(&bytes, "/nums").unwrap();
    assert_eq!(v, vec![-1, -2, -3]);
}

// ---- from_field_slice: signed integers ----

#[test]
fn field_slice_i16() {
    let data: Vec<i16> = (-50..50).collect();
    let bytes = beve::to_vec(&data).unwrap();
    let slice: Vec<i16> = beve::from_field_slice(&bytes, "", 45, 10).unwrap();
    let expected: Vec<i16> = (-5..5).collect();
    assert_eq!(slice, expected);
}

// ---- from_field_slice: zero-length slice ----

#[test]
fn field_slice_zero_count() {
    let data: Vec<u32> = vec![1, 2, 3];
    let bytes = beve::to_vec(&data).unwrap();
    let slice: Vec<u32> = beve::from_field_slice(&bytes, "", 1, 0).unwrap();
    assert!(slice.is_empty());
}

// ---- skip_value: typed arrays ----

#[test]
fn skip_typed_arrays() {
    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct Data {
        floats: Vec<f64>,
        target: u32,
    }
    let data = Data {
        floats: vec![1.0; 1000],
        target: 42,
    };
    let bytes = beve::to_vec(&data).unwrap();
    let v: u32 = beve::from_field(&bytes, "/target").unwrap();
    assert_eq!(v, 42);
}

// ---- skip_value: boolean array ----

#[test]
fn skip_bool_array() {
    #[derive(Serialize, Deserialize)]
    struct Data {
        flags: Vec<bool>,
        name: String,
    }
    let data = Data {
        flags: vec![true, false, true, true, false, false, true, false, true],
        name: "end".into(),
    };
    let bytes = beve::to_vec(&data).unwrap();
    let v: String = beve::from_field(&bytes, "/name").unwrap();
    assert_eq!(v, "end");
}

// ---- skip_value: string array ----

#[test]
fn skip_string_array() {
    #[derive(Serialize, Deserialize)]
    struct Data {
        words: Vec<String>,
        count: u32,
    }
    let data = Data {
        words: vec!["foo".into(), "bar".into(), "baz".into()],
        count: 3,
    };
    let bytes = beve::to_vec(&data).unwrap();
    let v: u32 = beve::from_field(&bytes, "/count").unwrap();
    assert_eq!(v, 3);
}

// ---- from_field: option fields ----

#[test]
fn field_option_some() {
    #[derive(Serialize, Deserialize)]
    struct Data {
        maybe: Option<u32>,
    }
    let data = Data { maybe: Some(7) };
    let bytes = beve::to_vec(&data).unwrap();
    let v: Option<u32> = beve::from_field(&bytes, "/maybe").unwrap();
    assert_eq!(v, Some(7));
}

#[test]
fn field_option_none() {
    #[derive(Serialize, Deserialize)]
    struct Data {
        maybe: Option<u32>,
        after: String,
    }
    let data = Data {
        maybe: None,
        after: "ok".into(),
    };
    let bytes = beve::to_vec(&data).unwrap();
    let v: Option<u32> = beve::from_field(&bytes, "/maybe").unwrap();
    assert_eq!(v, None);
    // Verify skip past None works
    let s: String = beve::from_field(&bytes, "/after").unwrap();
    assert_eq!(s, "ok");
}

// ---- from_field: enum variants ----

#[test]
fn field_enum_variant() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    enum Color {
        Red,
        Green,
        Blue,
    }
    #[derive(Serialize, Deserialize)]
    struct Data {
        color: Color,
        label: String,
    }
    let data = Data {
        color: Color::Green,
        label: "test".into(),
    };
    let bytes = beve::to_vec(&data).unwrap();
    let c: Color = beve::from_field(&bytes, "/color").unwrap();
    assert_eq!(c, Color::Green);
    let l: String = beve::from_field(&bytes, "/label").unwrap();
    assert_eq!(l, "test");
}

#[test]
fn field_enum_with_data() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    enum Shape {
        Circle(f64),
        Rect { w: f64, h: f64 },
    }
    #[derive(Serialize, Deserialize)]
    struct Data {
        shape: Shape,
    }
    let data = Data {
        shape: Shape::Rect { w: 3.0, h: 4.0 },
    };
    let bytes = beve::to_vec(&data).unwrap();
    let s: Shape = beve::from_field(&bytes, "/shape").unwrap();
    assert_eq!(s, Shape::Rect { w: 3.0, h: 4.0 });
}

// ---- from_field: skip over enum values ----

#[test]
fn skip_enum_then_read_next() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    enum Tag {
        A(u32),
        B(String),
    }
    #[derive(Serialize, Deserialize)]
    struct Data {
        tag: Tag,
        value: i64,
    }
    let data = Data {
        tag: Tag::B("hello".into()),
        value: -42,
    };
    let bytes = beve::to_vec(&data).unwrap();
    let v: i64 = beve::from_field(&bytes, "/value").unwrap();
    assert_eq!(v, -42);
}

// ---- from_field: empty containers ----

#[test]
fn field_empty_object() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Empty {}
    #[derive(Serialize, Deserialize)]
    struct Data {
        empty: Empty,
        after: u32,
    }
    let data = Data {
        empty: Empty {},
        after: 99,
    };
    let bytes = beve::to_vec(&data).unwrap();
    let v: u32 = beve::from_field(&bytes, "/after").unwrap();
    assert_eq!(v, 99);
}

#[test]
fn field_empty_vec() {
    #[derive(Serialize, Deserialize)]
    struct Data {
        items: Vec<u32>,
        count: u32,
    }
    let data = Data {
        items: vec![],
        count: 0,
    };
    let bytes = beve::to_vec(&data).unwrap();
    let items: Vec<u32> = beve::from_field(&bytes, "/items").unwrap();
    assert!(items.is_empty());
    let c: u32 = beve::from_field(&bytes, "/count").unwrap();
    assert_eq!(c, 0);
}

// ---- from_field: integer-keyed maps ----

#[test]
fn field_unsigned_integer_keyed_map() {
    let mut map = BTreeMap::new();
    map.insert(10u32, "ten".to_string());
    map.insert(20u32, "twenty".to_string());
    map.insert(30u32, "thirty".to_string());
    let bytes = beve::to_vec(&map).unwrap();

    let v: String = beve::from_field(&bytes, "/20").unwrap();
    assert_eq!(v, "twenty");
}

#[test]
fn field_signed_integer_keyed_map() {
    let mut map = BTreeMap::new();
    map.insert(-1i32, "neg_one".to_string());
    map.insert(0i32, "zero".to_string());
    map.insert(1i32, "one".to_string());
    let bytes = beve::to_vec(&map).unwrap();

    let v: String = beve::from_field(&bytes, "/0").unwrap();
    assert_eq!(v, "zero");

    let v: String = beve::from_field(&bytes, "/-1").unwrap();
    assert_eq!(v, "neg_one");
}

#[test]
fn field_integer_key_not_found() {
    let mut map = BTreeMap::new();
    map.insert(1u32, "one".to_string());
    let bytes = beve::to_vec(&map).unwrap();
    let res: Result<String, _> = beve::from_field(&bytes, "/999");
    assert!(res.is_err());
}

#[test]
fn field_integer_key_invalid_token() {
    let mut map = BTreeMap::new();
    map.insert(1u32, "one".to_string());
    let bytes = beve::to_vec(&map).unwrap();
    let res: Result<String, _> = beve::from_field(&bytes, "/not_a_number");
    assert!(res.is_err());
}

// ---- from_field: navigate into integer-keyed map value ----

#[test]
fn field_integer_key_nested() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Inner {
        data: Vec<f64>,
    }
    let mut map = BTreeMap::new();
    map.insert(5u32, Inner { data: vec![1.0, 2.0] });
    map.insert(10u32, Inner { data: vec![3.0, 4.0, 5.0] });
    let bytes = beve::to_vec(&map).unwrap();

    let v: Vec<f64> = beve::from_field(&bytes, "/10/data").unwrap();
    assert_eq!(v, vec![3.0, 4.0, 5.0]);
}

// ---- from_field: various numeric types ----

#[test]
fn field_various_numeric_widths() {
    #[derive(Serialize, Deserialize)]
    struct Nums {
        a: u8,
        b: u16,
        c: u32,
        d: u64,
        e: i8,
        f: i16,
        g: i32,
        h: i64,
        i: f32,
        j: f64,
    }
    let data = Nums {
        a: 1, b: 2, c: 3, d: 4, e: -1, f: -2, g: -3, h: -4, i: 1.5, j: 2.5,
    };
    let bytes = beve::to_vec(&data).unwrap();

    assert_eq!(beve::from_field::<u8>(&bytes, "/a").unwrap(), 1);
    assert_eq!(beve::from_field::<u16>(&bytes, "/b").unwrap(), 2);
    assert_eq!(beve::from_field::<u32>(&bytes, "/c").unwrap(), 3);
    assert_eq!(beve::from_field::<u64>(&bytes, "/d").unwrap(), 4);
    assert_eq!(beve::from_field::<i8>(&bytes, "/e").unwrap(), -1);
    assert_eq!(beve::from_field::<i16>(&bytes, "/f").unwrap(), -2);
    assert_eq!(beve::from_field::<i32>(&bytes, "/g").unwrap(), -3);
    assert_eq!(beve::from_field::<i64>(&bytes, "/h").unwrap(), -4);
    assert!((beve::from_field::<f32>(&bytes, "/i").unwrap() - 1.5).abs() < 1e-6);
    assert!((beve::from_field::<f64>(&bytes, "/j").unwrap() - 2.5).abs() < 1e-12);
}

// ---- from_field: navigate into scalar (error) ----

#[test]
fn field_navigate_into_number_is_error() {
    let bytes = beve::to_vec(&42u32).unwrap();
    let res: Result<u32, _> = beve::from_field(&bytes, "/0");
    assert!(res.is_err());
}

#[test]
fn field_navigate_into_string_is_error() {
    let bytes = beve::to_vec(&"hello").unwrap();
    let res: Result<String, _> = beve::from_field(&bytes, "/0");
    assert!(res.is_err());
}

#[test]
fn field_navigate_into_bool_is_error() {
    let bytes = beve::to_vec(&true).unwrap();
    let res: Result<bool, _> = beve::from_field(&bytes, "/0");
    assert!(res.is_err());
}

// ---- from_field: zero-copy borrowed str ----

#[test]
fn field_zero_copy_borrowed_str() {
    #[derive(Serialize, Deserialize)]
    struct Data {
        name: String,
    }
    let data = Data {
        name: "borrowed".into(),
    };
    let bytes = beve::to_vec(&data).unwrap();
    let v: &str = beve::from_field(&bytes, "/name").unwrap();
    assert_eq!(v, "borrowed");
}

// ---- from_field: complex data ----

#[test]
fn field_complex_f64_scalar() {
    use beve::Complex;
    #[derive(Serialize, Deserialize)]
    struct Data {
        z: Complex<f64>,
    }
    let data = Data {
        z: Complex { re: 1.0, im: 2.0 },
    };
    let bytes = beve::to_vec(&data).unwrap();
    let z: Complex<f64> = beve::from_field(&bytes, "/z").unwrap();
    assert!((z.re - 1.0).abs() < 1e-12);
    assert!((z.im - 2.0).abs() < 1e-12);
}

// ---- skip_value: complex scalar and array ----

#[test]
fn skip_complex_scalar() {
    use beve::Complex;
    #[derive(Serialize, Deserialize)]
    struct Data {
        z: Complex<f32>,
        after: u32,
    }
    let data = Data {
        z: Complex { re: 1.0, im: 2.0 },
        after: 42,
    };
    let bytes = beve::to_vec(&data).unwrap();
    let v: u32 = beve::from_field(&bytes, "/after").unwrap();
    assert_eq!(v, 42);
}

#[test]
fn skip_complex_array() {
    use beve::Complex;
    // Build a struct that has a complex array followed by another field.
    // We serialize manually: object with "zs" (complex array) + "after" (string).
    let complexes = vec![
        Complex { re: 1.0f64, im: 2.0 },
        Complex { re: 3.0, im: 4.0 },
        Complex { re: 5.0, im: 6.0 },
    ];
    #[derive(Serialize)]
    struct Data {
        zs: Vec<Complex<f64>>,
        after: String,
    }
    let data = Data {
        zs: complexes,
        after: "done".into(),
    };
    let bytes = beve::to_vec(&data).unwrap();
    let v: String = beve::from_field(&bytes, "/after").unwrap();
    assert_eq!(v, "done");
}

// ---- skip_value: nested objects ----

#[test]
fn skip_nested_objects() {
    #[derive(Serialize, Deserialize)]
    struct Inner {
        a: Vec<f64>,
        b: HashMap<String, u32>,
    }
    #[derive(Serialize, Deserialize)]
    struct Outer {
        first: Inner,
        target: bool,
    }
    let data = Outer {
        first: Inner {
            a: vec![1.0; 100],
            b: [("x".into(), 1), ("y".into(), 2)].into(),
        },
        target: true,
    };
    let bytes = beve::to_vec(&data).unwrap();
    let v: bool = beve::from_field(&bytes, "/target").unwrap();
    assert!(v);
}

// ---- from_field: first element of generic array (index 0) ----

#[test]
fn field_generic_array_index_zero() {
    // Use mixed types to ensure a generic array (not typed string array)
    let arr: Vec<beve::Value> = vec![
        beve::Value::String("first".into()),
        beve::Value::Number(beve::Number::U64(99)),
    ];
    let bytes = beve::to_vec(&arr).unwrap();
    let v: String = beve::from_field(&bytes, "/0").unwrap();
    assert_eq!(v, "first");
}

// ---- from_field: last element of generic array ----

#[test]
fn field_generic_array_last_element() {
    let arr: Vec<beve::Value> = vec![
        beve::Value::Number(beve::Number::U64(0)),
        beve::Value::Number(beve::Number::U64(1)),
        beve::Value::Number(beve::Number::U64(2)),
        beve::Value::Number(beve::Number::U64(3)),
        beve::Value::String("last".into()),
    ];
    let bytes = beve::to_vec(&arr).unwrap();
    let v: String = beve::from_field(&bytes, "/4").unwrap();
    assert_eq!(v, "last");
}

// ---- from_field: negative array index is error ----

#[test]
fn field_negative_array_index_is_error() {
    let arr: Vec<beve::Value> = vec![beve::Value::Null];
    let bytes = beve::to_vec(&arr).unwrap();
    let res: Result<(), _> = beve::from_field(&bytes, "/-1");
    assert!(res.is_err());
}

// ---- from_field: non-integer array index is error ----

#[test]
fn field_non_integer_array_index_is_error() {
    let arr: Vec<beve::Value> = vec![beve::Value::Null];
    let bytes = beve::to_vec(&arr).unwrap();
    let res: Result<(), _> = beve::from_field(&bytes, "/abc");
    assert!(res.is_err());
}

// ---- from_field_slice: all element byte widths ----

#[test]
fn field_slice_u8() {
    let data: Vec<u8> = (0..20).collect();
    let bytes = beve::to_vec(&data).unwrap();
    let slice: Vec<u8> = beve::from_field_slice(&bytes, "", 5, 3).unwrap();
    assert_eq!(slice, vec![5, 6, 7]);
}

#[test]
fn field_slice_u16() {
    let data: Vec<u16> = (1000..1020).collect();
    let bytes = beve::to_vec(&data).unwrap();
    let slice: Vec<u16> = beve::from_field_slice(&bytes, "", 10, 3).unwrap();
    assert_eq!(slice, vec![1010, 1011, 1012]);
}

#[test]
fn field_slice_u64() {
    let data: Vec<u64> = (0..10).map(|i| i * 1_000_000_000).collect();
    let bytes = beve::to_vec(&data).unwrap();
    let slice: Vec<u64> = beve::from_field_slice(&bytes, "", 8, 2).unwrap();
    assert_eq!(slice, vec![8_000_000_000, 9_000_000_000]);
}

#[test]
fn field_slice_i8() {
    let data: Vec<i8> = (-10..10).collect();
    let bytes = beve::to_vec(&data).unwrap();
    let slice: Vec<i8> = beve::from_field_slice(&bytes, "", 5, 5).unwrap();
    assert_eq!(slice, vec![-5, -4, -3, -2, -1]);
}

#[test]
fn field_slice_i32() {
    let data: Vec<i32> = (-100..100).collect();
    let bytes = beve::to_vec(&data).unwrap();
    let slice: Vec<i32> = beve::from_field_slice(&bytes, "", 95, 10).unwrap();
    let expected: Vec<i32> = (-5..5).collect();
    assert_eq!(slice, expected);
}

#[test]
fn field_slice_i64() {
    let data: Vec<i64> = (-5..5).collect();
    let bytes = beve::to_vec(&data).unwrap();
    let slice: Vec<i64> = beve::from_field_slice(&bytes, "", 0, 10).unwrap();
    let expected: Vec<i64> = (-5..5).collect();
    assert_eq!(slice, expected);
}

#[test]
fn field_slice_f32() {
    let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
    let bytes = beve::to_vec(&data).unwrap();
    let slice: Vec<f32> = beve::from_field_slice(&bytes, "", 50, 3).unwrap();
    assert!((slice[0] - 5.0).abs() < 1e-5);
    assert!((slice[1] - 5.1).abs() < 1e-5);
    assert!((slice[2] - 5.2).abs() < 1e-5);
}

// ---- from_field_slice: entire array ----

#[test]
fn field_slice_entire_array() {
    let data: Vec<u32> = vec![10, 20, 30];
    let bytes = beve::to_vec(&data).unwrap();
    let slice: Vec<u32> = beve::from_field_slice(&bytes, "", 0, 3).unwrap();
    assert_eq!(slice, vec![10, 20, 30]);
}

// ---- from_field_slice: last element only ----

#[test]
fn field_slice_last_element() {
    let data: Vec<u32> = vec![10, 20, 30];
    let bytes = beve::to_vec(&data).unwrap();
    let slice: Vec<u32> = beve::from_field_slice(&bytes, "", 2, 1).unwrap();
    assert_eq!(slice, vec![30]);
}

// ---- from_field_slice: start at boundary ----

#[test]
fn field_slice_start_equals_len_zero_count() {
    let data: Vec<u32> = vec![1, 2, 3];
    let bytes = beve::to_vec(&data).unwrap();
    let slice: Vec<u32> = beve::from_field_slice(&bytes, "", 3, 0).unwrap();
    assert!(slice.is_empty());
}

// ---- from_field_slice: bool array is rejected ----

#[test]
fn field_slice_bool_array_is_error() {
    let data: Vec<bool> = vec![true, false, true];
    let bytes = beve::to_vec(&data).unwrap();
    let res: Result<Vec<bool>, _> = beve::from_field_slice(&bytes, "", 0, 2);
    assert!(res.is_err());
}

// ---- from_field_slice: string array is rejected ----

#[test]
fn field_slice_string_array_is_error() {
    let data: Vec<String> = vec!["a".into(), "b".into()];
    let bytes = beve::to_vec(&data).unwrap();
    let res: Result<Vec<String>, _> = beve::from_field_slice(&bytes, "", 0, 1);
    assert!(res.is_err());
}

// ---- from_field_slice: object or generic array is rejected ----

#[test]
fn field_slice_on_object_is_error() {
    #[derive(Serialize)]
    struct Obj {
        x: u32,
    }
    let bytes = beve::to_vec(&Obj { x: 1 }).unwrap();
    let res: Result<Vec<u32>, _> = beve::from_field_slice(&bytes, "", 0, 1);
    assert!(res.is_err());
}

#[test]
fn field_slice_on_generic_array_is_error() {
    let arr: Vec<beve::Value> = vec![beve::Value::Null];
    let bytes = beve::to_vec(&arr).unwrap();
    let res: Result<Vec<u32>, _> = beve::from_field_slice(&bytes, "", 0, 1);
    assert!(res.is_err());
}

// ---- from_field: large object (many keys to skip) ----

#[test]
fn field_large_object_last_key() {
    let mut map = HashMap::new();
    for i in 0..100 {
        map.insert(format!("key_{i}"), i as u64);
    }
    let bytes = beve::to_vec(&map).unwrap();

    // Read a specific key
    let v: u64 = beve::from_field(&bytes, "/key_50").unwrap();
    assert_eq!(v, 50);
}

// ---- from_field: struct with many field types to skip over ----

#[test]
fn field_skip_varied_types() {
    #[derive(Serialize, Deserialize)]
    struct Mixed {
        flag: bool,
        count: u32,
        name: String,
        data: Vec<f64>,
        tags: Vec<String>,
        bits: Vec<bool>,
        nested: HashMap<String, u32>,
        target: i64,
    }
    let data = Mixed {
        flag: true,
        count: 42,
        name: "test".into(),
        data: vec![1.0, 2.0, 3.0],
        tags: vec!["a".into(), "b".into()],
        bits: vec![true, false, true],
        nested: [("x".into(), 1)].into(),
        target: -999,
    };
    let bytes = beve::to_vec(&data).unwrap();
    let v: i64 = beve::from_field(&bytes, "/target").unwrap();
    assert_eq!(v, -999);
}

// ---- from_field: deserialize sub-object ----

#[test]
fn field_deserialize_sub_object() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Inner {
        x: u32,
        y: u32,
    }
    #[derive(Serialize, Deserialize)]
    struct Outer {
        label: String,
        point: Inner,
    }
    let data = Outer {
        label: "origin".into(),
        point: Inner { x: 10, y: 20 },
    };
    let bytes = beve::to_vec(&data).unwrap();
    let p: Inner = beve::from_field(&bytes, "/point").unwrap();
    assert_eq!(p, Inner { x: 10, y: 20 });
}

// ---- JSON Pointer: root-only slash ----

#[test]
fn field_pointer_with_empty_key() {
    // "/" means: look up the key "" (empty string) in a map
    let mut map = HashMap::new();
    map.insert("".to_string(), 42u32);
    let bytes = beve::to_vec(&map).unwrap();
    let v: u32 = beve::from_field(&bytes, "/").unwrap();
    assert_eq!(v, 42);
}

// ---- JSON Pointer: multiple escapes in one segment ----

#[test]
fn field_multiple_escapes_in_segment() {
    let mut map = HashMap::new();
    map.insert("a~b/c~d".to_string(), 7u32);
    let bytes = beve::to_vec(&map).unwrap();
    let v: u32 = beve::from_field(&bytes, "/a~0b~1c~0d").unwrap();
    assert_eq!(v, 7);
}

// ---- skip_value: roundtrip correctness ----
// Verify that skip_value advances exactly past the value

#[test]
fn skip_value_advances_correctly() {
    // Serialize several values, then verify skip lands at each boundary
    #[derive(Serialize, Deserialize)]
    struct Row {
        a: u32,
        b: String,
        c: Vec<f64>,
    }
    let rows = vec![
        Row { a: 1, b: "one".into(), c: vec![1.0] },
        Row { a: 2, b: "two".into(), c: vec![2.0, 2.0] },
        Row { a: 3, b: "three".into(), c: vec![3.0, 3.0, 3.0] },
    ];
    let bytes = beve::to_vec(&rows).unwrap();

    // Read each element individually to verify skip works across boundaries
    for i in 0..3 {
        let r: Row = beve::from_field(&bytes, &format!("/{i}")).unwrap();
        assert_eq!(r.a, (i + 1) as u32);
    }
}

// ---- from_field_slice: deeply nested slice ----

#[test]
fn field_slice_deeply_nested() {
    #[derive(Serialize)]
    struct A {
        b: B,
    }
    #[derive(Serialize)]
    struct B {
        c: C,
    }
    #[derive(Serialize)]
    struct C {
        values: Vec<u32>,
    }
    let data = A {
        b: B {
            c: C {
                values: (0..50).collect(),
            },
        },
    };
    let bytes = beve::to_vec(&data).unwrap();
    let slice: Vec<u32> = beve::from_field_slice(&bytes, "/b/c/values", 20, 5).unwrap();
    assert_eq!(slice, vec![20, 21, 22, 23, 24]);
}
