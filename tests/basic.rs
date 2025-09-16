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
    rt(3.14f32);
    rt(-2.71828f64);
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
