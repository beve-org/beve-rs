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
