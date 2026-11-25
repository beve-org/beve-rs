//! A newtype struct is its inner value on the wire, and every decoder hands that
//! value back through `visit_newtype_struct`, as serde's contract expects. A
//! derived newtype visitor accepts nothing else, so before this every newtype
//! failed to decode.

use serde::{Deserialize, Serialize};
use std::io::Cursor;

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
struct Meters(f64);

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
struct Names(Vec<String>);

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
struct Point {
    x: i32,
    y: i32,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
struct Wrapped(Point);

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
struct Samples(#[serde(with = "beve::typed::f64")] Vec<f64>);

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Copy)]
struct Count(u16);

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Copy)]
struct Offset(i64);

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Copy, PartialOrd, Eq, Ord)]
struct Id(u32);

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Copy)]
struct Phasor(beve::Complex<f32>);

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, PartialOrd, Eq, Ord)]
struct Name(String);

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Copy)]
struct Flag(bool);

/// A unit-only enum.
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Copy, PartialOrd, Eq, Ord)]
enum Side {
    Left,
    Right,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
struct Record {
    distance: Meters,
    maybe: Option<Meters>,
    many: Vec<Meters>,
    samples: Samples,
}

/// Through the slice decoder, the streaming decoder and `Value`.
fn roundtrip<T>(value: &T)
where
    T: Serialize + for<'de> Deserialize<'de> + PartialEq + std::fmt::Debug,
{
    let bytes = beve::to_vec(value).unwrap();
    let mut streamed = Vec::new();
    beve::to_writer_streaming(&mut streamed, value).unwrap();
    assert_eq!(streamed, bytes, "to_writer_streaming");
    assert_eq!(
        beve::serialized_size(value).unwrap(),
        bytes.len() as u64,
        "serialized_size"
    );
    assert_eq!(&beve::from_slice::<T>(&bytes).unwrap(), value, "from_slice");
    assert_eq!(
        &beve::from_reader_streaming::<_, T>(Cursor::new(&bytes)).unwrap(),
        value,
        "from_reader_streaming"
    );
    let v: beve::Value = beve::from_slice(&bytes).unwrap();
    assert_eq!(
        &beve::from_value_ref::<T>(&v).unwrap(),
        value,
        "from_value_ref"
    );
    assert_eq!(&beve::from_value::<T>(v).unwrap(), value, "from_value");
}

#[test]
fn newtype_structs_roundtrip() {
    roundtrip(&Meters(2.5));
    roundtrip(&Names(vec!["a".into(), "b".into()]));
    roundtrip(&Names(vec![]));
    roundtrip(&Wrapped(Point { x: 1, y: -2 }));
    roundtrip(&Samples(vec![]));
    roundtrip(&Samples(vec![1.0, f64::INFINITY]));
    roundtrip(&Record {
        distance: Meters(1.0),
        maybe: Some(Meters(3.0)),
        many: vec![Meters(4.0), Meters(5.0)],
        samples: Samples(vec![6.0]),
    });
    // Arrays of newtypes are written as typed and complex arrays, whose
    // elements every decoder reads through its own element deserializer.
    roundtrip(&vec![Count(1), Count(u16::MAX)]);
    roundtrip(&vec![Offset(-1), Offset(i64::MAX)]);
    roundtrip(&vec![Phasor(beve::Complex { re: 1.0, im: -2.0 })]);
    roundtrip(&vec![Name("x".into()), Name("".into())]);
    roundtrip(&vec![Flag(true), Flag(false), Flag(true)]);
    // Newtype map keys.
    let keyed: std::collections::BTreeMap<Id, String> =
        [(Id(3), "c".to_string()), (Id(1), "a".to_string())].into();
    roundtrip(&keyed);
    let named: std::collections::BTreeMap<Name, u8> =
        [(Name("b".into()), 2), (Name("a".into()), 1)].into();
    roundtrip(&named);
    roundtrip(&Record {
        distance: Meters(1.0),
        maybe: None,
        many: vec![],
        samples: Samples(vec![]),
    });
}

#[test]
fn a_newtype_is_written_as_its_inner_value() {
    assert_eq!(
        beve::to_vec(&Meters(2.5)).unwrap(),
        beve::to_vec(&2.5f64).unwrap()
    );
    let point = Point { x: 1, y: 2 };
    assert_eq!(
        beve::to_vec(&Wrapped(point.clone())).unwrap(),
        beve::to_vec(&point).unwrap()
    );
    // So a newtype reads what its inner type wrote, and the reverse.
    let inner = beve::to_vec(&vec!["x".to_string()]).unwrap();
    assert_eq!(
        beve::from_slice::<Names>(&inner).unwrap(),
        Names(vec!["x".into()])
    );
    let outer = beve::to_vec(&Meters(7.0)).unwrap();
    assert_eq!(beve::from_slice::<f64>(&outer).unwrap(), 7.0);
}

/// Typed string arrays and string keys reach a newtype through an adapter; a
/// unit variant named by one still reads.
#[test]
fn string_arrays_and_keys_read_as_unit_variants_and_newtypes() {
    use std::collections::BTreeMap;
    let typed = beve::to_vec_str_slice(&["Right", "Left"]);
    assert_eq!(
        beve::from_slice::<Vec<Side>>(&typed).unwrap(),
        [Side::Right, Side::Left]
    );
    assert_eq!(
        beve::from_reader_streaming::<_, Vec<Side>>(Cursor::new(&typed)).unwrap(),
        [Side::Right, Side::Left]
    );
    assert_eq!(
        beve::from_slice::<Vec<Name>>(&typed).unwrap(),
        [Name("Right".into()), Name("Left".into())]
    );

    let keyed: BTreeMap<String, u8> = [("Left".to_string(), 1)].into();
    let bytes = beve::to_vec(&keyed).unwrap();
    let want: BTreeMap<Side, u8> = [(Side::Left, 1)].into();
    assert_eq!(
        beve::from_slice::<BTreeMap<Side, u8>>(&bytes).unwrap(),
        want
    );
    assert_eq!(
        beve::from_reader_streaming::<_, BTreeMap<Side, u8>>(Cursor::new(&bytes)).unwrap(),
        want
    );
}
