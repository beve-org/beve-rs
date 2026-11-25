//! `beve::typed::option::*`, `beve::typed::vec::*`, `beve::typed::usize` and
//! `beve::typed::isize`: round-trips through every decoder and JSON, the bytes
//! plain serde writes where it already writes a typed array, and a typed array
//! where it does not (an empty one).

use serde::{Deserialize, Serialize};
use std::io::Cursor;

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
struct Arrays {
    #[serde(with = "beve::typed::option::f64")]
    maybe: Option<Vec<f64>>,
    #[serde(with = "beve::typed::vec::f64")]
    rows: Vec<Vec<f64>>,
    #[serde(with = "beve::typed::vec::i32")]
    counts: Vec<Vec<i32>>,
    #[serde(with = "beve::typed::usize")]
    index: Vec<usize>,
    #[serde(with = "beve::typed::isize")]
    offset: Vec<isize>,
}

/// The same fields without the helpers.
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
struct Plain {
    maybe: Option<Vec<f64>>,
    rows: Vec<Vec<f64>>,
    counts: Vec<Vec<i32>>,
    index: Vec<usize>,
    offset: Vec<isize>,
}

fn full() -> Arrays {
    Arrays {
        maybe: Some(vec![0.5, -1.5]),
        rows: vec![vec![1.0, 2.0], vec![3.0]],
        counts: vec![vec![-1, 2, 3]],
        index: vec![0, 7, usize::MAX],
        offset: vec![isize::MIN, -1, 5],
    }
}

fn empty() -> Arrays {
    Arrays {
        maybe: Some(vec![]),
        rows: vec![vec![], vec![4.0], vec![]],
        counts: vec![],
        index: vec![],
        offset: vec![],
    }
}

fn plain(a: &Arrays) -> Plain {
    Plain {
        maybe: a.maybe.clone(),
        rows: a.rows.clone(),
        counts: a.counts.clone(),
        index: a.index.clone(),
        offset: a.offset.clone(),
    }
}

fn roundtrip(a: &Arrays) {
    let bytes = beve::to_vec(a).unwrap();
    let mut streamed = Vec::new();
    beve::to_writer_streaming(&mut streamed, a).unwrap();
    assert_eq!(streamed, bytes, "to_writer_streaming");
    assert_eq!(beve::serialized_size(a).unwrap(), bytes.len() as u64);
    let value: beve::Value = beve::from_slice(&bytes).unwrap();
    assert_eq!(&beve::from_value::<Arrays>(value).unwrap(), a, "from_value");
    assert_eq!(&beve::from_slice::<Arrays>(&bytes).unwrap(), a);
    assert_eq!(
        &beve::from_reader_streaming::<_, Arrays>(Cursor::new(&bytes)).unwrap(),
        a
    );
    let json = serde_json::to_string(a).unwrap();
    assert_eq!(
        json,
        serde_json::to_string(&plain(a)).unwrap(),
        "JSON is unchanged"
    );
    assert_eq!(&serde_json::from_str::<Arrays>(&json).unwrap(), a);
}

#[test]
fn nested_and_pointer_width_arrays_roundtrip() {
    roundtrip(&full());
    roundtrip(&empty());
    roundtrip(&Arrays {
        maybe: None,
        ..full()
    });
}

#[test]
fn non_empty_arrays_are_the_bytes_plain_serde_writes() {
    let a = full();
    assert_eq!(beve::to_vec(&a).unwrap(), beve::to_vec(&plain(&a)).unwrap());
    let none = Arrays { maybe: None, ..a };
    assert_eq!(
        beve::to_vec(&none).unwrap(),
        beve::to_vec(&plain(&none)).unwrap()
    );
}

#[test]
fn empty_arrays_are_written_typed() {
    let bytes = beve::to_vec(&empty()).unwrap();
    let field = |p: &str| beve::field_bytes(&bytes, p).unwrap();
    for (p, want) in [
        ("/maybe", &[][..]),
        ("/rows/0", &[]),
        ("/rows/1", &[4.0]),
        ("/rows/2", &[]),
    ] {
        assert_eq!(
            beve::read_typed_slice::<f64>(field(p)).unwrap(),
            want,
            "{p}"
        );
    }
    assert!(
        beve::read_typed_slice::<u64>(field("/index"))
            .unwrap()
            .is_empty()
    );
    assert!(
        beve::read_typed_slice::<i64>(field("/offset"))
            .unwrap()
            .is_empty()
    );

    // Plain serde writes them as generic arrays, which a strict read refuses.
    let plain_bytes = beve::to_vec(&plain(&empty())).unwrap();
    for p in ["/maybe", "/rows/0", "/index"] {
        let f = beve::field_bytes(&plain_bytes, p).unwrap();
        assert!(beve::read_typed_slice::<f64>(f).is_err(), "{p}");
    }
}

#[test]
fn what_plain_serde_wrote_still_decodes() {
    for a in [full(), empty()] {
        let bytes = beve::to_vec(&plain(&a)).unwrap();
        assert_eq!(beve::from_slice::<Arrays>(&bytes).unwrap(), a);
        assert_eq!(
            beve::from_reader_streaming::<_, Arrays>(Cursor::new(&bytes)).unwrap(),
            a
        );
    }
}
