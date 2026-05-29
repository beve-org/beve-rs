//! Exactness tests for [`beve::serialized_size`] and
//! [`beve::serialized_size_with_options`].
//!
//! The central invariant is that the measured length always equals the number of
//! bytes [`beve::to_writer_streaming`] actually writes for the same value and
//! options. `serialized_size` is the dual of `to_writer_streaming` (not of
//! `to_vec`), so every case below asserts
//! `serialized_size(value) == to_writer_streaming(value).len()`.

use std::collections::BTreeMap;

use beve::{Complex, EnumEncoding, Error, SerializerOptions, TypedSlice};
use half::{bf16, f16};
use serde::ser::{SerializeMap, SerializeSeq};
use serde::{Deserialize, Serialize, Serializer};

/// Assert the measured size equals the streamed length under default options.
fn assert_size_eq<T: Serialize>(value: &T) {
    let mut buf = Vec::new();
    beve::to_writer_streaming(&mut buf, value).expect("streaming serialize should succeed");
    let measured = beve::serialized_size(value).expect("serialized_size should succeed");
    assert_eq!(
        measured,
        buf.len() as u64,
        "serialized_size ({measured}) disagreed with streamed length ({})",
        buf.len()
    );
}

/// Assert the measured size equals the streamed length under the given options.
fn assert_size_eq_opts<T: Serialize>(value: &T, opts: SerializerOptions) {
    let mut buf = Vec::new();
    beve::to_writer_streaming_with_options(&mut buf, value, opts)
        .expect("streaming serialize should succeed");
    let measured =
        beve::serialized_size_with_options(value, opts).expect("serialized_size should succeed");
    assert_eq!(measured, buf.len() as u64);
}

// ---------------------------------------------------------------------------
// Scalars and strings
// ---------------------------------------------------------------------------

#[test]
fn scalars() {
    assert_size_eq(&true);
    assert_size_eq(&false);
    assert_size_eq(&0u8);
    assert_size_eq(&255u8);
    assert_size_eq(&-1i8);
    assert_size_eq(&12345u16);
    assert_size_eq(&-12345i16);
    assert_size_eq(&123_456_789u32);
    assert_size_eq(&-123_456_789i32);
    assert_size_eq(&123_456_789_012_345u64);
    assert_size_eq(&-123i64);
    assert_size_eq(&u128::MAX);
    assert_size_eq(&i128::MIN);
    assert_size_eq(&12.625f32);
    assert_size_eq(&98.5f64);
    assert_size_eq(&'A');
    assert_size_eq(&'€'); // multi-byte char -> multi-byte string
    assert_size_eq(&()); // unit -> null
}

#[test]
fn strings() {
    assert_size_eq(&String::new());
    assert_size_eq(&"hello".to_string());
    assert_size_eq(&"héllo wörld — 日本語 🎉".to_string()); // multi-byte UTF-8
    assert_size_eq(&"x".repeat(63));
    assert_size_eq(&"y".repeat(64));
    assert_size_eq(&"z".repeat(16_383));
    assert_size_eq(&"w".repeat(16_384));
}

// ---------------------------------------------------------------------------
// Typed arrays: numeric (every type, empty + non-empty), bool, string
// ---------------------------------------------------------------------------

#[test]
fn numeric_typed_arrays() {
    assert_size_eq(&Vec::<u8>::new());
    assert_size_eq(&vec![1u8, 2, 3]);
    assert_size_eq(&Vec::<i8>::new());
    assert_size_eq(&vec![-1i8, 2, -3]);
    assert_size_eq(&Vec::<u16>::new());
    assert_size_eq(&vec![1u16, 65535]);
    assert_size_eq(&Vec::<i16>::new());
    assert_size_eq(&vec![-1i16, 2]);
    assert_size_eq(&Vec::<u32>::new());
    assert_size_eq(&vec![1u32, 2, 3, 4]);
    assert_size_eq(&Vec::<i32>::new());
    assert_size_eq(&vec![-1i32, 2, -3]);
    assert_size_eq(&Vec::<u64>::new());
    assert_size_eq(&vec![1u64, 2]);
    assert_size_eq(&Vec::<i64>::new());
    assert_size_eq(&vec![-1i64, 2]);
    assert_size_eq(&Vec::<u128>::new());
    assert_size_eq(&vec![1u128, u128::MAX]);
    assert_size_eq(&Vec::<i128>::new());
    assert_size_eq(&vec![i128::MIN, 0, i128::MAX]);
    assert_size_eq(&Vec::<f32>::new());
    assert_size_eq(&vec![1.0f32, -2.5, 3.25]);
    assert_size_eq(&Vec::<f64>::new());
    assert_size_eq(&vec![1.0f64, -2.5, 3.25]);
}

#[test]
fn bool_arrays_bit_packed() {
    assert_size_eq(&Vec::<bool>::new());
    assert_size_eq(&vec![true]);
    // Exactly 8 bits -> a single packed byte.
    assert_size_eq(&vec![true, false, true, true, false, false, true, false]);
    // 9 bits -> two bytes (the second only partially filled).
    assert_size_eq(&vec![true; 9]);
    assert_size_eq(&vec![false; 100]);
}

#[test]
fn string_arrays() {
    assert_size_eq(&Vec::<String>::new());
    assert_size_eq(&vec!["a".to_string(), "bb".to_string(), "ccc".to_string()]);
    assert_size_eq(&vec!["日本語".to_string(), String::new()]);
}

// ---------------------------------------------------------------------------
// Tuples, heterogeneous/generic arrays, nested structs, options
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct TupleStruct(i32, String, Vec<f64>);

#[test]
fn tuples_and_heterogeneous() {
    assert_size_eq(&(1i32, "two".to_string(), 3.0f64));
    assert_size_eq(&(true, ()));
    assert_size_eq(&TupleStruct(7, "k".to_string(), vec![1.0, 2.0]));
    // Outer seq whose elements are themselves seqs -> generic array of typed arrays.
    assert_size_eq(&vec![vec![1i32, 2], vec![3, 4, 5]]);
    // Heterogeneous tuple containing a nested struct.
    assert_size_eq(&("label".to_string(), Inner { a: 1.0, b: "z".into() }));
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
struct Inner {
    a: f64,
    b: String,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
struct Outer {
    id: u32,
    inner: Inner,
    tags: Vec<String>,
    values: Vec<f64>,
    maybe: Option<i32>,
    nope: Option<i32>,
}

fn sample_outer() -> Outer {
    Outer {
        id: 9,
        inner: Inner {
            a: 0.0,
            b: "nested".into(),
        },
        tags: vec!["x".into(), "yy".into()],
        values: vec![1.5, -2.5, 3.5],
        maybe: Some(5),
        nope: None,
    }
}

#[test]
fn structs_and_options() {
    assert_size_eq(&Inner {
        a: 1.5,
        b: "hi".into(),
    });
    assert_size_eq(&sample_outer());
    assert_size_eq(&Some(42i32));
    assert_size_eq(&Option::<i32>::None);
    assert_size_eq(&Some("str".to_string()));
}

// ---------------------------------------------------------------------------
// Maps: string keys and integer keys
// ---------------------------------------------------------------------------

#[test]
fn maps() {
    let empty: BTreeMap<String, i32> = BTreeMap::new();
    assert_size_eq(&empty);

    let mut string_keyed: BTreeMap<String, i32> = BTreeMap::new();
    string_keyed.insert("alpha".into(), 1);
    string_keyed.insert("beta".into(), 2);
    string_keyed.insert("日本語".into(), 3);
    assert_size_eq(&string_keyed);

    let mut signed_keyed: BTreeMap<i32, String> = BTreeMap::new();
    signed_keyed.insert(-1, "neg".into());
    signed_keyed.insert(2, "two".into());
    assert_size_eq(&signed_keyed);

    let mut unsigned_keyed: BTreeMap<u64, f64> = BTreeMap::new();
    unsigned_keyed.insert(10, 1.0);
    unsigned_keyed.insert(20, 2.0);
    assert_size_eq(&unsigned_keyed);
}

// ---------------------------------------------------------------------------
// Enums: all variant kinds, both encodings (options parity)
// ---------------------------------------------------------------------------

#[derive(Serialize)]
enum E {
    Unit,
    Newtype(i32),
    Tuple(i32, String),
    Struct { x: f64, y: f64 },
}

#[test]
fn enum_variants_default() {
    let variants = [
        E::Unit,
        E::Newtype(7),
        E::Tuple(1, "a".into()),
        E::Struct { x: 1.0, y: 2.0 },
    ];
    for v in &variants {
        assert_size_eq(v);
    }
}

#[test]
fn enum_variants_options_parity() {
    let number = SerializerOptions {
        enum_encoding: EnumEncoding::Number,
    };
    let string = SerializerOptions {
        enum_encoding: EnumEncoding::String,
    };
    let variants = [
        E::Unit,
        E::Newtype(7),
        E::Tuple(1, "a".into()),
        E::Struct { x: 1.0, y: 2.0 },
    ];
    for v in &variants {
        // Enum tag width differs between encodings, so the size must track each.
        assert_size_eq_opts(v, number);
        assert_size_eq_opts(v, string);
    }
}

// ---------------------------------------------------------------------------
// Extension types: bf16 / f16 / complex
// ---------------------------------------------------------------------------

#[test]
fn half_floats_and_complex() {
    assert_size_eq(&bf16::from_f32(1.5));
    assert_size_eq(&f16::from_f32(2.5));
    assert_size_eq(&Complex { re: 1.0f64, im: -2.0f64 });
    assert_size_eq(&Complex { re: 3.0f32, im: 4.0f32 });
    // Typed complex array (extension encoding).
    assert_size_eq(&vec![
        Complex { re: 1.0f64, im: 2.0 },
        Complex { re: 3.0, im: 4.0 },
    ]);
    assert_size_eq(&vec![
        Complex { re: 0.5f32, im: -0.5 },
        Complex { re: 1.5, im: 2.5 },
    ]);
}

// ---------------------------------------------------------------------------
// Bulk-slice fast path: TypedSlice bodies (and cross-check vs typed_slice_size)
// ---------------------------------------------------------------------------

#[test]
fn typed_slice_bodies() {
    let f64s = vec![1.0f64, 2.0, 3.0, 4.0];
    assert_size_eq(&TypedSlice(&f64s));

    let empty: Vec<f64> = Vec::new();
    assert_size_eq(&TypedSlice(&empty));

    let u32s = vec![10u32, 20, 30];
    assert_size_eq(&TypedSlice(&u32s));

    // A TypedSlice body measures identically to the analytic, traversal-free
    // `typed_slice_size`, and to the bulk writer's actual output length.
    let mut buf = Vec::new();
    beve::to_writer_typed_slice(&mut buf, &f64s).unwrap();
    let measured = beve::serialized_size(&TypedSlice(&f64s)).unwrap();
    assert_eq!(measured, buf.len() as u64);
    assert_eq!(measured, beve::typed_slice_size(&f64s));
}

// ---------------------------------------------------------------------------
// SIZE-prefix boundaries (1/2/4-byte thresholds: 63 vs 64, 16383 vs 16384)
// ---------------------------------------------------------------------------

#[test]
fn size_prefix_boundaries() {
    for n in [0usize, 1, 63, 64, 16_383, 16_384] {
        // Typed-array element-count prefix.
        let arr: Vec<u16> = vec![7u16; n];
        assert_size_eq(&arr);
        // String byte-length prefix.
        let s = "a".repeat(n);
        assert_size_eq(&s);
    }
}

// ---------------------------------------------------------------------------
// Path-distinction guard: serde_bytes (serialize_bytes, O(1)) vs Vec<u8>
// (serialize_seq, O(N)) must measure to the same byte length.
// ---------------------------------------------------------------------------

/// A byte blob that serializes through `serialize_bytes` (the bulk, O(1)-measure
/// path), in contrast to a bare `Vec<u8>` which serializes element-by-element via
/// `serialize_seq` (the O(N)-measure path). Mirrors what `&[u8]` / `serde_bytes`
/// would do, without pulling in an extra dependency.
struct Blob(Vec<u8>);

impl Serialize for Blob {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(&self.0)
    }
}

#[test]
fn bytes_path_matches_vec_path() {
    let raw: Vec<u8> = (0..200u32).map(|x| x as u8).collect();
    let as_vec = raw.clone();
    let as_bytes = Blob(raw);

    // Each individually matches its own streamed length.
    assert_size_eq(&as_vec);
    assert_size_eq(&as_bytes);

    // And they measure to the same length despite reaching it via different
    // encoder paths: `serialize_bytes` (one bulk write, O(1) measure) vs a bare
    // `Vec<u8>` -> `serialize_seq` (element-by-element, O(N) measure). This pins
    // the size equivalence so swapping the body type for performance cannot
    // silently change the on-wire length.
    let vec_size = beve::serialized_size(&as_vec).unwrap();
    let bytes_size = beve::serialized_size(&as_bytes).unwrap();
    assert_eq!(vec_size, bytes_size);
}

// ---------------------------------------------------------------------------
// Rejection parity: unknown-length containers error identically for the measure
// and the real write.
// ---------------------------------------------------------------------------

struct UnknownLenSeq(Vec<i32>);

impl Serialize for UnknownLenSeq {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // `None` length -> the streaming serializer rejects this up front.
        let mut seq = serializer.serialize_seq(None)?;
        for x in &self.0 {
            seq.serialize_element(x)?;
        }
        seq.end()
    }
}

struct UnknownLenMap(Vec<(String, i32)>);

impl Serialize for UnknownLenMap {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        for (k, v) in &self.0 {
            map.serialize_entry(k, v)?;
        }
        map.end()
    }
}

#[test]
fn rejection_parity_unknown_len_seq() {
    let value = UnknownLenSeq(vec![1, 2, 3]);
    let mut buf = Vec::new();
    let stream_err = beve::to_writer_streaming(&mut buf, &value).unwrap_err();
    let size_err = beve::serialized_size(&value).unwrap_err();
    assert!(
        matches!(stream_err, Error::Unsupported(_)),
        "streaming should reject unknown-length seq, got {stream_err:?}"
    );
    assert!(
        matches!(size_err, Error::Unsupported(_)),
        "serialized_size should reject unknown-length seq, got {size_err:?}"
    );
}

#[test]
fn rejection_parity_unknown_len_map() {
    let value = UnknownLenMap(vec![("a".into(), 1), ("b".into(), 2)]);
    let mut buf = Vec::new();
    let stream_err = beve::to_writer_streaming(&mut buf, &value).unwrap_err();
    let size_err = beve::serialized_size(&value).unwrap_err();
    assert!(
        matches!(stream_err, Error::Unsupported(_)),
        "streaming should reject unknown-length map, got {stream_err:?}"
    );
    assert!(
        matches!(size_err, Error::Unsupported(_)),
        "serialized_size should reject unknown-length map, got {size_err:?}"
    );
}

// ---------------------------------------------------------------------------
// Local validation of the REPE-style framing use case: measure the body, write
// a length prefix, then stream the body and confirm it is exactly that length
// and reads back correctly.
// ---------------------------------------------------------------------------

#[test]
fn framing_body_length_is_exact_and_round_trips() {
    let value = sample_outer();

    let body_len = beve::serialized_size(&value).unwrap();

    let mut frame = Vec::new();
    frame.extend_from_slice(&body_len.to_le_bytes()); // advertise length up front
    let header_len = frame.len();
    beve::to_writer_streaming(&mut frame, &value).unwrap();

    let written_body = (frame.len() - header_len) as u64;
    assert_eq!(
        written_body, body_len,
        "streamed body length must equal the advertised serialized_size"
    );

    let decoded: Outer = beve::from_slice(&frame[header_len..]).unwrap();
    assert_eq!(decoded, value);
}
