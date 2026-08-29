//! An empty collection inside an enum variant still gets an array header.
//!
//! These three shapes are the ones a property test found in September 2025,
//! when a sequence that ended without ever serializing an element left
//! `SeqMode::Unknown` and wrote no header at all. The encoder resolves a
//! sequence's mode from its first element, so an empty sequence is the one case
//! that reaches the end with nothing to resolve, and each enclosing form
//! (newtype variant, newtype variant holding another enum, struct variant)
//! arrives there by a different route.
//!
//! The requirement is worth stating in the open rather than leaving to a
//! generated case: an empty collection is a value, not an absence, and it has
//! to occupy bytes on the wire. `fuzz/seeds/roundtrip/` keeps the generated
//! form of these too, but a seed file says nothing about what it is for.
//!
//! The unenclosed cases live with their own kind: `typed_slice.rs` covers the
//! empty typed slice and `complex_slice.rs` the empty complex one.

use beve::Value;
use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq, Serialize, Deserialize)]
enum Payload {
    Telemetry(Vec<f64>),
    MetaOnly(Meta),
    Snapshot { label: String, counters: Vec<u32> },
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
enum Meta {
    Codes(Vec<u32>),
}

/// Is there a well-formed empty array anywhere in this document?
///
/// The alternative is pinning the exact byte offset of the header, which would
/// fail the next time anything else about the encoding moves. What is being
/// asserted is that the empty collection survived as an empty collection, and
/// the decoder having produced an `Array([])` at all means a header was there
/// to produce it from.
fn has_empty_array(value: &Value) -> bool {
    match value {
        Value::Array(items) => items.is_empty() || items.iter().any(has_empty_array),
        Value::Object(entries) => entries.values().any(has_empty_array),
        _ => false,
    }
}

/// Encode, then require the bytes to validate, to decode back to the original,
/// and to still contain the empty array.
fn assert_empty_collection_survives<T>(value: &T)
where
    T: Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
{
    let bytes = beve::to_vec(value).expect("encode");
    assert!(
        !bytes.is_empty(),
        "an empty collection still occupies bytes"
    );

    beve::validate_slice(&bytes)
        .unwrap_or_else(|e| panic!("encoding of {value:?} failed validation: {e:?}"));

    let decoded: T =
        beve::from_slice(&bytes).unwrap_or_else(|e| panic!("decoding {value:?} back: {e:?}"));
    assert_eq!(&decoded, value);

    let as_value: Value = beve::from_slice(&bytes).expect("decode as Value");
    assert!(
        has_empty_array(&as_value),
        "no empty array in the decoded document for {value:?}; the header is missing"
    );
}

#[test]
fn an_empty_sequence_in_a_newtype_variant_keeps_its_header() {
    assert_empty_collection_survives(&Payload::Telemetry(Vec::new()));
}

#[test]
fn an_empty_sequence_in_a_nested_enum_keeps_its_header() {
    assert_empty_collection_survives(&Payload::MetaOnly(Meta::Codes(Vec::new())));
}

#[test]
fn an_empty_sequence_in_a_struct_variant_keeps_its_header() {
    assert_empty_collection_survives(&Payload::Snapshot {
        label: "a".to_string(),
        counters: Vec::new(),
    });
}
