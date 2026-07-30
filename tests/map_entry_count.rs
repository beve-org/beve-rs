//! A map body must match the header written ahead of it.
//!
//! The counterpart of `tests/skip_serializing_if.rs`, which covers the same
//! guard for structs. A known-length map writes its entry count into the header
//! on the first key and can never revise it, so a `len` that disagrees with the
//! body leaves a header promising entries the reader never finds. Serde's own
//! container impls always declare an honest `len`; only a hand-written
//! `Serialize` can get this wrong, and these tests pin that it is reported
//! rather than written out.

use serde::ser::{Error as _, SerializeMap};
use serde::{Deserialize, Serialize, Serializer};
use std::collections::BTreeMap;

/// Encodes `value` and returns the bytes, having first established that the
/// streaming writer agrees byte for byte, that `serialized_size` agrees, and
/// that `validate_slice` accepts the result.
///
/// `validate_slice` is the load-bearing check: `from_slice` stops at the end of
/// the first complete value and ignores what follows, so a round-trip alone
/// cannot see a body that overruns its header.
fn encoded<T: Serialize>(value: &T) -> Vec<u8> {
    let bytes = beve::to_vec(value).expect("buffered encode");

    let mut streamed = Vec::new();
    beve::to_writer_streaming(&mut streamed, value).expect("streaming encode");
    assert_eq!(streamed, bytes, "streaming diverged from buffered");

    assert_eq!(
        beve::serialized_size(value).expect("serialized_size"),
        bytes.len() as u64,
        "serialized_size disagreed with the encoder"
    );

    beve::validate_slice(&bytes).expect("encoded document must validate");
    bytes
}

// ---------------------------------------------------------------------------
// Honest maps are untouched.
// ---------------------------------------------------------------------------

#[test]
fn ordinary_maps_still_round_trip() {
    let map = BTreeMap::from([("a".to_string(), 1u32), ("b".to_string(), 2)]);
    assert_eq!(
        beve::from_slice::<BTreeMap<String, u32>>(&encoded(&map)).expect("decode"),
        map
    );

    let empty: BTreeMap<String, u32> = BTreeMap::new();
    assert_eq!(
        beve::from_slice::<BTreeMap<String, u32>>(&encoded(&empty)).expect("decode"),
        empty
    );

    // Integer keys take a different header and a different key-mode branch.
    let ints = BTreeMap::from([(1u64, "x".to_string()), (2, "y".to_string())]);
    assert_eq!(
        beve::from_slice::<BTreeMap<u64, String>>(&encoded(&ints)).expect("decode"),
        ints
    );
}

#[test]
fn nested_and_sequenced_maps_still_round_trip() {
    // A map reached as a sequence element is built at its own construction site
    // in the streaming writer, separate from the top-level one.
    let seq = vec![
        BTreeMap::from([("a".to_string(), 1u32)]),
        BTreeMap::new(),
        BTreeMap::from([("b".to_string(), 2), ("c".to_string(), 3)]),
    ];
    assert_eq!(
        beve::from_slice::<Vec<BTreeMap<String, u32>>>(&encoded(&seq)).expect("decode"),
        seq
    );

    let nested = BTreeMap::from([(
        "outer".to_string(),
        BTreeMap::from([("inner".to_string(), 1u32)]),
    )]);
    assert_eq!(
        beve::from_slice::<BTreeMap<String, BTreeMap<String, u32>>>(&encoded(&nested))
            .expect("decode"),
        nested
    );
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Flattened {
    id: u32,
    #[serde(flatten)]
    rest: BTreeMap<String, u32>,
}

#[test]
fn flatten_still_works() {
    // `#[serde(flatten)]` routes through `serialize_map(None)`, the
    // unknown-length path whose count is patched afterwards. It must stay clear
    // of the declared-length arm of the guard.
    let value = Flattened {
        id: 1,
        rest: BTreeMap::from([("extra".to_string(), 2u32)]),
    };
    let bytes = beve::to_vec(&value).expect("encode a flattened struct");
    beve::validate_slice(&bytes).expect("must validate");
    assert_eq!(
        beve::from_slice::<Flattened>(&bytes).expect("decode"),
        value
    );
}

// ---------------------------------------------------------------------------
// The guard: a hand-written `Serialize` that misreports `len`.
// ---------------------------------------------------------------------------

const KEYS: [&str; 4] = ["a", "b", "c", "d"];

/// Declares `declared` entries to `serialize_map` but writes `writes` of them.
/// `declared: None` selects the unknown-length path. If `dangling_key` is set,
/// one extra key is written with no value after it.
struct MiscountedMap {
    declared: Option<usize>,
    writes: usize,
    dangling_key: bool,
}

impl MiscountedMap {
    fn known(declared: usize, writes: usize) -> Self {
        Self {
            declared: Some(declared),
            writes,
            dangling_key: false,
        }
    }
}

impl Serialize for MiscountedMap {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut m = serializer.serialize_map(self.declared)?;
        for key in KEYS.iter().take(self.writes) {
            m.serialize_entry(key, &1u32)?;
        }
        if self.dangling_key {
            // A key with no value: the key bytes are already on the wire.
            m.serialize_key(KEYS[self.writes])?;
        }
        m.end().map_err(S::Error::custom)
    }
}

fn buffered_err(value: &MiscountedMap) -> String {
    beve::to_vec(value)
        .expect_err("a miscounted map must not encode")
        .to_string()
}

fn streamed_err(value: &MiscountedMap) -> String {
    let mut out = Vec::new();
    beve::to_writer_streaming(&mut out, value)
        .expect_err("a miscounted map must not stream")
        .to_string()
}

#[test]
fn declaring_more_entries_than_written_is_an_error() {
    // Header says 2, body has 1. This is the document `validate_slice` rejects
    // with "unexpected end of input" once it reaches a reader.
    let value = MiscountedMap::known(2, 1);
    for msg in [buffered_err(&value), streamed_err(&value)] {
        assert!(
            msg.contains("declared 2 entries") && msg.contains("serialized 1"),
            "unhelpful error: {msg}"
        );
    }
}

#[test]
fn declaring_fewer_entries_than_written_is_an_error() {
    let value = MiscountedMap::known(1, 2);
    for msg in [buffered_err(&value), streamed_err(&value)] {
        assert!(
            msg.contains("declared 1 entries") && msg.contains("serialized 2"),
            "unhelpful error: {msg}"
        );
    }
}

#[test]
fn declaring_entries_and_writing_none_is_an_error() {
    // Worth pinning separately: this case used to emit a *valid* empty object,
    // because the no-keys-written fallback writes its own zero-count header and
    // never consulted the declared length. Valid output, but not what the caller
    // said it would write, and the same mistake corrupts as soon as one entry
    // survives the filter.
    let value = MiscountedMap::known(2, 0);
    for msg in [buffered_err(&value), streamed_err(&value)] {
        assert!(
            msg.contains("declared 2 entries") && msg.contains("serialized 0"),
            "unhelpful error: {msg}"
        );
    }
}

#[test]
fn a_key_without_a_value_is_an_error() {
    let value = MiscountedMap {
        declared: Some(2),
        writes: 1,
        dangling_key: true,
    };
    for msg in [buffered_err(&value), streamed_err(&value)] {
        assert!(
            msg.contains("2 key(s) but 1 value(s)"),
            "unhelpful error: {msg}"
        );
    }
}

#[test]
fn a_dangling_key_is_caught_in_unknown_length_maps_too() {
    // The unknown-length path patches its count from the number of *values*, so
    // it cannot notice the orphaned key bytes on its own. Only the buffered
    // serializer accepts `serialize_map(None)`.
    let value = MiscountedMap {
        declared: None,
        writes: 1,
        dangling_key: true,
    };
    let msg = buffered_err(&value);
    assert!(
        msg.contains("2 key(s) but 1 value(s)"),
        "unhelpful error: {msg}"
    );
}

#[test]
fn the_guard_reaches_maps_nested_in_a_sequence() {
    let short = vec![MiscountedMap::known(2, 1)];

    let msg = beve::to_vec(&short)
        .expect_err("a miscounted map must not encode inside a sequence")
        .to_string();
    assert!(msg.contains("declared 2 entries"), "unhelpful error: {msg}");

    let mut out = Vec::new();
    let msg = beve::to_writer_streaming(&mut out, &short)
        .expect_err("a miscounted map must not stream inside a sequence")
        .to_string();
    assert!(msg.contains("declared 2 entries"), "unhelpful error: {msg}");
}

#[test]
fn an_honest_hand_written_impl_still_encodes() {
    // The guard must cost a correct impl nothing.
    let value = MiscountedMap::known(2, 2);
    let bytes = beve::to_vec(&value).expect("a matching count must encode");
    beve::validate_slice(&bytes).expect("an honest document must validate");
    assert_eq!(
        beve::from_slice::<BTreeMap<String, u32>>(&bytes).expect("decode"),
        BTreeMap::from([("a".to_string(), 1), ("b".to_string(), 1)])
    );

    let mut streamed = Vec::new();
    beve::to_writer_streaming(&mut streamed, &value).expect("a matching count must stream");
    assert_eq!(streamed, bytes);
}

#[test]
fn an_honest_unknown_length_map_still_encodes() {
    let value = MiscountedMap {
        declared: None,
        writes: 2,
        dangling_key: false,
    };
    let bytes = beve::to_vec(&value).expect("an unknown-length map must encode");
    beve::validate_slice(&bytes).expect("must validate");
    assert_eq!(
        beve::from_slice::<BTreeMap<String, u32>>(&bytes).expect("decode"),
        BTreeMap::from([("a".to_string(), 1), ("b".to_string(), 1)])
    );
}

#[test]
fn an_honest_empty_map_still_encodes() {
    for declared in [Some(0), None] {
        let value = MiscountedMap {
            declared,
            writes: 0,
            dangling_key: false,
        };
        let bytes = beve::to_vec(&value).expect("an empty map must encode");
        beve::validate_slice(&bytes).expect("must validate");
        assert!(
            beve::from_slice::<BTreeMap<String, u32>>(&bytes)
                .expect("decode")
                .is_empty()
        );
    }
}
