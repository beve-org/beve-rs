//! `#[serde(skip_serializing_if = ...)]` support.
//!
//! Serde's contract is that the `len` handed to `serialize_struct` is "the
//! number of data fields that will be serialized", excluding skipped ones, and
//! `serde_derive` honors it. So the object header a struct writes up front is
//! already correct by the time `skip_field` is reported, and `skip_field` is a
//! no-op. These tests pin that, and pin the `end` guard that catches a
//! hand-written `Serialize` declaring a `len` it does not deliver.

use serde::ser::{Error as _, SerializeStruct};
use serde::{Deserialize, Serialize, Serializer};
use std::collections::BTreeMap;

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Optional {
    always: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    sometimes: Option<u32>,
}

/// The shape left once `sometimes` is skipped. Encoding this must be
/// byte-identical to encoding `Optional { sometimes: None, .. }`.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct OnlyAlways {
    always: u32,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Flags {
    name: String,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    enabled: bool,
}

/// A skip on the *first* field, where a key/value ordering slip would show.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct SkipFirst {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    leading: Option<u32>,
    tail: u32,
}

/// A skip between two written fields.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct SkipMiddle {
    head: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    middle: Option<u32>,
    tail: u32,
}

/// Every field skipped, which must emit a `len 0` object header.
#[derive(Serialize, Deserialize, Debug, PartialEq, Default)]
struct AllSkipped {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    a: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    b: Option<u32>,
}

/// The shapes the three above reduce to once their skips are applied.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct OnlyTail {
    tail: u32,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct HeadTail {
    head: u32,
    tail: u32,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Empty {}

/// A struct that both skips a field of its own and holds one that skips.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Outer {
    inner: Optional,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    trailing: Option<u32>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
enum Variant {
    Detail {
        id: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        note: Option<String>,
    },
}

/// Encodes `value` and returns the bytes, having first established everything
/// that must hold of them regardless of the case under test:
///
/// - the streaming writer agrees byte for byte (it cannot revise a header it has
///   already flushed, so it is the path most at risk of desyncing),
/// - `serialized_size` agrees,
/// - and `validate_slice` accepts the document.
///
/// That last one is the load-bearing check. `from_slice` stops at the end of the
/// first complete value and ignores whatever follows (`src/de.rs`), so a
/// round-trip alone cannot see a body that overruns its header; only
/// `validate_slice` reports the trailing data. The 5.0.0 changelog cited exactly
/// that detector for the corruption this area is about.
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

#[test]
fn skipped_option_is_omitted_and_round_trips() {
    let value = Optional {
        always: 7,
        sometimes: None,
    };
    let bytes = encoded(&value);

    // The skipped field leaves no trace: the document is exactly the one-field
    // struct, not a two-field header with a missing body.
    assert_eq!(bytes, encoded(&OnlyAlways { always: 7 }));

    let back: Optional = beve::from_slice(&bytes).expect("decode");
    assert_eq!(back, value);
}

#[test]
fn present_option_is_written() {
    let value = Optional {
        always: 7,
        sometimes: Some(9),
    };
    let bytes = encoded(&value);
    let back: Optional = beve::from_slice(&bytes).expect("decode");
    assert_eq!(back, value);
    assert_ne!(bytes, encoded(&OnlyAlways { always: 7 }));
}

#[test]
fn skipping_a_bool_matches_the_option_case() {
    let value = Flags {
        name: "off".into(),
        enabled: false,
    };
    let back: Flags = beve::from_slice(&encoded(&value)).expect("decode");
    assert_eq!(back, value);
}

#[test]
fn a_skip_anywhere_in_the_field_order_is_clean() {
    // A trailing skip is the easy case: the header shrinks and the body simply
    // stops early. A leading or middle skip is where a key/value pairing slip
    // would surface, so each is compared against the shape it reduces to.
    let first = SkipFirst {
        leading: None,
        tail: 4,
    };
    assert_eq!(encoded(&first), encoded(&OnlyTail { tail: 4 }));
    assert_eq!(
        beve::from_slice::<SkipFirst>(&encoded(&first)).expect("decode"),
        first
    );

    let middle = SkipMiddle {
        head: 1,
        middle: None,
        tail: 2,
    };
    assert_eq!(encoded(&middle), encoded(&HeadTail { head: 1, tail: 2 }));
    assert_eq!(
        beve::from_slice::<SkipMiddle>(&encoded(&middle)).expect("decode"),
        middle
    );
}

#[test]
fn skipping_every_field_emits_an_empty_object() {
    let value = AllSkipped::default();
    let bytes = encoded(&value);

    // A string-keyed object header with a count of zero, and nothing after it.
    assert_eq!(bytes, encoded(&Empty {}));
    assert_eq!(
        beve::from_slice::<AllSkipped>(&bytes).expect("decode"),
        value
    );
}

#[test]
fn struct_variants_skip_too() {
    let value = Variant::Detail { id: 3, note: None };
    let back: Variant = beve::from_slice(&encoded(&value)).expect("decode");
    assert_eq!(back, value);
}

#[test]
fn structs_skipping_inside_a_sequence_are_clean() {
    // Sequence elements reach a *separate* pair of struct-serializer
    // construction sites in the streaming writer, one for plain structs and one
    // for struct variants. Mixing skipped and present fields in the same
    // sequence keeps a per-element header from being right by accident.
    let structs = vec![
        Optional {
            always: 1,
            sometimes: None,
        },
        Optional {
            always: 2,
            sometimes: Some(3),
        },
        Optional {
            always: 4,
            sometimes: None,
        },
    ];
    assert_eq!(
        beve::from_slice::<Vec<Optional>>(&encoded(&structs)).expect("decode"),
        structs
    );

    let variants = vec![
        Variant::Detail { id: 1, note: None },
        Variant::Detail {
            id: 2,
            note: Some("here".into()),
        },
        Variant::Detail { id: 3, note: None },
    ];
    assert_eq!(
        beve::from_slice::<Vec<Variant>>(&encoded(&variants)).expect("decode"),
        variants
    );
}

#[test]
fn a_nested_struct_may_skip_independently_of_its_parent() {
    let value = Outer {
        inner: Optional {
            always: 5,
            sometimes: None,
        },
        trailing: None,
    };
    assert_eq!(
        beve::from_slice::<Outer>(&encoded(&value)).expect("decode"),
        value
    );
}

// ---------------------------------------------------------------------------
// The `end` guard: a hand-written `Serialize` that misreports `len`.
// ---------------------------------------------------------------------------

/// Distinct `&'static str` keys, so an honest `Miscounted` is a well-formed
/// object rather than one with repeated keys.
const KEYS: [&str; 4] = ["a", "b", "c", "d"];

/// Declares `len` fields to `serialize_struct` but writes `writes` of them.
/// Serde's derive can never produce this; only a hand-written impl can.
struct Miscounted {
    declared: usize,
    writes: usize,
}

impl Serialize for Miscounted {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut s = serializer.serialize_struct("Miscounted", self.declared)?;
        for key in KEYS.iter().take(self.writes) {
            s.serialize_field(key, &1u32)?;
        }
        // Report the remainder as skipped, the way a conditional impl would.
        for key in KEYS.iter().take(self.declared).skip(self.writes) {
            s.skip_field(key)?;
        }
        s.end().map_err(S::Error::custom)
    }
}

#[test]
fn declaring_more_fields_than_written_is_an_error() {
    // Declaring 2 and writing 1 is precisely the corruption `skip_field` used to
    // refuse outright. It is still caught, just at `end`, where the tally is known.
    let err = beve::to_vec(&Miscounted {
        declared: 2,
        writes: 1,
    })
    .expect_err("a short body must not encode");
    let msg = err.to_string();
    assert!(
        msg.contains("declared 2 field(s)") && msg.contains("serialized 1"),
        "unhelpful error: {msg}"
    );
}

#[test]
fn declaring_fewer_fields_than_written_is_an_error() {
    let err = beve::to_vec(&Miscounted {
        declared: 1,
        writes: 2,
    })
    .expect_err("an overlong body must not encode");
    assert!(
        err.to_string().contains("declared 1 field(s)"),
        "unhelpful error: {err}"
    );
}

#[test]
fn an_honest_hand_written_impl_still_encodes() {
    // The guard must not cost a correct impl anything: the document validates
    // and decodes to exactly the fields that were written.
    let bytes = beve::to_vec(&Miscounted {
        declared: 2,
        writes: 2,
    })
    .expect("a matching count must encode");
    beve::validate_slice(&bytes).expect("an honest document must validate");

    let fields: BTreeMap<String, u32> = beve::from_slice(&bytes).expect("decode");
    assert_eq!(
        fields,
        BTreeMap::from([("a".to_string(), 1), ("b".to_string(), 1)])
    );
}

#[test]
fn an_honest_impl_that_skips_everything_encodes_an_empty_object() {
    let bytes = beve::to_vec(&Miscounted {
        declared: 0,
        writes: 0,
    })
    .expect("an all-skipped struct must encode");
    beve::validate_slice(&bytes).expect("an empty object must validate");
    assert_eq!(bytes, encoded(&Empty {}));
}

#[test]
fn the_streaming_serializer_guards_the_count_too() {
    let mut out = Vec::new();
    let err = beve::to_writer_streaming(
        &mut out,
        &Miscounted {
            declared: 2,
            writes: 1,
        },
    )
    .expect_err("a short body must not stream");
    assert!(
        err.to_string().contains("declared 2 field(s)"),
        "unhelpful error: {err}"
    );
}

#[test]
fn the_guard_reaches_structs_nested_in_a_sequence() {
    // Sequence elements build their struct serializer at their own call sites,
    // separate from the top-level ones exercised above. A miscount there must
    // not slip through.
    let short = vec![Miscounted {
        declared: 2,
        writes: 1,
    }];

    let err = beve::to_vec(&short).expect_err("a short body must not encode in a sequence");
    assert!(
        err.to_string().contains("declared 2 field(s)"),
        "unhelpful error: {err}"
    );

    let mut out = Vec::new();
    let err = beve::to_writer_streaming(&mut out, &short)
        .expect_err("a short body must not stream in a sequence");
    assert!(
        err.to_string().contains("declared 2 field(s)"),
        "unhelpful error: {err}"
    );
}
