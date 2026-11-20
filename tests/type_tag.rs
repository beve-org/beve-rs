//! The deprecated Version 1 type-tag extension through every decoder.
//!
//! The spec lays it out as `HEADER 0x0E | SIZE | VALUE`: the variant's
//! positional index is a compressed SIZE, not a headered value. That is what
//! Glaze writes. beve 1.x through 4.x wrote a headered number or string there
//! instead; those documents are not spec-conformant and no longer decode.

use beve::wire;
use serde::Deserialize;
use std::io::Cursor;

#[derive(Deserialize, Debug, PartialEq)]
enum Variant {
    Flag(bool),
    Word(u32),
}

#[derive(Deserialize, Debug, PartialEq)]
struct Doc {
    v: Variant,
    after: u8,
}

/// The reproduction from the issue: index 1 (SIZE `0x04`), then a u32.
const WORD: [u8; 7] = [0x0E, 0x04, 0x51, 0xEF, 0xBE, 0xAD, 0xDE];

/// `{"v": <WORD>, "after": 5}`, so a misread index desynchronizes the sibling.
fn doc_bytes() -> Vec<u8> {
    let mut out = Vec::new();
    wire::write_object_header(&mut out, 2);
    wire::write_object_key(&mut out, "v");
    out.extend_from_slice(&WORD);
    wire::write_object_key(&mut out, "after");
    beve::to_writer(&mut out, &5u8).unwrap();
    out
}

#[test]
fn skip_value_reads_the_index_as_a_size() {
    let mut pos = 0;
    beve::skip_value(&WORD, &mut pos).unwrap();
    assert_eq!(pos, WORD.len());

    let doc = doc_bytes();
    let mut pos = 0;
    beve::skip_value(&doc, &mut pos).unwrap();
    assert_eq!(pos, doc.len());
}

#[test]
fn every_decoder_reads_the_index_as_a_size() {
    let expected = Variant::Word(0xDEAD_BEEF);
    assert_eq!(beve::from_slice::<Variant>(&WORD).unwrap(), expected);
    assert_eq!(
        beve::from_reader_streaming::<_, Variant>(Cursor::new(&WORD)).unwrap(),
        expected
    );
    assert_eq!(
        beve::beve_slice_to_json_string(&WORD).unwrap(),
        "3735928559"
    );

    let doc = doc_bytes();
    let expected = Doc {
        v: Variant::Word(0xDEAD_BEEF),
        after: 5,
    };
    beve::validate_slice(&doc).unwrap();
    assert_eq!(beve::from_slice::<Doc>(&doc).unwrap(), expected);
    assert_eq!(
        beve::from_reader_streaming::<_, Doc>(Cursor::new(&doc)).unwrap(),
        expected
    );
    assert_eq!(
        beve::beve_slice_to_json_string(&doc).unwrap(),
        r#"{"v":3735928559,"after":5}"#
    );
    assert_eq!(beve::from_field::<u8>(&doc, "/after").unwrap(), 5);
}

/// An index past 63 takes a wider SIZE. Skipping and transcoding need no enum
/// to know the variant count, so they see any index.
#[test]
fn a_multi_byte_index_is_read_whole() {
    for index in [64u64, 16_384, 1 << 30] {
        let mut bytes = vec![0x0E];
        wire::write_size(&mut bytes, index);
        beve::to_writer(&mut bytes, &true).unwrap();

        let mut pos = 0;
        beve::skip_value(&bytes, &mut pos).unwrap();
        assert_eq!(pos, bytes.len(), "index {index}");
        assert_eq!(
            beve::beve_slice_to_json_string(&bytes).unwrap(),
            "true",
            "index {index}"
        );
    }
}

/// beve 1.x through 4.x wrote the index as a headered u32. Read as the spec
/// says, `0x51 0x01` is SIZE 84, a variant this enum does not have.
#[test]
fn the_headered_index_beve_4_wrote_is_rejected() {
    let bytes = [0x0E, 0x51, 0x01, 0x00, 0x00, 0x00, 0x11, 0x07];
    assert!(beve::from_slice::<Variant>(&bytes).is_err());
    assert!(beve::from_reader_streaming::<_, Variant>(Cursor::new(&bytes)).is_err());
}

#[derive(Deserialize, Debug)]
enum Unit {
    Alpha,
    Beta,
}

/// What beve 4.0.0 wrote for `vec![Unit::Beta, Unit::Alpha]`: the leading unit
/// variant under the type tag with a headered index and a null, then a bare
/// index. The type-tagged element is not spec form, so the sequence no longer
/// decodes.
#[test]
fn a_version_4_unit_variant_sequence_is_rejected() {
    let bytes = [
        0x05, 0x08, 0x0E, 0x51, 0x01, 0x00, 0x00, 0x00, 0x00, 0x51, 0x00, 0x00, 0x00, 0x00,
    ];
    assert!(beve::from_slice::<Vec<Unit>>(&bytes).is_err());
    assert!(beve::from_reader_streaming::<_, Vec<Unit>>(Cursor::new(&bytes)).is_err());
    assert!(beve::validate_slice(&bytes).is_err());
}

#[test]
fn a_truncated_index_is_an_error() {
    // A 2-byte SIZE cut after its first byte.
    let bytes = [0x0E, 0x01];
    let mut pos = 0;
    assert!(beve::skip_value(&bytes, &mut pos).is_err());
    assert!(beve::from_slice::<Variant>(&bytes).is_err());
    assert!(beve::from_reader_streaming::<_, Variant>(Cursor::new(&bytes)).is_err());
    assert!(beve::beve_slice_to_json_string(&bytes).is_err());
}
