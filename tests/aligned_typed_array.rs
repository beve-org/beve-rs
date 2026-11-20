//! Aligned typed arrays through every decoder.
//!
//! An aligned array (`HEADER 0x5C | NUMERIC_HEADER | SIZE | PADDING_LENGTH |
//! PADDING | DATA`) shares the typed-array category with boolean and string
//! arrays, told apart by the 3-bit field above it: 0 boolean, 1 string, 2
//! aligned, 3 through 7 undefined. Decoders that tested bit 5 alone read an
//! aligned array as a boolean one and desynchronized. Every decoder must read it
//! as the numeric array it wraps, and reject the undefined kinds.

use beve::{Value, wire};
use half::{bf16, f16};
use serde::{Deserialize, Serialize};
use std::io::Cursor;

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Doc {
    samples: Vec<f64>,
    after: u32,
}

/// `{"samples": <aligned f64 array>, "after": 7}`, and the same document with an
/// ordinary typed array, which every decoder must read identically.
fn aligned_and_plain_docs() -> (Vec<u8>, Vec<u8>, Doc) {
    let doc = Doc {
        samples: vec![1.0, -2.5, 3.25],
        after: 7,
    };
    let mut aligned = Vec::new();
    wire::write_object_header(&mut aligned, 2);
    wire::write_object_key(&mut aligned, "samples");
    beve::write_aligned_typed_slice(&mut aligned, &doc.samples);
    wire::write_object_key(&mut aligned, "after");
    beve::to_writer(&mut aligned, &doc.after).unwrap();
    (aligned, beve::to_vec(&doc).unwrap(), doc)
}

#[test]
fn skip_value_steps_over_an_aligned_array() {
    // The reproduction from the issue: two f64, padding length 3.
    let mut v = vec![0x5C, 0x64, 0x08, 0x03, 0xAA, 0xAA, 0xAA];
    v.extend_from_slice(&1.0f64.to_le_bytes());
    v.extend_from_slice(&2.0f64.to_le_bytes());
    let mut pos = 0;
    beve::skip_value(&v, &mut pos).unwrap();
    assert_eq!(pos, v.len());

    let (aligned, _, _) = aligned_and_plain_docs();
    let mut pos = 0;
    beve::skip_value(&aligned, &mut pos).unwrap();
    assert_eq!(pos, aligned.len());
}

#[test]
fn every_decoder_reads_an_aligned_array_as_its_numeric_array() {
    let (aligned, plain, doc) = aligned_and_plain_docs();
    assert_ne!(aligned, plain);

    beve::validate_slice(&aligned).unwrap();
    assert_eq!(beve::from_slice::<Doc>(&aligned).unwrap(), doc);
    assert_eq!(
        beve::from_reader_streaming::<_, Doc>(Cursor::new(&aligned)).unwrap(),
        doc
    );
    assert_eq!(
        beve::from_slice::<Value>(&aligned).unwrap(),
        beve::from_slice::<Value>(&plain).unwrap()
    );
    assert_eq!(
        beve::beve_slice_to_json_string(&aligned).unwrap(),
        beve::beve_slice_to_json_string(&plain).unwrap()
    );
    assert_eq!(beve::from_field::<u32>(&aligned, "/after").unwrap(), 7);
    assert_eq!(
        beve::from_field_slice::<Vec<f64>>(&aligned, "/samples", 1, 2).unwrap(),
        vec![-2.5, 3.25]
    );
}

/// A field using the bulk `typed::*` helper declines the aligned shape and
/// falls back to the element-wise path, which must then succeed.
#[test]
fn a_bulk_helper_field_reads_an_aligned_array() {
    #[derive(Deserialize, Debug, PartialEq)]
    struct Bulk {
        #[serde(with = "beve::typed::f64")]
        samples: Vec<f64>,
        after: u32,
    }
    let (aligned, _, doc) = aligned_and_plain_docs();
    let expected = Bulk {
        samples: doc.samples,
        after: doc.after,
    };
    assert_eq!(beve::from_slice::<Bulk>(&aligned).unwrap(), expected);
    assert_eq!(
        beve::from_reader_streaming::<_, Bulk>(Cursor::new(&aligned)).unwrap(),
        expected
    );
}

#[cfg(feature = "mat")]
#[test]
fn mat_export_reads_an_aligned_array_as_its_numeric_array() {
    use beve::{MatV73Options, RootBinding};

    let (aligned, plain, _) = aligned_and_plain_docs();
    let options = MatV73Options::default();
    let convert = |bytes: &[u8]| {
        beve::beve_slice_to_mat_v73_bytes(bytes, RootBinding::NamedVariable("doc"), &options)
            .unwrap()
    };
    assert_eq!(convert(&aligned), convert(&plain));
}

/// Every element type, at every padding length the alignment can call for, so
/// each decoder sees the padding run at each width.
#[test]
fn aligned_arrays_of_every_scalar_round_trip() {
    fn check<T>(values: &[T])
    where
        T: beve::BeveTypedSlice
            + Serialize
            + serde::de::DeserializeOwned
            + PartialEq
            + std::fmt::Debug,
    {
        for prefix in 0..16 {
            let mut bytes = vec![0u8; prefix];
            beve::write_aligned_typed_slice(&mut bytes, values);
            let body = &bytes[prefix..];

            let decoded: Vec<T> = beve::from_slice(body).unwrap();
            assert_eq!(decoded, values, "prefix={prefix}");
            let streamed: Vec<T> = beve::from_reader_streaming(Cursor::new(body)).unwrap();
            assert_eq!(streamed, values, "prefix={prefix}");
            let mut pos = 0;
            beve::skip_value(body, &mut pos).unwrap();
            assert_eq!(pos, body.len(), "prefix={prefix}");
            assert_eq!(
                beve::beve_slice_to_json_string(body).unwrap(),
                beve::beve_slice_to_json_string(&beve::to_vec(&values).unwrap()).unwrap(),
                "prefix={prefix}"
            );
        }
    }
    check(&[1i8, -2, 3]);
    check(&[1i16, -2, 3]);
    check(&[1i32, -2, 3]);
    check(&[1i64, -2, 3]);
    check(&[1i128, -2, 3]);
    check(&[1u8, 2, 3]);
    check(&[1u16, 2, 3]);
    check(&[1u32, 2, 3]);
    check(&[1u64, 2, 3]);
    check(&[1u128, 2, 3]);
    check(&[1.5f32, -2.0]);
    check(&[1.5f64, -2.0]);
    check(&[f16::from_f32(1.5), f16::from_f32(-2.0)]);
    check(&[bf16::from_f32(1.5), bf16::from_f32(-2.0)]);
    check::<f64>(&[]);
}

/// The bulk typed-slice readers read an aligned array as its numeric array too.
#[test]
fn the_bulk_readers_read_an_aligned_array() {
    let values = [1.0f64, -2.5, 3.25];
    let bytes = beve::to_vec_aligned_typed_slice(&values);
    assert_eq!(beve::read_typed_slice::<f64>(&bytes).unwrap(), values);
    assert_eq!(
        beve::read_typed_slice_from_reader::<f64, _>(Cursor::new(&bytes)).unwrap(),
        values
    );
    assert!(matches!(
        beve::read_typed_slice::<f32>(&bytes),
        Err(beve::Error::Mismatch(_))
    ));
}

/// Every decoder must reject `bytes`.
fn rejected_everywhere(bytes: &[u8], what: &str) {
    let mut pos = 0;
    assert!(beve::skip_value(bytes, &mut pos).is_err(), "{what}: skip");
    assert!(beve::validate_slice(bytes).is_err(), "{what}: validate");
    assert!(beve::from_slice::<Value>(bytes).is_err(), "{what}: Value");
    assert!(
        beve::from_reader_streaming::<_, Value>(Cursor::new(bytes)).is_err(),
        "{what}: streaming"
    );
    assert!(
        beve::beve_slice_to_json_string(bytes).is_err(),
        "{what}: JSON"
    );
    assert!(
        beve::from_field_slice::<Vec<f64>>(bytes, "", 0, 0).is_err(),
        "{what}: field slice"
    );
    #[cfg(feature = "mat")]
    {
        let options = beve::MatV73Options::default();
        assert!(
            beve::beve_slice_to_mat_v73_bytes(
                bytes,
                beve::RootBinding::NamedVariable("x"),
                &options
            )
            .is_err(),
            "{what}: MAT"
        );
    }
}

/// The padding aligns `DATA` to the element, so it is shorter than one element.
/// Glaze rejects a longer run, and so must every decoder here.
#[test]
fn padding_as_wide_as_an_element_is_rejected() {
    // f64 with 8 padding bytes, one past the most an 8-byte element needs.
    let mut f64_bytes = vec![0x5C, 0x64, 0x04, 0x08];
    f64_bytes.extend_from_slice(&[0; 8]);
    f64_bytes.extend_from_slice(&1.0f64.to_le_bytes());
    rejected_everywhere(&f64_bytes, "f64 padding 8");
    assert!(beve::read_aligned_typed_slice::<f64>(&f64_bytes).is_err());
    assert!(beve::read_typed_slice::<f64>(&f64_bytes).is_err());
    assert!(beve::read_typed_slice_from_reader::<f64, _>(Cursor::new(&f64_bytes)).is_err());

    // u8 needs no padding at all.
    let u8_bytes = [0x5C, 0x14, 0x04, 0x01, 0x00, 0x07];
    rejected_everywhere(&u8_bytes, "u8 padding 1");
    assert!(beve::read_aligned_typed_slice::<u8>(&u8_bytes).is_err());
}

/// Kinds 3 through 7 of the boolean/string/aligned category are undefined. Each
/// decoder must reject them rather than read them as a boolean array.
#[test]
fn undefined_kinds_are_rejected_by_every_decoder() {
    for kind in 3u8..=7 {
        // Typed array, category 3, kind in bits 5..7; SIZE 0.
        let bytes = [(kind << 5) | (3 << 3) | 4, 0x00];
        rejected_everywhere(&bytes, &format!("kind {kind}"));
        assert!(matches!(
            beve::from_field_slice::<Vec<f64>>(&bytes, "", 0, 0),
            Err(beve::Error::InvalidHeader(_))
        ));
    }
}

/// The header inside an aligned array must name a numeric typed array.
#[test]
fn an_aligned_array_wrapping_a_non_numeric_header_is_rejected() {
    // Inner header 0x1C is a boolean typed array.
    rejected_everywhere(&[0x5C, 0x1C, 0x04, 0x00, 0x01], "boolean inner header");
    // Inner float with byte code 7, a width BEVE does not define.
    rejected_everywhere(&[0x5C, 0xE4, 0x04, 0x00, 0x01], "float code 7 inner header");
}
