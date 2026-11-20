//! Aligned complex arrays through every decoder.
//!
//! The complex extension's COMPLEX HEADER carries a 3-bit kind: 0 a single
//! value, 1 an array, 2 an aligned array, 3 through 7 undefined. An aligned
//! complex array (`HEADER | COMPLEX HEADER | VALUE`) nests an aligned typed
//! array of the interleaved components, `2 * N` of them for `N` values, whose
//! numeric header must match the COMPLEX HEADER. Glaze writes it when aligned
//! arrays are enabled. Decoders that tested bit 0 alone read it as a single
//! value and desynchronized. Every decoder must read it as the complex array it
//! is, and reject the undefined kinds and malformed nesting.

use beve::{BeveTypedSlice, Complex, Value, wire};
use half::{bf16, f16};
use serde::Deserialize;
use std::io::Cursor;

/// Append an aligned complex array of `values` as Glaze writes it: the complex
/// extension, a COMPLEX HEADER of kind 2, then an aligned typed array of the
/// interleaved components, padded relative to the start of `out`.
fn write_aligned_complex<T: BeveTypedSlice + Copy>(out: &mut Vec<u8>, values: &[Complex<T>]) {
    out.push(0x1E);
    out.push((T::BYTE_CODE << 5) | (T::CLASS << 3) | 2);
    let components: Vec<T> = values.iter().flat_map(|c| [c.re, c.im]).collect();
    beve::write_aligned_typed_slice(out, &components);
}

fn c<T>(re: T, im: T) -> Complex<T> {
    Complex { re, im }
}

/// Bytes upstream Glaze writes for these vectors with `aligned_arrays` on.
#[test]
fn the_layout_matches_glaze() {
    let mut f64_bytes = Vec::new();
    write_aligned_complex(&mut f64_bytes, &[c(1.0f64, 2.0), c(-3.0, 4.5)]);
    assert_eq!(
        f64_bytes,
        [
            0x1e, 0x62, 0x5c, 0x64, 0x10, 0x02, 0x00, 0x00, //
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f, //
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, //
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0xc0, //
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x12, 0x40,
        ]
    );

    let mut i16_bytes = Vec::new();
    write_aligned_complex(&mut i16_bytes, &[c(1i16, -2), c(3, 4)]);
    assert_eq!(
        i16_bytes,
        [
            0x1e, 0x2a, 0x5c, 0x2c, 0x10, 0x00, 0x01, 0x00, 0xfe, 0xff, 0x03, 0x00, 0x04, 0x00
        ]
    );

    let mut empty = Vec::new();
    write_aligned_complex::<f64>(&mut empty, &[]);
    assert_eq!(empty, [0x1e, 0x62, 0x5c, 0x64, 0x00, 0x02, 0x00, 0x00]);
}

#[derive(Deserialize, Debug, PartialEq)]
struct Doc {
    id: u32,
    iq: Vec<Complex<f64>>,
    after: u32,
}

/// The same document as a bulk `complex_array::f64` field.
#[derive(Deserialize, Debug, PartialEq)]
struct BulkDoc {
    id: u32,
    #[serde(with = "beve::complex_array::f64")]
    iq: Vec<Complex<f64>>,
    after: u32,
}

/// `{"id": 9, "iq": <complex f64 array>, "after": 7}`, once with an aligned
/// complex array and once with a plain one, which every decoder must read
/// identically.
fn aligned_and_plain_docs() -> (Vec<u8>, Vec<u8>, Doc) {
    let doc = Doc {
        id: 9,
        iq: vec![c(1.0, 2.0), c(-3.0, 4.5), c(0.25, -0.5)],
        after: 7,
    };
    let build = |aligned: bool| {
        let mut out = Vec::new();
        wire::write_object_header(&mut out, 3);
        wire::write_object_key(&mut out, "id");
        beve::to_writer(&mut out, &doc.id).unwrap();
        wire::write_object_key(&mut out, "iq");
        if aligned {
            write_aligned_complex(&mut out, &doc.iq);
        } else {
            beve::write_complex_slice(&mut out, &doc.iq);
        }
        wire::write_object_key(&mut out, "after");
        beve::to_writer(&mut out, &doc.after).unwrap();
        out
    };
    (build(true), build(false), doc)
}

#[test]
fn skip_value_steps_over_an_aligned_complex_array() {
    let (aligned, _, _) = aligned_and_plain_docs();
    let mut pos = 0;
    beve::skip_value(&aligned, &mut pos).unwrap();
    assert_eq!(pos, aligned.len());
}

#[test]
fn every_decoder_reads_an_aligned_complex_array_as_its_complex_array() {
    let (aligned, plain, doc) = aligned_and_plain_docs();
    assert_ne!(aligned, plain);

    beve::validate_slice(&aligned).unwrap();
    assert_eq!(beve::from_slice::<Doc>(&aligned).unwrap(), doc);
    assert_eq!(
        beve::from_reader_streaming::<_, Doc>(Cursor::new(&aligned)).unwrap(),
        doc
    );
    let bulk = BulkDoc {
        id: doc.id,
        iq: doc.iq.clone(),
        after: doc.after,
    };
    assert_eq!(beve::from_slice::<BulkDoc>(&aligned).unwrap(), bulk);
    assert_eq!(
        beve::from_reader_streaming::<_, BulkDoc>(Cursor::new(&aligned)).unwrap(),
        bulk
    );
    assert_eq!(
        beve::from_slice::<Value>(&aligned).unwrap(),
        beve::from_slice::<Value>(&plain).unwrap()
    );
    assert_eq!(
        beve::from_reader_streaming::<_, Value>(Cursor::new(&aligned)).unwrap(),
        beve::from_slice::<Value>(&plain).unwrap()
    );
    assert_eq!(
        beve::beve_slice_to_json_string(&aligned).unwrap(),
        beve::beve_slice_to_json_string(&plain).unwrap()
    );
    assert_eq!(beve::from_field::<u32>(&aligned, "/after").unwrap(), 7);
    assert_eq!(
        beve::from_field::<Vec<Complex<f64>>>(&aligned, "/iq").unwrap(),
        doc.iq
    );
}

/// A bulk field of a narrower type widens an aligned source as it does a plain
/// one.
#[test]
fn a_bulk_field_widens_an_aligned_complex_array() {
    #[derive(Deserialize, Debug, PartialEq)]
    struct Wide {
        #[serde(with = "beve::complex_array::f64")]
        iq: Vec<Complex<f64>>,
    }
    let mut bytes = Vec::new();
    wire::write_object_header(&mut bytes, 1);
    wire::write_object_key(&mut bytes, "iq");
    write_aligned_complex(&mut bytes, &[c(1.5f32, -2.0), c(3.0, 0.5)]);
    let expected = Wide {
        iq: vec![c(1.5, -2.0), c(3.0, 0.5)],
    };
    assert_eq!(beve::from_slice::<Wide>(&bytes).unwrap(), expected);
    assert_eq!(
        beve::from_reader_streaming::<_, Wide>(Cursor::new(&bytes)).unwrap(),
        expected
    );
}

#[cfg(feature = "mat")]
#[test]
fn mat_export_reads_an_aligned_complex_array_as_its_complex_array() {
    use beve::{MatV73Options, RootBinding};

    let (aligned, plain, _) = aligned_and_plain_docs();
    let options = MatV73Options::default();
    let convert = |bytes: &[u8]| {
        beve::beve_slice_to_mat_v73_bytes(bytes, RootBinding::NamedVariable("doc"), &options)
            .unwrap()
    };
    assert_eq!(convert(&aligned), convert(&plain));
}

/// Every component type, at every padding length the alignment can call for.
#[test]
fn aligned_complex_arrays_of_every_scalar_round_trip() {
    fn check<T>(values: &[Complex<T>])
    where
        T: BeveTypedSlice + Copy + PartialEq + std::fmt::Debug,
        Complex<T>: serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
    {
        for prefix in 0..16 {
            let mut bytes = vec![0u8; prefix];
            write_aligned_complex(&mut bytes, values);
            let body = &bytes[prefix..];

            let decoded: Vec<Complex<T>> = beve::from_slice(body).unwrap();
            assert_eq!(decoded, values, "prefix={prefix}");
            let streamed: Vec<Complex<T>> = beve::from_reader_streaming(Cursor::new(body)).unwrap();
            assert_eq!(streamed, values, "prefix={prefix}");
            assert_eq!(
                beve::read_complex_slice::<T>(body).unwrap(),
                values,
                "prefix={prefix}"
            );
            assert_eq!(
                beve::read_complex_slice_from_reader::<T, _>(Cursor::new(body)).unwrap(),
                values,
                "prefix={prefix}"
            );
            let mut pos = 0;
            beve::skip_value(body, &mut pos).unwrap();
            assert_eq!(pos, body.len(), "prefix={prefix}");
            assert_eq!(
                beve::beve_slice_to_json_string(body).unwrap(),
                beve::beve_slice_to_json_string(&beve::to_vec_complex_slice(values)).unwrap(),
                "prefix={prefix}"
            );
        }
    }
    check(&[c(1i8, -2), c(3, 4)]);
    check(&[c(1i16, -2), c(3, 4)]);
    check(&[c(1i32, -2), c(3, 4)]);
    check(&[c(1i64, -2), c(3, 4)]);
    check(&[c(1u8, 2), c(3, 4)]);
    check(&[c(1u16, 2), c(3, 4)]);
    check(&[c(1u32, 2), c(3, 4)]);
    check(&[c(1u64, 2), c(3, 4)]);
    check(&[c(1.5f32, -2.0)]);
    check(&[c(1.5f64, -2.0), c(0.0, 8.25)]);
    check(&[c(f16::from_f32(1.5), f16::from_f32(-2.0))]);
    check(&[c(bf16::from_f32(1.5), bf16::from_f32(-2.0))]);
    check::<f64>(&[]);
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
        beve::from_slice::<Vec<Complex<f64>>>(bytes).is_err(),
        "{what}: Vec<Complex>"
    );
    assert!(
        beve::beve_slice_to_json_string(bytes).is_err(),
        "{what}: JSON"
    );
    assert!(
        beve::read_complex_slice::<f64>(bytes).is_err(),
        "{what}: read_complex_slice"
    );
    assert!(
        beve::read_complex_slice_from_reader::<f64, _>(Cursor::new(bytes)).is_err(),
        "{what}: read_complex_slice_from_reader"
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

/// A well-formed aligned complex f64 array of one value, `[1.0, 2.0]`, for the
/// malformed variants below to edit: `1E 62 5C 64 08 PAD(2) 00 00 DATA(16)`.
fn one_f64() -> Vec<u8> {
    let mut bytes = Vec::new();
    write_aligned_complex(&mut bytes, &[c(1.0f64, 2.0)]);
    assert_eq!(&bytes[..6], [0x1e, 0x62, 0x5c, 0x64, 0x08, 0x02]);
    bytes
}

#[test]
fn undefined_kinds_are_rejected_by_every_decoder() {
    for kind in 3u8..=7 {
        let mut bytes = one_f64();
        bytes[1] = 0x60 | kind;
        rejected_everywhere(&bytes, &format!("kind {kind}"));
    }
}

#[test]
fn malformed_aligned_complex_arrays_are_rejected_by_every_decoder() {
    // The nested value is a plain typed array, not an aligned one.
    let mut plain_inner = vec![0x1e, 0x62, 0x64, 0x08];
    plain_inner.extend_from_slice(&1.0f64.to_le_bytes());
    plain_inner.extend_from_slice(&2.0f64.to_le_bytes());
    rejected_everywhere(&plain_inner, "plain inner array");

    // The inner header names f32 components under an f64 COMPLEX HEADER.
    let mut mismatched = one_f64();
    mismatched[3] = 0x44;
    rejected_everywhere(&mismatched, "mismatched inner header");

    // An odd component count: 1 component cannot form a complex value.
    let mut odd = vec![0x1e, 0x62, 0x5c, 0x64, 0x04, 0x02, 0x00, 0x00];
    odd.extend_from_slice(&1.0f64.to_le_bytes());
    rejected_everywhere(&odd, "odd component count");

    // Padding as wide as a component.
    let mut wide_padding = vec![0x1e, 0x62, 0x5c, 0x64, 0x08, 0x08];
    wide_padding.extend_from_slice(&[0; 8]);
    wide_padding.extend_from_slice(&1.0f64.to_le_bytes());
    wide_padding.extend_from_slice(&2.0f64.to_le_bytes());
    rejected_everywhere(&wide_padding, "padding of 8");

    // Truncated at every byte.
    let whole = one_f64();
    for end in 1..whole.len() {
        rejected_everywhere(&whole[..end], &format!("truncated at {end}"));
    }
}
