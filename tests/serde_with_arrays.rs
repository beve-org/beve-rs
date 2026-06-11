//! `#[serde(with = "beve::typed::*" / "beve::complex_array::*")]` round-trips:
//! a struct whose numeric / complex array fields decode through the bulk path,
//! via both `from_slice` and `from_reader_streaming`, plus wire-compatibility
//! with the unannotated `Vec<T>` form and a foreign layout-compatible complex
//! element type.

use beve::Complex;
use serde::{Deserialize, Serialize};
use std::io::Cursor;

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Frame {
    id: u64,
    label: String,
    #[serde(with = "beve::typed::f64")]
    samples: Vec<f64>,
    #[serde(with = "beve::typed::i16")]
    counts: Vec<i16>,
    #[serde(with = "beve::complex_array::f32")]
    iq: Vec<Complex<f32>>,
    tail: i32,
}

fn sample_frame(n: usize) -> Frame {
    Frame {
        id: 0xFEED_BEEF,
        label: "bulk".to_string(),
        samples: (0..n).map(|i| i as f64 * 0.5 - 3.0).collect(),
        counts: (0..n).map(|i| (i as i64 % 60000 - 30000) as i16).collect(),
        iq: (0..n)
            .map(|i| Complex {
                re: i as f32 * 1e-3,
                im: -(i as f32) * 2e-3,
            })
            .collect(),
        tail: -42,
    }
}

fn roundtrip(frame: &Frame) {
    let bytes = beve::to_vec(frame).expect("encode");

    let from_slice: Frame = beve::from_slice(&bytes).expect("from_slice");
    assert_eq!(&from_slice, frame, "from_slice mismatch");

    let streamed: Frame = beve::from_reader_streaming(Cursor::new(&bytes)).expect("stream");
    assert_eq!(&streamed, frame, "from_reader_streaming mismatch");
}

#[test]
fn struct_with_bulk_array_fields_roundtrips() {
    for n in [0usize, 1, 2, 1000, 100_000] {
        roundtrip(&sample_frame(n));
    }
}

// A struct identical to `Frame` but with *unannotated* array fields, to confirm
// the `with` encoding is wire-compatible with plain `Vec<T>` and cross-decodes.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct PlainFrame {
    id: u64,
    label: String,
    samples: Vec<f64>,
    counts: Vec<i16>,
    iq: Vec<Complex<f32>>,
    tail: i32,
}

#[test]
fn with_encoding_is_wire_compatible_with_plain_vec() {
    let f = sample_frame(500);
    let plain = PlainFrame {
        id: f.id,
        label: f.label.clone(),
        samples: f.samples.clone(),
        counts: f.counts.clone(),
        iq: f.iq.clone(),
        tail: f.tail,
    };

    // Non-empty arrays encode identically whether annotated or not.
    assert_eq!(
        beve::to_vec(&f).unwrap(),
        beve::to_vec(&plain).unwrap(),
        "with-annotated and plain encodings must match for non-empty arrays"
    );

    // Cross-decode each into the other.
    let as_plain: PlainFrame = beve::from_slice(&beve::to_vec(&f).unwrap()).unwrap();
    assert_eq!(as_plain, plain);
    let as_frame: Frame = beve::from_slice(&beve::to_vec(&plain).unwrap()).unwrap();
    assert_eq!(as_frame, f);
}

// A foreign complex element type, layout-compatible with `Complex<f32>`
// (`#[repr(C)]`, two `f32` fields re/im) — the `num_complex::Complex<f32>` case.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
#[repr(C)]
struct Iq {
    re: f32,
    im: f32,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct ForeignFrame {
    #[serde(with = "beve::complex_array::f32")]
    iq: Vec<Iq>,
}

#[test]
fn foreign_layout_compatible_complex_roundtrips() {
    let frame = ForeignFrame {
        iq: (0..50_000)
            .map(|i| Iq {
                re: (i as f32).sin(),
                im: (i as f32).cos(),
            })
            .collect(),
    };
    let bytes = beve::to_vec(&frame).unwrap();

    let from_slice: ForeignFrame = beve::from_slice(&bytes).unwrap();
    assert_eq!(from_slice, frame);
    let streamed: ForeignFrame = beve::from_reader_streaming(Cursor::new(&bytes)).unwrap();
    assert_eq!(streamed, frame);

    // And it interops with a `Complex<f32>` view of the same field.
    #[derive(Deserialize, PartialEq, Debug)]
    struct AsComplex {
        #[serde(with = "beve::complex_array::f32")]
        iq: Vec<Complex<f32>>,
    }
    let as_complex: AsComplex = beve::from_slice(&bytes).unwrap();
    assert_eq!(as_complex.iq.len(), frame.iq.len());
    assert_eq!(
        as_complex.iq[7],
        Complex {
            re: frame.iq[7].re,
            im: frame.iq[7].im
        }
    );
}

#[test]
fn complex_f64_and_more_widths() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Wide {
        #[serde(with = "beve::complex_array::f64")]
        iq64: Vec<Complex<f64>>,
        #[serde(with = "beve::typed::u32")]
        idx: Vec<u32>,
    }
    let w = Wide {
        iq64: (0..10_000)
            .map(|i| Complex {
                re: i as f64,
                im: -(i as f64),
            })
            .collect(),
        idx: (0..10_000).map(|i| i as u32 * 7).collect(),
    };
    let bytes = beve::to_vec(&w).unwrap();
    assert_eq!(beve::from_slice::<Wide>(&bytes).unwrap(), w);
    assert_eq!(
        beve::from_reader_streaming::<_, Wide>(Cursor::new(&bytes)).unwrap(),
        w
    );
}
