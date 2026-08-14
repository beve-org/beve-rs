//! Bulk complex decode of a *different* component class than the destination.
//!
//! One claim, in every position it has to hold: **a `#[serde(with =
//! "beve::complex_array::*")]` field must decode to exactly what the same field
//! without the attribute decodes to.** The attribute buys throughput; it is not
//! allowed to buy or cost a single value, nor to turn a decode that works into
//! an error.
//!
//! The element-wise path is therefore the oracle throughout. Nothing here
//! asserts a hand-written expected value except where the oracle itself is the
//! thing under test.

use beve::Complex;
use serde::{Deserialize, Serialize};
use std::io::Cursor;

/// Element counts spanning empty, small, and the 1/2/4-byte SIZE-prefix
/// boundaries, where a miscomputed component count would surface.
const LENGTHS: &[usize] = &[0, 1, 2, 3, 7, 64, 100, 16383, 16384];

/// Build the source struct (a plain `Vec<Complex<S>>`, written element-wise),
/// the bulk destination struct, and the element-wise destination struct, then
/// assert the two destinations agree through both decoders.
macro_rules! check_pair {
    ($src:ty, $dst:ty, $with:literal, $mk:expr) => {{
        #[derive(Serialize)]
        struct Src {
            iq: Vec<Complex<$src>>,
        }
        #[derive(Deserialize)]
        struct Bulk {
            #[serde(with = $with)]
            iq: Vec<Complex<$dst>>,
        }
        #[derive(Deserialize)]
        struct Plain {
            iq: Vec<Complex<$dst>>,
        }

        for &n in LENGTHS {
            let src = Src {
                iq: (0..n).map($mk).collect(),
            };
            let bytes = beve::to_vec(&src).expect("encode");

            let expected = beve::from_slice::<Plain>(&bytes)
                .expect("element-wise decode")
                .iq;
            assert_eq!(expected.len(), n, "oracle length (n={n})");

            let bulk = beve::from_slice::<Bulk>(&bytes).expect("bulk decode").iq;
            assert_eq!(
                bulk,
                expected,
                "{} -> {}: bulk must equal element-wise (n={n})",
                stringify!($src),
                stringify!($dst),
            );

            let streamed = beve::from_reader_streaming::<_, Bulk>(Cursor::new(&bytes))
                .expect("streaming bulk decode")
                .iq;
            assert_eq!(
                streamed,
                expected,
                "{} -> {}: streaming bulk must equal element-wise (n={n})",
                stringify!($src),
                stringify!($dst),
            );
        }
    }};
}

/// A ramp that stays inside the source type's range while covering both signs
/// where the type has them.
macro_rules! ramp {
    ($t:ty, signed) => {
        |i: usize| Complex {
            re: (i % 100) as $t - 50 as $t,
            im: 50 as $t - (i % 100) as $t,
        }
    };
    ($t:ty, unsigned) => {
        |i: usize| Complex {
            re: (i % 100) as $t,
            im: (i % 37) as $t,
        }
    };
}

#[test]
fn every_signed_class_widens_into_f32_like_the_element_wise_path() {
    check_pair!(i8, f32, "beve::complex_array::f32", ramp!(i8, signed));
    check_pair!(i16, f32, "beve::complex_array::f32", ramp!(i16, signed));
    check_pair!(i32, f32, "beve::complex_array::f32", ramp!(i32, signed));
    check_pair!(i64, f32, "beve::complex_array::f32", ramp!(i64, signed));
    check_pair!(i128, f32, "beve::complex_array::f32", ramp!(i128, signed));
}

#[test]
fn every_unsigned_class_widens_into_f32_like_the_element_wise_path() {
    check_pair!(u8, f32, "beve::complex_array::f32", ramp!(u8, unsigned));
    check_pair!(u16, f32, "beve::complex_array::f32", ramp!(u16, unsigned));
    check_pair!(u32, f32, "beve::complex_array::f32", ramp!(u32, unsigned));
    check_pair!(u64, f32, "beve::complex_array::f32", ramp!(u64, unsigned));
    check_pair!(u128, f32, "beve::complex_array::f32", ramp!(u128, unsigned));
}

#[test]
fn every_signed_class_widens_into_f64_like_the_element_wise_path() {
    check_pair!(i8, f64, "beve::complex_array::f64", ramp!(i8, signed));
    check_pair!(i16, f64, "beve::complex_array::f64", ramp!(i16, signed));
    check_pair!(i32, f64, "beve::complex_array::f64", ramp!(i32, signed));
    check_pair!(i64, f64, "beve::complex_array::f64", ramp!(i64, signed));
    check_pair!(i128, f64, "beve::complex_array::f64", ramp!(i128, signed));
}

#[test]
fn every_unsigned_class_widens_into_f64_like_the_element_wise_path() {
    check_pair!(u8, f64, "beve::complex_array::f64", ramp!(u8, unsigned));
    check_pair!(u16, f64, "beve::complex_array::f64", ramp!(u16, unsigned));
    check_pair!(u32, f64, "beve::complex_array::f64", ramp!(u32, unsigned));
    check_pair!(u64, f64, "beve::complex_array::f64", ramp!(u64, unsigned));
    check_pair!(u128, f64, "beve::complex_array::f64", ramp!(u128, unsigned));
}

#[test]
fn float_classes_convert_between_each_other_like_the_element_wise_path() {
    check_pair!(f32, f64, "beve::complex_array::f64", |i: usize| Complex {
        re: i as f32 * 0.5 - 3.0,
        im: -(i as f32) * 0.25,
    });
    check_pair!(f64, f32, "beve::complex_array::f32", |i: usize| Complex {
        re: i as f64 * 0.5 - 3.0,
        im: -(i as f64) * 0.25,
    });
}

#[test]
fn half_classes_widen_like_the_element_wise_path() {
    use half::{bf16, f16};
    check_pair!(f16, f32, "beve::complex_array::f32", |i: usize| Complex {
        re: f16::from_f32(i as f32 * 0.5),
        im: f16::from_f32(-(i as f32)),
    });
    check_pair!(bf16, f32, "beve::complex_array::f32", |i: usize| Complex {
        re: bf16::from_f32(i as f32 * 0.5),
        im: bf16::from_f32(-(i as f32)),
    });
    check_pair!(f16, f64, "beve::complex_array::f64", |i: usize| Complex {
        re: f16::from_f32(i as f32 * 0.5),
        im: f16::from_f32(-(i as f32)),
    });
    check_pair!(bf16, f64, "beve::complex_array::f64", |i: usize| Complex {
        re: bf16::from_f32(i as f32 * 0.5),
        im: bf16::from_f32(-(i as f32)),
    });
}

/// Values where a narrowing conversion is observable. `f64 -> f32` overflows to
/// infinity, flushes to zero, and rounds; the bulk path has to do all three the
/// same way the element-wise path does.
#[test]
fn narrowing_specials_match_the_element_wise_path() {
    #[derive(Serialize)]
    struct Src {
        iq: Vec<Complex<f64>>,
    }
    #[derive(Deserialize)]
    struct Bulk {
        #[serde(with = "beve::complex_array::f32")]
        iq: Vec<Complex<f32>>,
    }
    #[derive(Deserialize)]
    struct Plain {
        iq: Vec<Complex<f32>>,
    }

    let specials = [
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::MAX,
        f64::MIN,
        f64::MIN_POSITIVE,
        5e-324, // subnormal f64, flushes to 0 in f32
        1e300,  // overflows f32
        -1e300,
        0.0,
        -0.0,
        1.0 / 3.0, // rounds
        f32::MAX as f64 * 1.0000001,
    ];
    let src = Src {
        iq: specials
            .iter()
            .flat_map(|&a| specials.iter().map(move |&b| Complex { re: a, im: b }))
            .collect(),
    };
    let bytes = beve::to_vec(&src).unwrap();

    let expected = beve::from_slice::<Plain>(&bytes).unwrap().iq;
    let bulk = beve::from_slice::<Bulk>(&bytes).unwrap().iq;
    let streamed = beve::from_reader_streaming::<_, Bulk>(Cursor::new(&bytes))
        .unwrap()
        .iq;

    // NaN != NaN, so compare bit patterns: the two paths must agree on the exact
    // f32 produced, quiet-NaN payload included.
    let bits = |v: &[Complex<f32>]| {
        v.iter()
            .map(|c| (c.re.to_bits(), c.im.to_bits()))
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(&bulk), bits(&expected), "buffered");
    assert_eq!(bits(&streamed), bits(&expected), "streaming");
}

/// Integer destinations keep the element-wise path's range checking. A value
/// that does not fit must fail, not wrap.
#[test]
fn an_out_of_range_integer_fails_like_the_element_wise_path() {
    #[derive(Serialize)]
    struct Src {
        iq: Vec<Complex<i32>>,
    }
    #[derive(Deserialize)]
    struct Bulk {
        #[serde(with = "beve::complex_array::i16")]
        #[allow(dead_code)]
        iq: Vec<Complex<i16>>,
    }
    #[derive(Deserialize)]
    struct Plain {
        #[allow(dead_code)]
        iq: Vec<Complex<i16>>,
    }

    // In range for i16: both paths accept.
    let ok = beve::to_vec(&Src {
        iq: vec![
            Complex {
                re: 30_000,
                im: -30_000
            };
            8
        ],
    })
    .unwrap();
    assert!(beve::from_slice::<Plain>(&ok).is_ok(), "premise");
    assert!(beve::from_slice::<Bulk>(&ok).is_ok());
    assert!(beve::from_reader_streaming::<_, Bulk>(Cursor::new(&ok)).is_ok());

    // Past i16::MAX: both paths must reject.
    let bad = beve::to_vec(&Src {
        iq: vec![Complex { re: 40_000, im: 0 }; 8],
    })
    .unwrap();
    assert!(beve::from_slice::<Plain>(&bad).is_err(), "premise");
    assert!(beve::from_slice::<Bulk>(&bad).is_err());
    assert!(beve::from_reader_streaming::<_, Bulk>(Cursor::new(&bad)).is_err());
}

/// An integer source into an integer destination of a different width still
/// decodes, through the element-wise fallback rather than a bulk copy. Same
/// contract, slower route.
#[test]
fn a_narrower_integer_source_decodes_through_the_fallback() {
    #[derive(Serialize)]
    struct Src {
        iq: Vec<Complex<i16>>,
    }
    #[derive(Deserialize)]
    struct Bulk {
        #[serde(with = "beve::complex_array::i32")]
        iq: Vec<Complex<i32>>,
    }
    #[derive(Deserialize)]
    struct Plain {
        iq: Vec<Complex<i32>>,
    }

    for &n in LENGTHS {
        let bytes = beve::to_vec(&Src {
            iq: (0..n)
                .map(|i| Complex {
                    re: (i % 1000) as i16 - 500,
                    im: 500 - (i % 1000) as i16,
                })
                .collect(),
        })
        .unwrap();
        let expected = beve::from_slice::<Plain>(&bytes).unwrap().iq;
        assert_eq!(
            beve::from_slice::<Bulk>(&bytes).unwrap().iq,
            expected,
            "n={n}"
        );
        assert_eq!(
            beve::from_reader_streaming::<_, Bulk>(Cursor::new(&bytes))
                .unwrap()
                .iq,
            expected,
            "n={n}"
        );
    }
}

/// The same-class path must still be the borrowing bulk copy it always was, and
/// an empty `Vec` must still decode: it encodes as a *generic* empty array, not
/// a complex one, so it reaches the marker as a shape the fast path declines.
#[test]
fn the_same_class_and_empty_cases_are_unchanged() {
    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct Same {
        #[serde(with = "beve::complex_array::f32")]
        iq: Vec<Complex<f32>>,
    }
    for &n in LENGTHS {
        let v = Same {
            iq: (0..n)
                .map(|i| Complex {
                    re: i as f32,
                    im: -(i as f32),
                })
                .collect(),
        };
        let bytes = beve::to_vec(&v).unwrap();
        assert_eq!(beve::from_slice::<Same>(&bytes).unwrap(), v, "n={n}");
        assert_eq!(
            beve::from_reader_streaming::<_, Same>(Cursor::new(&bytes)).unwrap(),
            v,
            "n={n}"
        );
    }
}

/// A truncated payload must be an error from either decoder, not a panic and not
/// a short `Vec`. The widening path sizes its read from the *source* width, so a
/// wrong width rule here would read past the end.
#[test]
fn a_truncated_widening_payload_errors_without_panicking() {
    #[derive(Serialize)]
    struct Src {
        iq: Vec<Complex<i16>>,
    }
    #[derive(Deserialize)]
    struct Bulk {
        #[serde(with = "beve::complex_array::f32")]
        #[allow(dead_code)]
        iq: Vec<Complex<f32>>,
    }

    let full = beve::to_vec(&Src {
        iq: (0..1000)
            .map(|i| Complex {
                re: i as i16,
                im: -(i as i16),
            })
            .collect(),
    })
    .unwrap();

    for cut in [1usize, 7, 64, 999, full.len() - 1] {
        let truncated = &full[..full.len() - cut];
        assert!(
            beve::from_slice::<Bulk>(truncated).is_err(),
            "buffered, cut={cut}"
        );
        assert!(
            beve::from_reader_streaming::<_, Bulk>(Cursor::new(truncated)).is_err(),
            "streaming, cut={cut}"
        );
    }
}
