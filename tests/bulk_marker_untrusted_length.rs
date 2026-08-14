//! A length on the wire is untrusted input, and the bulk `#[serde(with = ...)]`
//! markers must not size an allocation from one they have not backed with bytes.
//!
//! `Vec::with_capacity` answers an impossible request by aborting the process,
//! which no decoder can turn into an `Err` and no caller can catch. serde's own
//! `Vec` visitor clamps `size_hint` for exactly this reason; the hand-written
//! visitors behind these markers are the ones that skipped it.
//!
//! Each case below is a small, well-formed message whose element count has been
//! rewritten to claim 2^55 values. If a decode ever sizes a buffer from that
//! count again, these tests do not fail -- they take the whole test binary down
//! with them, which is the point.

use beve::Complex;
use serde::{Deserialize, Serialize};
use std::io::Cursor;

/// An element count no allocation can satisfy, small enough to still fit the
/// 62 bits a SIZE carries.
const FORGED_COUNT: u64 = 1 << 55;

/// Replace the element count preceding the trailing `payload_len`-byte payload
/// with `FORGED_COUNT`, leaving the payload itself as the handful of bytes it
/// was.
///
/// BEVE writes a count as a SIZE: the low two bits give the width (0 -> 1 byte,
/// 3 -> 8 bytes) and the value fills the remaining bits little-endian. A count
/// below 64 is one byte, `n << 2` -- asserted here, so this stops being a silent
/// no-op if the encoding ever moves.
fn forge_count(bytes: &[u8], payload_len: usize, actual: u64) -> Vec<u8> {
    let size_at = bytes.len() - payload_len - 1;
    assert_eq!(
        bytes[size_at],
        (actual as u8) << 2,
        "expected a 1-byte SIZE holding {actual} just before the payload"
    );

    let mut forged = bytes[..size_at].to_vec();
    forged.push((((FORGED_COUNT & 0x3f) as u8) << 2) | 0b11); // 8-byte SIZE
    forged.extend((0..7).map(|i| ((FORGED_COUNT >> 6) >> (i * 8)) as u8));
    forged.extend_from_slice(&bytes[size_at + 1..]);
    forged
}

/// A complex array read into a field of another *integer* class takes the
/// element-wise fallback, which is handed the wire's count with the payload
/// still unread -- the one route where a count reaches a visitor before any
/// bytes have vouched for it.
#[test]
fn a_forged_complex_array_count_errors_instead_of_aborting() {
    #[derive(Serialize)]
    struct Src {
        iq: Vec<Complex<i16>>,
    }
    #[derive(Deserialize)]
    struct Bulk {
        #[serde(with = "beve::complex_array::i32")]
        #[allow(dead_code)]
        iq: Vec<Complex<i32>>,
    }

    let bytes = beve::to_vec(&Src {
        iq: vec![Complex { re: 1i16, im: -1 }; 3],
    })
    .unwrap();
    // 3 values * 2 components * 2 bytes.
    let forged = forge_count(&bytes, 3 * 2 * 2, 3);

    assert!(beve::from_slice::<Bulk>(&forged).is_err(), "buffered");
    assert!(
        beve::from_reader_streaming::<_, Bulk>(Cursor::new(&forged)).is_err(),
        "streaming"
    );
}

/// The typed-array marker reaches the same visitor shape by a different route: a
/// class the bulk path declines falls through to the element-wise sequence,
/// which the streaming decoder builds from the count alone.
#[test]
fn a_forged_typed_array_count_errors_instead_of_aborting() {
    #[derive(Serialize)]
    struct Src {
        samples: Vec<i16>,
    }
    #[derive(Deserialize)]
    struct Bulk {
        #[serde(with = "beve::typed::i32")]
        #[allow(dead_code)]
        samples: Vec<i32>,
    }

    let bytes = beve::to_vec(&Src {
        samples: vec![1i16, -2, 3],
    })
    .unwrap();
    // 3 values * 2 bytes.
    let forged = forge_count(&bytes, 3 * 2, 3);

    assert!(beve::from_slice::<Bulk>(&forged).is_err(), "buffered");
    assert!(
        beve::from_reader_streaming::<_, Bulk>(Cursor::new(&forged)).is_err(),
        "streaming"
    );
}

/// The same message decoded without the annotations: the markers must not be the
/// only thing standing between a forged count and an abort, and this pins the
/// unannotated baseline the tests above are compared against.
#[test]
fn a_forged_count_errors_without_the_markers_too() {
    #[derive(Serialize)]
    struct Src {
        iq: Vec<Complex<i16>>,
    }
    #[derive(Deserialize)]
    struct Plain {
        #[allow(dead_code)]
        iq: Vec<Complex<i32>>,
    }

    let bytes = beve::to_vec(&Src {
        iq: vec![Complex { re: 1i16, im: -1 }; 3],
    })
    .unwrap();
    let forged = forge_count(&bytes, 3 * 2 * 2, 3);

    assert!(beve::from_slice::<Plain>(&forged).is_err(), "buffered");
    assert!(
        beve::from_reader_streaming::<_, Plain>(Cursor::new(&forged)).is_err(),
        "streaming"
    );
}
