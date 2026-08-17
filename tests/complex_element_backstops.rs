//! The run-time backstops on the bulk complex-array helpers.
//!
//! A wrong `unsafe impl ComplexElement` is the one way left to reach the
//! byte-reading paths with a `T` of the wrong shape. Width and alignment are the
//! two properties a check can see, and both helper families assert them; padding
//! and field order are held by the trait being `unsafe`, not by a test.

use serde::Serialize;

/// Two `f32` plus a third field: no padding, but 12 bytes where `Complex<f32>`
/// is 8. A deliberately wrong impl.
#[derive(Clone, Copy)]
#[repr(C)]
struct TooWide {
    re: f32,
    im: f32,
    extra: f32,
}

// SAFETY: deliberately FALSE — three components, not two. Drives the size
// backstop; never encodes successfully.
unsafe impl beve::ComplexElement for TooWide {
    type Component = f32;
}

/// Two `f32`, real first, no padding, but `#[repr(packed)]` drops it to align 1
/// where `Complex<f32>` needs 4.
#[derive(Clone, Copy)]
#[repr(C, packed)]
struct Packed {
    re: f32,
    im: f32,
}

// SAFETY: deliberately INCOMPLETE — truthful about layout and field order,
// wrong about alignment.
unsafe impl beve::ComplexElement for Packed {
    type Component = f32;
}

// Hand-written: a `#[repr(packed)]` field cannot be borrowed, so the derive does
// not apply. Satisfies the `serde(with)` family's `Serialize` bound; the assert
// under test fires before this runs.
impl Serialize for Packed {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let (re, im) = (self.re, self.im);
        (re, im).serialize(s)
    }
}

#[derive(Serialize)]
struct TooWideFrame {
    #[serde(serialize_with = "beve::complex::f32_array")]
    iq: Vec<TooWide>,
}

#[derive(Serialize)]
struct PackedFrame {
    #[serde(serialize_with = "beve::complex::f32_array")]
    iq: Vec<Packed>,
}

#[derive(Serialize)]
struct PackedWithFrame {
    #[serde(with = "beve::complex_array::f32")]
    iq: Vec<Packed>,
}

#[test]
#[should_panic(expected = "wrong element size")]
fn encode_helper_refuses_an_impl_of_the_wrong_width() {
    let frame = TooWideFrame {
        iq: vec![TooWide {
            re: 1.0,
            im: 2.0,
            extra: 3.0,
        }],
    };
    let _ = beve::to_vec(&frame);
}

#[test]
#[should_panic(expected = "wrong element alignment")]
fn encode_helper_refuses_an_underaligned_impl() {
    let frame = PackedFrame {
        iq: vec![Packed { re: 1.0, im: 2.0 }],
    };
    let _ = beve::to_vec(&frame);
}

/// The `serde(with)` family is a separate code path with its own copy of the
/// checks.
#[test]
#[should_panic(expected = "wrong element alignment")]
fn serde_with_family_refuses_an_underaligned_impl() {
    let frame = PackedWithFrame {
        iq: vec![Packed { re: 1.0, im: 2.0 }],
    };
    let _ = beve::to_vec(&frame);
}

/// An empty slice is checked too: `from_raw_parts` runs before any emptiness
/// check and requires an aligned pointer even at length zero.
#[test]
#[should_panic(expected = "wrong element alignment")]
fn an_empty_slice_is_checked_too() {
    let frame = PackedFrame { iq: Vec::new() };
    let _ = beve::to_vec(&frame);
}
