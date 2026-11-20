//! Low-level BEVE framing, for writers that emit a value piece by piece.
//!
//! The serde entry points ([`crate::to_vec`], [`crate::to_writer`]) and the
//! slice fast paths ([`crate::write_typed_slice`], [`crate::write_complex_slice`])
//! cover values that are in memory all at once. A writer that streams a payload
//! too large to hold, say a multi-gigabyte complex array arriving in chunks,
//! instead writes the framing itself and appends the payload bytes as they come.
//! This module is that framing: the SIZE codec and the headers that precede a
//! container's contents.
//!
//! Every writer appends to a `Vec<u8>` and emits exactly the bytes beve's own
//! encoders write, so a document assembled from these pieces is byte-identical
//! to one encoded whole. Complete values between the pieces (an object's field
//! values, a generic array's elements) are appended with [`crate::to_writer`],
//! which `&mut Vec<u8>` satisfies.
//!
//! ```
//! use beve::wire;
//!
//! // `{"id": 9, "iq": <complex f32 array of 2>}`, the samples appended last.
//! let mut out = Vec::new();
//! wire::write_object_header(&mut out, 2);
//! wire::write_object_key(&mut out, "id");
//! beve::to_writer(&mut out, &9u32).unwrap();
//! wire::write_object_key(&mut out, "iq");
//! wire::write_complex_array_header::<f32>(&mut out, 2);
//! for v in [1.0f32, 2.0, 3.0, 4.0] {
//!     out.extend_from_slice(&v.to_le_bytes());
//! }
//!
//! #[derive(serde::Serialize)]
//! struct Doc {
//!     id: u32,
//!     #[serde(with = "beve::complex_array::f32")]
//!     iq: Vec<beve::Complex<f32>>,
//! }
//! let iq = vec![beve::Complex { re: 1.0, im: 2.0 }, beve::Complex { re: 3.0, im: 4.0 }];
//! assert_eq!(out, beve::to_vec(&Doc { id: 9, iq }).unwrap());
//! ```

use crate::fast::{BeveTypedSlice, write_complex_header};
use crate::header::{KEY_STRING, TYPE_GENERIC_ARRAY, TYPE_OBJECT, TYPE_TYPED_ARRAY, make_header};

pub use crate::size::{read_size, size_encoded_len};

/// The largest value the SIZE codec can represent: 62 bits, since the two low
/// bits of the first byte carry the encoded width.
pub const MAX_SIZE: u64 = (1 << 62) - 1;

/// Append `n` in BEVE's compressed SIZE encoding (1, 2, 4 or 8 bytes; see
/// [`size_encoded_len`]).
///
/// # Panics
///
/// If `n` exceeds [`MAX_SIZE`]. The encoding has no room for the high bits, so
/// writing it would silently frame a different length. beve's own encoders skip
/// this check: their lengths come from in-memory collections, which cannot
/// reach 2^62.
#[inline]
pub fn write_size(out: &mut Vec<u8>, n: u64) {
    assert!(
        n <= MAX_SIZE,
        "BEVE SIZE {n} exceeds the 62-bit maximum {MAX_SIZE}"
    );
    crate::size::write_size(out, n);
}

/// Append the header and length of a generic (heterogeneous) array. `len`
/// complete values must follow.
///
/// # Panics
///
/// If `len` exceeds [`MAX_SIZE`].
#[inline]
pub fn write_generic_array_header(out: &mut Vec<u8>, len: u64) {
    out.push(TYPE_GENERIC_ARRAY);
    write_size(out, len);
}

/// Append the header and entry count of an object with string keys. `len`
/// entries must follow, each a key written with [`write_object_key`] and then
/// a complete value.
///
/// # Panics
///
/// If `len` exceeds [`MAX_SIZE`].
#[inline]
pub fn write_object_header(out: &mut Vec<u8>, len: u64) {
    out.push(make_header(TYPE_OBJECT, KEY_STRING, 0));
    write_size(out, len);
}

/// Append a string object key: its byte length, then its UTF-8 bytes. Keys
/// carry no header of their own; the object header fixes their type.
///
/// # Panics
///
/// If the key is longer than [`MAX_SIZE`] bytes.
#[inline]
pub fn write_object_key(out: &mut Vec<u8>, key: &str) {
    write_size(out, key.len() as u64);
    out.extend_from_slice(key.as_bytes());
}

/// Append the header and element count of a typed numeric array of `T`. The
/// payload that must follow is `len` elements of `T`, little-endian, with no
/// padding: `len * size_of::<T>()` bytes.
///
/// # Panics
///
/// If `len` exceeds [`MAX_SIZE`].
#[inline]
pub fn write_typed_array_header<T: BeveTypedSlice>(out: &mut Vec<u8>, len: u64) {
    out.push(make_header(TYPE_TYPED_ARRAY, T::CLASS, T::BYTE_CODE));
    write_size(out, len);
}

/// Append the extension header, complex header and element count of a complex
/// array whose components are `T`. The payload that must follow is `len`
/// complex values, each `re` then `im`, little-endian, with no padding:
/// `2 * len * size_of::<T>()` bytes.
///
/// # Panics
///
/// If `len` exceeds [`MAX_SIZE`].
#[inline]
pub fn write_complex_array_header<T: BeveTypedSlice>(out: &mut Vec<u8>, len: u64) {
    write_complex_header(out, true, T::CLASS, T::BYTE_CODE);
    write_size(out, len);
}
