//! Navigate into a BEVE payload using a [JSON Pointer (RFC 6901)][rfc6901]
//! and deserialize only the targeted value. Intermediate values are skipped
//! without allocation, making this efficient for extracting a single field
//! from a large payload.
//!
//! [rfc6901]: https://datatracker.ietf.org/doc/html/rfc6901
//!
//! # Quick reference
//!
//! | Function | Purpose |
//! |---|---|
//! | [`from_field`] | Navigate to a pointer and deserialize the value there. |
//! | [`from_field_slice`] | Navigate to a typed numeric array and deserialize a sub-range. |
//! | [`skip_value`] | Advance past one BEVE value without deserializing it. |
//!
//! # JSON Pointer syntax
//!
//! A JSON Pointer is either the empty string `""` (refers to the whole
//! document) or a sequence of `/`-separated reference tokens:
//!
//! | Pointer | Meaning |
//! |---|---|
//! | `""` | the entire document |
//! | `"/name"` | top-level object field `"name"` |
//! | `"/a/b/c"` | nested field lookup `a -> b -> c` |
//! | `"/0"` | index 0 of a generic array (0-based) |
//! | `"/2/data"` | index 2 of a generic array, then field `"data"` |
//! | `"/5"` | integer key `5` when navigating an integer-keyed object |
//!
//! Literal `~` and `/` inside a key are escaped as `~0` and `~1`
//! respectively (per RFC 6901, `~1` is unescaped first).
//!
//! # Navigable types
//!
//! - **Objects** with string, signed-integer, or unsigned-integer keys
//! - **Generic arrays** by 0-based element index
//!
//! Typed arrays, scalars, and strings can appear as the *leaf* target but
//! cannot be navigated *into* with further pointer segments.
//!
//! # Examples
//!
//! Extracting a nested field:
//!
//! ```rust
//! use std::collections::HashMap;
//! use serde::{Serialize, Deserialize};
//!
//! #[derive(Serialize, Deserialize)]
//! struct Outer {
//!     sensors: HashMap<String, Vec<f64>>,
//! }
//!
//! let data = Outer {
//!     sensors: [("channel0".into(), vec![1.0, 2.0, 3.0])].into(),
//! };
//! let bytes = beve::to_vec(&data).unwrap();
//! let ch: Vec<f64> = beve::from_field(&bytes, "/sensors/channel0").unwrap();
//! assert_eq!(ch, vec![1.0, 2.0, 3.0]);
//! ```
//!
//! Slicing a large array without loading all of it:
//!
//! ```rust
//! let data: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
//! let bytes = beve::to_vec(&data).unwrap();
//!
//! // Read only elements [100..103)
//! let slice: Vec<f64> = beve::from_field_slice(&bytes, "", 100, 3).unwrap();
//! assert_eq!(slice, vec![100.0, 101.0, 102.0]);
//! ```

use crate::error::{Error, Result};
use crate::header::*;
use crate::size::read_size;
use serde::Deserialize;
use simdutf8::basic::from_utf8;

/// Deserialize a value located at `pointer` inside a BEVE byte slice.
///
/// The `pointer` follows [RFC 6901](https://datatracker.ietf.org/doc/html/rfc6901).
/// See the [module documentation](self) for the full pointer syntax.
///
/// Supports zero-copy deserialization: types containing `&'de str` fields
/// borrow directly from the input buffer, just like [`crate::from_slice`].
///
/// # Errors
///
/// Returns an error if:
///
/// - The pointer is malformed (not empty and not starting with `/`).
/// - A pointer segment tries to navigate into a non-container type
///   (e.g. a number or string).
/// - An object key is not found.
/// - A generic-array index is out of bounds or not a valid integer.
/// - The targeted value cannot be deserialized into `T`.
///
/// # Examples
///
/// ```rust
/// use serde::{Serialize, Deserialize};
///
/// #[derive(Serialize, Deserialize)]
/// struct Config {
///     width: u32,
///     height: u32,
///     title: String,
/// }
///
/// let cfg = Config { width: 1920, height: 1080, title: "Main".into() };
/// let bytes = beve::to_vec(&cfg).unwrap();
///
/// let w: u32 = beve::from_field(&bytes, "/width").unwrap();
/// assert_eq!(w, 1920);
///
/// let t: &str = beve::from_field(&bytes, "/title").unwrap();
/// assert_eq!(t, "Main");
/// ```
pub fn from_field<'de, T: Deserialize<'de>>(bytes: &'de [u8], pointer: &str) -> Result<T> {
    let tokens = parse_json_pointer(pointer)?;
    let mut pos: usize = 0;
    navigate_to(bytes, &mut pos, &tokens)?;
    let mut de = crate::de::Deserializer::from_slice_at(bytes, pos);
    let value = T::deserialize(&mut de)?;
    Ok(value)
}

/// Deserialize a sub-range of a typed numeric array located at `pointer`.
///
/// `start` is 0-based and `count` is the number of elements to read.
/// Elements before `start` are skipped in constant time (a byte offset is
/// computed from the element width), so this is efficient even for large
/// arrays.
///
/// # Supported element types
///
/// All fixed-width numeric typed arrays: `u8`..`u64`, `i8`..`i64`,
/// `f32`, `f64`. Boolean and string typed arrays are **not** supported
/// for slicing.
///
/// # Errors
///
/// Returns an error if:
///
/// - Navigation to `pointer` fails (same rules as [`from_field`]).
/// - The target is not a typed numeric array.
/// - `start + count` exceeds the array length.
///
/// # Note
///
/// Because slicing constructs a temporary buffer, the target type must be
/// [`DeserializeOwned`](serde::de::DeserializeOwned) (i.e. cannot borrow from the input).
///
/// # Examples
///
/// ```rust
/// use serde::Serialize;
///
/// #[derive(Serialize)]
/// struct Recording {
///     samples: Vec<f32>,
/// }
///
/// let rec = Recording { samples: (0..10_000).map(|i| i as f32).collect() };
/// let bytes = beve::to_vec(&rec).unwrap();
///
/// // Read only samples [500..503)
/// let slice: Vec<f32> = beve::from_field_slice(&bytes, "/samples", 500, 3).unwrap();
/// assert_eq!(slice, vec![500.0, 501.0, 502.0]);
/// ```
pub fn from_field_slice<T: serde::de::DeserializeOwned>(
    bytes: &[u8],
    pointer: &str,
    start: usize,
    count: usize,
) -> Result<T> {
    let tokens = parse_json_pointer(pointer)?;
    let mut pos: usize = 0;
    navigate_to(bytes, &mut pos, &tokens)?;

    // Read the typed-array header and produce a sub-slice view
    let header = read_byte(bytes, &mut pos)?;
    let ty = parse_type(header);
    if ty != TYPE_TYPED_ARRAY {
        return Err(Error::InvalidType(
            "from_field_slice: target is not a typed array",
        ));
    }

    let cat = parse_subtype(header);
    if cat == ARRAY_BOOL_OR_STRING {
        return Err(Error::Unsupported(
            "slicing not supported for boolean or string typed arrays",
        ));
    }

    let bc = parse_byte_count_code(header);
    let elem_size = byte_count_to_bytes(bc)?;
    let len = read_size(bytes, &mut pos)? as usize;

    let end = start.checked_add(count).ok_or(Error::InvalidSize)?;
    if end > len {
        return Err(Error::msg(format!(
            "slice [{start}..{end}) out of bounds (array length {len})"
        )));
    }

    // Skip past `start` elements
    let skip_bytes = elem_size.checked_mul(start).ok_or(Error::InvalidSize)?;
    pos = pos.checked_add(skip_bytes).ok_or(Error::InvalidSize)?;

    // Build a synthetic typed-array payload: same header + count + sliced data
    let slice_data_len = elem_size.checked_mul(count).ok_or(Error::InvalidSize)?;
    if pos + slice_data_len > bytes.len() {
        return Err(Error::Eof);
    }

    let mut buf = Vec::with_capacity(1 + 8 + slice_data_len);
    buf.push(header);
    crate::size::write_size(count as u64, &mut buf);
    buf.extend_from_slice(&bytes[pos..pos + slice_data_len]);

    // Deserialize through the normal path from the synthetic buffer.
    // We cannot borrow from `bytes` through the synthetic buffer, so use
    // DeserializeOwned via from_slice which works for owned types.
    let mut de = crate::de::Deserializer::from_slice_at(&buf, 0);
    let value = T::deserialize(&mut de)?;
    Ok(value)
}

// ---------------------------------------------------------------------------
// JSON Pointer parsing (RFC 6901)
// ---------------------------------------------------------------------------

fn parse_json_pointer(pointer: &str) -> Result<Vec<String>> {
    if pointer.is_empty() {
        return Ok(Vec::new());
    }
    if !pointer.starts_with('/') {
        return Err(Error::msg(format!(
            "invalid JSON Pointer: must be empty or start with '/' (got '{pointer}')"
        )));
    }
    let parts: Vec<String> = pointer[1..]
        .split('/')
        .map(|p| {
            // Unescape: ~1 -> /, ~0 -> ~ (order per RFC 6901)
            p.replace("~1", "/").replace("~0", "~")
        })
        .collect();
    Ok(parts)
}

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------

fn navigate_to(input: &[u8], pos: &mut usize, tokens: &[String]) -> Result<()> {
    for token in tokens {
        let header = read_byte(input, pos)?;
        let ty = parse_type(header);

        match ty {
            TYPE_GENERIC_ARRAY => {
                let len = read_size(input, pos)? as usize;
                let idx: usize = token.parse().map_err(|_| {
                    Error::msg(format!(
                        "array index '{token}' is not a valid non-negative integer"
                    ))
                })?;
                if idx >= len {
                    return Err(Error::msg(format!(
                        "array index {idx} out of bounds (length {len})"
                    )));
                }
                for _ in 0..idx {
                    skip_value(input, pos)?;
                }
                // pos now points at element `idx`; continue to next token
            }
            TYPE_OBJECT => {
                let key_type = parse_subtype(header);
                let bc = parse_byte_count_code(header);
                let count = read_size(input, pos)? as usize;

                let found = match key_type {
                    KEY_STRING => find_string_key(input, pos, count, token)?,
                    KEY_SIGNED => {
                        let target: i64 = token.parse().map_err(|_| {
                            Error::msg(format!(
                                "object has integer keys but pointer token '{token}' is not an integer"
                            ))
                        })?;
                        find_signed_key(input, pos, count, bc, target)?
                    }
                    KEY_UNSIGNED => {
                        let target: u64 = token.parse().map_err(|_| {
                            Error::msg(format!(
                                "object has integer keys but pointer token '{token}' is not an integer"
                            ))
                        })?;
                        find_unsigned_key(input, pos, count, bc, target)?
                    }
                    _ => return Err(Error::InvalidHeader(header)),
                };

                if !found {
                    return Err(Error::msg(format!("key '{token}' not found in object")));
                }
                // pos now points at the value for the matched key
            }
            _ => {
                return Err(Error::msg(format!(
                    "cannot navigate into type {ty} at pointer segment '{token}'"
                )));
            }
        }
    }
    Ok(())
}

fn find_string_key(input: &[u8], pos: &mut usize, count: usize, target: &str) -> Result<bool> {
    for _ in 0..count {
        let key_len = read_size(input, pos)? as usize;
        if *pos + key_len > input.len() {
            return Err(Error::Eof);
        }
        let key_bytes = &input[*pos..*pos + key_len];
        *pos += key_len;
        let key = from_utf8(key_bytes).map_err(|_| Error::InvalidType("invalid utf-8 in key"))?;
        if key == target {
            return Ok(true);
        }
        skip_value(input, pos)?;
    }
    Ok(false)
}

fn find_signed_key(
    input: &[u8],
    pos: &mut usize,
    count: usize,
    byte_code: u8,
    target: i64,
) -> Result<bool> {
    let nbytes = byte_count_to_bytes(byte_code)?;
    for _ in 0..count {
        let key_val = read_signed_int(input, pos, nbytes)?;
        if key_val == target as i128 {
            return Ok(true);
        }
        skip_value(input, pos)?;
    }
    Ok(false)
}

fn find_unsigned_key(
    input: &[u8],
    pos: &mut usize,
    count: usize,
    byte_code: u8,
    target: u64,
) -> Result<bool> {
    let nbytes = byte_count_to_bytes(byte_code)?;
    for _ in 0..count {
        let key_val = read_unsigned_int(input, pos, nbytes)?;
        if key_val == target as u128 {
            return Ok(true);
        }
        skip_value(input, pos)?;
    }
    Ok(false)
}

// ---------------------------------------------------------------------------
// Skip a single BEVE value (advance `pos` past it without allocating)
// ---------------------------------------------------------------------------

/// Skip a single BEVE value starting at `input[*pos]`, advancing `*pos`
/// past the entire value without deserializing or allocating.
///
/// All BEVE types are handled: null, booleans, numbers, strings, objects
/// (with string or integer keys), typed arrays (numeric, boolean, string),
/// generic arrays, and extensions (enums, matrices, complex numbers).
///
/// This is the building block used by [`from_field`] to skip over
/// non-matching object values and array elements during navigation. It
/// can also be used directly when scanning a BEVE payload.
///
/// # Example
///
/// ```rust
/// // Serialize two values back-to-back into one buffer
/// let mut buf = beve::to_vec(&42u32).unwrap();
/// let second_start = buf.len();
/// buf.extend_from_slice(&beve::to_vec(&"hello").unwrap());
///
/// // Skip the first value
/// let mut pos = 0;
/// beve::skip_value(&buf, &mut pos).unwrap();
/// assert_eq!(pos, second_start);
/// ```
pub fn skip_value(input: &[u8], pos: &mut usize) -> Result<()> {
    let header = read_byte(input, pos)?;
    let ty = parse_type(header);

    match ty {
        TYPE_NULL_BOOL => {
            // Fully encoded in the header byte; nothing else to consume.
        }
        TYPE_NUMBER => {
            let bc = parse_byte_count_code(header);
            let nbytes = byte_count_to_bytes(bc)?;
            advance(input, pos, nbytes)?;
        }
        TYPE_STRING => {
            let len = read_size(input, pos)? as usize;
            advance(input, pos, len)?;
        }
        TYPE_OBJECT => {
            let key_type = parse_subtype(header);
            let bc = parse_byte_count_code(header);
            let count = read_size(input, pos)? as usize;

            match key_type {
                KEY_STRING => {
                    for _ in 0..count {
                        let key_len = read_size(input, pos)? as usize;
                        advance(input, pos, key_len)?; // skip key bytes
                        skip_value(input, pos)?; // skip value
                    }
                }
                KEY_SIGNED | KEY_UNSIGNED => {
                    let key_size = byte_count_to_bytes(bc)?;
                    for _ in 0..count {
                        advance(input, pos, key_size)?; // skip key
                        skip_value(input, pos)?; // skip value
                    }
                }
                _ => return Err(Error::InvalidHeader(header)),
            }
        }
        TYPE_TYPED_ARRAY => {
            let cat = parse_subtype(header);
            let bc = parse_byte_count_code(header);
            let len = read_size(input, pos)? as usize;

            if cat == ARRAY_BOOL_OR_STRING {
                let is_string = (header & 0b0010_0000) != 0;
                if is_string {
                    for _ in 0..len {
                        let s = read_size(input, pos)? as usize;
                        advance(input, pos, s)?;
                    }
                } else {
                    // boolean: ceil(len / 8) packed bytes
                    let packed = len.div_ceil(8);
                    advance(input, pos, packed)?;
                }
            } else {
                let elem_size = byte_count_to_bytes(bc)?;
                let total = elem_size.checked_mul(len).ok_or(Error::InvalidSize)?;
                advance(input, pos, total)?;
            }
        }
        TYPE_GENERIC_ARRAY => {
            let len = read_size(input, pos)? as usize;
            for _ in 0..len {
                skip_value(input, pos)?;
            }
        }
        TYPE_EXTENSION => {
            let ext = parse_extension_id(header);
            match ext {
                EXT_TYPE_TAG => {
                    // tag (number or string) + value
                    skip_value(input, pos)?; // tag
                    skip_value(input, pos)?; // value
                }
                EXT_DELIMITER => { /* nothing to skip */ }
                EXT_MATRICES => {
                    advance(input, pos, 1)?; // layout byte
                    skip_value(input, pos)?; // extents
                    skip_value(input, pos)?; // data
                }
                EXT_COMPLEX => {
                    let ch = read_byte(input, pos)?;
                    let is_array = (ch & 0x01) != 0;
                    let cbc = (ch >> 5) & 0x07;
                    let elem_bytes = byte_count_to_bytes(cbc)?;
                    if is_array {
                        let n = read_size(input, pos)? as usize;
                        let total = (2usize)
                            .checked_mul(n)
                            .and_then(|v| v.checked_mul(elem_bytes))
                            .ok_or(Error::InvalidSize)?;
                        advance(input, pos, total)?;
                    } else {
                        // scalar complex: 2 components
                        let total = 2usize
                            .checked_mul(elem_bytes)
                            .ok_or(Error::InvalidSize)?;
                        advance(input, pos, total)?;
                    }
                }
                _ => {
                    return Err(Error::Unsupported("cannot skip unsupported extension"));
                }
            }
        }
        _ => return Err(Error::InvalidHeader(header)),
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Low-level helpers
// ---------------------------------------------------------------------------

#[inline]
fn byte_count_to_bytes(code: u8) -> Result<usize> {
    match code {
        0 => Ok(1),
        1 => Ok(2),
        2 => Ok(4),
        3 => Ok(8),
        4 => Ok(16),
        _ => Err(Error::Unsupported("byte width > 16 not supported")),
    }
}

#[inline]
fn read_byte(input: &[u8], pos: &mut usize) -> Result<u8> {
    if *pos >= input.len() {
        return Err(Error::Eof);
    }
    let b = input[*pos];
    *pos += 1;
    Ok(b)
}

#[inline]
fn advance(input: &[u8], pos: &mut usize, n: usize) -> Result<()> {
    let end = pos.checked_add(n).ok_or(Error::InvalidSize)?;
    if end > input.len() {
        return Err(Error::Eof);
    }
    *pos = end;
    Ok(())
}

fn read_signed_int(input: &[u8], pos: &mut usize, nbytes: usize) -> Result<i128> {
    if *pos + nbytes > input.len() {
        return Err(Error::Eof);
    }
    let s = &input[*pos..*pos + nbytes];
    *pos += nbytes;
    let mut buf = [0u8; 16];
    buf[..nbytes].copy_from_slice(s);
    if nbytes < 16 && (s[nbytes - 1] & 0x80) != 0 {
        for b in &mut buf[nbytes..] {
            *b = 0xFF;
        }
    }
    Ok(i128::from_le_bytes(buf))
}

fn read_unsigned_int(input: &[u8], pos: &mut usize, nbytes: usize) -> Result<u128> {
    if *pos + nbytes > input.len() {
        return Err(Error::Eof);
    }
    let s = &input[*pos..*pos + nbytes];
    *pos += nbytes;
    let mut buf = [0u8; 16];
    buf[..nbytes].copy_from_slice(s);
    Ok(u128::from_le_bytes(buf))
}
