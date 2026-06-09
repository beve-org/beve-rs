//! BEVE - Binary Efficient Versatile Encoding
//!
//! High-performance, tagged binary format designed for scientific computing.
//! This crate provides a robust, fast, and ergonomic implementation with serde support.
//!
//! - Little-endian encoding
//! - Direct struct serialization via `serde::{Serialize, Deserialize}`
//! - Typed arrays for numeric, boolean, and string sequences when possible
//! - Object keys as strings or integer types
//! - Enum support via BEVE type-tag extension
//! - Zero-copy deserialization for `&str` fields (borrow directly from the input buffer)
//! - Validation helpers for checking BEVE payload integrity without deserialization
//! - Selective field loading via [JSON Pointer (RFC 6901)](https://datatracker.ietf.org/doc/html/rfc6901)
//!   with optional array slicing — see [`from_field`] and [`from_field_slice`]
//!
//! Example
//!
//! ```rust
//! use serde::{Serialize, Deserialize};
//!
//! #[derive(Serialize, Deserialize, Debug, PartialEq)]
//! struct Point { x: f64, y: f64 }
//!
//! let p = Point { x: 1.0, y: -2.0 };
//! let bytes = beve::to_vec(&p).unwrap();
//! let p2: Point = beve::from_slice(&bytes).unwrap();
//! assert_eq!(p, p2);
//! ```
//!
//! Validate a payload without decoding it into a concrete Rust type:
//!
//! ```rust
//! use std::io::Cursor;
//!
//! let bytes = beve::to_vec(&vec![1u32, 2, 3]).unwrap();
//! beve::validate_slice(&bytes).unwrap();
//! beve::validate_reader(Cursor::new(bytes)).unwrap();
//! ```

pub mod aligned;
mod de;
mod error;
mod ext;
pub mod fast;
pub mod field;
mod header;
mod json;
#[cfg(feature = "mat")]
pub mod mat;
#[cfg(feature = "mat")]
mod raw;
mod ser;
mod size;
mod streaming_de;
mod streaming_ser;
mod value;

pub use crate::aligned::{
    aligned_typed_slice_size, read_aligned_typed_slice, read_aligned_typed_slice_ref,
    to_vec_aligned_typed_slice, write_aligned_typed_slice,
};
pub use crate::de::{Deserializer, from_slice, validate_slice};
pub use crate::error::{Error, Result};
pub use crate::ext::{
    Complex, ComplexSlice, DecodedMatrix, Matrix, MatrixDecodeMode, MatrixLayout, MatrixOwned,
    RawMatrix, TypedSlice, complex, decode_matrix_slice,
};
pub use crate::fast::{
    BeveTypedSlice, complex_slice_size, read_complex_slice, read_typed_slice, to_vec_bool_slice,
    to_vec_complex_slice, to_vec_str_slice, to_vec_string_slice, to_vec_typed_slice,
    to_writer_complex_slice, to_writer_typed_slice, typed_slice_size, write_bool_slice,
    write_complex_slice, write_str_slice, write_string_slice, write_typed_slice,
};
pub use crate::field::{from_field, from_field_slice, skip_value};
pub use crate::json::{
    beve_slice_to_json, beve_slice_to_json_string, json_slice_to_beve, json_str_to_beve,
};
#[cfg(feature = "mat")]
pub use crate::mat::{
    Compression, InvalidNamePolicy, MatV73Options, NullPolicy, OneDimensionalMode, RootBinding,
    RowMajorPolicy, UnsupportedPolicy, beve_file_to_mat_v73_file, beve_slice_to_mat_v73_bytes,
    beve_slice_to_mat_v73_file,
};
pub use crate::ser::{
    EnumEncoding, Serializer, SerializerOptions, to_vec, to_vec_into, to_vec_into_with_options,
    to_vec_with_options,
};
pub use crate::streaming_de::{StreamingDeserializer, from_reader_streaming};
pub use crate::streaming_ser::{
    StreamingSerializer, serialized_size, serialized_size_with_options, to_writer_streaming,
    to_writer_streaming_with_options,
};
pub use crate::value::{
    BigInt, BigIntKey, Key, Number, Object, Value, ValueError, from_value, from_value_ref,
};

/// BEVE-specific utilities and helper types.
pub mod util {
    pub use crate::ext::MatrixLayout;
}

/// The single-byte data delimiter (BEVE extension 0).
///
/// Analogous to `\n` in Newline Delimited JSON (NDJSON). Write this byte
/// between consecutive BEVE values in a stream so that readers can
/// unambiguously identify record boundaries.
///
/// [`from_slice`], [`from_reader_streaming`], and their underlying
/// deserializers skip delimiter bytes transparently during deserialization.
/// Note that [`validate_slice`] expects exactly one value with no trailing
/// bytes, so it will reject delimiter-separated streams.
pub const DATA_DELIMITER: u8 = crate::header::make_extension_header(crate::header::EXT_DELIMITER);

/// Write a data delimiter byte to the given writer.
///
/// This is a convenience wrapper around writing [`DATA_DELIMITER`]. Use it
/// between consecutive BEVE values when streaming to a file.
///
/// # Example
///
/// ```rust
/// use serde::{Serialize, Deserialize};
/// use std::io::Cursor;
///
/// #[derive(Serialize, Deserialize, Debug, PartialEq)]
/// struct Record { id: u32, value: f64 }
///
/// let mut buf = Vec::new();
/// let r1 = Record { id: 1, value: 1.5 };
/// let r2 = Record { id: 2, value: 2.5 };
///
/// beve::to_writer_streaming(&mut buf, &r1).unwrap();
/// beve::write_delimiter(&mut buf).unwrap();
/// beve::to_writer_streaming(&mut buf, &r2).unwrap();
///
/// // Read back sequentially
/// let mut cursor = Cursor::new(&buf);
/// let back1: Record = beve::from_reader_streaming(&mut cursor).unwrap();
/// let back2: Record = beve::from_reader_streaming(&mut cursor).unwrap();
/// assert_eq!(back1, r1);
/// assert_eq!(back2, r2);
/// ```
pub fn write_delimiter<W: Write>(mut writer: W) -> Result<()> {
    writer
        .write_all(&[DATA_DELIMITER])
        .map_err(|e| Error::MessageOwned(e.to_string()))
}

use std::io::{Read, Write};

/// Serialize a value into any writer.
///
/// This serializes into an internal [`Vec<u8>`] in a single pass and then writes
/// those bytes into `writer`; it does **not** encode directly into the writer. The
/// name follows serde's `to_writer` convention (it accepts any [`Write`]) and is a
/// convenience, not a zero-copy guarantee.
///
/// For direct encoding into the writer with no intermediate `Vec`, use
/// [`to_writer_streaming`]. That path writes each BEVE token straight to the writer
/// in a single pass; the trade-off is that every container must have a known length
/// (it errors on `serialize_seq(None)` / `serialize_map(None)`), whereas this
/// function also supports unknown-length containers by back-patching sizes in the
/// buffer.
pub fn to_writer<W: Write, T: serde::Serialize>(mut writer: W, value: &T) -> Result<()> {
    let bytes = to_vec(value)?;
    writer
        .write_all(&bytes)
        .map_err(|e| Error::MessageOwned(e.to_string()))
}

/// Serialize a value into any writer with custom options.
///
/// Like [`to_writer`], this serializes into an intermediate [`Vec<u8>`] before
/// writing into `writer` rather than encoding directly. For direct, single-pass
/// streaming with no intermediate buffer, see [`to_writer_streaming_with_options`].
pub fn to_writer_with_options<W: Write, T: serde::Serialize>(
    mut writer: W,
    value: &T,
    opts: SerializerOptions,
) -> Result<()> {
    let bytes = to_vec_with_options(value, opts)?;
    writer
        .write_all(&bytes)
        .map_err(|e| Error::MessageOwned(e.to_string()))
}

/// Deserialize a value by reading all bytes from a reader into a buffer first.
pub fn from_reader<R: Read, T: serde::de::DeserializeOwned>(mut reader: R) -> Result<T> {
    let mut buf = Vec::new();
    reader
        .read_to_end(&mut buf)
        .map_err(|e| Error::MessageOwned(e.to_string()))?;
    from_slice(&buf)
}

/// Validate BEVE data from any reader without deserializing into a concrete type.
///
/// This enforces the same semantics as [`validate_slice`]: exactly one valid BEVE
/// value and no trailing bytes.
///
/// # Example
///
/// ```rust
/// use std::io::Cursor;
///
/// let bytes = beve::to_vec(&123u32).unwrap();
/// beve::validate_reader(Cursor::new(bytes)).unwrap();
/// ```
pub fn validate_reader<R: Read>(mut reader: R) -> Result<()> {
    let mut buf = Vec::new();
    reader
        .read_to_end(&mut buf)
        .map_err(|e| Error::MessageOwned(e.to_string()))?;
    validate_slice(&buf)
}
