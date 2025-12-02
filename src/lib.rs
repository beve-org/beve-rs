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

mod de;
mod error;
mod ext;
pub mod fast;
mod header;
mod json;
mod ser;
mod size;
mod value;

pub use crate::de::{from_slice, Deserializer};
pub use crate::error::{Error, Result};
pub use crate::ext::{Complex, ComplexSlice};
pub use crate::ext::{Matrix, MatrixLayout};
pub use crate::fast::{
    to_vec_bool_slice, to_vec_complex32, to_vec_complex32_slice, to_vec_complex64,
    to_vec_complex64_slice, to_vec_str_slice, to_vec_string_slice, to_vec_typed_slice,
    write_bool_slice, write_str_slice, write_string_slice, write_typed_slice, BeveTypedSlice,
};
pub use crate::json::{
    beve_slice_to_json, beve_slice_to_json_string, json_slice_to_beve, json_str_to_beve,
};
pub use crate::ser::{to_vec, to_vec_with_options, EnumEncoding, Serializer, SerializerOptions};
pub use crate::value::{
    from_value, from_value_ref, BigInt, BigIntKey, Key, Number, Object, Value, ValueError,
};

/// BEVE-specific utilities and helper types.
pub mod util {
    pub use crate::ext::MatrixLayout;
}

use std::io::{Read, Write};

/// Serialize a value to any writer. For unknown-length containers, this uses an internal buffer.
pub fn to_writer<W: Write, T: serde::Serialize>(mut writer: W, value: &T) -> Result<()> {
    let bytes = to_vec(value)?;
    writer
        .write_all(&bytes)
        .map_err(|e| Error::MessageOwned(e.to_string()))
}

/// Serialize a value to any writer with custom options.
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
