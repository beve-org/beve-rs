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

mod error;
mod size;
mod header;
mod ser;
mod de;

pub use crate::error::{Error, Result};
pub use crate::ser::{to_vec, Serializer};
pub use crate::de::{from_slice, Deserializer};

/// BEVE-specific utilities and helper types.
pub mod util {
    /// Matrix layout for the matrices extension (not yet implemented in serializer/deserializer).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum MatrixLayout { Right, Left }
}

