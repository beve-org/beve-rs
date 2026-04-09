//! Header construction/parsing for BEVE.

use crate::error::{Error, Result};

pub const TYPE_NULL_BOOL: u8 = 0;
pub const TYPE_NUMBER: u8 = 1;
pub const TYPE_STRING: u8 = 2;
pub const TYPE_OBJECT: u8 = 3;
pub const TYPE_TYPED_ARRAY: u8 = 4;
pub const TYPE_GENERIC_ARRAY: u8 = 5;
pub const TYPE_EXTENSION: u8 = 6;

// Number subtypes (bits 3..4)
pub const NUM_FLOAT: u8 = 0;
pub const NUM_SIGNED: u8 = 1;
pub const NUM_UNSIGNED: u8 = 2;

// Object key types (bits 3..4)
pub const KEY_STRING: u8 = 0;
pub const KEY_SIGNED: u8 = 1;
pub const KEY_UNSIGNED: u8 = 2;

// Typed array category (bits 3..4)
pub const ARRAY_FLOAT: u8 = 0;
pub const ARRAY_SIGNED: u8 = 1;
pub const ARRAY_UNSIGNED: u8 = 2;
pub const ARRAY_BOOL_OR_STRING: u8 = 3;

// Extension ids (bits 3..7)
/// Data delimiter extension id (like newline in NDJSON).
pub const EXT_DELIMITER: u8 = 0;
pub const EXT_TYPE_TAG: u8 = 1;
pub const EXT_MATRICES: u8 = 2;
pub const EXT_COMPLEX: u8 = 3;

#[inline]
pub fn make_header(ty: u8, subtype: u8, byte_count_code: u8) -> u8 {
    // Generic constructor for headers that follow: [byte_count:3][subtype:2][type:3]
    (byte_count_code << 5) | ((subtype & 0b11) << 3) | (ty & 0b111)
}

#[inline]
pub const fn make_extension_header(ext_id: u8) -> u8 {
    ((ext_id & 0x1f) << 3) | TYPE_EXTENSION
}

#[inline]
pub fn parse_type(header: u8) -> u8 {
    header & 0b111
}

#[inline]
pub fn parse_subtype(header: u8) -> u8 {
    (header >> 3) & 0b11
}

#[inline]
pub fn parse_byte_count_code(header: u8) -> u8 {
    header >> 5
}

#[inline]
pub fn parse_extension_id(header: u8) -> u8 {
    header >> 3
}

#[inline]
pub fn is_bool(header: u8) -> bool {
    (header & 0b111) == TYPE_NULL_BOOL && (header & 0b1000) != 0
}

#[inline]
pub fn bool_value(header: u8) -> Result<bool> {
    if !is_bool(header) {
        return Err(Error::InvalidType("not a boolean header"));
    }
    Ok((header & 0b10000) != 0)
}
