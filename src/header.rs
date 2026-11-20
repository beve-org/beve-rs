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

// Kinds within the ARRAY_BOOL_OR_STRING category (bits 5..7, the field other
// categories use for the byte count). 3 through 7 are undefined.
pub const ARRAY_KIND_BOOL: u8 = 0;
pub const ARRAY_KIND_STRING: u8 = 1;
/// Zero-copy aligned numeric array; see [`crate::aligned`] for its layout.
pub const ARRAY_KIND_ALIGNED: u8 = 2;

// Extension ids (bits 3..7)
/// Data delimiter extension id (like newline in NDJSON).
pub const EXT_DELIMITER: u8 = 0;
pub const EXT_TYPE_TAG: u8 = 1;
pub const EXT_MATRICES: u8 = 2;
pub const EXT_COMPLEX: u8 = 3;

/// Byte width of one numeric scalar of class `class` (float, signed or
/// unsigned) and `byte_code`, or `None` when the two name a width BEVE does not
/// define. The class values coincide across a number header's subtype, a typed
/// array's category and a complex header's class, so this one rule sizes a
/// number, a typed-array element and a complex component (`re` alone, or `im`
/// alone) alike.
///
/// This is the single source of truth for the rule, because the rule has one
/// exception that is easy to miss and silent when missed: the FLOAT class
/// encodes both of its half widths as 2 bytes — `bf16` at `byte_code` 0 and
/// `f16` at `byte_code` 1 — so it does **not** follow the generic
/// `1 << byte_code` width, which would give 1 and 2 there. A path that reaches
/// for the generic formula mis-sizes every `bf16`, and sizes `f16` correctly
/// only by coincidence. From `byte_code` 2 up the generic width holds for every
/// class, through the 16-byte widths at `byte_code` 4 (`i128`, `u128`, and the
/// 128-bit float Glaze writes for `long double` and `std::float128_t`).
#[inline]
pub fn numeric_width(class: u8, byte_code: u8) -> Option<usize> {
    match (class, byte_code) {
        (NUM_FLOAT, 0 | 1) => Some(2),
        (NUM_FLOAT | NUM_SIGNED | NUM_UNSIGNED, 0..=4) => Some(1usize << byte_code),
        _ => None,
    }
}

/// Byte width of an integer at `byte_code`, for the places whose class is fixed
/// to signed or unsigned: integer object keys and integer reads. Anything whose
/// class is read from the input goes through [`numeric_width`] instead.
#[inline]
pub fn integer_width(byte_code: u8) -> Result<usize> {
    numeric_width(NUM_UNSIGNED, byte_code)
        .ok_or(Error::Unsupported("byte width > 16 not supported"))
}

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
