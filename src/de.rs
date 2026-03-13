use half::{bf16, f16};
use serde::Deserialize;
use serde::de::value::{BoolDeserializer, BorrowedStrDeserializer};
use serde::de::{self, Visitor};
use serde::forward_to_deserialize_any;

use crate::error::{Error, Result};
use crate::header::*;
use crate::size::read_size;
use simdutf8::basic::from_utf8;

#[inline]
fn decode_utf8<'de>(bytes: &'de [u8], context: &'static str) -> Result<&'de str> {
    from_utf8(bytes).map_err(|_| Error::InvalidType(context))
}

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

#[derive(Clone, Copy)]
enum HalfKind {
    Bf16,
    F16,
}

impl HalfKind {
    #[inline]
    fn to_f32(self, bits: u16) -> f32 {
        match self {
            HalfKind::Bf16 => bf16::from_bits(bits).to_f32(),
            HalfKind::F16 => f16::from_bits(bits).to_f32(),
        }
    }

    #[inline]
    fn to_f64(self, bits: u16) -> f64 {
        self.to_f32(bits) as f64
    }
}

#[inline]
fn make_seq_unsigned<'de>(
    de: &mut Deserializer<'de>,
    len: usize,
    byte_code: u8,
) -> Result<SeqAccessUnsigned<'de>> {
    let elem_size = byte_count_to_bytes(byte_code)?;
    let total = elem_size.checked_mul(len).ok_or(Error::InvalidSize)?;
    let data = de.read_exact(total)?;
    Ok(SeqAccessUnsigned {
        data,
        remaining: len,
        elem_size,
        offset: 0,
    })
}

#[inline]
fn make_seq_signed<'de>(
    de: &mut Deserializer<'de>,
    len: usize,
    byte_code: u8,
) -> Result<SeqAccessSigned<'de>> {
    let elem_size = byte_count_to_bytes(byte_code)?;
    let total = elem_size.checked_mul(len).ok_or(Error::InvalidSize)?;
    let data = de.read_exact(total)?;
    Ok(SeqAccessSigned {
        data,
        remaining: len,
        elem_size,
        offset: 0,
    })
}

#[inline]
fn make_seq_half<'de>(
    de: &mut Deserializer<'de>,
    len: usize,
    kind: HalfKind,
) -> Result<SeqAccessHalf<'de>> {
    let total = len.checked_mul(2).ok_or(Error::InvalidSize)?;
    let data = de.read_exact(total)?;
    Ok(SeqAccessHalf {
        data,
        remaining: len,
        offset: 0,
        kind,
    })
}

#[inline]
fn make_seq_float32<'de>(de: &mut Deserializer<'de>, len: usize) -> Result<SeqAccessFloat32<'de>> {
    let total = 4usize.checked_mul(len).ok_or(Error::InvalidSize)?;
    let data = de.read_exact(total)?;
    Ok(SeqAccessFloat32 {
        data,
        remaining: len,
        offset: 0,
    })
}

#[inline]
fn make_seq_float64<'de>(de: &mut Deserializer<'de>, len: usize) -> Result<SeqAccessFloat64<'de>> {
    let total = 8usize.checked_mul(len).ok_or(Error::InvalidSize)?;
    let data = de.read_exact(total)?;
    Ok(SeqAccessFloat64 {
        data,
        remaining: len,
        offset: 0,
    })
}

pub struct Deserializer<'de> {
    input: &'de [u8],
    pos: usize,
}

/// Deserialize a value from a BEVE byte slice.
///
/// Supports zero-copy deserialization: types containing `&'de str` fields
/// borrow directly from the input buffer instead of allocating.
///
/// # Example
///
/// ```rust
/// use serde::Deserialize;
///
/// #[derive(Deserialize)]
/// struct Info<'a> {
///     name: &'a str,
///     count: u32,
/// }
///
/// let bytes = beve::to_vec(&("hello", 42u32)).unwrap();
/// ```
pub fn from_slice<'de, T: Deserialize<'de>>(bytes: &'de [u8]) -> Result<T> {
    let mut de = Deserializer {
        input: bytes,
        pos: 0,
    };
    let t = T::deserialize(&mut de)?;
    if de.pos != de.input.len() {
        // trailing bytes allowed? We'll ignore for now
    }
    Ok(t)
}

/// Validate that `bytes` contains exactly one well-formed BEVE value.
///
/// This performs structural validation only and does not materialize a target type.
/// The full input must be consumed by a single value; trailing bytes are rejected.
///
/// # Example
///
/// ```rust
/// let bytes = beve::to_vec(&vec![1u32, 2, 3]).unwrap();
/// beve::validate_slice(&bytes).unwrap();
/// ```
pub fn validate_slice(bytes: &[u8]) -> Result<()> {
    let mut de = Deserializer {
        input: bytes,
        pos: 0,
    };
    let _ = <de::IgnoredAny as serde::Deserialize>::deserialize(&mut de)?;
    if de.pos != de.input.len() {
        return Err(Error::InvalidType("unexpected trailing BEVE data"));
    }
    Ok(())
}

impl<'de> Deserializer<'de> {
    #[inline]
    fn remaining(&self) -> usize {
        self.input.len().saturating_sub(self.pos)
    }

    #[inline]
    fn peek_byte(&self) -> Result<u8> {
        self.input.get(self.pos).copied().ok_or(Error::Eof)
    }

    #[inline]
    fn read_byte(&mut self) -> Result<u8> {
        if self.pos >= self.input.len() {
            Err(Error::Eof)
        } else {
            let b = self.input[self.pos];
            self.pos += 1;
            Ok(b)
        }
    }

    #[inline]
    fn read_exact<'a>(&'a mut self, n: usize) -> Result<&'de [u8]> {
        if self.remaining() < n {
            return Err(Error::Eof);
        }
        let s = &self.input[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn parse_bool(&mut self, header: u8) -> Result<bool> {
        bool_value(header)
    }

    fn parse_signed(&mut self, code: u8) -> Result<i128> {
        let nbytes = match code {
            0 => 1,
            1 => 2,
            2 => 4,
            3 => 8,
            4 => 16,
            _ => return Err(Error::Unsupported("signed width > 128")),
        };
        let s = self.read_exact(nbytes)?;
        let mut buf = [0u8; 16];
        buf[..nbytes].copy_from_slice(s);
        // Sign-extend if the highest bit of the most significant byte is set
        if nbytes < 16 && (s[nbytes - 1] & 0x80) != 0 {
            for b in &mut buf[nbytes..] {
                *b = 0xFF;
            }
        }
        Ok(i128::from_le_bytes(buf))
    }

    fn parse_unsigned(&mut self, code: u8) -> Result<u128> {
        let nbytes = match code {
            0 => 1,
            1 => 2,
            2 => 4,
            3 => 8,
            4 => 16,
            _ => return Err(Error::Unsupported("unsigned width > 128")),
        };
        let s = self.read_exact(nbytes)?;
        let mut buf = [0u8; 16];
        buf[..nbytes].copy_from_slice(s);
        Ok(u128::from_le_bytes(buf))
    }

    fn parse_bf16_bits(&mut self) -> Result<u16> {
        let s = self.read_exact(2)?;
        Ok(u16::from_le_bytes([s[0], s[1]]))
    }

    fn parse_f16_bits(&mut self) -> Result<u16> {
        let s = self.read_exact(2)?;
        Ok(u16::from_le_bytes([s[0], s[1]]))
    }

    fn parse_f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.read_exact(4)?.try_into().unwrap()))
    }
    fn parse_f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.read_exact(8)?.try_into().unwrap()))
    }

    fn deserialize_half_newtype<V: Visitor<'de>>(
        &mut self,
        visitor: V,
        kind: HalfKind,
    ) -> Result<V::Value> {
        let header = self.peek_byte()?;
        if parse_type(header) == TYPE_NUMBER && parse_subtype(header) == NUM_FLOAT {
            let byte_code = parse_byte_count_code(header);
            let matches = match kind {
                HalfKind::Bf16 => byte_code == 0,
                HalfKind::F16 => byte_code == 1,
            };
            if matches {
                self.read_byte()?; // consume header
                let bits = match kind {
                    HalfKind::Bf16 => self.parse_bf16_bits()?,
                    HalfKind::F16 => self.parse_f16_bits()?,
                };
                let deser = HalfBitsDeserializer { kind, bits };
                return visitor.visit_newtype_struct(deser);
            }
        }
        serde::Deserializer::deserialize_any(self, visitor)
    }

    fn parse_string_borrowed(&mut self) -> Result<&'de str> {
        let len = read_size(self.input, &mut self.pos)? as usize;
        let s = self.read_exact(len)?;
        decode_utf8(s, "invalid utf-8")
    }

    fn read_enum_tag(&mut self) -> Result<EnumTag<'de>> {
        let header = self.read_byte()?;
        match parse_type(header) {
            TYPE_NUMBER => {
                let subtype = parse_subtype(header);
                let bc = parse_byte_count_code(header);
                match subtype {
                    NUM_UNSIGNED => Ok(EnumTag::Index(self.parse_unsigned(bc)? as u64)),
                    NUM_SIGNED => {
                        let v = self.parse_signed(bc)?;
                        if v < 0 {
                            Err(Error::InvalidType("negative enum index"))
                        } else {
                            Ok(EnumTag::Index(v as u64))
                        }
                    }
                    _ => Err(Error::InvalidType("enum tag must be integer")),
                }
            }
            TYPE_STRING => {
                let name = self.parse_string_borrowed()?;
                Ok(EnumTag::Name(name))
            }
            _ => Err(Error::InvalidType("unsupported enum tag")),
        }
    }

    fn deserialize_value<V: Visitor<'de>>(&mut self, visitor: V) -> Result<V::Value> {
        let header = self.read_byte()?;
        let ty = parse_type(header);
        match ty {
            TYPE_NULL_BOOL => {
                if header == 0 {
                    visitor.visit_unit()
                } else {
                    visitor.visit_bool(self.parse_bool(header)?)
                }
            }
            TYPE_NUMBER => {
                let class = parse_subtype(header);
                let bc = parse_byte_count_code(header);
                match class {
                    NUM_FLOAT => match bc {
                        0 => {
                            let bits = self.parse_bf16_bits()?;
                            let v = bf16::from_bits(bits).to_f32();
                            visitor.visit_f32(v)
                        }
                        1 => {
                            let bits = self.parse_f16_bits()?;
                            let v = f16::from_bits(bits).to_f32();
                            visitor.visit_f32(v)
                        }
                        2 => visitor.visit_f32(self.parse_f32()?),
                        3 => visitor.visit_f64(self.parse_f64()?),
                        4 => Err(Error::Unsupported("float128 not supported")),
                        _ => Err(Error::InvalidHeader(header)),
                    },
                    NUM_SIGNED => {
                        let v = self.parse_signed(bc)?;
                        // Choose the smallest fitting type for the visitor
                        if v >= i8::MIN as i128 && v <= i8::MAX as i128 {
                            visitor.visit_i8(v as i8)
                        } else if v >= i16::MIN as i128 && v <= i16::MAX as i128 {
                            visitor.visit_i16(v as i16)
                        } else if v >= i32::MIN as i128 && v <= i32::MAX as i128 {
                            visitor.visit_i32(v as i32)
                        } else if v >= i64::MIN as i128 && v <= i64::MAX as i128 {
                            visitor.visit_i64(v as i64)
                        } else {
                            visitor.visit_i128(v)
                        }
                    }
                    NUM_UNSIGNED => {
                        let v = self.parse_unsigned(bc)?;
                        if v <= u8::MAX as u128 {
                            visitor.visit_u8(v as u8)
                        } else if v <= u16::MAX as u128 {
                            visitor.visit_u16(v as u16)
                        } else if v <= u32::MAX as u128 {
                            visitor.visit_u32(v as u32)
                        } else if v <= u64::MAX as u128 {
                            visitor.visit_u64(v as u64)
                        } else {
                            visitor.visit_u128(v)
                        }
                    }
                    _ => Err(Error::InvalidHeader(header)),
                }
            }
            TYPE_STRING => {
                let s = self.parse_string_borrowed()?;
                visitor.visit_borrowed_str(s)
            }
            TYPE_OBJECT => {
                let key_type = parse_subtype(header);
                let bc = parse_byte_count_code(header);
                let count = read_size(self.input, &mut self.pos)? as usize;
                match key_type {
                    KEY_STRING => {
                        let access = MapAccessString {
                            de: self,
                            remaining: count,
                        };
                        visitor.visit_map(access)
                    }
                    KEY_SIGNED => {
                        let access = MapAccessSigned {
                            de: self,
                            remaining: count,
                            byte_code: bc,
                        };
                        visitor.visit_map(access)
                    }
                    KEY_UNSIGNED => {
                        let access = MapAccessUnsigned {
                            de: self,
                            remaining: count,
                            byte_code: bc,
                        };
                        visitor.visit_map(access)
                    }
                    _ => Err(Error::InvalidHeader(header)),
                }
            }
            TYPE_TYPED_ARRAY => {
                let cat = parse_subtype(header);
                let bc = parse_byte_count_code(header);
                let len = read_size(self.input, &mut self.pos)? as usize;
                match cat {
                    ARRAY_FLOAT => match bc {
                        0 => visitor.visit_seq(make_seq_half(self, len, HalfKind::Bf16)?),
                        1 => visitor.visit_seq(make_seq_half(self, len, HalfKind::F16)?),
                        2 => visitor.visit_seq(make_seq_float32(self, len)?),
                        3 => visitor.visit_seq(make_seq_float64(self, len)?),
                        _ => Err(Error::Unsupported(
                            "typed float arrays supported only for bf16/f16/f32/f64",
                        )),
                    },
                    ARRAY_SIGNED => visitor.visit_seq(make_seq_signed(self, len, bc)?),
                    ARRAY_UNSIGNED => visitor.visit_seq(make_seq_unsigned(self, len, bc)?),
                    ARRAY_BOOL_OR_STRING => {
                        if (header & 0b0010_0000) == 0 {
                            // boolean array
                            let packed_len = len.div_ceil(8);
                            let data = self.read_exact(packed_len)?;
                            visitor.visit_seq(SeqAccessBool {
                                data,
                                remaining: len,
                                byte_idx: 0,
                                bit_idx: 0,
                            })
                        } else {
                            // string array
                            visitor.visit_seq(SeqAccessString {
                                de: self,
                                remaining: len,
                            })
                        }
                    }
                    _ => Err(Error::InvalidHeader(header)),
                }
            }
            TYPE_GENERIC_ARRAY => {
                let len = read_size(self.input, &mut self.pos)? as usize;
                visitor.visit_seq(SeqAccessGeneric {
                    de: self,
                    remaining: len,
                })
            }
            TYPE_EXTENSION => {
                let ext = parse_extension_id(header);
                match ext {
                    EXT_TYPE_TAG => {
                        let tag = self.read_enum_tag()?;
                        let access = EnumAccess { de: self, tag };
                        visitor.visit_enum(access)
                    }
                    EXT_DELIMITER => visitor.visit_unit(),
                    EXT_COMPLEX => {
                        let ch = self.read_byte()?; // complex header
                        let is_array = (ch & 0x01) != 0;
                        let ty = (ch >> 3) & 0x03;
                        let bc = (ch >> 5) & 0x07;
                        if ty != NUM_FLOAT {
                            return Err(Error::Unsupported(
                                "only floating point complex supported",
                            ));
                        }
                        if !is_array {
                            // single complex -> present as 2-element sequence
                            visitor.visit_seq(ComplexPairAccess {
                                de: self,
                                byte_code: bc,
                                state: 0,
                            })
                        } else {
                            let len = read_size(self.input, &mut self.pos)? as usize;
                            visitor.visit_seq(SeqAccessComplexArray {
                                de: self,
                                remaining: len,
                                byte_code: bc,
                            })
                        }
                    }
                    EXT_MATRICES => {
                        // Present as a map with keys: layout, extents, value
                        let mheader = self.read_byte()?; // layout bit in bit0
                        visitor.visit_map(MatrixAccess {
                            de: self,
                            step: 0,
                            mheader,
                        })
                    }
                    _ => Err(Error::Unsupported("extension type not supported")),
                }
            }
            _ => Err(Error::InvalidHeader(header)),
        }
    }
}

impl<'de> serde::Deserializer<'de> for &mut Deserializer<'de> {
    type Error = Error;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_value(visitor)
    }

    fn deserialize_bool<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        let header = self.read_byte()?;
        if is_bool(header) {
            visitor.visit_bool(self.parse_bool(header)?)
        } else {
            Err(Error::InvalidType("expected bool"))
        }
    }

    fn deserialize_i8<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_any(visitor)
    }
    fn deserialize_i16<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_any(visitor)
    }
    fn deserialize_i32<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_any(visitor)
    }
    fn deserialize_i64<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_any(visitor)
    }
    fn deserialize_i128<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_any(visitor)
    }
    fn deserialize_u8<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_any(visitor)
    }
    fn deserialize_u16<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_any(visitor)
    }
    fn deserialize_u32<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_any(visitor)
    }
    fn deserialize_u64<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_any(visitor)
    }
    fn deserialize_u128<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_any(visitor)
    }
    fn deserialize_f32<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_any(visitor)
    }
    fn deserialize_f64<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_any(visitor)
    }
    fn deserialize_char<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_any(visitor)
    }
    fn deserialize_str<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_any(visitor)
    }
    fn deserialize_string<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_any(visitor)
    }
    fn deserialize_bytes<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        let header = self.peek_byte()?;
        let ty = parse_type(header);
        if ty == TYPE_TYPED_ARRAY
            && parse_subtype(header) == ARRAY_UNSIGNED
            && parse_byte_count_code(header) == 0
        {
            self.read_byte()?;
            let len = read_size(self.input, &mut self.pos)? as usize;
            let data = self.read_exact(len)?;
            return visitor.visit_borrowed_bytes(data);
        }
        // Empty sequences serialize as generic arrays; treat as empty bytes
        if ty == TYPE_GENERIC_ARRAY {
            self.read_byte()?;
            let len = read_size(self.input, &mut self.pos)? as usize;
            if len == 0 {
                return visitor.visit_borrowed_bytes(&[]);
            }
            // Non-empty generic array: fall through via visit_seq
            return visitor.visit_seq(SeqAccessGeneric {
                de: self,
                remaining: len,
            });
        }
        self.deserialize_any(visitor)
    }
    fn deserialize_byte_buf<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        let header = self.peek_byte()?;
        let ty = parse_type(header);
        if ty == TYPE_TYPED_ARRAY
            && parse_subtype(header) == ARRAY_UNSIGNED
            && parse_byte_count_code(header) == 0
        {
            self.read_byte()?;
            let len = read_size(self.input, &mut self.pos)? as usize;
            let data = self.read_exact(len)?;
            return visitor.visit_borrowed_bytes(data);
        }
        if ty == TYPE_GENERIC_ARRAY {
            self.read_byte()?;
            let len = read_size(self.input, &mut self.pos)? as usize;
            if len == 0 {
                return visitor.visit_borrowed_bytes(&[]);
            }
            return visitor.visit_seq(SeqAccessGeneric {
                de: self,
                remaining: len,
            });
        }
        self.deserialize_any(visitor)
    }

    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        let header = self.peek_byte()?;
        if header == 0 {
            self.read_byte()?;
            visitor.visit_none()
        } else {
            visitor.visit_some(self)
        }
    }

    fn deserialize_unit<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        let header = self.read_byte()?;
        if header == 0 {
            visitor.visit_unit()
        } else {
            Err(Error::InvalidType("expected null/unit"))
        }
    }

    fn deserialize_unit_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value> {
        self.deserialize_unit(visitor)
    }

    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        name: &'static str,
        visitor: V,
    ) -> Result<V::Value> {
        match name {
            "bf16" => self.deserialize_half_newtype(visitor, HalfKind::Bf16),
            "f16" => self.deserialize_half_newtype(visitor, HalfKind::F16),
            _ => self.deserialize_any(visitor),
        }
    }

    fn deserialize_seq<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_any(visitor)
    }
    fn deserialize_tuple<V: Visitor<'de>>(self, _len: usize, visitor: V) -> Result<V::Value> {
        self.deserialize_any(visitor)
    }
    fn deserialize_tuple_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _len: usize,
        visitor: V,
    ) -> Result<V::Value> {
        self.deserialize_any(visitor)
    }

    fn deserialize_map<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_any(visitor)
    }
    fn deserialize_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value> {
        self.deserialize_any(visitor)
    }

    fn deserialize_enum<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value> {
        let header = self.peek_byte()?;
        let ty = parse_type(header);
        match ty {
            TYPE_EXTENSION => {
                let _ = self.read_byte()?; // consume extension header
                let ext = parse_extension_id(header);
                match ext {
                    EXT_TYPE_TAG => {
                        let tag = self.read_enum_tag()?;
                        let access = EnumAccess { de: self, tag };
                        visitor.visit_enum(access)
                    }
                    _ => Err(Error::InvalidHeader(header)),
                }
            }
            TYPE_NUMBER => {
                let header = self.read_byte()?; // consume number header
                let subtype = parse_subtype(header);
                let idx: u64 = match subtype {
                    NUM_UNSIGNED => self.parse_unsigned(parse_byte_count_code(header))? as u64,
                    NUM_SIGNED => {
                        let v = self.parse_signed(parse_byte_count_code(header))?;
                        if v < 0 {
                            return Err(Error::InvalidType("negative enum index"));
                        }
                        v as u64
                    }
                    _ => return Err(Error::InvalidType("enum index must be integer")),
                };
                visitor.visit_enum(EnumIndexAccess { idx })
            }
            TYPE_STRING => {
                let _ = self.read_byte()?; // consume string header
                let s = self.parse_string_borrowed()?;
                visitor.visit_enum(EnumStrAccess { name: s })
            }
            _ => self.deserialize_any(visitor),
        }
    }

    fn deserialize_identifier<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_any(visitor)
    }
    fn deserialize_ignored_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_any(visitor)
    }
}

// =============== SeqAccess implementations ===============

struct SeqAccessGeneric<'a, 'de> {
    de: &'a mut Deserializer<'de>,
    remaining: usize,
}
impl<'de, 'a> de::SeqAccess<'de> for SeqAccessGeneric<'a, 'de> {
    type Error = Error;
    fn next_element_seed<T: de::DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;
        let val = seed.deserialize(&mut *self.de)?;
        Ok(Some(val))
    }
    fn size_hint(&self) -> Option<usize> {
        Some(self.remaining)
    }
}

struct SeqAccessUnsigned<'de> {
    data: &'de [u8],
    remaining: usize,
    elem_size: usize,
    offset: usize,
}
impl<'de> de::SeqAccess<'de> for SeqAccessUnsigned<'de> {
    type Error = Error;
    fn next_element_seed<T: de::DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;
        debug_assert!(self.offset + self.elem_size <= self.data.len());
        let chunk = &self.data[self.offset..self.offset + self.elem_size];
        self.offset += self.elem_size;
        let mut buf = [0u8; 16];
        buf[..self.elem_size].copy_from_slice(chunk);
        let v = u128::from_le_bytes(buf);
        let deser = NumDe::Unsigned(v);
        seed.deserialize(deser).map(Some)
    }
    fn size_hint(&self) -> Option<usize> {
        Some(self.remaining)
    }
}

struct SeqAccessSigned<'de> {
    data: &'de [u8],
    remaining: usize,
    elem_size: usize,
    offset: usize,
}
impl<'de> de::SeqAccess<'de> for SeqAccessSigned<'de> {
    type Error = Error;
    fn next_element_seed<T: de::DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;
        debug_assert!(self.offset + self.elem_size <= self.data.len());
        let chunk = &self.data[self.offset..self.offset + self.elem_size];
        self.offset += self.elem_size;
        let mut buf = [0u8; 16];
        buf[..self.elem_size].copy_from_slice(chunk);
        if self.elem_size < 16 && (chunk[self.elem_size - 1] & 0x80) != 0 {
            for b in &mut buf[self.elem_size..] {
                *b = 0xFF;
            }
        }
        let v = i128::from_le_bytes(buf);
        let deser = NumDe::Signed(v);
        seed.deserialize(deser).map(Some)
    }
    fn size_hint(&self) -> Option<usize> {
        Some(self.remaining)
    }
}

struct SeqAccessHalf<'de> {
    data: &'de [u8],
    remaining: usize,
    offset: usize,
    kind: HalfKind,
}
impl<'de> de::SeqAccess<'de> for SeqAccessHalf<'de> {
    type Error = Error;
    fn next_element_seed<T: de::DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;
        debug_assert!(self.offset + 2 <= self.data.len());
        let chunk = &self.data[self.offset..self.offset + 2];
        self.offset += 2;
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        let deser = NumDe::Half(self.kind, bits);
        seed.deserialize(deser).map(Some)
    }
    fn size_hint(&self) -> Option<usize> {
        Some(self.remaining)
    }
}

struct SeqAccessFloat32<'de> {
    data: &'de [u8],
    remaining: usize,
    offset: usize,
}
impl<'de> de::SeqAccess<'de> for SeqAccessFloat32<'de> {
    type Error = Error;
    fn next_element_seed<T: de::DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;
        debug_assert!(self.offset + 4 <= self.data.len());
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&self.data[self.offset..self.offset + 4]);
        self.offset += 4;
        let v = f32::from_le_bytes(buf);
        let deser = NumDe::F32(v);
        seed.deserialize(deser).map(Some)
    }
    fn size_hint(&self) -> Option<usize> {
        Some(self.remaining)
    }
}

struct SeqAccessFloat64<'de> {
    data: &'de [u8],
    remaining: usize,
    offset: usize,
}
impl<'de> de::SeqAccess<'de> for SeqAccessFloat64<'de> {
    type Error = Error;
    fn next_element_seed<T: de::DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;
        debug_assert!(self.offset + 8 <= self.data.len());
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&self.data[self.offset..self.offset + 8]);
        self.offset += 8;
        let v = f64::from_le_bytes(buf);
        let deser = NumDe::F64(v);
        seed.deserialize(deser).map(Some)
    }
    fn size_hint(&self) -> Option<usize> {
        Some(self.remaining)
    }
}

struct SeqAccessBool<'de> {
    data: &'de [u8],
    remaining: usize,
    byte_idx: usize,
    bit_idx: u8,
}
impl<'de> de::SeqAccess<'de> for SeqAccessBool<'de> {
    type Error = Error;
    fn next_element_seed<T: de::DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        let current = self.data[self.byte_idx];
        // LSB-first: bit 0 maps to the first element in the byte
        let bit = (current >> self.bit_idx) & 1;
        self.bit_idx += 1;
        if self.bit_idx == 8 {
            self.bit_idx = 0;
            self.byte_idx += 1;
        }
        self.remaining -= 1;
        seed.deserialize(BoolDeserializer::new(bit != 0)).map(Some)
    }
    fn size_hint(&self) -> Option<usize> {
        Some(self.remaining)
    }
}

struct SeqAccessString<'a, 'de> {
    de: &'a mut Deserializer<'de>,
    remaining: usize,
}
impl<'de, 'a> de::SeqAccess<'de> for SeqAccessString<'a, 'de> {
    type Error = Error;
    fn next_element_seed<T: de::DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        let s = self.de.parse_string_borrowed()?;
        self.remaining -= 1;
        seed.deserialize(BorrowedStrDeserializer::new(s)).map(Some)
    }
    fn size_hint(&self) -> Option<usize> {
        Some(self.remaining)
    }
}

// =============== MapAccess implementations ===============

struct MapAccessString<'a, 'de> {
    de: &'a mut Deserializer<'de>,
    remaining: usize,
}
impl<'de, 'a> de::MapAccess<'de> for MapAccessString<'a, 'de> {
    type Error = Error;
    fn next_key_seed<K: de::DeserializeSeed<'de>>(&mut self, seed: K) -> Result<Option<K::Value>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        let key_len = read_size(self.de.input, &mut self.de.pos)? as usize;
        let key_bytes = self.de.read_exact(key_len)?;
        let key = decode_utf8(key_bytes, "invalid utf-8 in key")?;
        seed.deserialize(BorrowedStrDeserializer::new(key))
            .map(Some)
    }
    fn next_value_seed<V: de::DeserializeSeed<'de>>(&mut self, seed: V) -> Result<V::Value> {
        self.remaining -= 1;
        seed.deserialize(&mut *self.de)
    }
}

// =============== Complex Extension Helpers ===============

struct ComplexPairAccess<'a, 'de> {
    de: &'a mut Deserializer<'de>,
    byte_code: u8,
    state: u8,
}
impl<'de, 'a> de::SeqAccess<'de> for ComplexPairAccess<'a, 'de> {
    type Error = Error;
    fn next_element_seed<T: de::DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>> {
        if self.state >= 2 {
            return Ok(None);
        }
        self.state += 1;
        let deser = match self.byte_code {
            2 => {
                let v = self.de.parse_f32()?;
                NumDe::F32(v)
            }
            3 => {
                let v = self.de.parse_f64()?;
                NumDe::F64(v)
            }
            _ => return Err(Error::Unsupported("unsupported complex float width")),
        };
        seed.deserialize(deser).map(Some)
    }
    fn size_hint(&self) -> Option<usize> {
        Some((2 - self.state) as usize)
    }
}

struct SeqAccessComplexArray<'a, 'de> {
    de: &'a mut Deserializer<'de>,
    remaining: usize,
    byte_code: u8,
}
impl<'de, 'a> de::SeqAccess<'de> for SeqAccessComplexArray<'a, 'de> {
    type Error = Error;
    fn next_element_seed<T: de::DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;
        let delem = ComplexElemDe {
            de: self.de,
            byte_code: self.byte_code,
        };
        seed.deserialize(delem).map(Some)
    }
    fn size_hint(&self) -> Option<usize> {
        Some(self.remaining)
    }
}

struct ComplexElemDe<'a, 'de> {
    de: &'a mut Deserializer<'de>,
    byte_code: u8,
}
impl<'de, 'a> de::Deserializer<'de> for ComplexElemDe<'a, 'de> {
    type Error = Error;
    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        visitor.visit_seq(ComplexPairAccess {
            de: self.de,
            byte_code: self.byte_code,
            state: 0,
        })
    }
    fn deserialize_seq<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_any(visitor)
    }
    fn deserialize_tuple<V: Visitor<'de>>(self, _len: usize, visitor: V) -> Result<V::Value> {
        self.deserialize_any(visitor)
    }
    forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string bytes byte_buf option unit unit_struct newtype_struct tuple_struct map struct enum identifier ignored_any
    }
}

// =============== Matrix Extension Helper ===============

struct MatrixAccess<'a, 'de> {
    de: &'a mut Deserializer<'de>,
    step: u8,
    mheader: u8,
}
impl<'de, 'a> de::MapAccess<'de> for MatrixAccess<'a, 'de> {
    type Error = Error;
    fn next_key_seed<K: de::DeserializeSeed<'de>>(&mut self, seed: K) -> Result<Option<K::Value>> {
        let key = match self.step {
            0 => "layout",
            1 => "extents",
            2 => "value",
            _ => return Ok(None),
        };
        self.step += 1;
        seed.deserialize(BorrowedStrDeserializer::new(key))
            .map(Some)
    }
    fn next_value_seed<V: de::DeserializeSeed<'de>>(&mut self, seed: V) -> Result<V::Value> {
        match self.step - 1 {
            0 => {
                let layout = if (self.mheader & 0x01) == 0 {
                    "layout_right"
                } else {
                    "layout_left"
                };
                seed.deserialize(BorrowedStrDeserializer::new(layout))
            }
            1 => {
                // extents as typed integer array (signed or unsigned)
                let h = self.de.read_byte()?;
                let len = read_size(self.de.input, &mut self.de.pos)? as usize;
                match h {
                    // Unsigned arrays
                    0x14 => seed.deserialize(SeqArrayDeUnsigned {
                        de: self.de,
                        len,
                        byte_code: 0,
                    }),
                    0x34 => seed.deserialize(SeqArrayDeUnsigned {
                        de: self.de,
                        len,
                        byte_code: 1,
                    }),
                    0x54 => seed.deserialize(SeqArrayDeUnsigned {
                        de: self.de,
                        len,
                        byte_code: 2,
                    }),
                    0x74 => seed.deserialize(SeqArrayDeUnsigned {
                        de: self.de,
                        len,
                        byte_code: 3,
                    }),
                    // Signed arrays
                    0x0c => seed.deserialize(SeqArrayDeSigned {
                        de: self.de,
                        len,
                        byte_code: 0,
                    }),
                    0x2c => seed.deserialize(SeqArrayDeSigned {
                        de: self.de,
                        len,
                        byte_code: 1,
                    }),
                    0x4c => seed.deserialize(SeqArrayDeSigned {
                        de: self.de,
                        len,
                        byte_code: 2,
                    }),
                    0x6c => seed.deserialize(SeqArrayDeSigned {
                        de: self.de,
                        len,
                        byte_code: 3,
                    }),
                    _ => Err(Error::InvalidHeader(h)),
                }
            }
            2 => {
                // value: just delegate to underlying deserializer at current position
                seed.deserialize(&mut *self.de)
            }
            _ => unreachable!(),
        }
    }
}

struct SeqArrayDeUnsigned<'a, 'de> {
    de: &'a mut Deserializer<'de>,
    len: usize,
    byte_code: u8,
}
impl<'de, 'a> de::Deserializer<'de> for SeqArrayDeUnsigned<'a, 'de> {
    type Error = Error;
    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        let access = make_seq_unsigned(self.de, self.len, self.byte_code)?;
        visitor.visit_seq(access)
    }
    fn deserialize_seq<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_any(visitor)
    }
    forward_to_deserialize_any! { bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string bytes byte_buf option unit unit_struct newtype_struct tuple tuple_struct map struct enum identifier ignored_any }
}

struct SeqArrayDeSigned<'a, 'de> {
    de: &'a mut Deserializer<'de>,
    len: usize,
    byte_code: u8,
}
impl<'de, 'a> de::Deserializer<'de> for SeqArrayDeSigned<'a, 'de> {
    type Error = Error;
    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        let access = make_seq_signed(self.de, self.len, self.byte_code)?;
        visitor.visit_seq(access)
    }
    fn deserialize_seq<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_any(visitor)
    }
    forward_to_deserialize_any! { bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string bytes byte_buf option unit unit_struct newtype_struct tuple tuple_struct map struct enum identifier ignored_any }
}

struct MapAccessSigned<'a, 'de> {
    de: &'a mut Deserializer<'de>,
    remaining: usize,
    byte_code: u8,
}
impl<'de, 'a> de::MapAccess<'de> for MapAccessSigned<'a, 'de> {
    type Error = Error;
    fn next_key_seed<K: de::DeserializeSeed<'de>>(&mut self, seed: K) -> Result<Option<K::Value>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        let v = self.de.parse_signed(self.byte_code)?;
        seed.deserialize(NumDe::Signed(v)).map(Some)
    }
    fn next_value_seed<V: de::DeserializeSeed<'de>>(&mut self, seed: V) -> Result<V::Value> {
        self.remaining -= 1;
        seed.deserialize(&mut *self.de)
    }
}

struct MapAccessUnsigned<'a, 'de> {
    de: &'a mut Deserializer<'de>,
    remaining: usize,
    byte_code: u8,
}
impl<'de, 'a> de::MapAccess<'de> for MapAccessUnsigned<'a, 'de> {
    type Error = Error;
    fn next_key_seed<K: de::DeserializeSeed<'de>>(&mut self, seed: K) -> Result<Option<K::Value>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        let v = self.de.parse_unsigned(self.byte_code)?;
        seed.deserialize(NumDe::Unsigned(v)).map(Some)
    }
    fn next_value_seed<V: de::DeserializeSeed<'de>>(&mut self, seed: V) -> Result<V::Value> {
        self.remaining -= 1;
        seed.deserialize(&mut *self.de)
    }
}

// =============== EnumAccess ===============

enum EnumTag<'de> {
    Index(u64),
    Name(&'de str),
}

struct EnumAccess<'a, 'de> {
    de: &'a mut Deserializer<'de>,
    tag: EnumTag<'de>,
}
impl<'de, 'a> de::EnumAccess<'de> for EnumAccess<'a, 'de> {
    type Error = Error;
    type Variant = VariantAccess<'a, 'de>;

    fn variant_seed<V: de::DeserializeSeed<'de>>(
        self,
        seed: V,
    ) -> Result<(V::Value, Self::Variant)> {
        match self.tag {
            EnumTag::Index(idx) => {
                let idx_de = NumDe::Unsigned(idx as u128);
                let v = seed.deserialize(idx_de)?;
                Ok((v, VariantAccess { de: self.de }))
            }
            EnumTag::Name(name) => {
                let v = seed.deserialize(BorrowedStrDeserializer::<Error>::new(name))?;
                Ok((v, VariantAccess { de: self.de }))
            }
        }
    }
}

struct VariantAccess<'a, 'de> {
    de: &'a mut Deserializer<'de>,
}
impl<'de, 'a> de::VariantAccess<'de> for VariantAccess<'a, 'de> {
    type Error = Error;
    fn unit_variant(self) -> Result<()> {
        // Consume optional VALUE if present
        if self.de.remaining() > 0 {
            let _ = serde::de::Deserializer::deserialize_ignored_any(self.de, de::IgnoredAny);
        }
        Ok(())
    }
    fn newtype_variant_seed<T: de::DeserializeSeed<'de>>(self, seed: T) -> Result<T::Value> {
        seed.deserialize(self.de)
    }
    fn tuple_variant<V: Visitor<'de>>(self, _len: usize, visitor: V) -> Result<V::Value> {
        de::Deserializer::deserialize_any(self.de, visitor)
    }
    fn struct_variant<V: Visitor<'de>>(
        self,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value> {
        de::Deserializer::deserialize_any(self.de, visitor)
    }
}

struct EnumIndexAccess {
    idx: u64,
}
impl<'de> de::EnumAccess<'de> for EnumIndexAccess {
    type Error = Error;
    type Variant = VariantAccessNoValue;
    fn variant_seed<V: de::DeserializeSeed<'de>>(
        self,
        seed: V,
    ) -> Result<(V::Value, Self::Variant)> {
        let v = seed.deserialize(NumDe::Unsigned(self.idx as u128))?;
        Ok((v, VariantAccessNoValue))
    }
}

struct EnumStrAccess<'de> {
    name: &'de str,
}
impl<'de> de::EnumAccess<'de> for EnumStrAccess<'de> {
    type Error = Error;
    type Variant = VariantAccessNoValue;
    fn variant_seed<V: de::DeserializeSeed<'de>>(
        self,
        seed: V,
    ) -> Result<(V::Value, Self::Variant)> {
        let v = seed.deserialize(BorrowedStrDeserializer::<Error>::new(self.name))?;
        Ok((v, VariantAccessNoValue))
    }
}

struct VariantAccessNoValue;
impl<'de> de::VariantAccess<'de> for VariantAccessNoValue {
    type Error = Error;
    fn unit_variant(self) -> Result<()> {
        Ok(())
    }
    fn newtype_variant_seed<T: de::DeserializeSeed<'de>>(self, _seed: T) -> Result<T::Value> {
        Err(Error::InvalidType("enum value not present"))
    }
    fn tuple_variant<V: Visitor<'de>>(self, _len: usize, _visitor: V) -> Result<V::Value> {
        Err(Error::InvalidType("enum value not present"))
    }
    fn struct_variant<V: Visitor<'de>>(
        self,
        _fields: &'static [&'static str],
        _visitor: V,
    ) -> Result<V::Value> {
        Err(Error::InvalidType("enum value not present"))
    }
}

// =============== Primitive Value Deserializers ===============

enum NumDe {
    Unsigned(u128),
    Signed(i128),
    F32(f32),
    F64(f64),
    Half(HalfKind, u16),
}
impl<'de> de::Deserializer<'de> for NumDe {
    type Error = Error;
    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        match self {
            NumDe::Unsigned(u) => {
                if u <= u8::MAX as u128 {
                    visitor.visit_u8(u as u8)
                } else if u <= u16::MAX as u128 {
                    visitor.visit_u16(u as u16)
                } else if u <= u32::MAX as u128 {
                    visitor.visit_u32(u as u32)
                } else if u <= u64::MAX as u128 {
                    visitor.visit_u64(u as u64)
                } else {
                    visitor.visit_u128(u)
                }
            }
            NumDe::Signed(i) => {
                if i >= i8::MIN as i128 && i <= i8::MAX as i128 {
                    visitor.visit_i8(i as i8)
                } else if i >= i16::MIN as i128 && i <= i16::MAX as i128 {
                    visitor.visit_i16(i as i16)
                } else if i >= i32::MIN as i128 && i <= i32::MAX as i128 {
                    visitor.visit_i32(i as i32)
                } else if i >= i64::MIN as i128 && i <= i64::MAX as i128 {
                    visitor.visit_i64(i as i64)
                } else {
                    visitor.visit_i128(i)
                }
            }
            NumDe::F32(f) => visitor.visit_f32(f),
            NumDe::F64(f) => visitor.visit_f64(f),
            NumDe::Half(kind, bits) => visitor.visit_f32(kind.to_f32(bits)),
        }
    }
    fn deserialize_u8<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        match self {
            NumDe::Unsigned(u) if u <= u8::MAX as u128 => visitor.visit_u8(u as u8),
            NumDe::Signed(i) if i >= 0 && i <= u8::MAX as i128 => visitor.visit_u8(i as u8),
            _ => Err(Error::Mismatch("expected u8")),
        }
    }
    fn deserialize_u16<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        match self {
            NumDe::Unsigned(u) if u <= u16::MAX as u128 => visitor.visit_u16(u as u16),
            NumDe::Signed(i) if i >= 0 && i <= u16::MAX as i128 => visitor.visit_u16(i as u16),
            NumDe::Half(_, bits) => visitor.visit_u16(bits),
            _ => Err(Error::Mismatch("expected u16")),
        }
    }
    fn deserialize_u32<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        match self {
            NumDe::Unsigned(u) if u <= u32::MAX as u128 => visitor.visit_u32(u as u32),
            NumDe::Signed(i) if i >= 0 && i <= u32::MAX as i128 => visitor.visit_u32(i as u32),
            _ => Err(Error::Mismatch("expected u32")),
        }
    }
    fn deserialize_u64<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        match self {
            NumDe::Unsigned(u) if u <= u64::MAX as u128 => visitor.visit_u64(u as u64),
            NumDe::Signed(i) if i >= 0 && i <= u64::MAX as i128 => visitor.visit_u64(i as u64),
            _ => Err(Error::Mismatch("expected u64")),
        }
    }
    fn deserialize_u128<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        match self {
            NumDe::Unsigned(u) => visitor.visit_u128(u),
            NumDe::Signed(i) if i >= 0 => visitor.visit_u128(i as u128),
            _ => Err(Error::Mismatch("expected u128")),
        }
    }
    fn deserialize_i8<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        match self {
            NumDe::Signed(i) if i >= i8::MIN as i128 && i <= i8::MAX as i128 => {
                visitor.visit_i8(i as i8)
            }
            NumDe::Unsigned(u) if u <= i8::MAX as u128 => visitor.visit_i8(u as i8),
            _ => Err(Error::Mismatch("expected i8")),
        }
    }
    fn deserialize_i16<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        match self {
            NumDe::Signed(i) if i >= i16::MIN as i128 && i <= i16::MAX as i128 => {
                visitor.visit_i16(i as i16)
            }
            NumDe::Unsigned(u) if u <= i16::MAX as u128 => visitor.visit_i16(u as i16),
            _ => Err(Error::Mismatch("expected i16")),
        }
    }
    fn deserialize_i32<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        match self {
            NumDe::Signed(i) if i >= i32::MIN as i128 && i <= i32::MAX as i128 => {
                visitor.visit_i32(i as i32)
            }
            NumDe::Unsigned(u) if u <= i32::MAX as u128 => visitor.visit_i32(u as i32),
            _ => Err(Error::Mismatch("expected i32")),
        }
    }
    fn deserialize_i64<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        match self {
            NumDe::Signed(i) if i >= i64::MIN as i128 && i <= i64::MAX as i128 => {
                visitor.visit_i64(i as i64)
            }
            NumDe::Unsigned(u) if u <= i64::MAX as u128 => visitor.visit_i64(u as i64),
            _ => Err(Error::Mismatch("expected i64")),
        }
    }
    fn deserialize_i128<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        match self {
            NumDe::Signed(i) => visitor.visit_i128(i),
            NumDe::Unsigned(u) if u <= i128::MAX as u128 => visitor.visit_i128(u as i128),
            _ => Err(Error::Mismatch("expected i128")),
        }
    }
    fn deserialize_f32<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        match self {
            NumDe::F32(f) => visitor.visit_f32(f),
            NumDe::F64(f) => visitor.visit_f32(f as f32),
            NumDe::Signed(i) => visitor.visit_f32(i as f32),
            NumDe::Unsigned(u) => visitor.visit_f32(u as f32),
            NumDe::Half(kind, bits) => visitor.visit_f32(kind.to_f32(bits)),
        }
    }
    fn deserialize_f64<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        match self {
            NumDe::F64(f) => visitor.visit_f64(f),
            NumDe::F32(f) => visitor.visit_f64(f as f64),
            NumDe::Signed(i) => visitor.visit_f64(i as f64),
            NumDe::Unsigned(u) => visitor.visit_f64(u as f64),
            NumDe::Half(kind, bits) => visitor.visit_f64(kind.to_f64(bits)),
        }
    }

    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value> {
        match self {
            NumDe::Half(kind, bits) => {
                let deser = HalfBitsDeserializer { kind, bits };
                visitor.visit_newtype_struct(deser)
            }
            _ => self.deserialize_any(visitor),
        }
    }

    forward_to_deserialize_any! {
        bool char str string bytes byte_buf option unit unit_struct seq tuple tuple_struct map struct enum identifier ignored_any
    }
}

struct HalfBitsDeserializer {
    kind: HalfKind,
    bits: u16,
}
impl<'de> de::Deserializer<'de> for HalfBitsDeserializer {
    type Error = Error;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        visitor.visit_f32(self.kind.to_f32(self.bits))
    }

    fn deserialize_u16<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        visitor.visit_u16(self.bits)
    }

    fn deserialize_f32<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        visitor.visit_f32(self.kind.to_f32(self.bits))
    }

    fn deserialize_f64<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        visitor.visit_f64(self.kind.to_f64(self.bits))
    }

    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value> {
        visitor.visit_newtype_struct(self)
    }

    forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u32 u64 u128 char str string bytes byte_buf option unit unit_struct seq tuple tuple_struct map struct enum identifier ignored_any
    }

    fn is_human_readable(&self) -> bool {
        false
    }
}
