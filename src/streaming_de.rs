//! Streaming deserializer that reads directly from a [`std::io::Read`] without buffering
//! the entire input. All types are deserialized incrementally from the reader with no
//! intermediate allocations beyond the output values themselves.

use std::io::Read;

use serde::de::{self, DeserializeOwned, Visitor};
use serde::forward_to_deserialize_any;

use crate::de::{
    EnumIndexAccess, HalfBitsDeserializer, HalfKind, NumDe, VariantAccessNoValue,
    byte_count_to_bytes,
};
use crate::error::{Error, Result};
use crate::header::*;
use crate::size::{read_size_from_reader, read_size_from_reader_with_first_byte};

// ---------------------------------------------------------------------------
// Core streaming deserializer
// ---------------------------------------------------------------------------

/// A BEVE deserializer that reads directly from a [`std::io::Read`].
///
/// Unlike [`from_slice`](crate::from_slice), this does not require the entire
/// input to be in memory. Typed arrays are bulk-read into temporary buffers;
/// all other types are read incrementally.
pub struct StreamingDeserializer<R: Read> {
    reader: R,
    peeked: Option<u8>,
}

impl<R: Read> StreamingDeserializer<R> {
    /// Create a new streaming deserializer.
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            peeked: None,
        }
    }

    /// Consume the deserializer and return the underlying reader.
    pub fn into_inner(self) -> R {
        self.reader
    }

    // -- low-level read helpers --

    #[inline]
    fn read_byte(&mut self) -> Result<u8> {
        if let Some(b) = self.peeked.take() {
            return Ok(b);
        }
        let mut buf = [0u8; 1];
        self.reader.read_exact(&mut buf).map_err(|_| Error::Eof)?;
        Ok(buf[0])
    }

    #[inline]
    fn peek_byte(&mut self) -> Result<u8> {
        if let Some(b) = self.peeked {
            return Ok(b);
        }
        let mut buf = [0u8; 1];
        self.reader.read_exact(&mut buf).map_err(|_| Error::Eof)?;
        self.peeked = Some(buf[0]);
        Ok(buf[0])
    }

    /// Read exactly `n` bytes into a new Vec.
    ///
    /// To avoid OOM from untrusted size fields, the initial allocation is capped
    /// and the buffer grows in chunks as data actually arrives from the reader.
    fn read_exact_vec(&mut self, n: usize) -> Result<Vec<u8>> {
        // Cap initial allocation to avoid OOM on bogus size fields.
        // 8 MB is large enough for typical payloads while limiting exposure.
        const INITIAL_CAP: usize = 8 * 1024 * 1024;
        let initial = n.min(INITIAL_CAP);
        let mut buf = vec![0u8; initial];
        self.read_exact_into(&mut buf)?;
        // If more bytes are needed, grow and read in chunks.
        if n > initial {
            let remaining = n - initial;
            buf.try_reserve(remaining)
                .map_err(|_| Error::Message("allocation failed for streaming read"))?;
            buf.resize(n, 0);
            self.reader
                .read_exact(&mut buf[initial..])
                .map_err(|e| -> Error { e.into() })?;
        }
        Ok(buf)
    }

    /// Read exactly `buf.len()` bytes, draining the peeked byte first if present.
    fn read_exact_into(&mut self, buf: &mut [u8]) -> Result<()> {
        if buf.is_empty() {
            return Ok(());
        }
        let start = if let Some(b) = self.peeked.take() {
            buf[0] = b;
            1
        } else {
            0
        };
        if start < buf.len() {
            self.reader.read_exact(&mut buf[start..])?;
        }
        Ok(())
    }

    /// Skip any leading data delimiter bytes (extension 0).
    fn skip_delimiters(&mut self) -> Result<()> {
        loop {
            let b = self.peek_byte()?;
            if b != make_extension_header(EXT_DELIMITER) {
                return Ok(());
            }
            self.read_byte()?; // consume the delimiter
        }
    }

    fn read_size(&mut self) -> Result<u64> {
        if let Some(b) = self.peeked.take() {
            read_size_from_reader_with_first_byte(&mut self.reader, b)
        } else {
            read_size_from_reader(&mut self.reader)
        }
    }

    // -- parse helpers --

    fn parse_signed(&mut self, code: u8) -> Result<i128> {
        let nbytes = byte_count_to_bytes(code)?;
        let mut buf = [0u8; 16];
        self.read_exact_into(&mut buf[..nbytes])?;
        if nbytes < 16 && (buf[nbytes - 1] & 0x80) != 0 {
            for b in &mut buf[nbytes..] {
                *b = 0xFF;
            }
        }
        Ok(i128::from_le_bytes(buf))
    }

    fn parse_unsigned(&mut self, code: u8) -> Result<u128> {
        let nbytes = byte_count_to_bytes(code)?;
        let mut buf = [0u8; 16];
        self.read_exact_into(&mut buf[..nbytes])?;
        Ok(u128::from_le_bytes(buf))
    }

    fn parse_bf16_bits(&mut self) -> Result<u16> {
        let mut buf = [0u8; 2];
        self.read_exact_into(&mut buf)?;
        Ok(u16::from_le_bytes(buf))
    }

    fn parse_f16_bits(&mut self) -> Result<u16> {
        self.parse_bf16_bits()
    }

    fn parse_f32(&mut self) -> Result<f32> {
        let mut buf = [0u8; 4];
        self.read_exact_into(&mut buf)?;
        Ok(f32::from_le_bytes(buf))
    }

    fn parse_f64(&mut self) -> Result<f64> {
        let mut buf = [0u8; 8];
        self.read_exact_into(&mut buf)?;
        Ok(f64::from_le_bytes(buf))
    }

    fn parse_string_owned(&mut self) -> Result<String> {
        let len = self.read_size()? as usize;
        let bytes = self.read_exact_vec(len)?;
        String::from_utf8(bytes).map_err(|_| Error::InvalidType("invalid utf-8"))
    }

    fn read_enum_tag(&mut self) -> Result<EnumTagOwned> {
        let header = self.read_byte()?;
        match parse_type(header) {
            TYPE_NUMBER => {
                let subtype = parse_subtype(header);
                let bc = parse_byte_count_code(header);
                match subtype {
                    NUM_UNSIGNED => Ok(EnumTagOwned::Index(self.parse_unsigned(bc)? as u64)),
                    NUM_SIGNED => {
                        let v = self.parse_signed(bc)?;
                        if v < 0 {
                            Err(Error::InvalidType("negative enum index"))
                        } else {
                            Ok(EnumTagOwned::Index(v as u64))
                        }
                    }
                    _ => Err(Error::InvalidType("enum tag must be integer")),
                }
            }
            TYPE_STRING => {
                let name = self.parse_string_owned()?;
                Ok(EnumTagOwned::Name(name))
            }
            _ => Err(Error::InvalidType("unsupported enum tag")),
        }
    }

    fn deserialize_half_newtype<'de, V: Visitor<'de>>(
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

    // -- main value dispatch --

    fn deserialize_value<'de, V: Visitor<'de>>(&mut self, visitor: V) -> Result<V::Value> {
        let header = self.read_byte()?;
        let ty = parse_type(header);
        match ty {
            TYPE_NULL_BOOL => {
                if header == 0 {
                    visitor.visit_unit()
                } else {
                    let v = (header & 0b10000) != 0;
                    visitor.visit_bool(v)
                }
            }
            TYPE_NUMBER => {
                let class = parse_subtype(header);
                let bc = parse_byte_count_code(header);
                match class {
                    NUM_FLOAT => match bc {
                        0 => {
                            let bits = self.parse_bf16_bits()?;
                            visitor.visit_f32(half::bf16::from_bits(bits).to_f32())
                        }
                        1 => {
                            let bits = self.parse_f16_bits()?;
                            visitor.visit_f32(half::f16::from_bits(bits).to_f32())
                        }
                        2 => visitor.visit_f32(self.parse_f32()?),
                        3 => visitor.visit_f64(self.parse_f64()?),
                        _ => Err(Error::InvalidHeader(header)),
                    },
                    NUM_SIGNED => {
                        let v = self.parse_signed(bc)?;
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
                let s = self.parse_string_owned()?;
                visitor.visit_string(s)
            }
            TYPE_OBJECT => {
                let key_type = parse_subtype(header);
                let bc = parse_byte_count_code(header);
                let count = self.read_size()? as usize;
                match key_type {
                    KEY_STRING => visitor.visit_map(MapAccessStringStreaming {
                        de: self,
                        remaining: count,
                    }),
                    KEY_SIGNED => visitor.visit_map(MapAccessSignedStreaming {
                        de: self,
                        remaining: count,
                        byte_code: bc,
                    }),
                    KEY_UNSIGNED => visitor.visit_map(MapAccessUnsignedStreaming {
                        de: self,
                        remaining: count,
                        byte_code: bc,
                    }),
                    _ => Err(Error::InvalidHeader(header)),
                }
            }
            TYPE_TYPED_ARRAY => {
                let cat = parse_subtype(header);
                let bc = parse_byte_count_code(header);
                let len = self.read_size()? as usize;
                match cat {
                    ARRAY_FLOAT => match bc {
                        0 => visitor.visit_seq(SeqAccessHalfStreaming {
                            de: self,
                            remaining: len,
                            kind: HalfKind::Bf16,
                        }),
                        1 => visitor.visit_seq(SeqAccessHalfStreaming {
                            de: self,
                            remaining: len,
                            kind: HalfKind::F16,
                        }),
                        2 => visitor.visit_seq(SeqAccessFloat32Streaming {
                            de: self,
                            remaining: len,
                        }),
                        3 => visitor.visit_seq(SeqAccessFloat64Streaming {
                            de: self,
                            remaining: len,
                        }),
                        _ => Err(Error::Unsupported("unsupported typed float array width")),
                    },
                    ARRAY_SIGNED => {
                        let elem_size = byte_count_to_bytes(bc)?;
                        visitor.visit_seq(SeqAccessSignedStreaming {
                            de: self,
                            remaining: len,
                            elem_size,
                        })
                    }
                    ARRAY_UNSIGNED => {
                        let elem_size = byte_count_to_bytes(bc)?;
                        visitor.visit_seq(SeqAccessUnsignedStreaming {
                            de: self,
                            remaining: len,
                            elem_size,
                        })
                    }
                    ARRAY_BOOL_OR_STRING => {
                        if (header & 0b0010_0000) == 0 {
                            visitor.visit_seq(SeqAccessBoolStreaming {
                                de: self,
                                remaining: len,
                                current_byte: 0,
                                bit_idx: 0,
                            })
                        } else {
                            visitor.visit_seq(SeqAccessStringStreaming {
                                de: self,
                                remaining: len,
                            })
                        }
                    }
                    _ => Err(Error::InvalidHeader(header)),
                }
            }
            TYPE_GENERIC_ARRAY => {
                let len = self.read_size()? as usize;
                visitor.visit_seq(SeqAccessGenericStreaming {
                    de: self,
                    remaining: len,
                })
            }
            TYPE_EXTENSION => {
                let ext = parse_extension_id(header);
                match ext {
                    EXT_TYPE_TAG => {
                        let tag = self.read_enum_tag()?;
                        visitor.visit_enum(EnumAccessStreaming { de: self, tag })
                    }
                    EXT_COMPLEX => {
                        let ch = self.read_byte()?;
                        let is_array = (ch & 0x01) != 0;
                        let class = (ch >> 3) & 0x03;
                        let bc = (ch >> 5) & 0x07;
                        if !is_array {
                            visitor.visit_seq(ComplexPairStreaming {
                                de: self,
                                class,
                                byte_code: bc,
                                state: 0,
                            })
                        } else {
                            let len = self.read_size()? as usize;
                            visitor.visit_seq(SeqAccessComplexStreaming {
                                de: self,
                                remaining: len,
                                class,
                                byte_code: bc,
                            })
                        }
                    }
                    EXT_MATRICES => {
                        let mheader = self.read_byte()?;
                        visitor.visit_map(MatrixAccessStreaming {
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

// ---------------------------------------------------------------------------
// serde::Deserializer impl
// ---------------------------------------------------------------------------

impl<'de, R: Read> serde::Deserializer<'de> for &mut StreamingDeserializer<R> {
    type Error = Error;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_value(visitor)
    }

    fn deserialize_bool<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        let header = self.read_byte()?;
        if is_bool(header) {
            visitor.visit_bool((header & 0b10000) != 0)
        } else {
            Err(Error::InvalidType("expected bool"))
        }
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
                self.read_byte()?;
                let ext = parse_extension_id(header);
                match ext {
                    EXT_TYPE_TAG => {
                        let tag = self.read_enum_tag()?;
                        visitor.visit_enum(EnumAccessStreaming { de: self, tag })
                    }
                    _ => Err(Error::InvalidHeader(header)),
                }
            }
            TYPE_NUMBER => {
                let header = self.read_byte()?;
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
                self.read_byte()?;
                let s = self.parse_string_owned()?;
                visitor.visit_enum(EnumStrAccessOwned { name: s })
            }
            _ => self.deserialize_any(visitor),
        }
    }

    fn deserialize_bytes<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        let header = self.peek_byte()?;
        let ty = parse_type(header);
        if ty == TYPE_TYPED_ARRAY
            && parse_subtype(header) == ARRAY_UNSIGNED
            && parse_byte_count_code(header) == 0
        {
            self.read_byte()?;
            let len = self.read_size()? as usize;
            let data = self.read_exact_vec(len)?;
            return visitor.visit_byte_buf(data);
        }
        if ty == TYPE_GENERIC_ARRAY {
            self.read_byte()?;
            let len = self.read_size()? as usize;
            if len == 0 {
                return visitor.visit_byte_buf(vec![]);
            }
            return visitor.visit_seq(SeqAccessGenericStreaming {
                de: self,
                remaining: len,
            });
        }
        self.deserialize_any(visitor)
    }

    fn deserialize_byte_buf<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_bytes(visitor)
    }

    forward_to_deserialize_any! {
        i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
        seq tuple tuple_struct map struct identifier ignored_any
    }

    fn is_human_readable(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Typed array SeqAccess (read directly from reader, no temp buffer)
// ---------------------------------------------------------------------------

struct SeqAccessUnsignedStreaming<'a, R: Read> {
    de: &'a mut StreamingDeserializer<R>,
    remaining: usize,
    elem_size: usize,
}
impl<'de, 'a, R: Read> de::SeqAccess<'de> for SeqAccessUnsignedStreaming<'a, R> {
    type Error = Error;
    fn next_element_seed<T: de::DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;
        let mut buf = [0u8; 16];
        self.de.read_exact_into(&mut buf[..self.elem_size])?;
        seed.deserialize(NumDe::Unsigned(u128::from_le_bytes(buf)))
            .map(Some)
    }
    fn size_hint(&self) -> Option<usize> {
        Some(self.remaining)
    }
}

struct SeqAccessSignedStreaming<'a, R: Read> {
    de: &'a mut StreamingDeserializer<R>,
    remaining: usize,
    elem_size: usize,
}
impl<'de, 'a, R: Read> de::SeqAccess<'de> for SeqAccessSignedStreaming<'a, R> {
    type Error = Error;
    fn next_element_seed<T: de::DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;
        let mut buf = [0u8; 16];
        self.de.read_exact_into(&mut buf[..self.elem_size])?;
        if self.elem_size < 16 && (buf[self.elem_size - 1] & 0x80) != 0 {
            for b in &mut buf[self.elem_size..] {
                *b = 0xFF;
            }
        }
        seed.deserialize(NumDe::Signed(i128::from_le_bytes(buf)))
            .map(Some)
    }
    fn size_hint(&self) -> Option<usize> {
        Some(self.remaining)
    }
}

struct SeqAccessHalfStreaming<'a, R: Read> {
    de: &'a mut StreamingDeserializer<R>,
    remaining: usize,
    kind: HalfKind,
}
impl<'de, 'a, R: Read> de::SeqAccess<'de> for SeqAccessHalfStreaming<'a, R> {
    type Error = Error;
    fn next_element_seed<T: de::DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;
        let bits = self.de.parse_bf16_bits()?;
        seed.deserialize(NumDe::Half(self.kind, bits)).map(Some)
    }
    fn size_hint(&self) -> Option<usize> {
        Some(self.remaining)
    }
}

struct SeqAccessFloat32Streaming<'a, R: Read> {
    de: &'a mut StreamingDeserializer<R>,
    remaining: usize,
}
impl<'de, 'a, R: Read> de::SeqAccess<'de> for SeqAccessFloat32Streaming<'a, R> {
    type Error = Error;
    fn next_element_seed<T: de::DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;
        seed.deserialize(NumDe::F32(self.de.parse_f32()?)).map(Some)
    }
    fn size_hint(&self) -> Option<usize> {
        Some(self.remaining)
    }
}

struct SeqAccessFloat64Streaming<'a, R: Read> {
    de: &'a mut StreamingDeserializer<R>,
    remaining: usize,
}
impl<'de, 'a, R: Read> de::SeqAccess<'de> for SeqAccessFloat64Streaming<'a, R> {
    type Error = Error;
    fn next_element_seed<T: de::DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;
        seed.deserialize(NumDe::F64(self.de.parse_f64()?)).map(Some)
    }
    fn size_hint(&self) -> Option<usize> {
        Some(self.remaining)
    }
}

struct SeqAccessBoolStreaming<'a, R: Read> {
    de: &'a mut StreamingDeserializer<R>,
    remaining: usize,
    current_byte: u8,
    bit_idx: u8,
}
impl<'de, 'a, R: Read> de::SeqAccess<'de> for SeqAccessBoolStreaming<'a, R> {
    type Error = Error;
    fn next_element_seed<T: de::DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        if self.bit_idx == 0 {
            self.current_byte = self.de.read_byte()?;
        }
        let bit = (self.current_byte >> self.bit_idx) & 1;
        self.bit_idx += 1;
        if self.bit_idx == 8 {
            self.bit_idx = 0;
        }
        self.remaining -= 1;
        seed.deserialize(de::value::BoolDeserializer::new(bit != 0))
            .map(Some)
    }
    fn size_hint(&self) -> Option<usize> {
        Some(self.remaining)
    }
}

// ---------------------------------------------------------------------------
// Streaming SeqAccess (element-by-element from reader)
// ---------------------------------------------------------------------------

struct SeqAccessStringStreaming<'a, R: Read> {
    de: &'a mut StreamingDeserializer<R>,
    remaining: usize,
}
impl<'de, 'a, R: Read> de::SeqAccess<'de> for SeqAccessStringStreaming<'a, R> {
    type Error = Error;
    fn next_element_seed<T: de::DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;
        let s = self.de.parse_string_owned()?;
        seed.deserialize(de::value::StringDeserializer::<Error>::new(s))
            .map(Some)
    }
    fn size_hint(&self) -> Option<usize> {
        Some(self.remaining)
    }
}

struct SeqAccessGenericStreaming<'a, R: Read> {
    de: &'a mut StreamingDeserializer<R>,
    remaining: usize,
}
impl<'de, 'a, R: Read> de::SeqAccess<'de> for SeqAccessGenericStreaming<'a, R> {
    type Error = Error;
    fn next_element_seed<T: de::DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;
        seed.deserialize(&mut *self.de).map(Some)
    }
    fn size_hint(&self) -> Option<usize> {
        Some(self.remaining)
    }
}

// ---------------------------------------------------------------------------
// MapAccess implementations
// ---------------------------------------------------------------------------

struct MapAccessStringStreaming<'a, R: Read> {
    de: &'a mut StreamingDeserializer<R>,
    remaining: usize,
}
impl<'de, 'a, R: Read> de::MapAccess<'de> for MapAccessStringStreaming<'a, R> {
    type Error = Error;
    fn next_key_seed<K: de::DeserializeSeed<'de>>(&mut self, seed: K) -> Result<Option<K::Value>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        let key = self.de.parse_string_owned()?;
        seed.deserialize(de::value::StringDeserializer::<Error>::new(key))
            .map(Some)
    }
    fn next_value_seed<V: de::DeserializeSeed<'de>>(&mut self, seed: V) -> Result<V::Value> {
        self.remaining -= 1;
        seed.deserialize(&mut *self.de)
    }
}

struct MapAccessSignedStreaming<'a, R: Read> {
    de: &'a mut StreamingDeserializer<R>,
    remaining: usize,
    byte_code: u8,
}
impl<'de, 'a, R: Read> de::MapAccess<'de> for MapAccessSignedStreaming<'a, R> {
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

struct MapAccessUnsignedStreaming<'a, R: Read> {
    de: &'a mut StreamingDeserializer<R>,
    remaining: usize,
    byte_code: u8,
}
impl<'de, 'a, R: Read> de::MapAccess<'de> for MapAccessUnsignedStreaming<'a, R> {
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

// ---------------------------------------------------------------------------
// Enum access
// ---------------------------------------------------------------------------

enum EnumTagOwned {
    Index(u64),
    Name(String),
}

struct EnumAccessStreaming<'a, R: Read> {
    de: &'a mut StreamingDeserializer<R>,
    tag: EnumTagOwned,
}
impl<'de, 'a, R: Read> de::EnumAccess<'de> for EnumAccessStreaming<'a, R> {
    type Error = Error;
    type Variant = VariantAccessStreaming<'a, R>;

    fn variant_seed<V: de::DeserializeSeed<'de>>(
        self,
        seed: V,
    ) -> Result<(V::Value, Self::Variant)> {
        match self.tag {
            EnumTagOwned::Index(idx) => {
                let v = seed.deserialize(NumDe::Unsigned(idx as u128))?;
                Ok((v, VariantAccessStreaming { de: self.de }))
            }
            EnumTagOwned::Name(name) => {
                let v = seed.deserialize(de::value::StringDeserializer::<Error>::new(name))?;
                Ok((v, VariantAccessStreaming { de: self.de }))
            }
        }
    }
}

struct VariantAccessStreaming<'a, R: Read> {
    de: &'a mut StreamingDeserializer<R>,
}
impl<'de, 'a, R: Read> de::VariantAccess<'de> for VariantAccessStreaming<'a, R> {
    type Error = Error;
    fn unit_variant(self) -> Result<()> {
        Ok(())
    }
    fn newtype_variant_seed<T: de::DeserializeSeed<'de>>(self, seed: T) -> Result<T::Value> {
        seed.deserialize(self.de)
    }
    fn tuple_variant<V: Visitor<'de>>(self, _len: usize, visitor: V) -> Result<V::Value> {
        serde::Deserializer::deserialize_any(self.de, visitor)
    }
    fn struct_variant<V: Visitor<'de>>(
        self,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value> {
        serde::Deserializer::deserialize_any(self.de, visitor)
    }
}

struct EnumStrAccessOwned {
    name: String,
}
impl<'de> de::EnumAccess<'de> for EnumStrAccessOwned {
    type Error = Error;
    type Variant = VariantAccessNoValue;
    fn variant_seed<V: de::DeserializeSeed<'de>>(
        self,
        seed: V,
    ) -> Result<(V::Value, Self::Variant)> {
        let v = seed.deserialize(de::value::StringDeserializer::<Error>::new(self.name))?;
        Ok((v, VariantAccessNoValue))
    }
}

// ---------------------------------------------------------------------------
// Complex extension helpers
// ---------------------------------------------------------------------------

struct ComplexPairStreaming<'a, R: Read> {
    de: &'a mut StreamingDeserializer<R>,
    class: u8,
    byte_code: u8,
    state: u8,
}
impl<'de, 'a, R: Read> de::SeqAccess<'de> for ComplexPairStreaming<'a, R> {
    type Error = Error;
    fn next_element_seed<T: de::DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>> {
        if self.state >= 2 {
            return Ok(None);
        }
        self.state += 1;
        let deser = match self.class {
            NUM_FLOAT => match self.byte_code {
                2 => NumDe::F32(self.de.parse_f32()?),
                3 => NumDe::F64(self.de.parse_f64()?),
                _ => return Err(Error::Unsupported("unsupported complex float width")),
            },
            NUM_SIGNED => NumDe::Signed(self.de.parse_signed(self.byte_code)?),
            NUM_UNSIGNED => NumDe::Unsigned(self.de.parse_unsigned(self.byte_code)?),
            _ => return Err(Error::Unsupported("unsupported complex number class")),
        };
        seed.deserialize(deser).map(Some)
    }
    fn size_hint(&self) -> Option<usize> {
        Some((2 - self.state) as usize)
    }
}

struct SeqAccessComplexStreaming<'a, R: Read> {
    de: &'a mut StreamingDeserializer<R>,
    remaining: usize,
    class: u8,
    byte_code: u8,
}
impl<'de, 'a, R: Read> de::SeqAccess<'de> for SeqAccessComplexStreaming<'a, R> {
    type Error = Error;
    fn next_element_seed<T: de::DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;
        seed.deserialize(ComplexElemDeStreaming {
            de: self.de,
            class: self.class,
            byte_code: self.byte_code,
        })
        .map(Some)
    }
    fn size_hint(&self) -> Option<usize> {
        Some(self.remaining)
    }
}

struct ComplexElemDeStreaming<'a, R: Read> {
    de: &'a mut StreamingDeserializer<R>,
    class: u8,
    byte_code: u8,
}
impl<'de, 'a, R: Read> serde::Deserializer<'de> for ComplexElemDeStreaming<'a, R> {
    type Error = Error;
    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        visitor.visit_seq(ComplexPairStreaming {
            de: self.de,
            class: self.class,
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
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string bytes byte_buf
        option unit unit_struct newtype_struct tuple_struct map struct enum identifier ignored_any
    }
}

// ---------------------------------------------------------------------------
// Matrix extension helper
// ---------------------------------------------------------------------------

struct MatrixAccessStreaming<'a, R: Read> {
    de: &'a mut StreamingDeserializer<R>,
    step: u8,
    mheader: u8,
}
impl<'de, 'a, R: Read> de::MapAccess<'de> for MatrixAccessStreaming<'a, R> {
    type Error = Error;
    fn next_key_seed<K: de::DeserializeSeed<'de>>(&mut self, seed: K) -> Result<Option<K::Value>> {
        let key = match self.step {
            0 => "layout",
            1 => "extents",
            2 => "value",
            _ => return Ok(None),
        };
        self.step += 1;
        seed.deserialize(de::value::BorrowedStrDeserializer::new(key))
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
                seed.deserialize(de::value::BorrowedStrDeserializer::new(layout))
            }
            1 => {
                // Extents: read typed integer array header then bulk-read
                let h = self.de.read_byte()?;
                let len = self.de.read_size()? as usize;
                let bc = parse_byte_count_code(h);
                let subtype = parse_subtype(h);
                let elem_size = byte_count_to_bytes(bc)?;
                match subtype {
                    ARRAY_UNSIGNED => seed.deserialize(SeqDeserializerWrapper {
                        de: self.de,
                        remaining: len,
                        elem_size,
                        signed: false,
                    }),
                    ARRAY_SIGNED => seed.deserialize(SeqDeserializerWrapper {
                        de: self.de,
                        remaining: len,
                        elem_size,
                        signed: true,
                    }),
                    _ => Err(Error::InvalidHeader(h)),
                }
            }
            2 => seed.deserialize(&mut *self.de),
            _ => unreachable!(),
        }
    }
}

// Wrapper to present a typed integer array as a Deserializer (for matrix extents).
struct SeqDeserializerWrapper<'a, R: Read> {
    de: &'a mut StreamingDeserializer<R>,
    remaining: usize,
    elem_size: usize,
    signed: bool,
}
impl<'de, 'a, R: Read> serde::Deserializer<'de> for SeqDeserializerWrapper<'a, R> {
    type Error = Error;
    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        if self.signed {
            visitor.visit_seq(SeqAccessSignedStreaming {
                de: self.de,
                remaining: self.remaining,
                elem_size: self.elem_size,
            })
        } else {
            visitor.visit_seq(SeqAccessUnsignedStreaming {
                de: self.de,
                remaining: self.remaining,
                elem_size: self.elem_size,
            })
        }
    }
    fn deserialize_seq<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_any(visitor)
    }
    forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string bytes byte_buf
        option unit unit_struct newtype_struct tuple tuple_struct map struct enum identifier ignored_any
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Deserialize a value by streaming from a reader without buffering the entire input.
///
/// Unlike [`from_reader`](crate::from_reader), this does not read all bytes into memory
/// first. It reads BEVE tokens incrementally from the reader with no intermediate
/// allocations beyond the output values themselves.
///
/// Wrap the reader in `BufReader` for file/network targets to batch syscalls.
///
/// # Example
///
/// ```rust
/// use serde::{Serialize, Deserialize};
/// use std::io::Cursor;
///
/// #[derive(Serialize, Deserialize, PartialEq, Debug)]
/// struct Point { x: f64, y: f64 }
///
/// let bytes = beve::to_vec(&Point { x: 1.0, y: 2.0 }).unwrap();
/// let p: Point = beve::from_reader_streaming(Cursor::new(bytes)).unwrap();
/// assert_eq!(p, Point { x: 1.0, y: 2.0 });
/// ```
pub fn from_reader_streaming<R: Read, T: DeserializeOwned>(reader: R) -> Result<T> {
    let mut de = StreamingDeserializer::new(reader);
    de.skip_delimiters()?;
    T::deserialize(&mut de)
}
