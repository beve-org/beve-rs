//! Streaming serializer that writes directly to a [`std::io::Write`] with zero internal buffering.
//!
//! Only supports known-length containers. Returns an error for `serialize_seq(None)` or
//! `serialize_map(None)`. Homogeneous sequences are encoded as typed arrays (detected from the
//! first element); heterogeneous sequences use generic arrays.

use std::io::Write;

use serde::ser::{self, Serialize};

use crate::error::{Error, Result};
use crate::ext::{NT_COMPLEX, NT_RAW_VALUE, typed_array_tag};
use crate::header::*;
use crate::ser::{
    BytesExtractor, EnumEncoding, SerializerOptions, TypedArrayWriteSink, U16Extractor,
};
use crate::size::encode_size_to_array;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline]
fn code_for_bytes(n: usize) -> u8 {
    match n {
        1 => 0,
        2 => 1,
        4 => 2,
        8 => 3,
        16 => 4,
        _ => 0,
    }
}

#[inline]
fn complex_header(class: u8, byte_code: u8, is_array: bool) -> u8 {
    ((byte_code & 0b111) << 5) | ((class & 0b11) << 3) | u8::from(is_array)
}

// ---------------------------------------------------------------------------
// Core streaming serializer
// ---------------------------------------------------------------------------

/// A BEVE serializer that writes directly to a [`std::io::Write`] with zero buffering.
///
/// All containers must have known lengths (structs, `Vec`, `HashMap`, etc.).
/// Homogeneous sequences are encoded as typed arrays; heterogeneous sequences
/// use generic arrays.
pub struct StreamingSerializer<W: Write> {
    writer: W,
    opts: SerializerOptions,
}

impl<W: Write> StreamingSerializer<W> {
    /// Create a new streaming serializer with default options.
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            opts: SerializerOptions::default(),
        }
    }

    /// Create a new streaming serializer with custom options.
    pub fn with_options(writer: W, opts: SerializerOptions) -> Self {
        Self { writer, opts }
    }

    /// Consume the serializer and return the underlying writer.
    pub fn into_inner(self) -> W {
        self.writer
    }

    // -- low-level write helpers --

    #[inline]
    fn write_all(&mut self, bytes: &[u8]) -> Result<()> {
        Ok(self.writer.write_all(bytes)?)
    }

    #[inline]
    fn write_byte(&mut self, b: u8) -> Result<()> {
        Ok(self.writer.write_all(&[b])?)
    }

    #[inline]
    fn write_size(&mut self, n: u64) -> Result<()> {
        let mut buf = [0u8; 8];
        let used = encode_size_to_array(n, &mut buf);
        Ok(self.writer.write_all(&buf[..used])?)
    }

    // -- BEVE value writers --

    fn write_null(&mut self) -> Result<()> {
        self.write_byte(0)
    }

    fn write_bool_value(&mut self, v: bool) -> Result<()> {
        self.write_byte(if v { 0x18 } else { 0x08 })
    }

    fn write_signed_value<const N: usize, T: Into<i128>>(&mut self, v: T) -> Result<()> {
        let code = code_for_bytes(N);
        let header = make_header(TYPE_NUMBER, NUM_SIGNED, code);
        self.write_byte(header)?;
        let x: i128 = v.into();
        let bytes = x.to_le_bytes();
        self.write_all(&bytes[..N])
    }

    fn write_unsigned_value<const N: usize, T: Into<u128>>(&mut self, v: T) -> Result<()> {
        let code = code_for_bytes(N);
        let header = make_header(TYPE_NUMBER, NUM_UNSIGNED, code);
        self.write_byte(header)?;
        let x: u128 = v.into();
        let bytes = x.to_le_bytes();
        self.write_all(&bytes[..N])
    }

    fn write_float32_value(&mut self, v: f32) -> Result<()> {
        self.write_byte(make_header(TYPE_NUMBER, NUM_FLOAT, 2))?;
        self.write_all(&v.to_le_bytes())
    }

    fn write_float64_value(&mut self, v: f64) -> Result<()> {
        self.write_byte(make_header(TYPE_NUMBER, NUM_FLOAT, 3))?;
        self.write_all(&v.to_le_bytes())
    }

    fn write_bfloat16_bits(&mut self, bits: u16) -> Result<()> {
        self.write_byte(make_header(TYPE_NUMBER, NUM_FLOAT, 0))?;
        self.write_all(&bits.to_le_bytes())
    }

    fn write_float16_bits(&mut self, bits: u16) -> Result<()> {
        self.write_byte(make_header(TYPE_NUMBER, NUM_FLOAT, 1))?;
        self.write_all(&bits.to_le_bytes())
    }

    fn write_str_value(&mut self, s: &str) -> Result<()> {
        self.write_byte(TYPE_STRING)?;
        self.write_size(s.len() as u64)?;
        self.write_all(s.as_bytes())
    }

    fn write_enum_tag(&mut self, variant_index: u32, variant_name: &'static str) -> Result<()> {
        match self.opts.enum_encoding {
            EnumEncoding::Number => self.write_unsigned_value::<4, _>(variant_index),
            EnumEncoding::String => self.write_str_value(variant_name),
        }
    }

    fn write_generic_array_header(&mut self, len: usize) -> Result<()> {
        self.write_byte(TYPE_GENERIC_ARRAY)?;
        self.write_size(len as u64)
    }

    fn write_bytes_typed_array(&mut self, bytes: &[u8]) -> Result<()> {
        let header = make_header(TYPE_TYPED_ARRAY, ARRAY_UNSIGNED, 0);
        self.write_byte(header)?;
        self.write_size(bytes.len() as u64)?;
        self.write_all(bytes)
    }

    /// Write a typed numeric array from an already little-endian byte payload in a
    /// single bulk `write_all`.
    ///
    /// The element count for the SIZE prefix is `payload.len() / elem_size`, exact
    /// by construction. Used by the [`crate::TypedSlice`] bulk-write dispatch: the
    /// borrowed payload is written without an intermediate copy, so with a counting
    /// sink the body costs one `len +=` (the O(1) measure).
    fn write_typed_array_bytes(
        &mut self,
        class: u8,
        byte_code: u8,
        elem_size: usize,
        payload: &[u8],
    ) -> Result<()> {
        let header = make_header(TYPE_TYPED_ARRAY, class, byte_code);
        self.write_byte(header)?;
        debug_assert_eq!(
            payload.len() % elem_size,
            0,
            "typed-array payload must be a whole number of elements"
        );
        let count = payload.len() / elem_size;
        self.write_size(count as u64)?;
        self.write_all(payload)
    }

    fn write_complex_single_payload(
        &mut self,
        class: u8,
        byte_code: u8,
        payload: &[u8],
    ) -> Result<()> {
        let elem_bytes = (1usize << byte_code) * 2;
        if payload.len() != elem_bytes {
            return Err(Error::Mismatch("invalid complex payload size"));
        }
        self.write_byte(make_extension_header(EXT_COMPLEX))?;
        self.write_byte(complex_header(class, byte_code, false))?;
        self.write_all(payload)
    }
}

// ---------------------------------------------------------------------------
// serde::Serializer for &mut StreamingSerializer<W>
// ---------------------------------------------------------------------------

impl<'a, W: Write> ser::Serializer for &'a mut StreamingSerializer<W> {
    type Ok = ();
    type Error = Error;

    type SerializeSeq = StreamingSeqSerializer<'a, W>;
    type SerializeTuple = StreamingSeqSerializer<'a, W>;
    type SerializeTupleStruct = StreamingSeqSerializer<'a, W>;
    type SerializeTupleVariant = StreamingVariantSeqSerializer<'a, W>;
    type SerializeMap = StreamingMapSerializer<'a, W>;
    type SerializeStruct = StreamingStructSerializer<'a, W>;
    type SerializeStructVariant = StreamingVariantStructSerializer<'a, W>;

    fn serialize_bool(self, v: bool) -> Result<()> {
        self.write_bool_value(v)
    }

    fn serialize_i8(self, v: i8) -> Result<()> {
        self.write_signed_value::<1, _>(v)
    }
    fn serialize_i16(self, v: i16) -> Result<()> {
        self.write_signed_value::<2, _>(v)
    }
    fn serialize_i32(self, v: i32) -> Result<()> {
        self.write_signed_value::<4, _>(v)
    }
    fn serialize_i64(self, v: i64) -> Result<()> {
        self.write_signed_value::<8, _>(v)
    }
    fn serialize_i128(self, v: i128) -> Result<()> {
        self.write_signed_value::<16, _>(v)
    }

    fn serialize_u8(self, v: u8) -> Result<()> {
        self.write_unsigned_value::<1, _>(v)
    }
    fn serialize_u16(self, v: u16) -> Result<()> {
        self.write_unsigned_value::<2, _>(v)
    }
    fn serialize_u32(self, v: u32) -> Result<()> {
        self.write_unsigned_value::<4, _>(v)
    }
    fn serialize_u64(self, v: u64) -> Result<()> {
        self.write_unsigned_value::<8, _>(v)
    }
    fn serialize_u128(self, v: u128) -> Result<()> {
        self.write_unsigned_value::<16, _>(v)
    }

    fn serialize_f32(self, v: f32) -> Result<()> {
        self.write_float32_value(v)
    }
    fn serialize_f64(self, v: f64) -> Result<()> {
        self.write_float64_value(v)
    }

    fn serialize_char(self, v: char) -> Result<()> {
        let mut buf = [0u8; 4];
        let s = v.encode_utf8(&mut buf);
        self.serialize_str(s)
    }

    fn serialize_str(self, v: &str) -> Result<()> {
        self.write_str_value(v)
    }

    fn serialize_bytes(self, v: &[u8]) -> Result<()> {
        self.write_bytes_typed_array(v)
    }

    fn serialize_none(self) -> Result<()> {
        self.write_null()
    }

    fn serialize_some<T: ?Sized + Serialize>(self, value: &T) -> Result<()> {
        value.serialize(self)
    }

    fn serialize_unit(self) -> Result<()> {
        self.write_null()
    }

    fn serialize_unit_struct(self, _name: &'static str) -> Result<()> {
        self.write_null()
    }

    fn serialize_unit_variant(
        self,
        _name: &'static str,
        variant_index: u32,
        variant: &'static str,
    ) -> Result<()> {
        self.write_enum_tag(variant_index, variant)
    }

    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        name: &'static str,
        value: &T,
    ) -> Result<()> {
        match name {
            "bf16" => {
                let bits = value.serialize(U16Extractor)?;
                self.write_bfloat16_bits(bits)
            }
            "f16" => {
                let bits = value.serialize(U16Extractor)?;
                self.write_float16_bits(bits)
            }
            NT_COMPLEX => {
                let payload = value.serialize(BytesExtractor)?;
                if payload.len() < 2 {
                    return Err(Error::Mismatch("invalid complex payload"));
                }
                self.write_complex_single_payload(payload[0], payload[1], &payload[2..])
            }
            NT_RAW_VALUE => {
                let raw = value.serialize(BytesExtractor)?;
                self.write_all(&raw)
            }
            _ => {
                if let Some((class, byte_code, elem_size)) = typed_array_tag(name) {
                    // `TypedSlice<T>` field: write the borrowed payload straight to
                    // the writer with no copy (one `write_all` of the body).
                    value.serialize(TypedArrayWriteSink(move |bytes: &[u8]| {
                        self.write_typed_array_bytes(class, byte_code, elem_size, bytes)
                    }))
                } else {
                    value.serialize(self)
                }
            }
        }
    }

    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        variant_index: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<()> {
        self.write_byte(make_extension_header(EXT_TYPE_TAG))?;
        self.write_enum_tag(variant_index, variant)?;
        value.serialize(self)
    }

    fn serialize_seq(self, len: Option<usize>) -> Result<Self::SerializeSeq> {
        let len = len.ok_or(Error::Unsupported(
            "streaming serializer requires known-length sequences",
        ))?;
        // Defer header — the first element determines typed vs generic encoding
        Ok(StreamingSeqSerializer {
            ser: self,
            len,
            mode: SeqMode::Detecting,
            count: 0,
        })
    }

    fn serialize_tuple(self, len: usize) -> Result<Self::SerializeTuple> {
        // Tuples are heterogeneous — always generic
        self.write_generic_array_header(len)?;
        Ok(StreamingSeqSerializer {
            ser: self,
            len,
            mode: SeqMode::Generic,
            count: 0,
        })
    }

    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleStruct> {
        self.write_generic_array_header(len)?;
        Ok(StreamingSeqSerializer {
            ser: self,
            len,
            mode: SeqMode::Generic,
            count: 0,
        })
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleVariant> {
        self.write_byte(make_extension_header(EXT_TYPE_TAG))?;
        self.write_enum_tag(variant_index, variant)?;
        self.write_generic_array_header(len)?;
        Ok(StreamingVariantSeqSerializer {
            inner: StreamingSeqSerializer {
                ser: self,
                len,
                mode: SeqMode::Generic,
                count: 0,
            },
        })
    }

    fn serialize_map(self, len: Option<usize>) -> Result<Self::SerializeMap> {
        let len = len.ok_or(Error::Unsupported(
            "streaming serializer requires known-length maps",
        ))?;
        Ok(StreamingMapSerializer {
            ser: self,
            len,
            mode: KeyMode::Unknown,
            count: 0,
        })
    }

    fn serialize_struct(self, _name: &'static str, len: usize) -> Result<Self::SerializeStruct> {
        self.write_byte(TYPE_OBJECT | (KEY_STRING << 3))?;
        self.write_size(len as u64)?;
        Ok(StreamingStructSerializer { ser: self })
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStructVariant> {
        self.write_byte(make_extension_header(EXT_TYPE_TAG))?;
        self.write_enum_tag(variant_index, variant)?;
        self.write_byte(TYPE_OBJECT | (KEY_STRING << 3))?;
        self.write_size(len as u64)?;
        Ok(StreamingVariantStructSerializer {
            inner: StreamingStructSerializer { ser: self },
        })
    }

    fn is_human_readable(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Sequence serializer — typed array detection from first element
// ---------------------------------------------------------------------------

enum SeqMode {
    /// No elements seen yet; first element determines the encoding.
    Detecting,
    /// Generic array (header already written).
    Generic,
    /// Typed unsigned numeric array.
    TypedUnsigned { byte_code: u8 },
    /// Typed signed numeric array.
    TypedSigned { byte_code: u8 },
    /// Typed float array.
    TypedFloat { byte_code: u8 },
    /// Typed boolean array (bit-packed).
    TypedBool { byte_acc: u8, bit_idx: u8 },
    /// Typed string array (size-prefixed UTF-8, no per-element header).
    TypedString,
    /// Typed complex array (extension encoding).
    TypedComplex { class: u8, byte_code: u8 },
}

impl SeqMode {
    fn type_name(&self) -> &'static str {
        match self {
            SeqMode::Detecting => "unknown",
            SeqMode::Generic => "generic",
            SeqMode::TypedUnsigned { .. } => "unsigned",
            SeqMode::TypedSigned { .. } => "signed",
            SeqMode::TypedFloat { .. } => "float",
            SeqMode::TypedBool { .. } => "bool",
            SeqMode::TypedString => "string",
            SeqMode::TypedComplex { .. } => "complex",
        }
    }
}

fn typed_array_mismatch(expected: &SeqMode, got: &str) -> Error {
    Error::msg(format!(
        "streaming typed array expected {} elements, got {}; \
         use to_vec for heterogeneous sequences",
        expected.type_name(),
        got
    ))
}

pub struct StreamingSeqSerializer<'a, W: Write> {
    ser: &'a mut StreamingSerializer<W>,
    len: usize,
    mode: SeqMode,
    count: usize,
}

impl<'a, W: Write> StreamingSeqSerializer<'a, W> {
    /// Switch to generic mode if still detecting. Error if already in a typed mode.
    fn ensure_generic(&mut self) -> Result<()> {
        match self.mode {
            SeqMode::Generic => Ok(()),
            SeqMode::Detecting => {
                self.ser.write_generic_array_header(self.len)?;
                self.mode = SeqMode::Generic;
                Ok(())
            }
            _ => Err(typed_array_mismatch(&self.mode, "generic")),
        }
    }
}

// Element serializer — intercepts the first element to detect the type.
struct StreamingElemSer<'a, 'b, W: Write> {
    seq: &'b mut StreamingSeqSerializer<'a, W>,
}

impl<'a, 'b, W: Write> StreamingElemSer<'a, 'b, W> {
    #[inline(always)]
    fn emit_signed(&mut self, v: i128, bytes: usize) -> Result<()> {
        let byte_code = code_for_bytes(bytes);
        if let SeqMode::TypedSigned { byte_code: bc } = self.seq.mode
            && bc == byte_code
        {
            return self.seq.ser.write_all(&v.to_le_bytes()[..bytes]);
        }
        self.emit_signed_cold(v, bytes, byte_code)
    }

    #[cold]
    #[inline(never)]
    fn emit_signed_cold(&mut self, v: i128, bytes: usize, byte_code: u8) -> Result<()> {
        match self.seq.mode {
            SeqMode::Detecting => {
                let header = make_header(TYPE_TYPED_ARRAY, ARRAY_SIGNED, byte_code);
                self.seq.ser.write_byte(header)?;
                self.seq.ser.write_size(self.seq.len as u64)?;
                self.seq.mode = SeqMode::TypedSigned { byte_code };
            }
            _ => return Err(typed_array_mismatch(&self.seq.mode, "signed")),
        }
        self.seq.ser.write_all(&v.to_le_bytes()[..bytes])
    }

    #[inline(always)]
    fn emit_unsigned(&mut self, v: u128, bytes: usize) -> Result<()> {
        let byte_code = code_for_bytes(bytes);
        if let SeqMode::TypedUnsigned { byte_code: bc } = self.seq.mode
            && bc == byte_code
        {
            return self.seq.ser.write_all(&v.to_le_bytes()[..bytes]);
        }
        self.emit_unsigned_cold(v, bytes, byte_code)
    }

    #[cold]
    #[inline(never)]
    fn emit_unsigned_cold(&mut self, v: u128, bytes: usize, byte_code: u8) -> Result<()> {
        match self.seq.mode {
            SeqMode::Detecting => {
                let header = make_header(TYPE_TYPED_ARRAY, ARRAY_UNSIGNED, byte_code);
                self.seq.ser.write_byte(header)?;
                self.seq.ser.write_size(self.seq.len as u64)?;
                self.seq.mode = SeqMode::TypedUnsigned { byte_code };
            }
            _ => return Err(typed_array_mismatch(&self.seq.mode, "unsigned")),
        }
        self.seq.ser.write_all(&v.to_le_bytes()[..bytes])
    }

    #[inline(always)]
    fn emit_float(&mut self, raw: &[u8], byte_code: u8) -> Result<()> {
        if let SeqMode::TypedFloat { byte_code: bc } = self.seq.mode
            && bc == byte_code
        {
            return self.seq.ser.write_all(raw);
        }
        self.emit_float_cold(raw, byte_code)
    }

    #[cold]
    #[inline(never)]
    fn emit_float_cold(&mut self, raw: &[u8], byte_code: u8) -> Result<()> {
        match self.seq.mode {
            SeqMode::Detecting => {
                let header = make_header(TYPE_TYPED_ARRAY, ARRAY_FLOAT, byte_code);
                self.seq.ser.write_byte(header)?;
                self.seq.ser.write_size(self.seq.len as u64)?;
                self.seq.mode = SeqMode::TypedFloat { byte_code };
            }
            _ => return Err(typed_array_mismatch(&self.seq.mode, "float")),
        }
        self.seq.ser.write_all(raw)
    }

    #[inline(always)]
    fn emit_complex(&mut self, class: u8, byte_code: u8, payload: &[u8]) -> Result<()> {
        if let SeqMode::TypedComplex {
            class: c,
            byte_code: bc,
        } = self.seq.mode
            && c == class
            && bc == byte_code
        {
            return self.seq.ser.write_all(payload);
        }
        self.emit_complex_cold(class, byte_code, payload)
    }

    #[cold]
    #[inline(never)]
    fn emit_complex_cold(&mut self, class: u8, byte_code: u8, payload: &[u8]) -> Result<()> {
        let elem_bytes = (1usize << byte_code) * 2;
        if payload.len() != elem_bytes {
            return Err(Error::Mismatch("invalid complex payload size"));
        }
        match self.seq.mode {
            SeqMode::Detecting => {
                self.seq
                    .ser
                    .write_byte(make_extension_header(EXT_COMPLEX))?;
                self.seq
                    .ser
                    .write_byte(complex_header(class, byte_code, true))?;
                self.seq.ser.write_size(self.seq.len as u64)?;
                self.seq.mode = SeqMode::TypedComplex { class, byte_code };
            }
            _ => return Err(typed_array_mismatch(&self.seq.mode, "complex")),
        }
        self.seq.ser.write_all(payload)
    }
}

impl<'a, 'b, W: Write> StreamingElemSer<'a, 'b, W> {
    #[cold]
    #[inline(never)]
    fn serialize_bool_cold(&mut self, v: bool) -> Result<()> {
        match self.seq.mode {
            SeqMode::Detecting => {
                let header = make_header(TYPE_TYPED_ARRAY, ARRAY_BOOL_OR_STRING, 0);
                self.seq.ser.write_byte(header)?;
                self.seq.ser.write_size(self.seq.len as u64)?;
                self.seq.mode = SeqMode::TypedBool {
                    byte_acc: 0,
                    bit_idx: 0,
                };
            }
            _ => return Err(typed_array_mismatch(&self.seq.mode, "bool")),
        }
        // First element — accumulate the bit
        if v && let SeqMode::TypedBool { byte_acc, .. } = &mut self.seq.mode {
            *byte_acc |= 1;
        }
        if let SeqMode::TypedBool { bit_idx, .. } = &mut self.seq.mode {
            *bit_idx = 1;
        }
        Ok(())
    }

    #[cold]
    #[inline(never)]
    fn serialize_str_cold(&mut self, v: &str) -> Result<()> {
        match self.seq.mode {
            SeqMode::Detecting => {
                let header = make_header(TYPE_TYPED_ARRAY, ARRAY_BOOL_OR_STRING, 1);
                self.seq.ser.write_byte(header)?;
                self.seq.ser.write_size(self.seq.len as u64)?;
                self.seq.mode = SeqMode::TypedString;
            }
            _ => return Err(typed_array_mismatch(&self.seq.mode, "string")),
        }
        self.seq.ser.write_size(v.len() as u64)?;
        self.seq.ser.write_all(v.as_bytes())
    }
}

impl<'a, 'b, W: Write> ser::Serializer for &'b mut StreamingElemSer<'a, 'b, W> {
    type Ok = ();
    type Error = Error;

    type SerializeSeq = StreamingSeqSerializer<'b, W>;
    type SerializeTuple = StreamingSeqSerializer<'b, W>;
    type SerializeTupleStruct = StreamingSeqSerializer<'b, W>;
    type SerializeTupleVariant = StreamingVariantSeqSerializer<'b, W>;
    type SerializeMap = StreamingMapSerializer<'b, W>;
    type SerializeStruct = StreamingStructSerializer<'b, W>;
    type SerializeStructVariant = StreamingVariantStructSerializer<'b, W>;

    #[inline]
    fn serialize_bool(self, v: bool) -> Result<()> {
        if let SeqMode::TypedBool {
            byte_acc, bit_idx, ..
        } = &mut self.seq.mode
        {
            if v {
                *byte_acc |= 1 << *bit_idx;
            }
            *bit_idx += 1;
            if *bit_idx == 8 {
                self.seq.ser.write_byte(*byte_acc)?;
                *byte_acc = 0;
                *bit_idx = 0;
            }
            return Ok(());
        }
        self.serialize_bool_cold(v)
    }

    #[inline]
    fn serialize_i8(self, v: i8) -> Result<()> {
        self.emit_signed(v as i128, 1)
    }
    #[inline]
    fn serialize_i16(self, v: i16) -> Result<()> {
        self.emit_signed(v as i128, 2)
    }
    #[inline]
    fn serialize_i32(self, v: i32) -> Result<()> {
        self.emit_signed(v as i128, 4)
    }
    #[inline]
    fn serialize_i64(self, v: i64) -> Result<()> {
        self.emit_signed(v as i128, 8)
    }
    #[inline]
    fn serialize_i128(self, v: i128) -> Result<()> {
        self.emit_signed(v, 16)
    }

    #[inline]
    fn serialize_u8(self, v: u8) -> Result<()> {
        self.emit_unsigned(v as u128, 1)
    }
    #[inline]
    fn serialize_u16(self, v: u16) -> Result<()> {
        self.emit_unsigned(v as u128, 2)
    }
    #[inline]
    fn serialize_u32(self, v: u32) -> Result<()> {
        self.emit_unsigned(v as u128, 4)
    }
    #[inline]
    fn serialize_u64(self, v: u64) -> Result<()> {
        self.emit_unsigned(v as u128, 8)
    }
    #[inline]
    fn serialize_u128(self, v: u128) -> Result<()> {
        self.emit_unsigned(v, 16)
    }

    #[inline]
    fn serialize_f32(self, v: f32) -> Result<()> {
        self.emit_float(&v.to_le_bytes(), 2)
    }
    #[inline]
    fn serialize_f64(self, v: f64) -> Result<()> {
        self.emit_float(&v.to_le_bytes(), 3)
    }

    fn serialize_char(self, v: char) -> Result<()> {
        let mut buf = [0u8; 4];
        self.serialize_str(v.encode_utf8(&mut buf))
    }

    #[inline]
    fn serialize_str(self, v: &str) -> Result<()> {
        if let SeqMode::TypedString = self.seq.mode {
            self.seq.ser.write_size(v.len() as u64)?;
            return self.seq.ser.write_all(v.as_bytes());
        }
        self.serialize_str_cold(v)
    }

    fn serialize_bytes(self, v: &[u8]) -> Result<()> {
        self.seq.ensure_generic()?;
        self.seq.ser.write_bytes_typed_array(v)
    }

    fn serialize_none(self) -> Result<()> {
        self.seq.ensure_generic()?;
        self.seq.ser.write_null()
    }

    fn serialize_some<T: ?Sized + Serialize>(self, v: &T) -> Result<()> {
        v.serialize(self)
    }

    fn serialize_unit(self) -> Result<()> {
        self.seq.ensure_generic()?;
        self.seq.ser.write_null()
    }

    fn serialize_unit_struct(self, _name: &'static str) -> Result<()> {
        self.serialize_unit()
    }

    fn serialize_unit_variant(
        self,
        _name: &'static str,
        variant_index: u32,
        variant: &'static str,
    ) -> Result<()> {
        self.seq.ensure_generic()?;
        self.seq.ser.write_enum_tag(variant_index, variant)
    }

    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        name: &'static str,
        value: &T,
    ) -> Result<()> {
        match name {
            "bf16" => {
                let bits = value.serialize(U16Extractor)?;
                self.emit_float(&bits.to_le_bytes(), 0)
            }
            "f16" => {
                let bits = value.serialize(U16Extractor)?;
                self.emit_float(&bits.to_le_bytes(), 1)
            }
            NT_COMPLEX => {
                let payload = value.serialize(BytesExtractor)?;
                if payload.len() < 2 {
                    return Err(Error::Mismatch("invalid complex payload"));
                }
                self.emit_complex(payload[0], payload[1], &payload[2..])
            }
            NT_RAW_VALUE => {
                self.seq.ensure_generic()?;
                let raw = value.serialize(BytesExtractor)?;
                self.seq.ser.write_all(&raw)
            }
            _ => {
                if let Some((class, byte_code, elem_size)) = typed_array_tag(name) {
                    // A typed array is a full VALUE, so it becomes a generic-array
                    // element (matching how `NT_RAW_VALUE` is handled here).
                    self.seq.ensure_generic()?;
                    let ser = &mut *self.seq.ser;
                    value.serialize(TypedArrayWriteSink(move |bytes: &[u8]| {
                        ser.write_typed_array_bytes(class, byte_code, elem_size, bytes)
                    }))
                } else {
                    value.serialize(self)
                }
            }
        }
    }

    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        variant_index: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<()> {
        self.seq.ensure_generic()?;
        self.seq
            .ser
            .write_byte(make_extension_header(EXT_TYPE_TAG))?;
        self.seq.ser.write_enum_tag(variant_index, variant)?;
        value.serialize(&mut *self.seq.ser)
    }

    fn serialize_seq(self, len: Option<usize>) -> Result<Self::SerializeSeq> {
        self.seq.ensure_generic()?;
        let len = len.ok_or(Error::Unsupported(
            "streaming serializer requires known-length sequences",
        ))?;
        Ok(StreamingSeqSerializer {
            ser: self.seq.ser,
            len,
            mode: SeqMode::Detecting,
            count: 0,
        })
    }

    fn serialize_tuple(self, len: usize) -> Result<Self::SerializeTuple> {
        self.seq.ensure_generic()?;
        self.seq.ser.write_generic_array_header(len)?;
        Ok(StreamingSeqSerializer {
            ser: self.seq.ser,
            len,
            mode: SeqMode::Generic,
            count: 0,
        })
    }

    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleStruct> {
        self.seq.ensure_generic()?;
        self.seq.ser.write_generic_array_header(len)?;
        Ok(StreamingSeqSerializer {
            ser: self.seq.ser,
            len,
            mode: SeqMode::Generic,
            count: 0,
        })
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleVariant> {
        self.seq.ensure_generic()?;
        self.seq
            .ser
            .write_byte(make_extension_header(EXT_TYPE_TAG))?;
        self.seq.ser.write_enum_tag(variant_index, variant)?;
        self.seq.ser.write_generic_array_header(len)?;
        Ok(StreamingVariantSeqSerializer {
            inner: StreamingSeqSerializer {
                ser: self.seq.ser,
                len,
                mode: SeqMode::Generic,
                count: 0,
            },
        })
    }

    fn serialize_map(self, len: Option<usize>) -> Result<Self::SerializeMap> {
        self.seq.ensure_generic()?;
        let len = len.ok_or(Error::Unsupported(
            "streaming serializer requires known-length maps",
        ))?;
        Ok(StreamingMapSerializer {
            ser: self.seq.ser,
            len,
            mode: KeyMode::Unknown,
            count: 0,
        })
    }

    fn serialize_struct(self, _name: &'static str, len: usize) -> Result<Self::SerializeStruct> {
        self.seq.ensure_generic()?;
        self.seq.ser.write_byte(TYPE_OBJECT | (KEY_STRING << 3))?;
        self.seq.ser.write_size(len as u64)?;
        Ok(StreamingStructSerializer { ser: self.seq.ser })
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStructVariant> {
        self.seq.ensure_generic()?;
        self.seq
            .ser
            .write_byte(make_extension_header(EXT_TYPE_TAG))?;
        self.seq.ser.write_enum_tag(variant_index, variant)?;
        self.seq.ser.write_byte(TYPE_OBJECT | (KEY_STRING << 3))?;
        self.seq.ser.write_size(len as u64)?;
        Ok(StreamingVariantStructSerializer {
            inner: StreamingStructSerializer { ser: self.seq.ser },
        })
    }

    fn is_human_readable(&self) -> bool {
        false
    }
}

// -- SerializeSeq / SerializeTuple / SerializeTupleStruct --

impl<'a, W: Write> ser::SerializeSeq for StreamingSeqSerializer<'a, W> {
    type Ok = ();
    type Error = Error;

    #[inline]
    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<()> {
        if matches!(self.mode, SeqMode::Generic) {
            value.serialize(&mut *self.ser)?;
        } else {
            let mut elem = StreamingElemSer { seq: self };
            value.serialize(&mut elem)?;
        }
        self.count += 1;
        Ok(())
    }

    fn end(self) -> Result<()> {
        // Flush remaining bool bits
        if let SeqMode::TypedBool { byte_acc, bit_idx } = self.mode
            && bit_idx != 0
        {
            self.ser.write_byte(byte_acc)?;
        }
        // Empty sequence with no elements seen — write generic header
        if matches!(self.mode, SeqMode::Detecting) {
            self.ser.write_generic_array_header(self.len)?;
        }
        Ok(())
    }
}

impl<'a, W: Write> ser::SerializeTuple for StreamingSeqSerializer<'a, W> {
    type Ok = ();
    type Error = Error;

    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<()> {
        ser::SerializeSeq::serialize_element(self, value)
    }

    fn end(self) -> Result<()> {
        ser::SerializeSeq::end(self)
    }
}

impl<'a, W: Write> ser::SerializeTupleStruct for StreamingSeqSerializer<'a, W> {
    type Ok = ();
    type Error = Error;

    fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<()> {
        ser::SerializeSeq::serialize_element(self, value)
    }

    fn end(self) -> Result<()> {
        ser::SerializeSeq::end(self)
    }
}

// ---------------------------------------------------------------------------
// Tuple variant serializer
// ---------------------------------------------------------------------------

pub struct StreamingVariantSeqSerializer<'a, W: Write> {
    inner: StreamingSeqSerializer<'a, W>,
}

impl<'a, W: Write> ser::SerializeTupleVariant for StreamingVariantSeqSerializer<'a, W> {
    type Ok = ();
    type Error = Error;

    fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<()> {
        ser::SerializeSeq::serialize_element(&mut self.inner, value)
    }

    fn end(self) -> Result<()> {
        ser::SerializeSeq::end(self.inner)
    }
}

// ---------------------------------------------------------------------------
// Map serializer (key mode detection, known length)
// ---------------------------------------------------------------------------

enum KeyMode {
    Unknown,
    String,
    Signed(u8),
    Unsigned(u8),
}

pub struct StreamingMapSerializer<'a, W: Write> {
    ser: &'a mut StreamingSerializer<W>,
    len: usize,
    mode: KeyMode,
    count: usize,
}

impl<'a, W: Write> ser::SerializeMap for StreamingMapSerializer<'a, W> {
    type Ok = ();
    type Error = Error;

    fn serialize_key<T: ?Sized + Serialize>(&mut self, key: &T) -> Result<()> {
        struct KeySer<'a, 'b, W: Write> {
            map: &'b mut StreamingMapSerializer<'a, W>,
        }

        impl<'a, 'b, W: Write> KeySer<'a, 'b, W> {
            fn emit_signed_key(&mut self, v: i128, bytes: usize) -> Result<()> {
                let code = code_for_bytes(bytes);
                match self.map.mode {
                    KeyMode::Unknown => {
                        self.map
                            .ser
                            .write_byte(make_header(TYPE_OBJECT, KEY_SIGNED, code))?;
                        self.map.ser.write_size(self.map.len as u64)?;
                        self.map.mode = KeyMode::Signed(code);
                    }
                    KeyMode::Signed(c) if c == code => {}
                    _ => {
                        return Err(Error::Mismatch(
                            "object keys must be homogeneous signed integers of same width",
                        ));
                    }
                }
                let raw = v.to_le_bytes();
                self.map.ser.write_all(&raw[..bytes])
            }

            fn emit_unsigned_key(&mut self, v: u128, bytes: usize) -> Result<()> {
                let code = code_for_bytes(bytes);
                match self.map.mode {
                    KeyMode::Unknown => {
                        self.map
                            .ser
                            .write_byte(make_header(TYPE_OBJECT, KEY_UNSIGNED, code))?;
                        self.map.ser.write_size(self.map.len as u64)?;
                        self.map.mode = KeyMode::Unsigned(code);
                    }
                    KeyMode::Unsigned(c) if c == code => {}
                    _ => {
                        return Err(Error::Mismatch(
                            "object keys must be homogeneous unsigned integers of same width",
                        ));
                    }
                }
                let raw = v.to_le_bytes();
                self.map.ser.write_all(&raw[..bytes])
            }
        }

        impl<'a, 'b, W: Write> ser::Serializer for &'b mut KeySer<'a, 'b, W> {
            type Ok = ();
            type Error = Error;
            type SerializeSeq = ser::Impossible<(), Error>;
            type SerializeTuple = ser::Impossible<(), Error>;
            type SerializeTupleStruct = ser::Impossible<(), Error>;
            type SerializeTupleVariant = ser::Impossible<(), Error>;
            type SerializeMap = ser::Impossible<(), Error>;
            type SerializeStruct = ser::Impossible<(), Error>;
            type SerializeStructVariant = ser::Impossible<(), Error>;

            fn serialize_str(self, v: &str) -> Result<()> {
                match self.map.mode {
                    KeyMode::Unknown => {
                        self.map.ser.write_byte(TYPE_OBJECT | (KEY_STRING << 3))?;
                        self.map.ser.write_size(self.map.len as u64)?;
                        self.map.mode = KeyMode::String;
                    }
                    KeyMode::String => {}
                    _ => return Err(Error::Mismatch("object keys must be homogeneous")),
                }
                self.map.ser.write_size(v.len() as u64)?;
                self.map.ser.write_all(v.as_bytes())
            }

            fn serialize_char(self, v: char) -> Result<()> {
                let mut buf = [0; 4];
                self.serialize_str(v.encode_utf8(&mut buf))
            }

            fn serialize_i8(self, v: i8) -> Result<()> {
                self.emit_signed_key(v as i128, 1)
            }
            fn serialize_i16(self, v: i16) -> Result<()> {
                self.emit_signed_key(v as i128, 2)
            }
            fn serialize_i32(self, v: i32) -> Result<()> {
                self.emit_signed_key(v as i128, 4)
            }
            fn serialize_i64(self, v: i64) -> Result<()> {
                self.emit_signed_key(v as i128, 8)
            }
            fn serialize_i128(self, v: i128) -> Result<()> {
                self.emit_signed_key(v, 16)
            }

            fn serialize_u8(self, v: u8) -> Result<()> {
                self.emit_unsigned_key(v as u128, 1)
            }
            fn serialize_u16(self, v: u16) -> Result<()> {
                self.emit_unsigned_key(v as u128, 2)
            }
            fn serialize_u32(self, v: u32) -> Result<()> {
                self.emit_unsigned_key(v as u128, 4)
            }
            fn serialize_u64(self, v: u64) -> Result<()> {
                self.emit_unsigned_key(v as u128, 8)
            }
            fn serialize_u128(self, v: u128) -> Result<()> {
                self.emit_unsigned_key(v, 16)
            }

            fn serialize_bool(self, _v: bool) -> Result<()> {
                Err(Error::InvalidType("boolean not allowed as object key"))
            }
            fn serialize_f32(self, _v: f32) -> Result<()> {
                Err(Error::InvalidType("float not allowed as object key"))
            }
            fn serialize_f64(self, _v: f64) -> Result<()> {
                Err(Error::InvalidType("float not allowed as object key"))
            }
            fn serialize_bytes(self, _v: &[u8]) -> Result<()> {
                Err(Error::InvalidType("bytes not allowed as object key"))
            }
            fn serialize_none(self) -> Result<()> {
                Err(Error::InvalidType("none not allowed as object key"))
            }
            fn serialize_some<T: ?Sized + Serialize>(self, _value: &T) -> Result<()> {
                Err(Error::InvalidType("option not allowed as object key"))
            }
            fn serialize_unit(self) -> Result<()> {
                Err(Error::InvalidType("unit not allowed as object key"))
            }
            fn serialize_unit_struct(self, _name: &'static str) -> Result<()> {
                Err(Error::InvalidType("unit struct not allowed as object key"))
            }
            fn serialize_unit_variant(
                self,
                _name: &'static str,
                _idx: u32,
                _var: &'static str,
            ) -> Result<()> {
                Err(Error::InvalidType("enum not allowed as object key"))
            }
            fn serialize_newtype_struct<T: ?Sized + Serialize>(
                self,
                _: &'static str,
                _: &T,
            ) -> Result<()> {
                Err(Error::InvalidType("newtype not allowed as object key"))
            }
            fn serialize_newtype_variant<T: ?Sized + Serialize>(
                self,
                _: &'static str,
                _: u32,
                _: &'static str,
                _: &T,
            ) -> Result<()> {
                Err(Error::InvalidType("enum not allowed as object key"))
            }
            fn serialize_seq(self, _: Option<usize>) -> Result<Self::SerializeSeq> {
                Err(Error::InvalidType("seq not allowed as object key"))
            }
            fn serialize_tuple(self, _: usize) -> Result<Self::SerializeTuple> {
                Err(Error::InvalidType("tuple not allowed as object key"))
            }
            fn serialize_tuple_struct(
                self,
                _: &'static str,
                _: usize,
            ) -> Result<Self::SerializeTupleStruct> {
                Err(Error::InvalidType("tuple struct not allowed as object key"))
            }
            fn serialize_tuple_variant(
                self,
                _: &'static str,
                _: u32,
                _: &'static str,
                _: usize,
            ) -> Result<Self::SerializeTupleVariant> {
                Err(Error::InvalidType(
                    "tuple variant not allowed as object key",
                ))
            }
            fn serialize_map(self, _: Option<usize>) -> Result<Self::SerializeMap> {
                Err(Error::InvalidType("map not allowed as object key"))
            }
            fn serialize_struct(self, _: &'static str, _: usize) -> Result<Self::SerializeStruct> {
                Err(Error::InvalidType("struct not allowed as object key"))
            }
            fn serialize_struct_variant(
                self,
                _: &'static str,
                _: u32,
                _: &'static str,
                _: usize,
            ) -> Result<Self::SerializeStructVariant> {
                Err(Error::InvalidType(
                    "struct variant not allowed as object key",
                ))
            }
            fn is_human_readable(&self) -> bool {
                false
            }
        }

        let mut ks = KeySer { map: self };
        key.serialize(&mut ks)
    }

    fn serialize_value<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<()> {
        value.serialize(&mut *self.ser)?;
        self.count += 1;
        Ok(())
    }

    fn end(self) -> Result<()> {
        if matches!(self.mode, KeyMode::Unknown) {
            self.ser.write_byte(TYPE_OBJECT | (KEY_STRING << 3))?;
            self.ser.write_size(0)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Struct serializer (always string-keyed)
// ---------------------------------------------------------------------------

pub struct StreamingStructSerializer<'a, W: Write> {
    ser: &'a mut StreamingSerializer<W>,
}

impl<'a, W: Write> ser::SerializeStruct for StreamingStructSerializer<'a, W> {
    type Ok = ();
    type Error = Error;

    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<()> {
        self.ser.write_size(key.len() as u64)?;
        self.ser.write_all(key.as_bytes())?;
        value.serialize(&mut *self.ser)
    }

    fn end(self) -> Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Struct variant serializer
// ---------------------------------------------------------------------------

pub struct StreamingVariantStructSerializer<'a, W: Write> {
    inner: StreamingStructSerializer<'a, W>,
}

impl<'a, W: Write> ser::SerializeStructVariant for StreamingVariantStructSerializer<'a, W> {
    type Ok = ();
    type Error = Error;

    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<()> {
        ser::SerializeStruct::serialize_field(&mut self.inner, key, value)
    }

    fn end(self) -> Result<()> {
        ser::SerializeStruct::end(self.inner)
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Serialize directly to a writer with zero internal buffering.
///
/// All containers must have known lengths (structs, `Vec`, `HashMap`, etc.).
/// Returns an error if an unknown-length sequence or map is encountered.
/// Homogeneous sequences (e.g. `Vec<f64>`) are encoded as compact typed arrays.
/// Heterogeneous sequences fall back to generic arrays.
///
/// # Example
///
/// ```rust
/// use serde::{Serialize, Deserialize};
///
/// #[derive(Serialize, Deserialize)]
/// struct Point { x: f64, y: f64 }
///
/// let mut buf = Vec::new();
/// beve::to_writer_streaming(&mut buf, &Point { x: 1.0, y: 2.0 }).unwrap();
/// let p: Point = beve::from_slice(&buf).unwrap();
/// assert_eq!(p.x, 1.0);
/// ```
pub fn to_writer_streaming<W: Write, T: Serialize>(writer: W, value: &T) -> Result<()> {
    let mut ser = StreamingSerializer::new(writer);
    value.serialize(&mut ser)?;
    Ok(ser.into_inner().flush()?)
}

/// Serialize directly to a writer with zero internal buffering and custom options.
///
/// See [`to_writer_streaming`] for details.
pub fn to_writer_streaming_with_options<W: Write, T: Serialize>(
    writer: W,
    value: &T,
    opts: SerializerOptions,
) -> Result<()> {
    let mut ser = StreamingSerializer::with_options(writer, opts);
    value.serialize(&mut ser)?;
    Ok(ser.into_inner().flush()?)
}
