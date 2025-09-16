use serde::ser::{self, Serialize};

use crate::error::{Error, Result};
use crate::header::*;
use crate::size::{encode_size_to_array, write_size};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnumEncoding {
    Number,
    String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SerializerOptions {
    pub enum_encoding: EnumEncoding,
}

impl Default for SerializerOptions {
    fn default() -> Self {
        // Default to numbers for interop with Glaze
        Self {
            enum_encoding: EnumEncoding::Number,
        }
    }
}

pub struct Serializer {
    pub(crate) buf: Vec<u8>,
    pub(crate) opts: SerializerOptions,
}

impl Serializer {
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            opts: SerializerOptions::default(),
        }
    }
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            buf: Vec::with_capacity(cap),
            opts: SerializerOptions::default(),
        }
    }
    pub fn with_options(opts: SerializerOptions) -> Self {
        Self {
            buf: Vec::new(),
            opts,
        }
    }
    pub fn with_capacity_and_options(cap: usize, opts: SerializerOptions) -> Self {
        Self {
            buf: Vec::with_capacity(cap),
            opts,
        }
    }
    pub fn into_vec(self) -> Vec<u8> {
        self.buf
    }

    #[inline]
    fn push(&mut self, b: u8) {
        self.buf.push(b)
    }

    #[inline]
    fn extend_from_slice(&mut self, s: &[u8]) {
        self.buf.extend_from_slice(s)
    }

    #[inline]
    fn reserve_size_patch(&mut self) -> SizePatch {
        let pos = self.buf.len();
        self.buf.extend_from_slice(&[0u8; 8]);
        SizePatch { pos }
    }

    #[inline]
    fn finalize_size_patch(&mut self, patch: SizePatch, count: usize) {
        let payload_start = patch.pos + 8;
        let payload_end = self.buf.len();
        let mut tmp = [0u8; 8];
        let used = encode_size_to_array(count as u64, &mut tmp);
        self.buf[patch.pos..patch.pos + used].copy_from_slice(&tmp[..used]);
        let delta = 8 - used;
        if delta > 0 {
            if payload_start < payload_end {
                self.buf
                    .copy_within(payload_start..payload_end, patch.pos + used);
            }
            self.buf.truncate(payload_end - delta);
        }
    }

    // ----- helpers to write full VALUEs -----
    fn write_null(&mut self) {
        self.push(0)
    }

    fn write_bool_value(&mut self, v: bool) {
        self.push(if v { 0x18 } else { 0x08 });
    }

    fn code_for_bytes(n: usize) -> u8 {
        match n {
            1 => 0,
            2 => 1,
            4 => 2,
            8 => 3,
            16 => 4,
            32 => 5,
            64 => 6,
            128 => 7,
            _ => 0,
        }
    }

    fn write_signed_value<const N: usize, T: Into<i128>>(&mut self, v: T) {
        let code = Self::code_for_bytes(N);
        let header = make_header(TYPE_NUMBER, NUM_SIGNED, code);
        self.push(header);
        let x: i128 = v.into();
        let bytes = x.to_le_bytes();
        self.extend_from_slice(&bytes[..N]);
    }

    fn write_unsigned_value<const N: usize, T: Into<u128>>(&mut self, v: T) {
        let code = Self::code_for_bytes(N);
        let header = make_header(TYPE_NUMBER, NUM_UNSIGNED, code);
        self.push(header);
        let x: u128 = v.into();
        let bytes = x.to_le_bytes();
        self.extend_from_slice(&bytes[..N]);
    }

    fn write_float32_value(&mut self, v: f32) {
        let header = make_header(TYPE_NUMBER, NUM_FLOAT, 2);
        self.push(header);
        self.extend_from_slice(&v.to_le_bytes());
    }

    fn write_float64_value(&mut self, v: f64) {
        let header = make_header(TYPE_NUMBER, NUM_FLOAT, 3);
        self.push(header);
        self.extend_from_slice(&v.to_le_bytes());
    }

    fn write_str_value(&mut self, s: &str) {
        self.push(TYPE_STRING);
        write_size(s.len() as u64, &mut self.buf);
        self.extend_from_slice(s.as_bytes());
    }

    fn write_enum_tag(&mut self, variant_index: u32, variant_name: &'static str) {
        match self.opts.enum_encoding {
            EnumEncoding::Number => {
                // Match Glaze behavior: emit 32-bit unsigned discriminant.
                self.write_unsigned_value::<4, _>(variant_index as u32);
            }
            EnumEncoding::String => {
                self.write_str_value(variant_name);
            }
        }
    }

    fn write_bytes_typed_array(&mut self, bytes: &[u8]) {
        // Typed array of u8
        let header = make_header(TYPE_TYPED_ARRAY, ARRAY_UNSIGNED, 0);
        self.push(header);
        write_size(bytes.len() as u64, &mut self.buf);
        self.extend_from_slice(bytes);
    }

    fn write_generic_array_header(&mut self, len: usize) {
        self.push(TYPE_GENERIC_ARRAY);
        write_size(len as u64, &mut self.buf);
    }

    fn write_typed_array_header_numeric(&mut self, class: u8, byte_code: u8, len: usize) {
        let header = make_header(TYPE_TYPED_ARRAY, class, byte_code);
        self.push(header);
        write_size(len as u64, &mut self.buf);
    }

    fn write_typed_array_header_bool(&mut self, len: usize) {
        let header = make_header(TYPE_TYPED_ARRAY, ARRAY_BOOL_OR_STRING, 0);
        self.push(header);
        write_size(len as u64, &mut self.buf);
    }

    fn write_typed_array_header_string(&mut self, len: usize) {
        let header = make_header(
            TYPE_TYPED_ARRAY,
            ARRAY_BOOL_OR_STRING,
            1 << 0, /* bit5 set via byte_code=1? */
        );
        // For boolean/string category, we use bit5 to indicate string (1) vs boolean (0).
        // Our make_header uses byte_code in bits 5..7; setting to 1 sets the bit5.
        self.push(header);
        write_size(len as u64, &mut self.buf);
    }
}

pub fn to_vec<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let mut ser = Serializer::new();
    value.serialize(&mut ser)?;
    Ok(ser.into_vec())
}

pub fn to_vec_with_options<T: Serialize>(value: &T, opts: SerializerOptions) -> Result<Vec<u8>> {
    let mut ser = Serializer::with_options(opts);
    value.serialize(&mut ser)?;
    Ok(ser.into_vec())
}

impl<'a> ser::Serializer for &'a mut Serializer {
    type Ok = ();
    type Error = Error;

    type SerializeSeq = SeqSerializer<'a>;
    type SerializeTuple = SeqSerializer<'a>;
    type SerializeTupleStruct = SeqSerializer<'a>;
    type SerializeTupleVariant = VariantSeqSerializer<'a>;
    type SerializeMap = MapSerializer<'a>;
    type SerializeStruct = StructSerializer<'a>;
    type SerializeStructVariant = VariantStructSerializer<'a>;

    fn serialize_bool(self, v: bool) -> Result<()> {
        self.write_bool_value(v);
        Ok(())
    }

    fn serialize_i8(self, v: i8) -> Result<()> {
        self.write_signed_value::<1, _>(v);
        Ok(())
    }
    fn serialize_i16(self, v: i16) -> Result<()> {
        self.write_signed_value::<2, _>(v);
        Ok(())
    }
    fn serialize_i32(self, v: i32) -> Result<()> {
        self.write_signed_value::<4, _>(v);
        Ok(())
    }
    fn serialize_i64(self, v: i64) -> Result<()> {
        self.write_signed_value::<8, _>(v);
        Ok(())
    }
    fn serialize_i128(self, v: i128) -> Result<()> {
        self.write_signed_value::<16, _>(v);
        Ok(())
    }

    fn serialize_u8(self, v: u8) -> Result<()> {
        self.write_unsigned_value::<1, _>(v);
        Ok(())
    }
    fn serialize_u16(self, v: u16) -> Result<()> {
        self.write_unsigned_value::<2, _>(v);
        Ok(())
    }
    fn serialize_u32(self, v: u32) -> Result<()> {
        self.write_unsigned_value::<4, _>(v);
        Ok(())
    }
    fn serialize_u64(self, v: u64) -> Result<()> {
        self.write_unsigned_value::<8, _>(v);
        Ok(())
    }
    fn serialize_u128(self, v: u128) -> Result<()> {
        self.write_unsigned_value::<16, _>(v);
        Ok(())
    }

    fn serialize_f32(self, v: f32) -> Result<()> {
        self.write_float32_value(v);
        Ok(())
    }
    fn serialize_f64(self, v: f64) -> Result<()> {
        self.write_float64_value(v);
        Ok(())
    }

    fn serialize_char(self, v: char) -> Result<()> {
        let mut buf = [0u8; 4];
        let s = v.encode_utf8(&mut buf);
        self.serialize_str(s)
    }

    fn serialize_str(self, v: &str) -> Result<()> {
        self.write_str_value(v);
        Ok(())
    }

    fn serialize_bytes(self, v: &[u8]) -> Result<()> {
        self.write_bytes_typed_array(v);
        Ok(())
    }

    fn serialize_none(self) -> Result<()> {
        self.write_null();
        Ok(())
    }

    fn serialize_some<T: ?Sized + Serialize>(self, value: &T) -> Result<()> {
        value.serialize(self)
    }

    fn serialize_unit(self) -> Result<()> {
        self.write_null();
        Ok(())
    }
    fn serialize_unit_struct(self, _name: &'static str) -> Result<()> {
        self.write_null();
        Ok(())
    }

    fn serialize_unit_variant(
        self,
        _name: &'static str,
        variant_index: u32,
        variant: &'static str,
    ) -> Result<()> {
        self.write_enum_tag(variant_index, variant);
        Ok(())
    }

    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        variant_index: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<()> {
        // Variants with values require the type-tag extension for round-trip
        self.push(make_extension_header(EXT_TYPE_TAG));
        self.write_enum_tag(variant_index, variant);
        value.serialize(self)
    }

    fn serialize_seq(self, len: Option<usize>) -> Result<Self::SerializeSeq> {
        Ok(SeqSerializer::new(self, len))
    }

    fn serialize_tuple(self, len: usize) -> Result<Self::SerializeTuple> {
        self.serialize_seq(Some(len))
    }

    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleStruct> {
        self.serialize_seq(Some(len))
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleVariant> {
        // Emit type-tag header+index, then the value as a generic array (tuple)
        self.push(make_extension_header(EXT_TYPE_TAG));
        self.write_enum_tag(variant_index, variant);
        Ok(VariantSeqSerializer::new(self, len))
    }

    fn serialize_map(self, len: Option<usize>) -> Result<Self::SerializeMap> {
        Ok(MapSerializer::new(self, len))
    }

    fn serialize_struct(self, _name: &'static str, len: usize) -> Result<Self::SerializeStruct> {
        Ok(StructSerializer::new(self, len))
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStructVariant> {
        // Emit type tag first
        self.push(make_extension_header(EXT_TYPE_TAG));
        self.write_enum_tag(variant_index, variant);
        Ok(VariantStructSerializer::new(self, len))
    }

    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        value: &T,
    ) -> Result<()> {
        value.serialize(self)
    }

    // Note: No `serialize_enum`; enum variants dispatch through variant helpers.

    fn is_human_readable(&self) -> bool {
        false
    }
}

// =============== Sequence Serializer ===============

enum SeqMode {
    Unknown,
    Generic {
        patch: Option<SizePatch>,
    },
    TypedUnsigned {
        byte_code: u8,
        patch: Option<SizePatch>,
    },
    TypedSigned {
        byte_code: u8,
        patch: Option<SizePatch>,
    },
    TypedFloat {
        byte_code: u8,
        patch: Option<SizePatch>,
    },
    TypedBool {
        byte_acc: u8,
        bit_idx: u8,
        patch: Option<SizePatch>,
    },
    TypedString {
        patch: Option<SizePatch>,
    },
}

pub struct SeqSerializer<'a> {
    ser: &'a mut Serializer,
    len: Option<usize>,
    mode: SeqMode,
    count: usize,
}

impl<'a> SeqSerializer<'a> {
    fn new(ser: &'a mut Serializer, len: Option<usize>) -> Self {
        Self {
            ser,
            len,
            mode: SeqMode::Unknown,
            count: 0,
        }
    }

    fn start_generic_if_needed(&mut self) {
        if !matches!(self.mode, SeqMode::Generic { .. }) {
            match self.len {
                Some(n) => {
                    self.ser.write_generic_array_header(n);
                    self.mode = SeqMode::Generic { patch: None };
                }
                None => {
                    self.ser.push(TYPE_GENERIC_ARRAY);
                    let p = self.ser.reserve_size_patch();
                    self.mode = SeqMode::Generic { patch: Some(p) };
                }
            }
        }
    }

    fn start_typed_bool_if_needed(&mut self) {
        if !matches!(self.mode, SeqMode::TypedBool { .. }) {
            match self.len {
                Some(n) => {
                    self.ser.write_typed_array_header_bool(n);
                    self.mode = SeqMode::TypedBool {
                        byte_acc: 0,
                        bit_idx: 0,
                        patch: None,
                    };
                }
                None => {
                    let header = make_header(TYPE_TYPED_ARRAY, ARRAY_BOOL_OR_STRING, 0);
                    self.ser.push(header);
                    let p = self.ser.reserve_size_patch();
                    self.mode = SeqMode::TypedBool {
                        byte_acc: 0,
                        bit_idx: 0,
                        patch: Some(p),
                    };
                }
            }
        }
    }

    fn start_typed_string_if_needed(&mut self) {
        if !matches!(self.mode, SeqMode::TypedString { .. }) {
            match self.len {
                Some(n) => {
                    self.ser.write_typed_array_header_string(n);
                    self.mode = SeqMode::TypedString { patch: None };
                }
                None => {
                    let header = make_header(TYPE_TYPED_ARRAY, ARRAY_BOOL_OR_STRING, 1);
                    self.ser.push(header);
                    let p = self.ser.reserve_size_patch();
                    self.mode = SeqMode::TypedString { patch: Some(p) };
                }
            }
        }
    }
}

struct SeqElemSer<'a, 'b> {
    seq: &'b mut SeqSerializer<'a>,
}

impl<'a, 'b> ser::Serializer for &'b mut SeqElemSer<'a, 'b> {
    type Ok = ();
    type Error = Error;

    type SerializeSeq = SeqSerializer<'b>;
    type SerializeTuple = SeqSerializer<'b>;
    type SerializeTupleStruct = SeqSerializer<'b>;
    type SerializeTupleVariant = VariantSeqSerializer<'b>;
    type SerializeMap = MapSerializer<'b>;
    type SerializeStruct = StructSerializer<'b>;
    type SerializeStructVariant = VariantStructSerializer<'b>;

    fn serialize_bool(self, v: bool) -> Result<()> {
        self.seq.start_typed_bool_if_needed();
        if let SeqMode::TypedBool {
            byte_acc, bit_idx, ..
        } = &mut self.seq.mode
        {
            // MSB-first packing: first element goes to bit7
            if v {
                *byte_acc |= 1 << (7 - *bit_idx);
            }
            *bit_idx += 1;
            if *bit_idx == 8 {
                self.seq.ser.push(*byte_acc);
                *byte_acc = 0;
                *bit_idx = 0;
            }
            self.seq.count += 1;
            Ok(())
        } else {
            unreachable!()
        }
    }

    fn serialize_i8(self, v: i8) -> Result<()> {
        self.serialize_signed(v, 1)
    }
    fn serialize_i16(self, v: i16) -> Result<()> {
        self.serialize_signed(v, 2)
    }
    fn serialize_i32(self, v: i32) -> Result<()> {
        self.serialize_signed(v, 4)
    }
    fn serialize_i64(self, v: i64) -> Result<()> {
        self.serialize_signed(v, 8)
    }
    fn serialize_i128(self, v: i128) -> Result<()> {
        self.serialize_signed(v, 16)
    }

    fn serialize_u8(self, v: u8) -> Result<()> {
        self.serialize_unsigned(v, 1)
    }
    fn serialize_u16(self, v: u16) -> Result<()> {
        self.serialize_unsigned(v, 2)
    }
    fn serialize_u32(self, v: u32) -> Result<()> {
        self.serialize_unsigned(v, 4)
    }
    fn serialize_u64(self, v: u64) -> Result<()> {
        self.serialize_unsigned(v, 8)
    }
    fn serialize_u128(self, v: u128) -> Result<()> {
        self.serialize_unsigned(v, 16)
    }

    fn serialize_f32(self, v: f32) -> Result<()> {
        self.serialize_float_by(v.to_le_bytes().as_slice(), 2)
    }
    fn serialize_f64(self, v: f64) -> Result<()> {
        self.serialize_float_by(v.to_le_bytes().as_slice(), 3)
    }

    fn serialize_char(self, v: char) -> Result<()> {
        let mut buf = [0u8; 4];
        self.serialize_str(v.encode_utf8(&mut buf))
    }

    fn serialize_str(self, v: &str) -> Result<()> {
        self.seq.start_typed_string_if_needed();
        write_size(v.len() as u64, &mut self.seq.ser.buf);
        self.seq.ser.extend_from_slice(v.as_bytes());
        self.seq.count += 1;
        Ok(())
    }

    fn serialize_bytes(self, v: &[u8]) -> Result<()> {
        // Treat as unsigned typed array of u8 nested sequence-of-bytes: fallback to generic element
        self.seq.start_generic_if_needed();
        // Write full VALUE for bytes as a typed array of u8
        let header = make_header(TYPE_TYPED_ARRAY, ARRAY_UNSIGNED, 0);
        self.seq.ser.push(header);
        write_size(v.len() as u64, &mut self.seq.ser.buf);
        self.seq.ser.extend_from_slice(v);
        self.seq.count += 1;
        Ok(())
    }

    fn serialize_none(self) -> Result<()> {
        self.serialize_unit()
    }
    fn serialize_some<T: ?Sized + Serialize>(self, v: &T) -> Result<()> {
        v.serialize(self)
    }
    fn serialize_unit(self) -> Result<()> {
        self.seq.start_generic_if_needed();
        self.seq.ser.write_null();
        self.seq.count += 1;
        Ok(())
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
        // As a generic element: extension type-tag with null value
        self.seq.start_generic_if_needed();
        let h = make_extension_header(EXT_TYPE_TAG);
        self.seq.ser.push(h);
        self.seq.ser.write_enum_tag(variant_index, variant);
        self.seq.ser.write_null();
        self.seq.count += 1;
        Ok(())
    }

    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        value: &T,
    ) -> Result<()> {
        value.serialize(self)
    }

    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        variant_index: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<()> {
        self.seq.start_generic_if_needed();
        let h = make_extension_header(EXT_TYPE_TAG);
        self.seq.ser.push(h);
        self.seq.ser.write_enum_tag(variant_index, variant);
        // Reborrow underlying serializer as &mut Serializer to satisfy trait
        let ser = &mut *self.seq.ser;
        value.serialize(ser)
    }

    fn serialize_seq(self, len: Option<usize>) -> Result<Self::SerializeSeq> {
        self.seq.start_generic_if_needed();
        self.seq.count += 1;
        Ok(SeqSerializer::new(self.seq.ser, len))
    }
    fn serialize_tuple(self, len: usize) -> Result<Self::SerializeTuple> {
        self.seq.start_generic_if_needed();
        self.seq.count += 1;
        Ok(SeqSerializer::new(self.seq.ser, Some(len)))
    }
    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleStruct> {
        self.seq.start_generic_if_needed();
        self.seq.count += 1;
        Ok(SeqSerializer::new(self.seq.ser, Some(len)))
    }
    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleVariant> {
        self.seq.start_generic_if_needed();
        self.seq.count += 1;
        let h = make_extension_header(EXT_TYPE_TAG);
        self.seq.ser.push(h);
        self.seq.ser.write_enum_tag(variant_index, variant);
        Ok(VariantSeqSerializer::new(self.seq.ser, len))
    }
    fn serialize_map(self, len: Option<usize>) -> Result<Self::SerializeMap> {
        self.seq.start_generic_if_needed();
        self.seq.count += 1;
        Ok(MapSerializer::new(self.seq.ser, len))
    }
    fn serialize_struct(self, _name: &'static str, len: usize) -> Result<Self::SerializeStruct> {
        self.seq.start_generic_if_needed();
        self.seq.count += 1;
        Ok(StructSerializer::new(self.seq.ser, len))
    }
    fn serialize_struct_variant(
        self,
        _name: &'static str,
        variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStructVariant> {
        self.seq.start_generic_if_needed();
        self.seq.count += 1;
        let h = make_extension_header(EXT_TYPE_TAG);
        self.seq.ser.push(h);
        self.seq.ser.write_enum_tag(variant_index, variant);
        Ok(VariantStructSerializer::new(self.seq.ser, len))
    }

    // helper impls
}

impl<'a, 'b> SeqElemSer<'a, 'b> {
    fn serialize_signed<T: Into<i128>>(&mut self, v: T, bytes: usize) -> Result<()> {
        let byte_code = Serializer::code_for_bytes(bytes);
        match self.seq.mode {
            SeqMode::Unknown => match self.seq.len {
                Some(n) => {
                    self.seq
                        .ser
                        .write_typed_array_header_numeric(ARRAY_SIGNED, byte_code, n);
                    self.seq.mode = SeqMode::TypedSigned {
                        byte_code,
                        patch: None,
                    };
                }
                None => {
                    let header = make_header(TYPE_TYPED_ARRAY, ARRAY_SIGNED, byte_code);
                    self.seq.ser.push(header);
                    let p = self.seq.ser.reserve_size_patch();
                    self.seq.mode = SeqMode::TypedSigned {
                        byte_code,
                        patch: Some(p),
                    };
                }
            },
            SeqMode::TypedSigned { byte_code: bc, .. } if bc == byte_code => {}
            _ => {
                return Err(Error::Mismatch(
                    "heterogeneous sequence types (signed integer mix)",
                ))
            }
        }
        let x: i128 = v.into();
        let bytes_le = x.to_le_bytes();
        // Stream directly into output; unknown-length arrays use backpatching
        self.seq.ser.extend_from_slice(&bytes_le[..bytes]);
        self.seq.count += 1;
        Ok(())
    }

    fn serialize_unsigned<T: Into<u128>>(&mut self, v: T, bytes: usize) -> Result<()> {
        let byte_code = Serializer::code_for_bytes(bytes);
        match self.seq.mode {
            SeqMode::Unknown => match self.seq.len {
                Some(n) => {
                    self.seq
                        .ser
                        .write_typed_array_header_numeric(ARRAY_UNSIGNED, byte_code, n);
                    self.seq.mode = SeqMode::TypedUnsigned {
                        byte_code,
                        patch: None,
                    };
                }
                None => {
                    let header = make_header(TYPE_TYPED_ARRAY, ARRAY_UNSIGNED, byte_code);
                    self.seq.ser.push(header);
                    let p = self.seq.ser.reserve_size_patch();
                    self.seq.mode = SeqMode::TypedUnsigned {
                        byte_code,
                        patch: Some(p),
                    };
                }
            },
            SeqMode::TypedUnsigned { byte_code: bc, .. } if bc == byte_code => {}
            _ => {
                return Err(Error::Mismatch(
                    "heterogeneous sequence types (unsigned integer mix)",
                ))
            }
        }
        let x: u128 = v.into();
        let bytes_le = x.to_le_bytes();
        self.seq.ser.extend_from_slice(&bytes_le[..bytes]);
        self.seq.count += 1;
        Ok(())
    }

    fn serialize_float_by(&mut self, raw: &[u8], byte_code: u8) -> Result<()> {
        match self.seq.mode {
            SeqMode::Unknown => match self.seq.len {
                Some(n) => {
                    self.seq
                        .ser
                        .write_typed_array_header_numeric(ARRAY_FLOAT, byte_code, n);
                    self.seq.mode = SeqMode::TypedFloat {
                        byte_code,
                        patch: None,
                    };
                }
                None => {
                    let header = make_header(TYPE_TYPED_ARRAY, ARRAY_FLOAT, byte_code);
                    self.seq.ser.push(header);
                    let p = self.seq.ser.reserve_size_patch();
                    self.seq.mode = SeqMode::TypedFloat {
                        byte_code,
                        patch: Some(p),
                    };
                }
            },
            SeqMode::TypedFloat { byte_code: bc, .. } if bc == byte_code => {}
            _ => return Err(Error::Mismatch("heterogeneous sequence types (float mix)")),
        }
        self.seq.ser.extend_from_slice(raw);
        self.seq.count += 1;
        Ok(())
    }
}

impl<'a> ser::SerializeSeq for SeqSerializer<'a> {
    type Ok = ();
    type Error = Error;

    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<()> {
        if matches!(self.mode, SeqMode::Generic { .. }) {
            value.serialize(&mut *self.ser)?;
            self.count += 1;
            Ok(())
        } else {
            let mut elem = SeqElemSer { seq: self };
            value.serialize(&mut elem)
        }
    }

    fn end(mut self) -> Result<()> {
        if let SeqMode::TypedBool {
            byte_acc, bit_idx, ..
        } = &mut self.mode
        {
            if *bit_idx != 0 {
                self.ser.push(*byte_acc);
            }
        }
        if self.len.is_none() {
            match self.mode {
                SeqMode::Unknown => {
                    self.ser.push(TYPE_GENERIC_ARRAY);
                    let mut tmp = [0u8; 8];
                    let used = encode_size_to_array(0, &mut tmp);
                    self.ser.extend_from_slice(&tmp[..used]);
                }
                SeqMode::Generic { patch: Some(p) } => self.ser.finalize_size_patch(p, self.count),
                SeqMode::TypedUnsigned { patch: Some(p), .. } => {
                    self.ser.finalize_size_patch(p, self.count)
                }
                SeqMode::TypedSigned { patch: Some(p), .. } => {
                    self.ser.finalize_size_patch(p, self.count)
                }
                SeqMode::TypedFloat { patch: Some(p), .. } => {
                    self.ser.finalize_size_patch(p, self.count)
                }
                SeqMode::TypedBool { patch: Some(p), .. } => {
                    self.ser.finalize_size_patch(p, self.count)
                }
                SeqMode::TypedString { patch: Some(p) } => {
                    self.ser.finalize_size_patch(p, self.count)
                }
                _ => {}
            }
        }
        Ok(())
    }
}

impl<'a> ser::SerializeTuple for SeqSerializer<'a> {
    type Ok = ();
    type Error = Error;
    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<()> {
        ser::SerializeSeq::serialize_element(self, value)
    }
    fn end(self) -> Result<()> {
        ser::SerializeSeq::end(self)
    }
}

impl<'a> ser::SerializeTupleStruct for SeqSerializer<'a> {
    type Ok = ();
    type Error = Error;
    fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<()> {
        ser::SerializeSeq::serialize_element(self, value)
    }
    fn end(self) -> Result<()> {
        ser::SerializeSeq::end(self)
    }
}

// =============== Tuple Variant Serializer ===============

pub struct VariantSeqSerializer<'a> {
    inner: SeqSerializer<'a>,
}

impl<'a> VariantSeqSerializer<'a> {
    fn new(ser: &'a mut Serializer, len: usize) -> Self {
        let mut inner = SeqSerializer::new(ser, Some(len));
        // For tuple variants, we encode as a generic array as the VALUE following the tag
        inner.start_generic_if_needed();
        Self { inner }
    }
}

impl<'a> ser::SerializeTupleVariant for VariantSeqSerializer<'a> {
    type Ok = ();
    type Error = Error;
    fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<()> {
        ser::SerializeSeq::serialize_element(&mut self.inner, value)
    }
    fn end(self) -> Result<()> {
        ser::SerializeSeq::end(self.inner)
    }
}

// =============== Map/Struct Serializer ===============

enum KeyMode {
    Unknown,
    String,
    Signed(u8),
    Unsigned(u8),
}

pub struct MapSerializer<'a> {
    ser: &'a mut Serializer,
    len: Option<usize>,
    mode: KeyMode,
    count: usize,
    patch: Option<SizePatch>,
}

impl<'a> MapSerializer<'a> {
    fn new(ser: &'a mut Serializer, len: Option<usize>) -> Self {
        Self {
            ser,
            len,
            mode: KeyMode::Unknown,
            count: 0,
            patch: None,
        }
    }
}

impl<'a> ser::SerializeMap for MapSerializer<'a> {
    type Ok = ();
    type Error = Error;

    fn serialize_key<T: ?Sized + Serialize>(&mut self, key: &T) -> Result<()> {
        // We'll serialize key immediately; value will follow in serialize_value
        struct KeySer<'a, 'b> {
            map: &'b mut MapSerializer<'a>,
        }
        impl<'a, 'b> ser::Serializer for &'b mut KeySer<'a, 'b> {
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
                        if let Some(n) = self.map.len {
                            self.map.ser.push(TYPE_OBJECT | (KEY_STRING << 3));
                            write_size(n as u64, &mut self.map.ser.buf);
                        } else {
                            self.map.ser.push(TYPE_OBJECT | (KEY_STRING << 3));
                            self.map.patch = Some(self.map.ser.reserve_size_patch());
                        }
                        self.map.mode = KeyMode::String;
                    }
                    KeyMode::String => {}
                    _ => return Err(Error::Mismatch("object keys must be homogeneous")),
                }
                write_size(v.len() as u64, &mut self.map.ser.buf);
                self.map.ser.extend_from_slice(v.as_bytes());
                Ok(())
            }
            fn serialize_i8(self, v: i8) -> Result<()> {
                let bytes = 1;
                let code = Serializer::code_for_bytes(bytes);
                match self.map.mode {
                    KeyMode::Unknown => {
                        if let Some(n) = self.map.len {
                            self.map
                                .ser
                                .push(make_header(TYPE_OBJECT, KEY_SIGNED, code));
                            write_size(n as u64, &mut self.map.ser.buf);
                        } else {
                            self.map
                                .ser
                                .push(make_header(TYPE_OBJECT, KEY_SIGNED, code));
                            self.map.patch = Some(self.map.ser.reserve_size_patch());
                        }
                        self.map.mode = KeyMode::Signed(code);
                    }
                    KeyMode::Signed(c) if c == code => {}
                    _ => {
                        return Err(Error::Mismatch(
                            "object keys must be homogeneous signed integers of same width",
                        ))
                    }
                }
                let raw = (v as i128).to_le_bytes();
                self.map.ser.extend_from_slice(&raw[..bytes]);
                Ok(())
            }
            fn serialize_i16(self, v: i16) -> Result<()> {
                let bytes = 2;
                let code = Serializer::code_for_bytes(bytes);
                match self.map.mode {
                    KeyMode::Unknown => {
                        if let Some(n) = self.map.len {
                            self.map
                                .ser
                                .push(make_header(TYPE_OBJECT, KEY_SIGNED, code));
                            write_size(n as u64, &mut self.map.ser.buf);
                        } else {
                            self.map
                                .ser
                                .push(make_header(TYPE_OBJECT, KEY_SIGNED, code));
                            self.map.patch = Some(self.map.ser.reserve_size_patch());
                        }
                        self.map.mode = KeyMode::Signed(code);
                    }
                    KeyMode::Signed(c) if c == code => {}
                    _ => {
                        return Err(Error::Mismatch(
                            "object keys must be homogeneous signed integers of same width",
                        ))
                    }
                }
                let raw = (v as i128).to_le_bytes();
                self.map.ser.extend_from_slice(&raw[..bytes]);
                Ok(())
            }
            fn serialize_i32(self, v: i32) -> Result<()> {
                let bytes = 4;
                let code = Serializer::code_for_bytes(bytes);
                match self.map.mode {
                    KeyMode::Unknown => {
                        if let Some(n) = self.map.len {
                            self.map
                                .ser
                                .push(make_header(TYPE_OBJECT, KEY_SIGNED, code));
                            write_size(n as u64, &mut self.map.ser.buf);
                        } else {
                            self.map
                                .ser
                                .push(make_header(TYPE_OBJECT, KEY_SIGNED, code));
                            self.map.patch = Some(self.map.ser.reserve_size_patch());
                        }
                        self.map.mode = KeyMode::Signed(code);
                    }
                    KeyMode::Signed(c) if c == code => {}
                    _ => {
                        return Err(Error::Mismatch(
                            "object keys must be homogeneous signed integers of same width",
                        ))
                    }
                }
                let raw = (v as i128).to_le_bytes();
                self.map.ser.extend_from_slice(&raw[..bytes]);
                Ok(())
            }
            fn serialize_i64(self, v: i64) -> Result<()> {
                let bytes = 8;
                let code = Serializer::code_for_bytes(bytes);
                match self.map.mode {
                    KeyMode::Unknown => {
                        if let Some(n) = self.map.len {
                            self.map
                                .ser
                                .push(make_header(TYPE_OBJECT, KEY_SIGNED, code));
                            write_size(n as u64, &mut self.map.ser.buf);
                        } else {
                            self.map
                                .ser
                                .push(make_header(TYPE_OBJECT, KEY_SIGNED, code));
                            self.map.patch = Some(self.map.ser.reserve_size_patch());
                        }
                        self.map.mode = KeyMode::Signed(code);
                    }
                    KeyMode::Signed(c) if c == code => {}
                    _ => {
                        return Err(Error::Mismatch(
                            "object keys must be homogeneous signed integers of same width",
                        ))
                    }
                }
                let raw = (v as i128).to_le_bytes();
                self.map.ser.extend_from_slice(&raw[..bytes]);
                Ok(())
            }
            fn serialize_i128(self, v: i128) -> Result<()> {
                let bytes = 16;
                let code = Serializer::code_for_bytes(bytes);
                match self.map.mode {
                    KeyMode::Unknown => {
                        if let Some(n) = self.map.len {
                            self.map
                                .ser
                                .push(make_header(TYPE_OBJECT, KEY_SIGNED, code));
                            write_size(n as u64, &mut self.map.ser.buf);
                        } else {
                            self.map
                                .ser
                                .push(make_header(TYPE_OBJECT, KEY_SIGNED, code));
                            self.map.patch = Some(self.map.ser.reserve_size_patch());
                        }
                        self.map.mode = KeyMode::Signed(code);
                    }
                    KeyMode::Signed(c) if c == code => {}
                    _ => {
                        return Err(Error::Mismatch(
                            "object keys must be homogeneous signed integers of same width",
                        ))
                    }
                }
                let raw = v.to_le_bytes();
                self.map.ser.extend_from_slice(&raw[..bytes]);
                Ok(())
            }
            fn serialize_u8(self, v: u8) -> Result<()> {
                let bytes = 1;
                let code = Serializer::code_for_bytes(bytes);
                match self.map.mode {
                    KeyMode::Unknown => {
                        if let Some(n) = self.map.len {
                            self.map
                                .ser
                                .push(make_header(TYPE_OBJECT, KEY_UNSIGNED, code));
                            write_size(n as u64, &mut self.map.ser.buf);
                        } else {
                            self.map
                                .ser
                                .push(make_header(TYPE_OBJECT, KEY_UNSIGNED, code));
                            self.map.patch = Some(self.map.ser.reserve_size_patch());
                        }
                        self.map.mode = KeyMode::Unsigned(code);
                    }
                    KeyMode::Unsigned(c) if c == code => {}
                    _ => {
                        return Err(Error::Mismatch(
                            "object keys must be homogeneous unsigned integers of same width",
                        ))
                    }
                }
                let raw = (v as u128).to_le_bytes();
                self.map.ser.extend_from_slice(&raw[..bytes]);
                Ok(())
            }
            fn serialize_u16(self, v: u16) -> Result<()> {
                let bytes = 2;
                let code = Serializer::code_for_bytes(bytes);
                match self.map.mode {
                    KeyMode::Unknown => {
                        if let Some(n) = self.map.len {
                            self.map
                                .ser
                                .push(make_header(TYPE_OBJECT, KEY_UNSIGNED, code));
                            write_size(n as u64, &mut self.map.ser.buf);
                        } else {
                            self.map
                                .ser
                                .push(make_header(TYPE_OBJECT, KEY_UNSIGNED, code));
                            self.map.patch = Some(self.map.ser.reserve_size_patch());
                        }
                        self.map.mode = KeyMode::Unsigned(code);
                    }
                    KeyMode::Unsigned(c) if c == code => {}
                    _ => {
                        return Err(Error::Mismatch(
                            "object keys must be homogeneous unsigned integers of same width",
                        ))
                    }
                }
                let raw = (v as u128).to_le_bytes();
                self.map.ser.extend_from_slice(&raw[..bytes]);
                Ok(())
            }
            fn serialize_u32(self, v: u32) -> Result<()> {
                let bytes = 4;
                let code = Serializer::code_for_bytes(bytes);
                match self.map.mode {
                    KeyMode::Unknown => {
                        if let Some(n) = self.map.len {
                            self.map
                                .ser
                                .push(make_header(TYPE_OBJECT, KEY_UNSIGNED, code));
                            write_size(n as u64, &mut self.map.ser.buf);
                        } else {
                            self.map
                                .ser
                                .push(make_header(TYPE_OBJECT, KEY_UNSIGNED, code));
                            self.map.patch = Some(self.map.ser.reserve_size_patch());
                        }
                        self.map.mode = KeyMode::Unsigned(code);
                    }
                    KeyMode::Unsigned(c) if c == code => {}
                    _ => {
                        return Err(Error::Mismatch(
                            "object keys must be homogeneous unsigned integers of same width",
                        ))
                    }
                }
                let raw = (v as u128).to_le_bytes();
                self.map.ser.extend_from_slice(&raw[..bytes]);
                Ok(())
            }
            fn serialize_u64(self, v: u64) -> Result<()> {
                let bytes = 8;
                let code = Serializer::code_for_bytes(bytes);
                match self.map.mode {
                    KeyMode::Unknown => {
                        if let Some(n) = self.map.len {
                            self.map
                                .ser
                                .push(make_header(TYPE_OBJECT, KEY_UNSIGNED, code));
                            write_size(n as u64, &mut self.map.ser.buf);
                        } else {
                            self.map
                                .ser
                                .push(make_header(TYPE_OBJECT, KEY_UNSIGNED, code));
                            self.map.patch = Some(self.map.ser.reserve_size_patch());
                        }
                        self.map.mode = KeyMode::Unsigned(code);
                    }
                    KeyMode::Unsigned(c) if c == code => {}
                    _ => {
                        return Err(Error::Mismatch(
                            "object keys must be homogeneous unsigned integers of same width",
                        ))
                    }
                }
                let raw = (v as u128).to_le_bytes();
                self.map.ser.extend_from_slice(&raw[..bytes]);
                Ok(())
            }
            fn serialize_u128(self, v: u128) -> Result<()> {
                let bytes = 16;
                let code = Serializer::code_for_bytes(bytes);
                match self.map.mode {
                    KeyMode::Unknown => {
                        if let Some(n) = self.map.len {
                            self.map
                                .ser
                                .push(make_header(TYPE_OBJECT, KEY_UNSIGNED, code));
                            write_size(n as u64, &mut self.map.ser.buf);
                        } else {
                            self.map
                                .ser
                                .push(make_header(TYPE_OBJECT, KEY_UNSIGNED, code));
                            self.map.patch = Some(self.map.ser.reserve_size_patch());
                        }
                        self.map.mode = KeyMode::Unsigned(code);
                    }
                    KeyMode::Unsigned(c) if c == code => {}
                    _ => {
                        return Err(Error::Mismatch(
                            "object keys must be homogeneous unsigned integers of same width",
                        ))
                    }
                }
                let raw = v.to_le_bytes();
                self.map.ser.extend_from_slice(&raw[..bytes]);
                Ok(())
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
            fn serialize_char(self, v: char) -> Result<()> {
                let mut buf = [0; 4];
                self.serialize_str(v.encode_utf8(&mut buf))
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

            // no additional helpers inside trait impl
        }

        let mut ks = KeySer { map: self };
        key.serialize(&mut ks)
    }

    fn serialize_value<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<()> {
        value.serialize(&mut *self.ser)?;
        self.count += 1;
        Ok(())
    }

    fn end(mut self) -> Result<()> {
        if self.len.is_none() {
            if let Some(p) = self.patch.take() {
                self.ser.finalize_size_patch(p, self.count);
            } else {
                // empty map, default to string-keyed object
                self.ser.push(TYPE_OBJECT | (KEY_STRING << 3));
                let mut tmp = [0u8; 8];
                let used = encode_size_to_array(0, &mut tmp);
                self.ser.extend_from_slice(&tmp[..used]);
            }
        }
        Ok(())
    }
}

pub struct StructSerializer<'a> {
    map: MapSerializer<'a>,
}

impl<'a> StructSerializer<'a> {
    fn new(ser: &'a mut Serializer, len: usize) -> Self {
        let mut map = MapSerializer::new(ser, Some(len));
        // Emit header for string-keyed object immediately
        map.ser.push(TYPE_OBJECT | (KEY_STRING << 3));
        write_size(len as u64, &mut map.ser.buf);
        map.mode = KeyMode::String;
        Self { map }
    }
}

impl<'a> ser::SerializeStruct for StructSerializer<'a> {
    type Ok = ();
    type Error = Error;
    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<()> {
        write_size(key.len() as u64, &mut self.map.ser.buf);
        self.map.ser.extend_from_slice(key.as_bytes());
        value.serialize(&mut *self.map.ser)
    }
    fn end(self) -> Result<()> {
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct SizePatch {
    pos: usize,
}

pub struct VariantStructSerializer<'a> {
    inner: StructSerializer<'a>,
}

impl<'a> VariantStructSerializer<'a> {
    fn new(ser: &'a mut Serializer, len: usize) -> Self {
        // After type-tag header and index, we write the struct as the VALUE object
        let inner = StructSerializer::new(ser, len);
        Self { inner }
    }
}

impl<'a> ser::SerializeStructVariant for VariantStructSerializer<'a> {
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
