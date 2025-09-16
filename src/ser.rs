use core::mem;
use serde::ser::{self, Serialize};

use crate::error::{Error, Result};
use crate::header::*;
use crate::size::write_size;

pub struct Serializer {
    pub(crate) buf: Vec<u8>,
}

impl Serializer {
    pub fn new() -> Self { Self { buf: Vec::new() } }
    pub fn with_capacity(cap: usize) -> Self { Self { buf: Vec::with_capacity(cap) } }
    pub fn into_vec(self) -> Vec<u8> { self.buf }

    #[inline]
    fn push(&mut self, b: u8) { self.buf.push(b) }

    #[inline]
    fn extend_from_slice(&mut self, s: &[u8]) { self.buf.extend_from_slice(s) }

    // ----- helpers to write full VALUEs -----
    fn write_null(&mut self) { self.push(0) }

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
        let header = make_header(TYPE_TYPED_ARRAY, ARRAY_BOOL_OR_STRING, 1 << 0 /* bit5 set via byte_code=1? */);
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

    fn serialize_i8(self, v: i8) -> Result<()> { self.write_signed_value::<1, _>(v); Ok(()) }
    fn serialize_i16(self, v: i16) -> Result<()> { self.write_signed_value::<2, _>(v); Ok(()) }
    fn serialize_i32(self, v: i32) -> Result<()> { self.write_signed_value::<4, _>(v); Ok(()) }
    fn serialize_i64(self, v: i64) -> Result<()> { self.write_signed_value::<8, _>(v); Ok(()) }
    fn serialize_i128(self, v: i128) -> Result<()> { self.write_signed_value::<16, _>(v); Ok(()) }

    fn serialize_u8(self, v: u8) -> Result<()> { self.write_unsigned_value::<1, _>(v); Ok(()) }
    fn serialize_u16(self, v: u16) -> Result<()> { self.write_unsigned_value::<2, _>(v); Ok(()) }
    fn serialize_u32(self, v: u32) -> Result<()> { self.write_unsigned_value::<4, _>(v); Ok(()) }
    fn serialize_u64(self, v: u64) -> Result<()> { self.write_unsigned_value::<8, _>(v); Ok(()) }
    fn serialize_u128(self, v: u128) -> Result<()> { self.write_unsigned_value::<16, _>(v); Ok(()) }

    fn serialize_f32(self, v: f32) -> Result<()> { self.write_float32_value(v); Ok(()) }
    fn serialize_f64(self, v: f64) -> Result<()> { self.write_float64_value(v); Ok(()) }

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

    fn serialize_none(self) -> Result<()> { self.write_null(); Ok(()) }

    fn serialize_some<T: ?Sized + Serialize>(self, value: &T) -> Result<()> {
        value.serialize(self)
    }

    fn serialize_unit(self) -> Result<()> { self.write_null(); Ok(()) }
    fn serialize_unit_struct(self, _name: &'static str) -> Result<()> { self.write_null(); Ok(()) }

    fn serialize_unit_variant(self, _name: &'static str, variant_index: u32, _variant: &'static str) -> Result<()> {
        // Extension: Type Tag: HEADER | SIZE(tag) | VALUE(null)
        self.push(make_extension_header(EXT_TYPE_TAG));
        write_size(variant_index as u64, &mut self.buf);
        self.write_null();
        Ok(())
    }

    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        variant_index: u32,
        _variant: &'static str,
        value: &T,
    ) -> Result<()> {
        self.push(make_extension_header(EXT_TYPE_TAG));
        write_size(variant_index as u64, &mut self.buf);
        value.serialize(self)
    }

    fn serialize_seq(self, len: Option<usize>) -> Result<Self::SerializeSeq> {
        Ok(SeqSerializer::new(self, len))
    }

    fn serialize_tuple(self, len: usize) -> Result<Self::SerializeTuple> {
        self.serialize_seq(Some(len))
    }

    fn serialize_tuple_struct(self, _name: &'static str, len: usize) -> Result<Self::SerializeTupleStruct> {
        self.serialize_seq(Some(len))
    }

    fn serialize_tuple_variant(self, _name: &'static str, variant_index: u32, _variant: &'static str, len: usize) -> Result<Self::SerializeTupleVariant> {
        // Emit type-tag header+index, then the value as a generic array (tuple)
        self.push(make_extension_header(EXT_TYPE_TAG));
        write_size(variant_index as u64, &mut self.buf);
        Ok(VariantSeqSerializer::new(self, len))
    }

    fn serialize_map(self, len: Option<usize>) -> Result<Self::SerializeMap> {
        Ok(MapSerializer::new(self, len))
    }

    fn serialize_struct(self, _name: &'static str, len: usize) -> Result<Self::SerializeStruct> {
        Ok(StructSerializer::new(self, len))
    }

    fn serialize_struct_variant(self, _name: &'static str, variant_index: u32, _variant: &'static str, len: usize) -> Result<Self::SerializeStructVariant> {
        // Emit type tag first
        self.push(make_extension_header(EXT_TYPE_TAG));
        write_size(variant_index as u64, &mut self.buf);
        Ok(VariantStructSerializer::new(self, len))
    }

    fn serialize_newtype_struct<T: ?Sized + Serialize>(self, _name: &'static str, value: &T) -> Result<()> {
        value.serialize(self)
    }

    fn serialize_bytes_newtype(self, v: &[u8]) -> Result<()> { // non-standard helper; not used by serde
        self.serialize_bytes(v)
    }

    fn serialize_enum(self, _name: &'static str, _variant_index: u32, _variant: &'static str, _value: &dyn Serialize) -> Result<()> {
        Err(Error::Unsupported("serialize_enum should be dispatched through variant helpers"))
    }

    fn is_human_readable(&self) -> bool { false }
}

// =============== Sequence Serializer ===============

enum SeqMode {
    Unknown,
    Generic,
    TypedUnsigned(u8), // bytecode
    TypedSigned(u8),
    TypedFloat(u8),
    TypedBool { byte_acc: u8, bit_idx: u8 },
    TypedString,
}

pub struct SeqSerializer<'a> {
    ser: &'a mut Serializer,
    len: Option<usize>,
    mode: SeqMode,
    count: usize,
    // For unknown-length sequences we buffer and write at end
    buffer: Vec<u8>,
}

impl<'a> SeqSerializer<'a> {
    fn new(ser: &'a mut Serializer, len: Option<usize>) -> Self {
        Self { ser, len, mode: SeqMode::Unknown, count: 0, buffer: Vec::new() }
    }

    fn ensure_generic_header(&mut self) {
        match self.len {
            Some(n) if self.count == 0 => {
                self.ser.write_generic_array_header(n);
                self.mode = SeqMode::Generic;
            }
            None => {
                if let SeqMode::Unknown = self.mode {
                    self.mode = SeqMode::Generic;
                }
            }
            _ => {}
        }
    }
}

struct SeqElemSer<'a, 'b> {
    seq: &'b mut SeqSerializer<'a>,
}

impl<'a, 'b> ser::Serializer for &'b mut SeqElemSer<'a, 'b> {
    type Ok = ();
    type Error = Error;

    type SerializeSeq = ser::Impossible<(), Error>;
    type SerializeTuple = ser::Impossible<(), Error>;
    type SerializeTupleStruct = ser::Impossible<(), Error>;
    type SerializeTupleVariant = ser::Impossible<(), Error>;
    type SerializeMap = ser::Impossible<(), Error>;
    type SerializeStruct = ser::Impossible<(), Error>;
    type SerializeStructVariant = ser::Impossible<(), Error>;

    fn serialize_bool(self, v: bool) -> Result<()> {
        match &mut self.seq.mode {
            SeqMode::Unknown => {
                match self.seq.len {
                    Some(n) => {
                        self.seq.ser.write_typed_array_header_bool(n);
                        self.seq.mode = SeqMode::TypedBool { byte_acc: 0, bit_idx: 0 };
                    }
                    None => {
                        self.seq.mode = SeqMode::TypedBool { byte_acc: 0, bit_idx: 0 };
                    }
                }
            }
            SeqMode::TypedBool { .. } => {}
            _ => return Err(Error::Mismatch("heterogeneous sequence: expected boolean consistent with first element")),
        }

        match &mut self.seq.mode {
            SeqMode::TypedBool { byte_acc, bit_idx } => {
                if self.seq.len.is_some() {
                    if v { *byte_acc |= 1 << *bit_idx; }
                    *bit_idx += 1;
                    if *bit_idx == 8 {
                        self.seq.ser.push(*byte_acc);
                        *byte_acc = 0;
                        *bit_idx = 0;
                    }
                } else {
                    if v { *byte_acc |= 1 << *bit_idx; }
                    *bit_idx += 1;
                    if *bit_idx == 8 {
                        self.seq.buffer.push(*byte_acc);
                        *byte_acc = 0;
                        *bit_idx = 0;
                    }
                }
                self.seq.count += 1;
                Ok(())
            }
            _ => unreachable!(),
        }
    }

    fn serialize_i8(self, v: i8) -> Result<()> { self.serialize_signed(v, 1) }
    fn serialize_i16(self, v: i16) -> Result<()> { self.serialize_signed(v, 2) }
    fn serialize_i32(self, v: i32) -> Result<()> { self.serialize_signed(v, 4) }
    fn serialize_i64(self, v: i64) -> Result<()> { self.serialize_signed(v, 8) }
    fn serialize_i128(self, v: i128) -> Result<()> { self.serialize_signed(v, 16) }

    fn serialize_u8(self, v: u8) -> Result<()> { self.serialize_unsigned(v, 1) }
    fn serialize_u16(self, v: u16) -> Result<()> { self.serialize_unsigned(v, 2) }
    fn serialize_u32(self, v: u32) -> Result<()> { self.serialize_unsigned(v, 4) }
    fn serialize_u64(self, v: u64) -> Result<()> { self.serialize_unsigned(v, 8) }
    fn serialize_u128(self, v: u128) -> Result<()> { self.serialize_unsigned(v, 16) }

    fn serialize_f32(self, v: f32) -> Result<()> { self.serialize_float_by(v.to_le_bytes().as_slice(), 2) }
    fn serialize_f64(self, v: f64) -> Result<()> { self.serialize_float_by(v.to_le_bytes().as_slice(), 3) }

    fn serialize_char(self, v: char) -> Result<()> {
        let mut buf = [0u8; 4];
        self.serialize_str(v.encode_utf8(&mut buf))
    }

    fn serialize_str(self, v: &str) -> Result<()> {
        match self.seq.mode {
            SeqMode::Unknown => {
                match self.seq.len {
                    Some(n) => {
                        self.seq.ser.write_typed_array_header_string(n);
                    }
                    None => {}
                }
                self.seq.mode = SeqMode::TypedString;
            }
            SeqMode::TypedString => {}
            _ => return Err(Error::Mismatch("heterogeneous sequence types (expected all strings)")),
        }

        // Strings in typed string array are written as SIZE | DATA (no per-element header)
        if self.seq.len.is_some() {
            write_size(v.len() as u64, &mut self.seq.ser.buf);
            self.seq.ser.extend_from_slice(v.as_bytes());
        } else {
            write_size(v.len() as u64, &mut self.seq.buffer);
            self.seq.buffer.extend_from_slice(v.as_bytes());
        }
        self.seq.count += 1;
        Ok(())
    }

    fn serialize_bytes(self, v: &[u8]) -> Result<()> {
        // Treat as unsigned typed array of u8 nested sequence-of-bytes: fallback to generic element
        self.seq.ensure_generic_header();
        // Write full VALUE for bytes as a typed array of u8
        let header = make_header(TYPE_TYPED_ARRAY, ARRAY_UNSIGNED, 0);
        if self.seq.len.is_some() {
            self.seq.ser.push(header);
            write_size(v.len() as u64, &mut self.seq.ser.buf);
            self.seq.ser.extend_from_slice(v);
        } else {
            self.seq.buffer.push(header);
            write_size(v.len() as u64, &mut self.seq.buffer);
            self.seq.buffer.extend_from_slice(v);
        }
        self.seq.count += 1;
        Ok(())
    }

    fn serialize_none(self) -> Result<()> { self.serialize_unit() }
    fn serialize_some<T: ?Sized + Serialize>(self, v: &T) -> Result<()> { v.serialize(self) }
    fn serialize_unit(self) -> Result<()> {
        self.seq.ensure_generic_header();
        if self.seq.len.is_some() {
            self.seq.ser.write_null();
        } else {
            self.seq.buffer.push(0);
        }
        self.seq.count += 1;
        Ok(())
    }

    fn serialize_unit_struct(self, _name: &'static str) -> Result<()> { self.serialize_unit() }

    fn serialize_unit_variant(self, _name: &'static str, variant_index: u32, _variant: &'static str) -> Result<()> {
        // As a generic element: extension type-tag with null value
        self.seq.ensure_generic_header();
        let h = make_extension_header(EXT_TYPE_TAG);
        if self.seq.len.is_some() {
            self.seq.ser.push(h);
            write_size(variant_index as u64, &mut self.seq.ser.buf);
            self.seq.ser.write_null();
        } else {
            self.seq.buffer.push(h);
            write_size(variant_index as u64, &mut self.seq.buffer);
            self.seq.buffer.push(0);
        }
        self.seq.count += 1;
        Ok(())
    }

    fn serialize_newtype_struct<T: ?Sized + Serialize>(self, _name: &'static str, value: &T) -> Result<()> { value.serialize(self) }

    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        variant_index: u32,
        _variant: &'static str,
        value: &T,
    ) -> Result<()> {
        self.seq.ensure_generic_header();
        let h = make_extension_header(EXT_TYPE_TAG);
        if self.seq.len.is_some() {
            self.seq.ser.push(h);
            write_size(variant_index as u64, &mut self.seq.ser.buf);
            value.serialize(&mut self.seq.ser)
        } else {
            self.seq.buffer.push(h);
            write_size(variant_index as u64, &mut self.seq.buffer);
            let mut tmp = Serializer { buf: Vec::new() };
            value.serialize(&mut tmp)?;
            self.seq.buffer.extend_from_slice(&tmp.buf);
            Ok(())
        }
    }

    fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq> { Err(Error::Unsupported("nested seq via seq element")) }
    fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple> { Err(Error::Unsupported("nested tuple via seq element")) }
    fn serialize_tuple_struct(self, _name: &'static str, _len: usize) -> Result<Self::SerializeTupleStruct> { Err(Error::Unsupported("nested tuple struct via seq element")) }
    fn serialize_tuple_variant(self, _name: &'static str, _variant_index: u32, _variant: &'static str, _len: usize) -> Result<Self::SerializeTupleVariant> { Err(Error::Unsupported("nested tuple variant via seq element")) }
    fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap> { Err(Error::Unsupported("map in typed sequence element")) }
    fn serialize_struct(self, _name: &'static str, _len: usize) -> Result<Self::SerializeStruct> { Err(Error::Unsupported("struct in typed sequence element")) }
    fn serialize_struct_variant(self, _name: &'static str, _variant_index: u32, _variant: &'static str, _len: usize) -> Result<Self::SerializeStructVariant> { Err(Error::Unsupported("struct variant in typed sequence element")) }

    fn serialize_f32(self, v: f32) -> Result<()> where Self: Sized { self.serialize_float_by(v.to_le_bytes().as_slice(), 2) }

    // helper impls
}

impl<'a, 'b> SeqElemSer<'a, 'b> {
    fn serialize_signed<T: Into<i128>>(self, v: T, bytes: usize) -> Result<()> {
        let byte_code = Serializer::code_for_bytes(bytes);
        match self.seq.mode {
            SeqMode::Unknown => {
                match self.seq.len {
                    Some(n) => {
                        self.seq.ser.write_typed_array_header_numeric(ARRAY_SIGNED, byte_code, n);
                    }
                    None => {}
                }
                self.seq.mode = SeqMode::TypedSigned(byte_code);
            }
            SeqMode::TypedSigned(bc) if bc == byte_code => {}
            _ => return Err(Error::Mismatch("heterogeneous sequence types (signed integer mix)")),
        }
        let x: i128 = v.into();
        let bytes_le = x.to_le_bytes();
        if self.seq.len.is_some() {
            self.seq.ser.extend_from_slice(&bytes_le[..bytes]);
        } else {
            self.seq.buffer.extend_from_slice(&bytes_le[..bytes]);
        }
        self.seq.count += 1;
        Ok(())
    }

    fn serialize_unsigned<T: Into<u128>>(self, v: T, bytes: usize) -> Result<()> {
        let byte_code = Serializer::code_for_bytes(bytes);
        match self.seq.mode {
            SeqMode::Unknown => {
                match self.seq.len {
                    Some(n) => self.seq.ser.write_typed_array_header_numeric(ARRAY_UNSIGNED, byte_code, n),
                    None => {}
                }
                self.seq.mode = SeqMode::TypedUnsigned(byte_code);
            }
            SeqMode::TypedUnsigned(bc) if bc == byte_code => {}
            _ => return Err(Error::Mismatch("heterogeneous sequence types (unsigned integer mix)")),
        }
        let x: u128 = v.into();
        let bytes_le = x.to_le_bytes();
        if self.seq.len.is_some() {
            self.seq.ser.extend_from_slice(&bytes_le[..bytes]);
        } else {
            self.seq.buffer.extend_from_slice(&bytes_le[..bytes]);
        }
        self.seq.count += 1;
        Ok(())
    }

    fn serialize_float_by(self, raw: &[u8], byte_code: u8) -> Result<()> {
        match self.seq.mode {
            SeqMode::Unknown => {
                match self.seq.len {
                    Some(n) => self.seq.ser.write_typed_array_header_numeric(ARRAY_FLOAT, byte_code, n),
                    None => {}
                }
                self.seq.mode = SeqMode::TypedFloat(byte_code);
            }
            SeqMode::TypedFloat(bc) if bc == byte_code => {}
            _ => return Err(Error::Mismatch("heterogeneous sequence types (float mix)")),
        }
        if self.seq.len.is_some() {
            self.seq.ser.extend_from_slice(raw);
        } else {
            self.seq.buffer.extend_from_slice(raw);
        }
        self.seq.count += 1;
        Ok(())
    }
}

impl<'a> ser::SerializeSeq for SeqSerializer<'a> {
    type Ok = ();
    type Error = Error;

    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<()> {
        // If we are already in generic mode, just write full value(s)
        if matches!(self.mode, SeqMode::Generic) {
            if self.len.is_some() {
                value.serialize(&mut *self.ser)
            } else {
                let mut tmp = Serializer { buf: Vec::new() };
                value.serialize(&mut tmp)?;
                self.buffer.extend_from_slice(&tmp.buf);
                self.count += 1;
                Ok(())
            }
        } else {
            let mut elem = SeqElemSer { seq: self };
            value.serialize(&mut elem)
        }
    }

    fn end(mut self) -> Result<()> {
        // Flush boolean bits
        if let SeqMode::TypedBool { byte_acc, bit_idx } = &mut self.mode {
            if self.len.is_some() {
                if *bit_idx != 0 { self.ser.push(*byte_acc); }
            } else {
                if *bit_idx != 0 { self.buffer.push(*byte_acc); }
            }
        }
        if self.len.is_none() {
            // Unknown length; now write header + size + buffer
            match self.mode {
                SeqMode::Unknown => {
                    // zero-length sequence; choose generic for compatibility
                    self.ser.write_generic_array_header(0);
                }
                SeqMode::Generic => {
                    self.ser.write_generic_array_header(self.count);
                    self.ser.extend_from_slice(&self.buffer);
                }
                SeqMode::TypedUnsigned(bc) => {
                    self.ser.write_typed_array_header_numeric(ARRAY_UNSIGNED, bc, self.count);
                    self.ser.extend_from_slice(&self.buffer);
                }
                SeqMode::TypedSigned(bc) => {
                    self.ser.write_typed_array_header_numeric(ARRAY_SIGNED, bc, self.count);
                    self.ser.extend_from_slice(&self.buffer);
                }
                SeqMode::TypedFloat(bc) => {
                    self.ser.write_typed_array_header_numeric(ARRAY_FLOAT, bc, self.count);
                    self.ser.extend_from_slice(&self.buffer);
                }
                SeqMode::TypedBool { .. } => {
                    self.ser.write_typed_array_header_bool(self.count);
                    self.ser.extend_from_slice(&self.buffer);
                }
                SeqMode::TypedString => {
                    self.ser.write_typed_array_header_string(self.count);
                    self.ser.extend_from_slice(&self.buffer);
                }
            }
        }
        Ok(())
    }
}

impl<'a> ser::SerializeTuple for SeqSerializer<'a> {
    type Ok = ();
    type Error = Error;
    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<()> { ser::SerializeSeq::serialize_element(self, value) }
    fn end(self) -> Result<()> { ser::SerializeSeq::end(self) }
}

impl<'a> ser::SerializeTupleStruct for SeqSerializer<'a> {
    type Ok = ();
    type Error = Error;
    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<()> { ser::SerializeSeq::serialize_element(self, value) }
    fn end(self) -> Result<()> { ser::SerializeSeq::end(self) }
}

// =============== Tuple Variant Serializer ===============

pub struct VariantSeqSerializer<'a> {
    ser: &'a mut Serializer,
    inner: SeqSerializer<'a>,
}

impl<'a> VariantSeqSerializer<'a> {
    fn new(ser: &'a mut Serializer, len: usize) -> Self {
        let mut inner = SeqSerializer::new(ser, Some(len));
        // For tuple variants, we encode as a generic array as the VALUE following the tag
        inner.ensure_generic_header();
        Self { ser: inner.ser, inner }
    }
}

impl<'a> ser::SerializeTupleVariant for VariantSeqSerializer<'a> {
    type Ok = ();
    type Error = Error;
    fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<()> { self.inner.serialize_element(value) }
    fn end(self) -> Result<()> { self.inner.end() }
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
    buffer: Vec<u8>,
}

impl<'a> MapSerializer<'a> {
    fn new(ser: &'a mut Serializer, len: Option<usize>) -> Self {
        Self { ser, len, mode: KeyMode::Unknown, count: 0, buffer: Vec::new() }
    }
}

impl<'a> ser::SerializeMap for MapSerializer<'a> {
    type Ok = ();
    type Error = Error;

    fn serialize_key<T: ?Sized + Serialize>(&mut self, key: &T) -> Result<()> {
        // We'll serialize key immediately; value will follow in serialize_value
        struct KeySer<'a, 'b> { map: &'b mut MapSerializer<'a> }
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
                        if let Some(n) = self.map.len { self.map.ser.push(TYPE_OBJECT | (KEY_STRING << 3)); write_size(n as u64, &mut self.map.ser.buf); }
                        self.map.mode = KeyMode::String;
                    }
                    KeyMode::String => {}
                    _ => return Err(Error::Mismatch("object keys must be homogeneous")),
                }
                if self.map.len.is_some() {
                    write_size(v.len() as u64, &mut self.map.ser.buf);
                    self.map.ser.extend_from_slice(v.as_bytes());
                } else {
                    write_size(v.len() as u64, &mut self.map.buffer);
                    self.map.buffer.extend_from_slice(v.as_bytes());
                }
                Ok(())
            }
            fn serialize_i8(self, v: i8) -> Result<()> { self.serialize_signed(v, 1) }
            fn serialize_i16(self, v: i16) -> Result<()> { self.serialize_signed(v, 2) }
            fn serialize_i32(self, v: i32) -> Result<()> { self.serialize_signed(v, 4) }
            fn serialize_i64(self, v: i64) -> Result<()> { self.serialize_signed(v, 8) }
            fn serialize_i128(self, v: i128) -> Result<()> { self.serialize_signed(v, 16) }
            fn serialize_u8(self, v: u8) -> Result<()> { self.serialize_unsigned(v, 1) }
            fn serialize_u16(self, v: u16) -> Result<()> { self.serialize_unsigned(v, 2) }
            fn serialize_u32(self, v: u32) -> Result<()> { self.serialize_unsigned(v, 4) }
            fn serialize_u64(self, v: u64) -> Result<()> { self.serialize_unsigned(v, 8) }
            fn serialize_u128(self, v: u128) -> Result<()> { self.serialize_unsigned(v, 16) }
            fn serialize_bool(self, _v: bool) -> Result<()> { Err(Error::InvalidType("boolean not allowed as object key")) }
            fn serialize_f32(self, _v: f32) -> Result<()> { Err(Error::InvalidType("float not allowed as object key")) }
            fn serialize_f64(self, _v: f64) -> Result<()> { Err(Error::InvalidType("float not allowed as object key")) }
            fn serialize_bytes(self, _v: &[u8]) -> Result<()> { Err(Error::InvalidType("bytes not allowed as object key")) }
            fn serialize_char(self, v: char) -> Result<()> { let mut buf=[0;4]; self.serialize_str(v.encode_utf8(&mut buf)) }
            fn serialize_none(self) -> Result<()> { Err(Error::InvalidType("none not allowed as object key")) }
            fn serialize_some<T: ?Sized + Serialize>(self, _value:&T) -> Result<()> { Err(Error::InvalidType("option not allowed as object key")) }
            fn serialize_unit(self) -> Result<()> { Err(Error::InvalidType("unit not allowed as object key")) }
            fn serialize_unit_struct(self, _name: &'static str) -> Result<()> { Err(Error::InvalidType("unit struct not allowed as object key")) }
            fn serialize_unit_variant(self, _name: &'static str,_idx:u32,_var:&'static str) -> Result<()> { Err(Error::InvalidType("enum not allowed as object key")) }
            fn serialize_newtype_struct<T: ?Sized + Serialize>(self,_:&'static str,_:&T)->Result<()> { Err(Error::InvalidType("newtype not allowed as object key")) }
            fn serialize_newtype_variant<T: ?Sized + Serialize>(self,_:&'static str,_:u32,_:&'static str,_:&T)->Result<()> { Err(Error::InvalidType("enum not allowed as object key")) }
            fn serialize_seq(self,_:Option<usize>)->Result<Self::SerializeSeq>{ Err(Error::InvalidType("seq not allowed as object key")) }
            fn serialize_tuple(self,_:usize)->Result<Self::SerializeTuple>{ Err(Error::InvalidType("tuple not allowed as object key")) }
            fn serialize_tuple_struct(self,_:&'static str,_:usize)->Result<Self::SerializeTupleStruct>{ Err(Error::InvalidType("tuple struct not allowed as object key")) }
            fn serialize_tuple_variant(self,_:&'static str,_:u32,_:&'static str,_:usize)->Result<Self::SerializeTupleVariant>{ Err(Error::InvalidType("tuple variant not allowed as object key")) }
            fn serialize_map(self,_:Option<usize>)->Result<Self::SerializeMap>{ Err(Error::InvalidType("map not allowed as object key")) }
            fn serialize_struct(self,_:&'static str,_:usize)->Result<Self::SerializeStruct>{ Err(Error::InvalidType("struct not allowed as object key")) }
            fn serialize_struct_variant(self,_:&'static str,_:u32,_:&'static str,_:usize)->Result<Self::SerializeStructVariant>{ Err(Error::InvalidType("struct variant not allowed as object key")) }
            fn is_human_readable(&self)->bool{ false }

            // helpers for int keys
            fn serialize_signed<T: Into<i128>>(self, v: T, bytes: usize) -> Result<()> {
                let code = Serializer::code_for_bytes(bytes);
                match self.map.mode {
                    KeyMode::Unknown => {
                        if let Some(n) = self.map.len { self.map.ser.push(make_header(TYPE_OBJECT, KEY_SIGNED, code)); write_size(n as u64, &mut self.map.ser.buf); }
                        self.map.mode = KeyMode::Signed(code);
                    }
                    KeyMode::Signed(c) if c == code => {}
                    _ => return Err(Error::Mismatch("object keys must be homogeneous signed integers of same width")),
                }
                let x: i128 = v.into();
                let raw = x.to_le_bytes();
                if self.map.len.is_some() { self.map.ser.extend_from_slice(&raw[..bytes]); } else { self.map.buffer.extend_from_slice(&raw[..bytes]); }
                Ok(())
            }
            fn serialize_unsigned<T: Into<u128>>(self, v: T, bytes: usize) -> Result<()> {
                let code = Serializer::code_for_bytes(bytes);
                match self.map.mode {
                    KeyMode::Unknown => {
                        if let Some(n) = self.map.len { self.map.ser.push(make_header(TYPE_OBJECT, KEY_UNSIGNED, code)); write_size(n as u64, &mut self.map.ser.buf); }
                        self.map.mode = KeyMode::Unsigned(code);
                    }
                    KeyMode::Unsigned(c) if c == code => {}
                    _ => return Err(Error::Mismatch("object keys must be homogeneous unsigned integers of same width")),
                }
                let x: u128 = v.into();
                let raw = x.to_le_bytes();
                if self.map.len.is_some() { self.map.ser.extend_from_slice(&raw[..bytes]); } else { self.map.buffer.extend_from_slice(&raw[..bytes]); }
                Ok(())
            }
        }

        let mut ks = KeySer { map: self };
        key.serialize(&mut ks)
    }

    fn serialize_value<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<()> {
        if self.len.is_some() {
            value.serialize(&mut *self.ser)?;
        } else {
            let mut tmp = Serializer { buf: Vec::new() };
            value.serialize(&mut tmp)?;
            self.buffer.extend_from_slice(&tmp.buf);
        }
        self.count += 1;
        Ok(())
    }

    fn end(mut self) -> Result<()> {
        if self.len.is_none() {
            // finalize unknown length map
            let header = match self.mode {
                KeyMode::Unknown => TYPE_OBJECT, // string object with zero fields by default
                KeyMode::String => TYPE_OBJECT | (KEY_STRING << 3),
                KeyMode::Signed(code) => make_header(TYPE_OBJECT, KEY_SIGNED, code),
                KeyMode::Unsigned(code) => make_header(TYPE_OBJECT, KEY_UNSIGNED, code),
            };
            self.ser.push(header);
            write_size(self.count as u64, &mut self.ser.buf);
            self.ser.extend_from_slice(&self.buffer);
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
    fn serialize_field<T: ?Sized + Serialize>(&mut self, key: &'static str, value: &T) -> Result<()> {
        write_size(key.len() as u64, &mut self.map.ser.buf);
        self.map.ser.extend_from_slice(key.as_bytes());
        value.serialize(&mut *self.map.ser)
    }
    fn end(self) -> Result<()> { Ok(()) }
}

pub struct VariantStructSerializer<'a> {
    ser: &'a mut Serializer,
    inner: StructSerializer<'a>,
}

impl<'a> VariantStructSerializer<'a> {
    fn new(ser: &'a mut Serializer, len: usize) -> Self {
        // After type-tag header and index, we write the struct as the VALUE object
        let inner = StructSerializer::new(ser, len);
        Self { ser, inner }
    }
}

impl<'a> ser::SerializeStructVariant for VariantStructSerializer<'a> {
    type Ok = ();
    type Error = Error;
    fn serialize_field<T: ?Sized + Serialize>(&mut self, key: &'static str, value: &T) -> Result<()> { self.inner.serialize_field(key, value) }
    fn end(self) -> Result<()> { self.inner.end() }
}

