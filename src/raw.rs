use crate::error::{Error, Result};
use crate::header::*;
use crate::size::read_size;
use half::{bf16, f16};
use simdutf8::basic::from_utf8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TypedArrayClass {
    Float,
    Signed,
    Unsigned,
    Bool,
    String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ObjectHeader {
    pub(crate) key_type: u8,
    pub(crate) byte_code: u8,
    pub(crate) len: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TypedArrayHeader {
    pub(crate) class: TypedArrayClass,
    pub(crate) byte_code: u8,
    pub(crate) len: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ComplexHeader {
    pub(crate) is_array: bool,
    pub(crate) byte_code: u8,
}

pub(crate) struct Reader<'a> {
    input: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(input: &'a [u8]) -> Self {
        Self { input, pos: 0 }
    }

    pub(crate) fn is_finished(&self) -> bool {
        self.pos >= self.input.len()
    }

    pub(crate) fn read_header(&mut self) -> Result<u8> {
        self.read_byte()
    }

    pub(crate) fn read_size(&mut self) -> Result<usize> {
        read_size(self.input, &mut self.pos)
            .map_err(|_| Error::InvalidSize)?
            .try_into()
            .map_err(|_| Error::InvalidSize)
    }

    pub(crate) fn read_string(&mut self) -> Result<&'a str> {
        let len = self.read_size()?;
        let bytes = self.read_exact(len)?;
        from_utf8(bytes).map_err(|_| Error::InvalidType("invalid utf-8"))
    }

    pub(crate) fn read_signed(&mut self, byte_code: u8) -> Result<i128> {
        let nbytes = byte_count_to_bytes(byte_code)?;
        let bytes = self.read_exact(nbytes)?;
        let mut buf = [0u8; 16];
        buf[..nbytes].copy_from_slice(bytes);
        if nbytes < 16 && (bytes[nbytes - 1] & 0x80) != 0 {
            for b in &mut buf[nbytes..] {
                *b = 0xff;
            }
        }
        Ok(i128::from_le_bytes(buf))
    }

    pub(crate) fn read_unsigned(&mut self, byte_code: u8) -> Result<u128> {
        let nbytes = byte_count_to_bytes(byte_code)?;
        let bytes = self.read_exact(nbytes)?;
        let mut buf = [0u8; 16];
        buf[..nbytes].copy_from_slice(bytes);
        Ok(u128::from_le_bytes(buf))
    }

    pub(crate) fn read_f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.read_exact(4)?.try_into().unwrap()))
    }

    pub(crate) fn read_f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.read_exact(8)?.try_into().unwrap()))
    }

    pub(crate) fn read_f16_as_f32(&mut self) -> Result<f32> {
        let bytes = self.read_exact(2)?;
        Ok(f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())
    }

    pub(crate) fn read_bf16_as_f32(&mut self) -> Result<f32> {
        let bytes = self.read_exact(2)?;
        Ok(bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())
    }

    pub(crate) fn read_object_header(&mut self, header: u8) -> Result<ObjectHeader> {
        if parse_type(header) != TYPE_OBJECT {
            return Err(Error::InvalidType("expected object"));
        }
        Ok(ObjectHeader {
            key_type: parse_subtype(header),
            byte_code: parse_byte_count_code(header),
            len: self.read_size()?,
        })
    }

    pub(crate) fn read_typed_array_header(&mut self, header: u8) -> Result<TypedArrayHeader> {
        if parse_type(header) != TYPE_TYPED_ARRAY {
            return Err(Error::InvalidType("expected typed array"));
        }
        let class = match parse_subtype(header) {
            ARRAY_FLOAT => TypedArrayClass::Float,
            ARRAY_SIGNED => TypedArrayClass::Signed,
            ARRAY_UNSIGNED => TypedArrayClass::Unsigned,
            ARRAY_BOOL_OR_STRING => {
                if (header & 0b0010_0000) == 0 {
                    TypedArrayClass::Bool
                } else {
                    TypedArrayClass::String
                }
            }
            _ => return Err(Error::InvalidHeader(header)),
        };
        Ok(TypedArrayHeader {
            class,
            byte_code: parse_byte_count_code(header),
            len: self.read_size()?,
        })
    }

    pub(crate) fn read_complex_header(&mut self) -> Result<ComplexHeader> {
        let header = self.read_byte()?;
        let class = (header >> 3) & 0b11;
        if class != NUM_FLOAT {
            return Err(Error::Unsupported(
                "only floating-point complex values are supported",
            ));
        }
        Ok(ComplexHeader {
            is_array: (header & 0x01) != 0,
            byte_code: (header >> 5) & 0b111,
        })
    }

    pub(crate) fn read_matrix_extents(&mut self) -> Result<Vec<usize>> {
        let header = self.read_header()?;
        let info = self.read_typed_array_header(header)?;
        let len = info.len;
        let mut extents = Vec::with_capacity(len);
        match info.class {
            TypedArrayClass::Unsigned => {
                for _ in 0..len {
                    let value = self.read_unsigned(info.byte_code)?;
                    extents.push(value.try_into().map_err(|_| Error::InvalidSize)?);
                }
            }
            TypedArrayClass::Signed => {
                for _ in 0..len {
                    let value = self.read_signed(info.byte_code)?;
                    if value < 0 {
                        return Err(Error::InvalidSize);
                    }
                    extents.push(value.try_into().map_err(|_| Error::InvalidSize)?);
                }
            }
            _ => {
                return Err(Error::InvalidType(
                    "matrix extents must be integer typed arrays",
                ))
            }
        }
        Ok(extents)
    }

    pub(crate) fn read_exact(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.remaining() < n {
            return Err(Error::Eof);
        }
        let start = self.pos;
        self.pos += n;
        Ok(&self.input[start..start + n])
    }

    pub(crate) fn remaining(&self) -> usize {
        self.input.len().saturating_sub(self.pos)
    }

    fn peek_byte(&self) -> Result<u8> {
        self.input.get(self.pos).copied().ok_or(Error::Eof)
    }

    fn read_byte(&mut self) -> Result<u8> {
        let byte = self.peek_byte()?;
        self.pos += 1;
        Ok(byte)
    }
}

pub(crate) fn byte_count_to_bytes(code: u8) -> Result<usize> {
    match code {
        0 => Ok(1),
        1 => Ok(2),
        2 => Ok(4),
        3 => Ok(8),
        4 => Ok(16),
        _ => Err(Error::Unsupported("byte width > 16 not supported")),
    }
}
