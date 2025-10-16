use crate::error::{Error, Result};
use crate::header::*;
use crate::size::{encode_size_to_array, read_size, write_size};
use half::{bf16, f16};
use std::char;
use std::str;

const MAX_RECURSION_DEPTH: usize = 256;

/// Convert JSON bytes into BEVE bytes without building an intermediate DOM.
pub fn json_slice_to_beve(json: &[u8]) -> Result<Vec<u8>> {
    let mut parser = JsonParser::new(json);
    parser.skip_ws();
    let mut builder = BeveBuilder::new();
    parser.parse_value(&mut builder, 0)?;
    parser.skip_ws();
    if !parser.is_eof() {
        return Err(Error::InvalidType("unexpected trailing JSON data"));
    }
    Ok(builder.into_vec())
}

/// Convert a JSON string into BEVE bytes without building an intermediate DOM.
pub fn json_str_to_beve(json: &str) -> Result<Vec<u8>> {
    json_slice_to_beve(json.as_bytes())
}

/// Convert BEVE bytes into compact JSON bytes.
pub fn beve_slice_to_json(beve_bytes: &[u8]) -> Result<Vec<u8>> {
    let mut reader = BeveReader::new(beve_bytes);
    let mut out = Vec::new();
    while !reader.is_finished() {
        write_value(&mut reader, &mut out, 0)?;
    }
    Ok(out)
}

/// Convert BEVE bytes into a JSON string.
pub fn beve_slice_to_json_string(beve_bytes: &[u8]) -> Result<String> {
    let bytes = beve_slice_to_json(beve_bytes)?;
    String::from_utf8(bytes).map_err(|_| Error::InvalidType("json output is not valid utf-8"))
}

struct BeveReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> BeveReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn is_finished(&self) -> bool {
        self.pos >= self.buf.len()
    }

    fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    fn read_byte(&mut self) -> Result<u8> {
        if self.pos >= self.buf.len() {
            return Err(Error::Eof);
        }
        let b = self.buf[self.pos];
        self.pos += 1;
        Ok(b)
    }

    fn read_exact(&mut self, len: usize) -> Result<&'a [u8]> {
        if self.remaining() < len {
            return Err(Error::Eof);
        }
        let start = self.pos;
        self.pos += len;
        Ok(&self.buf[start..start + len])
    }

    fn read_size(&mut self) -> Result<u64> {
        let mut idx = self.pos;
        let n = read_size(self.buf, &mut idx)?;
        self.pos = idx;
        Ok(n)
    }
}

struct BeveBuilder {
    buf: Vec<u8>,
}

impl BeveBuilder {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }

    fn into_vec(self) -> Vec<u8> {
        self.buf
    }

    fn push(&mut self, b: u8) {
        self.buf.push(b);
    }

    fn extend(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    fn write_null(&mut self) {
        self.push(0);
    }

    fn write_bool(&mut self, value: bool) {
        self.push(if value { 0x18 } else { 0x08 });
    }

    fn write_string(&mut self, value: &str) {
        self.push(TYPE_STRING);
        write_size(value.len() as u64, &mut self.buf);
        self.extend(value.as_bytes());
    }

    fn write_unsigned(&mut self, value: u128) {
        let (code, bytes) = match value {
            v if v <= u8::MAX as u128 => (0, 1),
            v if v <= u16::MAX as u128 => (1, 2),
            v if v <= u32::MAX as u128 => (2, 4),
            v if v <= u64::MAX as u128 => (3, 8),
            _ => (4, 16),
        };
        self.push(make_header(TYPE_NUMBER, NUM_UNSIGNED, code));
        let le = value.to_le_bytes();
        self.extend(&le[..bytes]);
    }

    fn write_signed(&mut self, value: i128) {
        let (code, bytes) = if value >= i8::MIN as i128 && value <= i8::MAX as i128 {
            (0, 1)
        } else if value >= i16::MIN as i128 && value <= i16::MAX as i128 {
            (1, 2)
        } else if value >= i32::MIN as i128 && value <= i32::MAX as i128 {
            (2, 4)
        } else if value >= i64::MIN as i128 && value <= i64::MAX as i128 {
            (3, 8)
        } else {
            (4, 16)
        };
        self.push(make_header(TYPE_NUMBER, NUM_SIGNED, code));
        let le = value.to_le_bytes();
        self.extend(&le[..bytes]);
    }

    fn write_float64(&mut self, value: f64) -> Result<()> {
        if !value.is_finite() {
            return Err(Error::Unsupported(
                "non-finite float not representable in JSON",
            ));
        }
        self.push(make_header(TYPE_NUMBER, NUM_FLOAT, 3));
        self.extend(&value.to_le_bytes());
        Ok(())
    }

    fn reserve_patch(&mut self) -> Patch {
        let pos = self.buf.len();
        self.buf.extend_from_slice(&[0u8; 8]);
        Patch { pos }
    }

    fn finalize_patch(&mut self, patch: Patch, count: usize) {
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

    fn begin_array(&mut self) -> Patch {
        self.push(TYPE_GENERIC_ARRAY);
        self.reserve_patch()
    }

    fn begin_object(&mut self) -> Patch {
        self.push(TYPE_OBJECT | (KEY_STRING << 3));
        self.reserve_patch()
    }
}

#[derive(Clone, Copy)]
struct Patch {
    pos: usize,
}

struct JsonParser<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(src: &'a [u8]) -> Self {
        Self { src, pos: 0 }
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.src.len()
    }

    fn skip_ws(&mut self) {
        while self.pos < self.src.len() {
            match self.src[self.pos] {
                b' ' | b'\t' | b'\n' | b'\r' => self.pos += 1,
                _ => break,
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        if self.pos >= self.src.len() {
            None
        } else {
            let b = self.src[self.pos];
            self.pos += 1;
            Some(b)
        }
    }

    fn expect_byte(&mut self, expected: u8) -> Result<()> {
        match self.advance() {
            Some(b) if b == expected => Ok(()),
            _ => Err(Error::InvalidType("unexpected JSON character")),
        }
    }

    fn parse_value(&mut self, builder: &mut BeveBuilder, depth: usize) -> Result<()> {
        if depth >= MAX_RECURSION_DEPTH {
            return Err(Error::Unsupported("json recursion depth exceeded"));
        }
        self.skip_ws();
        let ch = self.peek().ok_or(Error::Eof)?;
        match ch {
            b'n' => {
                self.expect_literal(b"null")?;
                builder.write_null();
            }
            b't' => {
                self.expect_literal(b"true")?;
                builder.write_bool(true);
            }
            b'f' => {
                self.expect_literal(b"false")?;
                builder.write_bool(false);
            }
            b'"' => {
                let s = self.parse_string()?;
                builder.write_string(&s);
            }
            b'[' => {
                self.advance();
                self.skip_ws();
                let patch = builder.begin_array();
                let mut count = 0usize;
                if self.consume_if(b']') {
                    builder.finalize_patch(patch, count);
                    return Ok(());
                }
                loop {
                    self.parse_value(builder, depth + 1)?;
                    count += 1;
                    self.skip_ws();
                    if self.consume_if(b',') {
                        self.skip_ws();
                        continue;
                    }
                    self.expect_byte(b']')?;
                    break;
                }
                builder.finalize_patch(patch, count);
            }
            b'{' => {
                self.advance();
                self.skip_ws();
                let patch = builder.begin_object();
                let mut count = 0usize;
                if self.consume_if(b'}') {
                    builder.finalize_patch(patch, count);
                    return Ok(());
                }
                loop {
                    self.skip_ws();
                    if self.peek() != Some(b'"') {
                        return Err(Error::InvalidType("json object keys must be strings"));
                    }
                    let key = self.parse_string()?;
                    write_size(key.len() as u64, &mut builder.buf);
                    builder.extend(key.as_bytes());
                    self.skip_ws();
                    self.expect_byte(b':')?;
                    self.skip_ws();
                    self.parse_value(builder, depth + 1)?;
                    count += 1;
                    self.skip_ws();
                    if self.consume_if(b',') {
                        self.skip_ws();
                        continue;
                    }
                    self.expect_byte(b'}')?;
                    break;
                }
                builder.finalize_patch(patch, count);
            }
            b'-' | b'0'..=b'9' => {
                let number = self.parse_number()?;
                match number {
                    ParsedNumber::Unsigned(v) => builder.write_unsigned(v),
                    ParsedNumber::Signed(v) => builder.write_signed(v),
                    ParsedNumber::Float(v) => builder.write_float64(v)?,
                }
            }
            _ => return Err(Error::InvalidType("unexpected JSON token")),
        }
        Ok(())
    }

    fn consume_if(&mut self, expected: u8) -> bool {
        if self.peek() == Some(expected) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect_literal(&mut self, literal: &[u8]) -> Result<()> {
        for &b in literal {
            if self.advance() != Some(b) {
                return Err(Error::InvalidType("unexpected JSON literal"));
            }
        }
        Ok(())
    }

    fn parse_string(&mut self) -> Result<String> {
        if self.advance() != Some(b'"') {
            return Err(Error::InvalidType("expected string"));
        }
        let mut result = String::new();
        let mut start = self.pos;
        while let Some(b) = self.advance() {
            if b == b'"' {
                // Fast path: copy contiguous slice.
                result.push_str(
                    str::from_utf8(&self.src[start..self.pos - 1])
                        .map_err(|_| Error::InvalidType("invalid utf-8 in JSON string"))?,
                );
                return Ok(result);
            } else if b == b'\\' {
                result.push_str(
                    str::from_utf8(&self.src[start..self.pos - 1])
                        .map_err(|_| Error::InvalidType("invalid utf-8 in JSON string"))?,
                );
                let escaped = self
                    .advance()
                    .ok_or(Error::InvalidType("unterminated escape in JSON string"))?;
                match escaped {
                    b'"' => result.push('"'),
                    b'\\' => result.push('\\'),
                    b'/' => result.push('/'),
                    b'b' => result.push('\u{0008}'),
                    b'f' => result.push('\u{000C}'),
                    b'n' => result.push('\n'),
                    b'r' => result.push('\r'),
                    b't' => result.push('\t'),
                    b'u' => {
                        let code = self.parse_hex_quad()?;
                        if (0xD800..=0xDBFF).contains(&code) {
                            // High surrogate; expect a following low surrogate.
                            if self.advance() != Some(b'\\') || self.advance() != Some(b'u') {
                                return Err(Error::InvalidType("invalid unicode surrogate pair"));
                            }
                            let low = self.parse_hex_quad()?;
                            if !(0xDC00..=0xDFFF).contains(&low) {
                                return Err(Error::InvalidType("invalid unicode surrogate pair"));
                            }
                            let composed =
                                0x10000 + (((code - 0xD800) as u32) << 10) + (low - 0xDC00) as u32;
                            if let Some(ch) = char::from_u32(composed) {
                                result.push(ch);
                            } else {
                                return Err(Error::InvalidType("invalid unicode codepoint"));
                            }
                        } else if let Some(ch) = char::from_u32(code as u32) {
                            result.push(ch);
                        } else {
                            return Err(Error::InvalidType("invalid unicode codepoint"));
                        }
                    }
                    _ => return Err(Error::InvalidType("invalid escape in JSON string")),
                }
                start = self.pos;
            } else if b < 0x20 {
                return Err(Error::InvalidType("control character in JSON string"));
            }
        }
        Err(Error::InvalidType("unterminated JSON string"))
    }

    fn parse_hex_quad(&mut self) -> Result<u16> {
        let mut value: u16 = 0;
        for _ in 0..4 {
            let b = self
                .advance()
                .ok_or(Error::InvalidType("unexpected end in \\u escape"))?;
            let digit = match b {
                b'0'..=b'9' => (b - b'0') as u16,
                b'a'..=b'f' => (10 + b - b'a') as u16,
                b'A'..=b'F' => (10 + b - b'A') as u16,
                _ => return Err(Error::InvalidType("invalid hex digit in \\u escape")),
            };
            value = (value << 4) | digit;
        }
        Ok(value)
    }

    fn parse_number(&mut self) -> Result<ParsedNumber> {
        let start = self.pos;
        let mut is_float = false;

        if self.peek() == Some(b'-') {
            self.pos += 1;
        }

        match self.advance() {
            Some(b'0') => {}
            Some(b'1'..=b'9') => {
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            _ => return Err(Error::InvalidType("invalid number")),
        }

        if let Some(b'.') = self.peek() {
            is_float = true;
            self.pos += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(Error::InvalidType("invalid number"));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }

        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            is_float = true;
            self.pos += 1;
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.pos += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(Error::InvalidType("invalid number exponent"));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }

        let slice = &self.src[start..self.pos];
        if !is_float {
            let negative = slice[0] == b'-';
            let digits = if negative { &slice[1..] } else { slice };
            if digits.len() > 1 && digits[0] == b'0' {
                return Err(Error::InvalidType("leading zeros not allowed"));
            }
            if negative {
                let s = str::from_utf8(slice).map_err(|_| Error::InvalidType("invalid number"))?;
                let value = s.parse::<i128>().map_err(|_| {
                    Error::Unsupported("integer outside representable range for BEVE")
                })?;
                Ok(ParsedNumber::Signed(value))
            } else {
                let s = str::from_utf8(slice).map_err(|_| Error::InvalidType("invalid number"))?;
                let value = s.parse::<u128>().map_err(|_| {
                    Error::Unsupported("integer outside representable range for BEVE")
                })?;
                Ok(ParsedNumber::Unsigned(value))
            }
        } else {
            let s = str::from_utf8(slice).map_err(|_| Error::InvalidType("invalid number"))?;
            let value = s
                .parse::<f64>()
                .map_err(|_| Error::InvalidType("invalid floating point literal"))?;
            if !value.is_finite() {
                return Err(Error::Unsupported(
                    "non-finite float not representable in JSON",
                ));
            }
            Ok(ParsedNumber::Float(value))
        }
    }
}

enum ParsedNumber {
    Unsigned(u128),
    Signed(i128),
    Float(f64),
}

fn write_value(reader: &mut BeveReader<'_>, out: &mut Vec<u8>, depth: usize) -> Result<()> {
    if depth >= MAX_RECURSION_DEPTH {
        return Err(Error::Unsupported("beve recursion depth exceeded"));
    }
    let header = reader.read_byte()?;
    let ty = parse_type(header);
    match ty {
        TYPE_NULL_BOOL => {
            if header == 0 {
                out.extend_from_slice(b"null");
            } else if is_bool(header) {
                if bool_value(header)? {
                    out.extend_from_slice(b"true");
                } else {
                    out.extend_from_slice(b"false");
                }
            } else {
                return Err(Error::InvalidHeader(header));
            }
        }
        TYPE_NUMBER => {
            write_number(reader, out, header)?;
        }
        TYPE_STRING => {
            let len = reader.read_size()? as usize;
            let bytes = reader.read_exact(len)?;
            let s = str::from_utf8(bytes)
                .map_err(|_| Error::InvalidType("invalid utf-8 in beve string"))?;
            write_json_string(out, s);
        }
        TYPE_OBJECT => {
            write_object(reader, out, header, depth)?;
        }
        TYPE_GENERIC_ARRAY => {
            let len = reader.read_size()? as usize;
            out.push(b'[');
            for i in 0..len {
                if i > 0 {
                    out.push(b',');
                }
                write_value(reader, out, depth + 1)?;
            }
            out.push(b']');
        }
        TYPE_TYPED_ARRAY => {
            write_typed_array(reader, out, header)?;
        }
        TYPE_EXTENSION => {
            write_extension(reader, out, header, depth)?;
        }
        _ => return Err(Error::InvalidHeader(header)),
    }
    Ok(())
}

fn write_number(reader: &mut BeveReader<'_>, out: &mut Vec<u8>, header: u8) -> Result<()> {
    let subtype = parse_subtype(header);
    let byte_code = parse_byte_count_code(header);
    match subtype {
        NUM_FLOAT => match byte_code {
            0 => {
                let raw = reader.read_exact(2)?;
                let bits = u16::from_le_bytes([raw[0], raw[1]]);
                let value = bf16::from_bits(bits).to_f32() as f64;
                out.extend_from_slice(value.to_string().as_bytes());
            }
            1 => {
                let raw = reader.read_exact(2)?;
                let bits = u16::from_le_bytes([raw[0], raw[1]]);
                let value = f16::from_bits(bits).to_f32() as f64;
                out.extend_from_slice(value.to_string().as_bytes());
            }
            2 => {
                let raw = reader.read_exact(4)?;
                let mut arr = [0u8; 4];
                arr.copy_from_slice(raw);
                let value = f32::from_le_bytes(arr) as f64;
                out.extend_from_slice(value.to_string().as_bytes());
            }
            3 => {
                let raw = reader.read_exact(8)?;
                let mut arr = [0u8; 8];
                arr.copy_from_slice(raw);
                let value = f64::from_le_bytes(arr);
                out.extend_from_slice(value.to_string().as_bytes());
            }
            _ => return Err(Error::Unsupported("float byte width not supported")),
        },
        NUM_SIGNED => {
            let bytes = match byte_code {
                0 => 1,
                1 => 2,
                2 => 4,
                3 => 8,
                4 => 16,
                _ => return Err(Error::Unsupported("signed width not supported")),
            };
            let raw = reader.read_exact(bytes)?;
            let mut buf = [0u8; 16];
            buf[..bytes].copy_from_slice(raw);
            if (raw[bytes - 1] & 0x80) != 0 && bytes < 16 {
                for b in &mut buf[bytes..] {
                    *b = 0xFF;
                }
            }
            let value = i128::from_le_bytes(buf);
            out.extend_from_slice(value.to_string().as_bytes());
        }
        NUM_UNSIGNED => {
            let bytes = match byte_code {
                0 => 1,
                1 => 2,
                2 => 4,
                3 => 8,
                4 => 16,
                _ => return Err(Error::Unsupported("unsigned width not supported")),
            };
            let raw = reader.read_exact(bytes)?;
            let mut buf = [0u8; 16];
            buf[..bytes].copy_from_slice(raw);
            let value = u128::from_le_bytes(buf);
            out.extend_from_slice(value.to_string().as_bytes());
        }
        _ => return Err(Error::InvalidHeader(header)),
    }
    Ok(())
}

fn write_object(
    reader: &mut BeveReader<'_>,
    out: &mut Vec<u8>,
    header: u8,
    depth: usize,
) -> Result<()> {
    let key_type = parse_subtype(header);
    if key_type != KEY_STRING {
        return Err(Error::Unsupported(
            "json conversion supports string keys only",
        ));
    }
    let len = reader.read_size()? as usize;
    out.push(b'{');
    for i in 0..len {
        if i > 0 {
            out.push(b',');
        }
        let key_len = reader.read_size()? as usize;
        let key_bytes = reader.read_exact(key_len)?;
        let key = str::from_utf8(key_bytes)
            .map_err(|_| Error::InvalidType("invalid utf-8 in object key"))?;
        write_json_string(out, key);
        out.push(b':');
        write_value(reader, out, depth + 1)?;
    }
    out.push(b'}');
    Ok(())
}

fn write_typed_array(reader: &mut BeveReader<'_>, out: &mut Vec<u8>, header: u8) -> Result<()> {
    let value_type = parse_subtype(header);
    let byte_code = parse_byte_count_code(header);
    out.push(b'[');
    match value_type {
        ARRAY_FLOAT => write_typed_float_array(reader, out, byte_code)?,
        ARRAY_SIGNED => write_typed_signed_array(reader, out, byte_code)?,
        ARRAY_UNSIGNED => write_typed_unsigned_array(reader, out, byte_code)?,
        ARRAY_BOOL_OR_STRING => {
            if byte_code == 0 {
                write_typed_bool_array(reader, out)?;
            } else {
                write_typed_string_array(reader, out)?;
            }
        }
        _ => return Err(Error::InvalidHeader(header)),
    }
    out.push(b']');
    Ok(())
}

fn write_typed_unsigned_array(
    reader: &mut BeveReader<'_>,
    out: &mut Vec<u8>,
    byte_code: u8,
) -> Result<()> {
    let bytes_per = match byte_code {
        0 => 1,
        1 => 2,
        2 => 4,
        3 => 8,
        4 => 16,
        _ => return Err(Error::Unsupported("unsigned width not supported")),
    };
    let len = reader.read_size()? as usize;
    for i in 0..len {
        if i > 0 {
            out.push(b',');
        }
        let raw = reader.read_exact(bytes_per)?;
        let mut buf = [0u8; 16];
        buf[..bytes_per].copy_from_slice(raw);
        let value = u128::from_le_bytes(buf);
        out.extend_from_slice(value.to_string().as_bytes());
    }
    Ok(())
}

fn write_typed_signed_array(
    reader: &mut BeveReader<'_>,
    out: &mut Vec<u8>,
    byte_code: u8,
) -> Result<()> {
    let bytes_per = match byte_code {
        0 => 1,
        1 => 2,
        2 => 4,
        3 => 8,
        4 => 16,
        _ => return Err(Error::Unsupported("signed width not supported")),
    };
    let len = reader.read_size()? as usize;
    for i in 0..len {
        if i > 0 {
            out.push(b',');
        }
        let raw = reader.read_exact(bytes_per)?;
        let mut buf = [0u8; 16];
        buf[..bytes_per].copy_from_slice(raw);
        if bytes_per < 16 && (raw[bytes_per - 1] & 0x80) != 0 {
            for b in &mut buf[bytes_per..] {
                *b = 0xFF;
            }
        }
        let value = i128::from_le_bytes(buf);
        out.extend_from_slice(value.to_string().as_bytes());
    }
    Ok(())
}

fn write_typed_float_array(
    reader: &mut BeveReader<'_>,
    out: &mut Vec<u8>,
    byte_code: u8,
) -> Result<()> {
    let len = reader.read_size()? as usize;
    for i in 0..len {
        if i > 0 {
            out.push(b',');
        }
        let value = match byte_code {
            0 => {
                let raw = reader.read_exact(2)?;
                let bits = u16::from_le_bytes([raw[0], raw[1]]);
                bf16::from_bits(bits).to_f32() as f64
            }
            1 => {
                let raw = reader.read_exact(2)?;
                let bits = u16::from_le_bytes([raw[0], raw[1]]);
                f16::from_bits(bits).to_f32() as f64
            }
            2 => {
                let raw = reader.read_exact(4)?;
                let mut arr = [0u8; 4];
                arr.copy_from_slice(raw);
                f32::from_le_bytes(arr) as f64
            }
            3 => {
                let raw = reader.read_exact(8)?;
                let mut arr = [0u8; 8];
                arr.copy_from_slice(raw);
                f64::from_le_bytes(arr)
            }
            _ => return Err(Error::Unsupported("float width not supported")),
        };
        out.extend_from_slice(value.to_string().as_bytes());
    }
    Ok(())
}

fn write_typed_bool_array(reader: &mut BeveReader<'_>, out: &mut Vec<u8>) -> Result<()> {
    let len = reader.read_size()? as usize;
    let packed = len.div_ceil(8);
    let bytes = reader.read_exact(packed)?;
    for idx in 0..len {
        if idx > 0 {
            out.push(b',');
        }
        let byte = bytes[idx / 8];
        let bit = idx % 8;
        if (byte & (1 << bit)) != 0 {
            out.extend_from_slice(b"true");
        } else {
            out.extend_from_slice(b"false");
        }
    }
    Ok(())
}

fn write_typed_string_array(reader: &mut BeveReader<'_>, out: &mut Vec<u8>) -> Result<()> {
    let len = reader.read_size()? as usize;
    for i in 0..len {
        if i > 0 {
            out.push(b',');
        }
        let slen = reader.read_size()? as usize;
        let bytes = reader.read_exact(slen)?;
        let s = str::from_utf8(bytes)
            .map_err(|_| Error::InvalidType("invalid utf-8 in typed string array"))?;
        write_json_string(out, s);
    }
    Ok(())
}

fn write_extension(
    reader: &mut BeveReader<'_>,
    out: &mut Vec<u8>,
    header: u8,
    depth: usize,
) -> Result<()> {
    let ext_id = parse_extension_id(header);
    match ext_id {
        EXT_DELIMITER => {
            out.push(b'\n');
        }
        EXT_TYPE_TAG => {
            // Skip variant tag and emit the associated value only.
            skip_value(reader, depth + 1)?;
            write_value(reader, out, depth + 1)?;
        }
        EXT_COMPLEX => write_complex_extension(reader, out)?,
        EXT_MATRICES => write_matrix_extension(reader, out, depth)?,
        _ => {
            return Err(Error::Unsupported(
                "extension not supported for JSON conversion",
            ))
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum ComplexNumeric {
    Float,
    Signed,
    Unsigned,
}

enum ComplexValue {
    Float(f64),
    Signed(i128),
    Unsigned(u128),
}

impl ComplexValue {
    fn write_json(&self, out: &mut Vec<u8>) {
        match self {
            ComplexValue::Float(v) => out.extend_from_slice(v.to_string().as_bytes()),
            ComplexValue::Signed(v) => out.extend_from_slice(v.to_string().as_bytes()),
            ComplexValue::Unsigned(v) => out.extend_from_slice(v.to_string().as_bytes()),
        }
    }
}

fn parse_complex_header_byte(header: u8) -> Result<(bool, ComplexNumeric, u8)> {
    let form = header & 0x07;
    let is_array = match form {
        0 => false,
        1 => true,
        _ => return Err(Error::Unsupported("complex extension form not supported")),
    };
    let numeric = match (header >> 3) & 0x03 {
        NUM_FLOAT => ComplexNumeric::Float,
        NUM_SIGNED => ComplexNumeric::Signed,
        NUM_UNSIGNED => ComplexNumeric::Unsigned,
        _ => {
            return Err(Error::Unsupported(
                "complex extension numeric type not supported",
            ))
        }
    };
    let byte_code = (header >> 5) & 0x07;
    Ok((is_array, numeric, byte_code))
}

fn complex_elem_len(kind: ComplexNumeric, byte_code: u8) -> Result<usize> {
    match kind {
        ComplexNumeric::Float => match byte_code {
            0 | 1 => Ok(2),
            2 => Ok(4),
            3 => Ok(8),
            _ => Err(Error::Unsupported("complex float width unsupported")),
        },
        ComplexNumeric::Signed | ComplexNumeric::Unsigned => match byte_code {
            0 => Ok(1),
            1 => Ok(2),
            2 => Ok(4),
            3 => Ok(8),
            4 => Ok(16),
            _ => Err(Error::Unsupported("complex width not supported")),
        },
    }
}

fn write_complex_extension(reader: &mut BeveReader<'_>, out: &mut Vec<u8>) -> Result<()> {
    let header = reader.read_byte()?;
    let (is_array, numeric, byte_code) = parse_complex_header_byte(header)?;
    if !is_array {
        let re = read_complex_component(reader, numeric, byte_code)?;
        let im = read_complex_component(reader, numeric, byte_code)?;
        out.push(b'[');
        re.write_json(out);
        out.push(b',');
        im.write_json(out);
        out.push(b']');
    } else {
        let len = reader.read_size()? as usize;
        out.push(b'[');
        for i in 0..len {
            if i > 0 {
                out.push(b',');
            }
            let re = read_complex_component(reader, numeric, byte_code)?;
            let im = read_complex_component(reader, numeric, byte_code)?;
            out.push(b'[');
            re.write_json(out);
            out.push(b',');
            im.write_json(out);
            out.push(b']');
        }
        out.push(b']');
    }
    Ok(())
}

fn read_complex_component(
    reader: &mut BeveReader<'_>,
    kind: ComplexNumeric,
    byte_code: u8,
) -> Result<ComplexValue> {
    match kind {
        ComplexNumeric::Float => {
            let value = match byte_code {
                0 => {
                    let raw = reader.read_exact(2)?;
                    let bits = u16::from_le_bytes([raw[0], raw[1]]);
                    bf16::from_bits(bits).to_f32() as f64
                }
                1 => {
                    let raw = reader.read_exact(2)?;
                    let bits = u16::from_le_bytes([raw[0], raw[1]]);
                    f16::from_bits(bits).to_f32() as f64
                }
                2 => {
                    let raw = reader.read_exact(4)?;
                    let mut arr = [0u8; 4];
                    arr.copy_from_slice(raw);
                    f32::from_le_bytes(arr) as f64
                }
                3 => {
                    let raw = reader.read_exact(8)?;
                    let mut arr = [0u8; 8];
                    arr.copy_from_slice(raw);
                    f64::from_le_bytes(arr)
                }
                _ => return Err(Error::Unsupported("complex float width unsupported")),
            };
            Ok(ComplexValue::Float(value))
        }
        ComplexNumeric::Signed => {
            let elem_bytes = complex_elem_len(kind, byte_code)?;
            let raw = reader.read_exact(elem_bytes)?;
            let mut buf = [0u8; 16];
            buf[..elem_bytes].copy_from_slice(raw);
            if elem_bytes < 16 && (raw[elem_bytes - 1] & 0x80) != 0 {
                for b in &mut buf[elem_bytes..] {
                    *b = 0xFF;
                }
            }
            let value = i128::from_le_bytes(buf);
            Ok(ComplexValue::Signed(value))
        }
        ComplexNumeric::Unsigned => {
            let elem_bytes = complex_elem_len(kind, byte_code)?;
            let raw = reader.read_exact(elem_bytes)?;
            let mut buf = [0u8; 16];
            buf[..elem_bytes].copy_from_slice(raw);
            let value = u128::from_le_bytes(buf);
            Ok(ComplexValue::Unsigned(value))
        }
    }
}

fn write_matrix_extension(
    reader: &mut BeveReader<'_>,
    out: &mut Vec<u8>,
    depth: usize,
) -> Result<()> {
    let header = reader.read_byte()?;
    let layout = if (header & 0x01) != 0 {
        "layout_left"
    } else {
        "layout_right"
    };
    out.extend_from_slice(b"{\"layout\":");
    write_json_string(out, layout);
    out.extend_from_slice(b",\"extents\":");
    write_value(reader, out, depth + 1)?;
    out.extend_from_slice(b",\"value\":");
    write_value(reader, out, depth + 1)?;
    out.push(b'}');
    Ok(())
}

fn skip_value(reader: &mut BeveReader<'_>, depth: usize) -> Result<()> {
    if depth >= MAX_RECURSION_DEPTH {
        return Err(Error::Unsupported("beve recursion depth exceeded"));
    }
    let header = reader.read_byte()?;
    let ty = parse_type(header);
    match ty {
        TYPE_NULL_BOOL => {}
        TYPE_NUMBER => {
            let byte_code = parse_byte_count_code(header);
            let len = match parse_subtype(header) {
                NUM_FLOAT => match byte_code {
                    0 | 1 => 2,
                    2 => 4,
                    3 => 8,
                    _ => return Err(Error::Unsupported("float width not supported")),
                },
                NUM_SIGNED | NUM_UNSIGNED => match byte_code {
                    0 => 1,
                    1 => 2,
                    2 => 4,
                    3 => 8,
                    4 => 16,
                    _ => return Err(Error::Unsupported("integer width not supported")),
                },
                _ => return Err(Error::InvalidHeader(header)),
            };
            reader.read_exact(len)?;
        }
        TYPE_STRING => {
            let len = reader.read_size()? as usize;
            reader.read_exact(len)?;
        }
        TYPE_OBJECT => {
            let len = reader.read_size()? as usize;
            for _ in 0..len {
                let key_len = reader.read_size()? as usize;
                reader.read_exact(key_len)?;
                skip_value(reader, depth + 1)?;
            }
        }
        TYPE_GENERIC_ARRAY => {
            let len = reader.read_size()? as usize;
            for _ in 0..len {
                skip_value(reader, depth + 1)?;
            }
        }
        TYPE_TYPED_ARRAY => {
            let value_type = parse_subtype(header);
            let byte_code = parse_byte_count_code(header);
            match value_type {
                ARRAY_FLOAT => {
                    let len = reader.read_size()? as usize;
                    let elem_bytes: usize = match byte_code {
                        0 | 1 => 2usize,
                        2 => 4usize,
                        3 => 8usize,
                        _ => return Err(Error::Unsupported("float width not supported")),
                    };
                    let total = len.checked_mul(elem_bytes).ok_or(Error::InvalidSize)?;
                    reader.read_exact(total)?;
                }
                ARRAY_SIGNED | ARRAY_UNSIGNED => {
                    let len = reader.read_size()? as usize;
                    let elem_bytes: usize = match byte_code {
                        0 => 1usize,
                        1 => 2usize,
                        2 => 4usize,
                        3 => 8usize,
                        4 => 16usize,
                        _ => return Err(Error::Unsupported("integer width not supported")),
                    };
                    let total = len.checked_mul(elem_bytes).ok_or(Error::InvalidSize)?;
                    reader.read_exact(total)?;
                }
                ARRAY_BOOL_OR_STRING => {
                    if byte_code == 0 {
                        let len = reader.read_size()? as usize;
                        let packed = len.div_ceil(8);
                        reader.read_exact(packed)?;
                    } else {
                        let len = reader.read_size()? as usize;
                        for _ in 0..len {
                            let slen = reader.read_size()? as usize;
                            reader.read_exact(slen)?;
                        }
                    }
                }
                _ => return Err(Error::InvalidHeader(header)),
            }
        }
        TYPE_EXTENSION => {
            let ext_id = parse_extension_id(header);
            match ext_id {
                EXT_DELIMITER => {}
                EXT_TYPE_TAG => {
                    skip_value(reader, depth + 1)?;
                    skip_value(reader, depth + 1)?;
                }
                EXT_COMPLEX => {
                    let ch = reader.read_byte()?;
                    let (is_array, numeric, byte_code) = parse_complex_header_byte(ch)?;
                    let elem_bytes = complex_elem_len(numeric, byte_code)?;
                    if !is_array {
                        let total = elem_bytes.checked_mul(2usize).ok_or(Error::InvalidSize)?;
                        reader.read_exact(total)?;
                    } else {
                        let len = reader.read_size()? as usize;
                        let total = elem_bytes
                            .checked_mul(2usize)
                            .and_then(|v| v.checked_mul(len))
                            .ok_or(Error::InvalidSize)?;
                        reader.read_exact(total)?;
                    }
                }
                EXT_MATRICES => {
                    reader.read_byte()?; // matrix header
                    skip_value(reader, depth + 1)?;
                    skip_value(reader, depth + 1)?;
                }
                _ => {
                    return Err(Error::Unsupported(
                        "extension not supported for JSON conversion",
                    ))
                }
            }
        }
        _ => return Err(Error::InvalidHeader(header)),
    }
    Ok(())
}

fn write_json_string(out: &mut Vec<u8>, value: &str) {
    out.push(b'"');
    let bytes = value.as_bytes();
    let mut start = 0;
    for (idx, &b) in bytes.iter().enumerate() {
        let escape = match b {
            b'"' => Some(b'"'),
            b'\\' => Some(b'\\'),
            b'\n' => Some(b'n'),
            b'\r' => Some(b'r'),
            b'\t' => Some(b't'),
            b'\x08' => Some(b'b'),
            b'\x0C' => Some(b'f'),
            _ if b < 0x20 => None,
            _ => continue,
        };
        if let Some(esc) = escape {
            if start < idx {
                out.extend_from_slice(&bytes[start..idx]);
            }
            out.extend_from_slice(&[b'\\', esc]);
            start = idx + 1;
        } else if b < 0x20 {
            if start < idx {
                out.extend_from_slice(&bytes[start..idx]);
            }
            out.extend_from_slice(b"\\u00");
            let hi = b >> 4;
            let lo = b & 0x0F;
            out.push(nibble_to_hex(hi));
            out.push(nibble_to_hex(lo));
            start = idx + 1;
        }
    }
    if start < bytes.len() {
        out.extend_from_slice(&bytes[start..]);
    }
    out.push(b'"');
}

fn nibble_to_hex(n: u8) -> u8 {
    match n {
        0..=9 => b'0' + n,
        10..=15 => b'a' + (n - 10),
        _ => b'0',
    }
}
