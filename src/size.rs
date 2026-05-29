//! Compressed unsigned integer encoding/decoding used for SIZE in BEVE.
//! Uses the first two LSBits to encode byte-length: 0->1, 1->2, 2->4, 3->8.
//! Remaining bits across bytes hold the value in little-endian order.

use crate::error::{Error, Result};

#[inline]
pub fn write_size(mut n: u64, out: &mut Vec<u8>) {
    if n < (1 << 6) {
        // 1 byte: 2-bit code 0, 6-bit value
        let b = (n as u8) << 2;
        out.push(b);
        return;
    }
    if n < (1 << 14) {
        // 2 bytes: 2-bit code 1
        let mut b0 = ((n & 0x3f) as u8) << 2;
        b0 |= 0b01;
        out.push(b0);
        n >>= 6;
        out.push(n as u8);
        return;
    }
    if n < (1 << 30) {
        // 4 bytes: 2-bit code 2
        let mut b0 = ((n & 0x3f) as u8) << 2;
        b0 |= 0b10;
        out.push(b0);
        n >>= 6;
        out.push(n as u8);
        out.push((n >> 8) as u8);
        out.push((n >> 16) as u8);
        return;
    }
    // 8 bytes: 2-bit code 3; up to 62 bits of value
    let mut b0 = ((n & 0x3f) as u8) << 2;
    b0 |= 0b11;
    out.push(b0);
    n >>= 6;
    for i in 0..7 {
        out.push((n >> (i * 8)) as u8);
    }
}

/// Number of bytes the SIZE codec emits for `n`.
///
/// This is the width [`write_size`] / [`encode_size_to_array`] would produce for
/// `n`, computed without writing any bytes. It shares the same 1/2/4/8-byte
/// thresholds so an analytic size (e.g. [`crate::typed_slice_size`]) cannot drift
/// from what the encoders actually emit.
#[inline]
pub fn size_encoded_len(n: u64) -> usize {
    if n < (1 << 6) {
        1
    } else if n < (1 << 14) {
        2
    } else if n < (1 << 30) {
        4
    } else {
        8
    }
}

/// Write a SIZE value directly to a writer (streaming counterpart of [`write_size`]).
#[inline]
pub(crate) fn write_size_to_writer<W: std::io::Write>(w: &mut W, n: u64) -> std::io::Result<()> {
    let mut buf = [0u8; 8];
    let used = encode_size_to_array(n, &mut buf);
    w.write_all(&buf[..used])
}

/// Encode SIZE into the provided 8-byte buffer and return the number of bytes used.
#[inline]
pub fn encode_size_to_array(mut n: u64, out: &mut [u8; 8]) -> usize {
    if n < (1 << 6) {
        out[0] = (n as u8) << 2;
        1
    } else if n < (1 << 14) {
        out[0] = (((n & 0x3f) as u8) << 2) | 0b01;
        n >>= 6;
        out[1] = n as u8;
        2
    } else if n < (1 << 30) {
        out[0] = (((n & 0x3f) as u8) << 2) | 0b10;
        n >>= 6;
        out[1] = n as u8;
        out[2] = (n >> 8) as u8;
        out[3] = (n >> 16) as u8;
        4
    } else {
        out[0] = (((n & 0x3f) as u8) << 2) | 0b11;
        n >>= 6;
        for i in 0..7 {
            out[i + 1] = (n >> (i * 8)) as u8;
        }
        8
    }
}

#[inline]
pub fn read_size(input: &[u8], pos: &mut usize) -> Result<u64> {
    if *pos >= input.len() {
        return Err(Error::Eof);
    }
    let b0 = input[*pos];
    *pos += 1;
    let code = b0 & 0b11;
    let mut value: u64 = (b0 >> 2) as u64; // lower 6 bits
    match code {
        0 => Ok(value), // 1 byte
        1 => {
            if *pos >= input.len() {
                return Err(Error::Eof);
            }
            value |= (input[*pos] as u64) << 6;
            *pos += 1;
            Ok(value)
        }
        2 => {
            if input.len() < *pos + 3 {
                return Err(Error::Eof);
            }
            value |= (input[*pos] as u64) << 6;
            value |= (input[*pos + 1] as u64) << 14;
            value |= (input[*pos + 2] as u64) << 22;
            *pos += 3;
            Ok(value)
        }
        _ => {
            if input.len() < *pos + 7 {
                return Err(Error::Eof);
            }
            // bytes 1..7 (7 bytes)
            for i in 0..7 {
                value |= (input[*pos + i] as u64) << (6 + 8 * i as u64);
            }
            *pos += 7;
            Ok(value)
        }
    }
}

/// Read a SIZE value from a reader (streaming variant of [`read_size`]).
#[inline]
pub(crate) fn read_size_from_reader<R: std::io::Read>(reader: &mut R) -> Result<u64> {
    let mut b0_buf = [0u8; 1];
    reader.read_exact(&mut b0_buf).map_err(|_| Error::Eof)?;
    read_size_from_reader_with_first_byte(reader, b0_buf[0])
}

/// Decode a SIZE value given the already-read first byte and a reader for the remaining bytes.
#[inline]
pub(crate) fn read_size_from_reader_with_first_byte<R: std::io::Read>(
    reader: &mut R,
    b0: u8,
) -> Result<u64> {
    let code = b0 & 0b11;
    let mut value: u64 = (b0 >> 2) as u64;
    match code {
        0 => Ok(value),
        1 => {
            let mut buf = [0u8; 1];
            reader.read_exact(&mut buf).map_err(|_| Error::Eof)?;
            value |= (buf[0] as u64) << 6;
            Ok(value)
        }
        2 => {
            let mut buf = [0u8; 3];
            reader.read_exact(&mut buf).map_err(|_| Error::Eof)?;
            value |= (buf[0] as u64) << 6;
            value |= (buf[1] as u64) << 14;
            value |= (buf[2] as u64) << 22;
            Ok(value)
        }
        _ => {
            let mut buf = [0u8; 7];
            reader.read_exact(&mut buf).map_err(|_| Error::Eof)?;
            for (i, &b) in buf.iter().enumerate() {
                value |= (b as u64) << (6 + 8 * i as u64);
            }
            Ok(value)
        }
    }
}
