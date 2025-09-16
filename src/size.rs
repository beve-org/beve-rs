//! Compressed unsigned integer encoding/decoding used for SIZE in BEVE.
//! Uses the first two LSBits to encode byte-length: 0->1, 1->2, 2->4, 3->8.
//! Remaining bits across bytes hold the value in little-endian order.

use crate::error::{Error, Result};

#[inline]
pub fn byte_count_to_bytes(count_code: u8) -> usize {
    match count_code {
        0 => 1,
        1 => 2,
        2 => 4,
        _ => 8,
    }
}

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

#[inline]
pub fn read_size(input: &[u8], pos: &mut usize) -> Result<u64> {
    if *pos >= input.len() { return Err(Error::Eof); }
    let b0 = input[*pos];
    *pos += 1;
    let code = b0 & 0b11;
    let mut value: u64 = (b0 >> 2) as u64; // lower 6 bits
    match code {
        0 => Ok(value), // 1 byte
        1 => {
            if *pos >= input.len() { return Err(Error::Eof); }
            value |= (input[*pos] as u64) << 6;
            *pos += 1;
            Ok(value)
        }
        2 => {
            if input.len() < *pos + 3 { return Err(Error::Eof); }
            value |= (input[*pos] as u64) << 6;
            value |= (input[*pos + 1] as u64) << 14;
            value |= (input[*pos + 2] as u64) << 22;
            *pos += 3;
            Ok(value)
        }
        _ => {
            if input.len() < *pos + 7 { return Err(Error::Eof); }
            // bytes 1..7 (7 bytes)
            for i in 0..7 {
                value |= (input[*pos + i] as u64) << (6 + 8 * i as u64);
            }
            *pos += 7;
            Ok(value)
        }
    }
}

