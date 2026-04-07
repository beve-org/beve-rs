use crate::ext::Complex;
use crate::header::*;
use crate::size::write_size;
use core::ptr;
use half::{bf16, f16};

/// Write a typed array header and length for numeric arrays.
#[inline]
fn write_typed_array_header_numeric(out: &mut Vec<u8>, class: u8, byte_code: u8, len: usize) {
    let header = ((byte_code & 0b111) << 5) | ((class & 0b11) << 3) | (TYPE_TYPED_ARRAY & 0b111);
    out.push(header);
    write_size(len as u64, out);
}

/// Write a typed array header for boolean arrays.
#[inline]
fn write_typed_array_header_bool(out: &mut Vec<u8>, len: usize) {
    let header = ((ARRAY_BOOL_OR_STRING & 0b11) << 3) | (TYPE_TYPED_ARRAY & 0b111);
    out.push(header);
    write_size(len as u64, out);
}

/// Write a typed array header for string arrays.
#[inline]
fn write_typed_array_header_string(out: &mut Vec<u8>, len: usize) {
    // For boolean/string category, we use byte_code=1 to indicate string arrays
    let header = ((1u8) << 5) | ((ARRAY_BOOL_OR_STRING & 0b11) << 3) | (TYPE_TYPED_ARRAY & 0b111);
    out.push(header);
    write_size(len as u64, out);
}

/// A trait implemented for scalar numeric types supported by BEVE typed arrays.
pub trait BeveTypedSlice: Sized {
    /// Typed array class (ARRAY_SIGNED/ARRAY_UNSIGNED/ARRAY_FLOAT)
    const CLASS: u8;
    /// Byte code (0:1B, 1:2B, 2:4B, 3:8B, 4:16B)
    const BYTE_CODE: u8;
    /// Number of bytes per element
    const ELEM_SIZE: usize;
    /// Encode one element to little-endian bytes, appending to `out`.
    fn write_one_le(v: &Self, out: &mut Vec<u8>);
}

macro_rules! impl_beve_typed_int {
    ($t:ty, $class:expr, $code:expr) => {
        impl BeveTypedSlice for $t {
            const CLASS: u8 = $class;
            const BYTE_CODE: u8 = $code;
            const ELEM_SIZE: usize = core::mem::size_of::<$t>();
            #[inline]
            fn write_one_le(v: &Self, out: &mut Vec<u8>) {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
    };
}

impl_beve_typed_int!(i8, ARRAY_SIGNED, 0);
impl_beve_typed_int!(i16, ARRAY_SIGNED, 1);
impl_beve_typed_int!(i32, ARRAY_SIGNED, 2);
impl_beve_typed_int!(i64, ARRAY_SIGNED, 3);
impl_beve_typed_int!(i128, ARRAY_SIGNED, 4);

impl_beve_typed_int!(u8, ARRAY_UNSIGNED, 0);
impl_beve_typed_int!(u16, ARRAY_UNSIGNED, 1);
impl_beve_typed_int!(u32, ARRAY_UNSIGNED, 2);
impl_beve_typed_int!(u64, ARRAY_UNSIGNED, 3);
impl_beve_typed_int!(u128, ARRAY_UNSIGNED, 4);

impl BeveTypedSlice for f32 {
    const CLASS: u8 = ARRAY_FLOAT;
    const BYTE_CODE: u8 = 2; // 4 bytes
    const ELEM_SIZE: usize = core::mem::size_of::<f32>();
    #[inline]
    fn write_one_le(v: &Self, out: &mut Vec<u8>) {
        out.extend_from_slice(&v.to_le_bytes());
    }
}
impl BeveTypedSlice for f64 {
    const CLASS: u8 = ARRAY_FLOAT;
    const BYTE_CODE: u8 = 3; // 8 bytes
    const ELEM_SIZE: usize = core::mem::size_of::<f64>();
    #[inline]
    fn write_one_le(v: &Self, out: &mut Vec<u8>) {
        out.extend_from_slice(&v.to_le_bytes());
    }
}
impl BeveTypedSlice for bf16 {
    const CLASS: u8 = ARRAY_FLOAT;
    const BYTE_CODE: u8 = 0; // Special-case brain float (2 bytes)
    const ELEM_SIZE: usize = core::mem::size_of::<bf16>();
    #[inline]
    fn write_one_le(v: &Self, out: &mut Vec<u8>) {
        out.extend_from_slice(&v.to_le_bytes());
    }
}
impl BeveTypedSlice for f16 {
    const CLASS: u8 = ARRAY_FLOAT;
    const BYTE_CODE: u8 = 1; // 2 bytes
    const ELEM_SIZE: usize = core::mem::size_of::<f16>();
    #[inline]
    fn write_one_le(v: &Self, out: &mut Vec<u8>) {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

/// Write a typed numeric array directly to `out` without serde.
pub fn write_typed_slice<T: BeveTypedSlice>(out: &mut Vec<u8>, slice: &[T]) {
    let payload = core::mem::size_of_val(slice);
    // Reserve for header byte, size prefix (<=8 bytes), and payload.
    out.reserve(1 + 8 + payload);
    write_typed_array_header_numeric(out, T::CLASS, T::BYTE_CODE, slice.len());
    if slice.is_empty() {
        return;
    }
    #[cfg(target_endian = "little")]
    {
        let start = out.len();
        unsafe {
            let dst = out.as_mut_ptr().add(start);
            ptr::copy_nonoverlapping(slice.as_ptr() as *const u8, dst, payload);
            out.set_len(start + payload);
        }
    }
    #[cfg(not(target_endian = "little"))]
    {
        for v in slice {
            T::write_one_le(v, out);
        }
    }
}

/// Encode a typed numeric slice to a new Vec<u8> (BEVE typed array).
pub fn to_vec_typed_slice<T: BeveTypedSlice>(slice: &[T]) -> Vec<u8> {
    let payload = slice.len() * T::ELEM_SIZE;
    let mut out = Vec::with_capacity(1 + 8 + payload);
    write_typed_slice(&mut out, slice);
    out
}

/// Write a boolean typed array (bit-packed) to `out`.
pub fn write_bool_slice(out: &mut Vec<u8>, slice: &[bool]) {
    write_typed_array_header_bool(out, slice.len());
    // Pack LSB-first per BEVE spec: bit0 holds the first element
    let mut acc: u8 = 0;
    let mut idx: u8 = 0; // counts elements within the current byte
    for &b in slice {
        if b {
            acc |= 1 << idx;
        }
        idx += 1;
        if idx == 8 {
            out.push(acc);
            acc = 0;
            idx = 0;
        }
    }
    if idx != 0 {
        out.push(acc);
    }
}

/// Encode a boolean slice to a new Vec<u8> (BEVE typed boolean array).
pub fn to_vec_bool_slice(slice: &[bool]) -> Vec<u8> {
    let mut out = Vec::new();
    write_bool_slice(&mut out, slice);
    out
}

/// Write a typed string array to `out` (each element as SIZE | UTF-8 DATA, no per-element header).
pub fn write_str_slice(out: &mut Vec<u8>, slice: &[&str]) {
    write_typed_array_header_string(out, slice.len());
    for s in slice {
        write_size(s.len() as u64, out);
        out.extend_from_slice(s.as_bytes());
    }
}

/// Encode `&[&str]` to a new Vec<u8> (BEVE typed string array).
pub fn to_vec_str_slice(slice: &[&str]) -> Vec<u8> {
    let mut out = Vec::new();
    write_str_slice(&mut out, slice);
    out
}

/// Write a typed string array from `&[String]` to `out`.
pub fn write_string_slice(out: &mut Vec<u8>, slice: &[String]) {
    write_typed_array_header_string(out, slice.len());
    for s in slice {
        write_size(s.len() as u64, out);
        out.extend_from_slice(s.as_bytes());
    }
}

/// Encode `&[String]` to a new Vec<u8> (BEVE typed string array).
pub fn to_vec_string_slice(slice: &[String]) -> Vec<u8> {
    let mut out = Vec::new();
    write_string_slice(&mut out, slice);
    out
}

// -------- Complex numbers (extension) --------

#[inline]
fn write_complex_header(out: &mut Vec<u8>, is_array: bool, class: u8, byte_code: u8) {
    // Extension header + complex header
    out.push(((EXT_COMPLEX & 0x1f) << 3) | (TYPE_EXTENSION & 0b111));
    let h = ((byte_code & 0b111) << 5) | ((class & 0b11) << 3) | (if is_array { 1 } else { 0 });
    out.push(h);
}

/// Write a complex array for any scalar type implementing `BeveTypedSlice`.
pub fn write_complex_slice<T: BeveTypedSlice>(out: &mut Vec<u8>, slice: &[Complex<T>]) {
    let payload = core::mem::size_of_val(slice);
    out.reserve(2 + 8 + payload);
    write_complex_header(out, true, T::CLASS, T::BYTE_CODE);
    write_size(slice.len() as u64, out);
    if !slice.is_empty() {
        #[cfg(target_endian = "little")]
        {
            let start = out.len();
            unsafe {
                let dst = out.as_mut_ptr().add(start);
                ptr::copy_nonoverlapping(slice.as_ptr() as *const u8, dst, payload);
                out.set_len(start + payload);
            }
        }
        #[cfg(not(target_endian = "little"))]
        {
            for c in slice {
                T::write_one_le(&c.re, out);
                T::write_one_le(&c.im, out);
            }
        }
    }
}

/// Encode a complex slice to a new Vec<u8> (BEVE complex extension array).
pub fn to_vec_complex_slice<T: BeveTypedSlice>(slice: &[Complex<T>]) -> Vec<u8> {
    let payload = core::mem::size_of_val(slice);
    let mut out = Vec::with_capacity(2 + 8 + payload);
    write_complex_slice(&mut out, slice);
    out
}

pub fn to_vec_complex64(re: f64, im: f64) -> Vec<u8> {
    let mut out = Vec::new();
    write_complex_header(&mut out, false, ARRAY_FLOAT, 3);
    out.extend_from_slice(&re.to_le_bytes());
    out.extend_from_slice(&im.to_le_bytes());
    out
}

pub fn to_vec_complex32(re: f32, im: f32) -> Vec<u8> {
    let mut out = Vec::new();
    write_complex_header(&mut out, false, ARRAY_FLOAT, 2);
    out.extend_from_slice(&re.to_le_bytes());
    out.extend_from_slice(&im.to_le_bytes());
    out
}

pub fn to_vec_complex64_slice(slice: &[Complex<f64>]) -> Vec<u8> {
    to_vec_complex_slice(slice)
}

pub fn to_vec_complex32_slice(slice: &[Complex<f32>]) -> Vec<u8> {
    to_vec_complex_slice(slice)
}

// -------- Matrices (extension) --------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixLayoutFast {
    Right,
    Left,
}

pub fn to_vec_matrix_f64(layout: MatrixLayoutFast, extents: &[u64], data: &[f64]) -> Vec<u8> {
    let mut out = Vec::new();
    // Extension: matrices
    out.push(((EXT_MATRICES & 0x1f) << 3) | (TYPE_EXTENSION & 0b111));
    // Matrix header: bit0 layout (0 row-major/right, 1 column-major/left per spec)
    let mh = match layout {
        MatrixLayoutFast::Right => 0u8,
        MatrixLayoutFast::Left => 1u8,
    };
    out.push(mh);
    // Extents as typed array of u64 (0x74)
    write_typed_array_header_numeric(&mut out, ARRAY_UNSIGNED, 3, extents.len());
    for &e in extents {
        out.extend_from_slice(&e.to_le_bytes());
    }
    // Data as typed array of f64
    write_typed_array_header_numeric(&mut out, ARRAY_FLOAT, 3, data.len());
    for &v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}
