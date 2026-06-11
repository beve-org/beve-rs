use crate::error::{Error, Result};
use crate::ext::Complex;
use crate::header::*;
use crate::size::{
    read_size, read_size_from_reader, size_encoded_len, write_size, write_size_to_writer,
};
// Only the little-endian bulk-copy paths use raw `ptr`; on big-endian they fall
// back to per-element conversion, so the import would otherwise be unused there.
#[cfg(target_endian = "little")]
use core::ptr;
use half::{bf16, f16};
use std::io::{Read, Write};

/// Write a typed array header and length for numeric arrays.
#[inline]
fn write_typed_array_header_numeric(out: &mut Vec<u8>, class: u8, byte_code: u8, len: usize) {
    out.push(make_header(TYPE_TYPED_ARRAY, class, byte_code));
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

/// Encode a typed numeric slice to a new `Vec<u8>` (BEVE typed array).
pub fn to_vec_typed_slice<T: BeveTypedSlice>(slice: &[T]) -> Vec<u8> {
    let payload = slice.len() * T::ELEM_SIZE;
    let mut out = Vec::with_capacity(1 + 8 + payload);
    write_typed_slice(&mut out, slice);
    out
}

/// Write a BEVE typed numeric array straight to a writer in a single bulk write.
///
/// Emits the typed-array header byte, the SIZE prefix, then the element payload.
/// On little-endian targets the payload is written with one `write_all` of the
/// slice reinterpreted as bytes (no allocation, no per-element conversion); on
/// big-endian targets it falls back to per-element little-endian conversion.
///
/// The bytes produced are identical to [`write_typed_slice`] (the `Vec<u8>`
/// primitive) and, for a non-empty slice, to serializing the equivalent
/// `Vec<T>` through [`crate::to_writer_streaming`]. See [`typed_slice_size`] for
/// the matching analytic length, and [`crate::TypedSlice`] to reach this path
/// through serde for a slice nested inside a derived struct.
///
/// # Example
///
/// ```rust
/// let data = [1.0f64, 2.0, 3.0];
/// let mut buf = Vec::new();
/// beve::to_writer_typed_slice(&mut buf, &data).unwrap();
/// assert_eq!(buf.len() as u64, beve::typed_slice_size(&data));
/// let back: Vec<f64> = beve::from_slice(&buf).unwrap();
/// assert_eq!(back, data);
/// ```
pub fn to_writer_typed_slice<W: Write, T: BeveTypedSlice>(mut w: W, slice: &[T]) -> Result<()> {
    w.write_all(&[make_header(TYPE_TYPED_ARRAY, T::CLASS, T::BYTE_CODE)])?;
    write_size_to_writer(&mut w, slice.len() as u64)?;
    if slice.is_empty() {
        return Ok(());
    }
    #[cfg(target_endian = "little")]
    {
        // Sound: every `BeveTypedSlice` type is a fixed-width scalar with no
        // padding and every bit pattern valid, so reinterpreting the slice as
        // bytes mirrors `write_typed_slice`'s `copy_nonoverlapping`. `size_of_val`
        // gives the exact payload length.
        let bytes = unsafe {
            core::slice::from_raw_parts(slice.as_ptr() as *const u8, core::mem::size_of_val(slice))
        };
        w.write_all(bytes)?;
    }
    #[cfg(not(target_endian = "little"))]
    {
        // Rare big-endian path: convert each element to little-endian. One small
        // reused scratch buffer, no per-element allocation.
        let mut scratch = Vec::with_capacity(T::ELEM_SIZE);
        for v in slice {
            scratch.clear();
            T::write_one_le(v, &mut scratch);
            w.write_all(&scratch)?;
        }
    }
    Ok(())
}

/// Exact streaming-encoded byte length of [`to_writer_typed_slice`] for `slice`.
///
/// Closed-form and O(1) in the element count: `header byte + SIZE prefix width +
/// payload`. The SIZE width reuses the same codec thresholds as the bytes the
/// primitive actually writes, so the two cannot drift.
///
/// # Example
///
/// ```rust
/// let data = [1u32, 2, 3, 4];
/// let mut buf = Vec::new();
/// beve::to_writer_typed_slice(&mut buf, &data).unwrap();
/// assert_eq!(beve::typed_slice_size(&data), buf.len() as u64);
/// ```
pub fn typed_slice_size<T: BeveTypedSlice>(slice: &[T]) -> u64 {
    1 // header byte
        + size_encoded_len(slice.len() as u64) as u64 // SIZE prefix width
        + core::mem::size_of_val(slice) as u64 // payload
}

/// Decode a BEVE typed numeric array (the bytes produced by [`write_typed_slice`]
/// / [`to_vec_typed_slice`] / [`to_writer_typed_slice`], or by serializing a
/// `Vec<T>` of a supported scalar) into a `Vec<T>` in a single bounds-checked
/// bulk read — the read counterpart of the bulk writers.
///
/// Like the writers, this is opt-in and requires the caller to name the element
/// type `T`: serde's `Deserialize` for `Vec<T>` pulls elements one at a time
/// through a visitor and never sees the contiguous little-endian block, so the
/// generic `from_slice` path cannot take it automatically. On little-endian
/// targets the whole payload is moved into the result with one
/// `copy_nonoverlapping` after a single bounds check (a single bulk copy, not
/// zero-copy: the input bytes may be unaligned for `T`); on big-endian targets it
/// copies and then byte-reverses each element in place.
///
/// Errors if `input` is not a typed *numeric* array, if the on-wire class /
/// byte-code do not match `T`, or if the payload is truncated. Any bytes after
/// the array are ignored (matching [`crate::from_slice`]); the function returns
/// only the decoded `Vec`, so it does not report how many bytes were consumed.
/// Boolean and string arrays use a different on-wire layout (bit-packed /
/// length-prefixed) and are not `BeveTypedSlice` types, so they are rejected by
/// the class check rather than misread.
///
/// # Example
///
/// ```rust
/// let data = [1.0f64, -2.5, 3.25];
/// let bytes = beve::to_vec_typed_slice(&data);
/// let back = beve::read_typed_slice::<f64>(&bytes).unwrap();
/// assert_eq!(back, data);
/// ```
pub fn read_typed_slice<T: BeveTypedSlice>(input: &[u8]) -> Result<Vec<T>> {
    let mut pos = 0usize;

    // Header byte: must be a typed array whose class and byte-code name `T`.
    let header = *input.get(pos).ok_or(Error::Eof)?;
    pos += 1;
    if parse_type(header) != TYPE_TYPED_ARRAY {
        return Err(Error::InvalidType("not a typed array"));
    }
    if parse_subtype(header) != T::CLASS || parse_byte_count_code(header) != T::BYTE_CODE {
        return Err(Error::Mismatch("typed array element type does not match T"));
    }

    let len = read_size(input, &mut pos)? as usize;
    let payload = len.checked_mul(T::ELEM_SIZE).ok_or(Error::InvalidSize)?;
    if input.len() - pos < payload {
        return Err(Error::Eof);
    }
    let src = &input[pos..pos + payload];

    let mut out: Vec<T> = Vec::with_capacity(len);
    if len != 0 {
        // Every `BeveTypedSlice` type is a fixed-width scalar with no padding and
        // every bit pattern valid, so `len` values occupy exactly `payload`
        // contiguous bytes — the same layout the writers emit. The destination
        // `Vec` is allocated with `T`'s alignment, and `src` is exactly `payload`
        // bytes by the bounds check above.
        #[cfg(target_endian = "little")]
        {
            // Wire payload and target are both little-endian, so the bytes are
            // already the in-memory representation: one bulk copy, no per-element
            // work — the mirror of the writer's `copy_nonoverlapping`.
            unsafe {
                ptr::copy_nonoverlapping(src.as_ptr(), out.as_mut_ptr() as *mut u8, payload);
                out.set_len(len);
            }
        }
        #[cfg(not(target_endian = "little"))]
        {
            // Big-endian fallback: copy the little-endian payload, then byte-reverse
            // each element in place. On a big-endian target the native byte order is
            // the reverse of the wire's little-endian, and every element is exactly
            // `ELEM_SIZE` wide with no padding, so reversing each `ELEM_SIZE` chunk
            // across the whole payload yields the correct native values. Correct, and
            // the rare path (matching the writer, whose bulk copy is also LE-only).
            unsafe {
                let dst = out.as_mut_ptr() as *mut u8;
                core::ptr::copy_nonoverlapping(src.as_ptr(), dst, payload);
                let bytes = core::slice::from_raw_parts_mut(dst, payload);
                for elem in bytes.chunks_exact_mut(T::ELEM_SIZE) {
                    elem.reverse();
                }
                out.set_len(len);
            }
        }
    }
    Ok(out)
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

/// Encode a boolean slice to a new `Vec<u8>` (BEVE typed boolean array).
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

/// Encode `&[&str]` to a new `Vec<u8>` (BEVE typed string array).
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

/// Encode `&[String]` to a new `Vec<u8>` (BEVE typed string array).
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

/// Encode a complex slice to a new `Vec<u8>` (BEVE complex extension array).
pub fn to_vec_complex_slice<T: BeveTypedSlice>(slice: &[Complex<T>]) -> Vec<u8> {
    let payload = core::mem::size_of_val(slice);
    let mut out = Vec::with_capacity(2 + 8 + payload);
    write_complex_slice(&mut out, slice);
    out
}

/// Write a BEVE complex array straight to a writer in a single bulk write — the
/// streaming counterpart of [`write_complex_slice`] / [`to_vec_complex_slice`],
/// mirroring [`to_writer_typed_slice`] for the complex extension.
///
/// Emits the two-byte complex extension header, the SIZE prefix, then the
/// interleaved `(re, im)` payload. On little-endian targets the payload is
/// written with one `write_all` of the slice reinterpreted as bytes (no
/// allocation, no per-element conversion); on big-endian targets it falls back to
/// per-scalar little-endian conversion.
///
/// The bytes produced are identical to [`write_complex_slice`] (the `Vec<u8>`
/// primitive). See [`complex_slice_size`] for the matching analytic length, and
/// [`read_complex_slice`] for the bulk read counterpart.
///
/// # Example
///
/// ```rust
/// use beve::Complex;
/// let data = [Complex { re: 1.0f64, im: -2.0 }, Complex { re: 3.5, im: 4.25 }];
/// let mut buf = Vec::new();
/// beve::to_writer_complex_slice(&mut buf, &data).unwrap();
/// assert_eq!(buf.len() as u64, beve::complex_slice_size(&data));
/// let back = beve::read_complex_slice::<f64>(&buf).unwrap();
/// assert_eq!(back, data);
/// ```
pub fn to_writer_complex_slice<W: Write, T: BeveTypedSlice>(
    mut w: W,
    slice: &[Complex<T>],
) -> Result<()> {
    // Two-byte complex extension header (extension header + complex header),
    // identical to `write_complex_header(.., is_array = true, ..)`.
    w.write_all(&[
        ((EXT_COMPLEX & 0x1f) << 3) | (TYPE_EXTENSION & 0b111),
        ((T::BYTE_CODE & 0b111) << 5) | ((T::CLASS & 0b11) << 3) | 1,
    ])?;
    write_size_to_writer(&mut w, slice.len() as u64)?;
    if slice.is_empty() {
        return Ok(());
    }
    #[cfg(target_endian = "little")]
    {
        // Sound for the same reasons as `to_writer_typed_slice`: `Complex<T>` for a
        // `BeveTypedSlice` `T` is two fixed-width scalars with no padding and every
        // bit pattern valid, so reinterpreting the slice as bytes mirrors
        // `write_complex_slice`'s `copy_nonoverlapping`. `size_of_val` gives the
        // exact interleaved `(re, im)` payload length.
        let bytes = unsafe {
            core::slice::from_raw_parts(slice.as_ptr() as *const u8, core::mem::size_of_val(slice))
        };
        w.write_all(bytes)?;
    }
    #[cfg(not(target_endian = "little"))]
    {
        // Rare big-endian path: convert each scalar to little-endian, re then im.
        // One small reused scratch buffer, no per-element allocation.
        let mut scratch = Vec::with_capacity(T::ELEM_SIZE);
        for c in slice {
            scratch.clear();
            T::write_one_le(&c.re, &mut scratch);
            T::write_one_le(&c.im, &mut scratch);
            w.write_all(&scratch)?;
        }
    }
    Ok(())
}

/// Exact streaming-encoded byte length of [`to_writer_complex_slice`] for `slice`.
///
/// Closed-form and O(1) in the element count: `2-byte complex header + SIZE prefix
/// width + payload`, where the payload is the interleaved `(re, im)` block. The
/// SIZE width reuses the same codec thresholds as the bytes the primitive actually
/// writes, so the two cannot drift.
///
/// # Example
///
/// ```rust
/// use beve::Complex;
/// let data = [Complex { re: 1.0f32, im: 2.0 }, Complex { re: 3.0, im: 4.0 }];
/// let mut buf = Vec::new();
/// beve::to_writer_complex_slice(&mut buf, &data).unwrap();
/// assert_eq!(beve::complex_slice_size(&data), buf.len() as u64);
/// ```
pub fn complex_slice_size<T: BeveTypedSlice>(slice: &[Complex<T>]) -> u64 {
    2 // two-byte complex extension header
        + size_encoded_len(slice.len() as u64) as u64 // SIZE prefix width
        + core::mem::size_of_val(slice) as u64 // interleaved (re, im) payload
}

/// Decode a BEVE complex array (the bytes produced by [`write_complex_slice`] /
/// [`to_vec_complex_slice`]) into a `Vec<Complex<T>>` in a single bounds-checked
/// bulk read — the read counterpart of the bulk writer.
///
/// Like the writer, this is opt-in and requires the caller to name the element
/// type `T`: serde's `Deserialize` for `Vec<Complex<T>>` pulls elements one at a
/// time through nested visitors and never sees the contiguous little-endian
/// block, so the generic `from_slice` path cannot take it automatically. On
/// little-endian targets the whole payload is moved into the result with one
/// `copy_nonoverlapping` after a single bounds check (a single bulk copy, not
/// zero-copy: the input bytes may be unaligned for `Complex<T>`); on big-endian
/// targets it copies and then byte-reverses each scalar in place.
///
/// Errors if `input` is not a complex *array* extension, if the on-wire
/// class/byte-code do not match `T`, or if the payload is truncated. Any bytes
/// after the array are ignored (matching [`crate::from_slice`]); the function
/// returns only the decoded `Vec`, so it does not report how many bytes were
/// consumed.
///
/// # Example
///
/// ```rust
/// use beve::Complex;
/// let data = [Complex { re: 1.0f64, im: -2.0 }, Complex { re: 3.5, im: 4.25 }];
/// let bytes = beve::to_vec_complex_slice(&data);
/// let back = beve::read_complex_slice::<f64>(&bytes).unwrap();
/// assert_eq!(back, data);
/// ```
pub fn read_complex_slice<T: BeveTypedSlice>(input: &[u8]) -> Result<Vec<Complex<T>>> {
    let mut pos = 0usize;

    // Extension header byte: must be the complex extension.
    let header = *input.get(pos).ok_or(Error::Eof)?;
    pos += 1;
    if parse_type(header) != TYPE_EXTENSION || parse_extension_id(header) != EXT_COMPLEX {
        return Err(Error::InvalidType("not a complex extension"));
    }

    // Complex header byte: bit0 = array flag, bits3-4 = class, bits5-7 = byte code.
    let ch = *input.get(pos).ok_or(Error::Eof)?;
    pos += 1;
    let is_array = (ch & 0x01) != 0;
    let class = (ch >> 3) & 0x03;
    let byte_code = (ch >> 5) & 0x07;
    if !is_array {
        return Err(Error::InvalidType(
            "expected a complex array, found a scalar complex",
        ));
    }
    if class != T::CLASS || byte_code != T::BYTE_CODE {
        return Err(Error::Mismatch("complex element type does not match T"));
    }

    let len = read_size(input, &mut pos)? as usize;
    // Two scalars (re, im) per complex value.
    let payload = len
        .checked_mul(2 * T::ELEM_SIZE)
        .ok_or(Error::InvalidSize)?;
    if input.len() - pos < payload {
        return Err(Error::Eof);
    }
    let src = &input[pos..pos + payload];

    let mut out: Vec<Complex<T>> = Vec::with_capacity(len);
    if len != 0 {
        // `Complex<T>` is `#[repr(C)]` over two fixed-width, no-padding scalars, so
        // `len` values occupy exactly `payload` contiguous bytes laid out as
        // [re, im, re, im, ...] — the same layout `write_complex_slice` writes. The
        // destination `Vec` is allocated with `Complex<T>`'s alignment, and `src` is
        // exactly `payload` bytes by the bounds check above.
        #[cfg(target_endian = "little")]
        {
            // The wire payload is little-endian and so is the target, so the bytes
            // are already the in-memory representation: one bulk copy, no per-element
            // work — the mirror of the writer's `copy_nonoverlapping`.
            unsafe {
                ptr::copy_nonoverlapping(src.as_ptr(), out.as_mut_ptr() as *mut u8, payload);
                out.set_len(len);
            }
        }
        #[cfg(not(target_endian = "little"))]
        {
            // Big-endian fallback: copy the little-endian payload, then byte-reverse
            // each scalar in place. On a big-endian target the native byte order is
            // the reverse of the wire's little-endian, and every scalar is exactly
            // `ELEM_SIZE` wide with no padding, so reversing each `ELEM_SIZE` chunk
            // across the whole payload yields the correct native values. Correct, and
            // the rare path (matching the writer, whose bulk copy is also LE-only).
            unsafe {
                let dst = out.as_mut_ptr() as *mut u8;
                core::ptr::copy_nonoverlapping(src.as_ptr(), dst, payload);
                let bytes = core::slice::from_raw_parts_mut(dst, payload);
                for elem in bytes.chunks_exact_mut(T::ELEM_SIZE) {
                    elem.reverse();
                }
                out.set_len(len);
            }
        }
    }
    Ok(out)
}

/// Streaming counterpart of [`read_typed_slice`]: read a BEVE typed numeric array
/// straight from a reader into a `Vec<T>` at bulk speed, without first holding the
/// encoded payload in memory.
///
/// This is the bulk alternative to deserializing a `Vec<T>` through serde
/// (`from_reader_streaming`), which visits every element through the serde
/// machinery. It reads the array header, validates the element type names `T`,
/// then drains exactly `len * size_of::<T>()` little-endian payload bytes directly
/// into the result buffer. Memory is bounded to the result `Vec<T>`, and the
/// reader is left positioned immediately after the array, so it composes with
/// further reads from the same stream.
///
/// The payload is read in capped steps, so a corrupt or hostile length field can
/// fail the allocation gracefully (`Error`) rather than forcing one huge up-front
/// allocation; bytes are only committed to the `Vec` once they have actually
/// arrived.
pub fn read_typed_slice_from_reader<T: BeveTypedSlice, R: Read>(mut reader: R) -> Result<Vec<T>> {
    let mut hdr = [0u8; 1];
    reader.read_exact(&mut hdr).map_err(|_| Error::Eof)?;
    let header = hdr[0];
    if parse_type(header) != TYPE_TYPED_ARRAY {
        return Err(Error::InvalidType("not a typed array"));
    }
    if parse_subtype(header) != T::CLASS || parse_byte_count_code(header) != T::BYTE_CODE {
        return Err(Error::Mismatch("typed array element type does not match T"));
    }
    let len = read_size_from_reader(&mut reader)? as usize;
    read_le_payload::<T, R>(&mut reader, len, T::ELEM_SIZE)
}

/// Streaming counterpart of [`read_complex_slice`]: read a BEVE complex array
/// straight from a reader into a `Vec<Complex<T>>` at bulk speed.
///
/// The complex counterpart of [`read_typed_slice_from_reader`]; see it for the
/// memory and allocation behavior. Each value is two `T` scalars `[re, im]`, so a
/// `Vec<Complex<T>>` of `len` values occupies `len * 2 * size_of::<T>()` contiguous
/// little-endian payload bytes, the same layout [`write_complex_slice`] writes.
pub fn read_complex_slice_from_reader<T: BeveTypedSlice, R: Read>(
    mut reader: R,
) -> Result<Vec<Complex<T>>> {
    // The complex extension header byte and the complex header byte are adjacent.
    let mut hdr = [0u8; 2];
    reader.read_exact(&mut hdr).map_err(|_| Error::Eof)?;
    if parse_type(hdr[0]) != TYPE_EXTENSION || parse_extension_id(hdr[0]) != EXT_COMPLEX {
        return Err(Error::InvalidType("not a complex extension"));
    }
    let ch = hdr[1];
    let is_array = (ch & 0x01) != 0;
    let class = (ch >> 3) & 0x03;
    let byte_code = (ch >> 5) & 0x07;
    if !is_array {
        return Err(Error::InvalidType(
            "expected a complex array, found a scalar complex",
        ));
    }
    if class != T::CLASS || byte_code != T::BYTE_CODE {
        return Err(Error::Mismatch("complex element type does not match T"));
    }
    let len = read_size_from_reader(&mut reader)? as usize;
    // Byte-swap unit (big-endian) is the scalar width `T::ELEM_SIZE`; the element
    // stride `size_of::<Complex<T>>()` covers the two scalars.
    read_le_payload::<Complex<T>, R>(&mut reader, len, T::ELEM_SIZE)
}

/// Read `len` fixed-layout little-endian values of `E` from `reader` into a
/// `Vec<E>`, the streaming bulk core shared by the typed and complex readers.
///
/// `E` is `T` (typed) or `Complex<T>` (complex); both are `#[repr(C)]`-compatible
/// over fixed-width scalars with no padding and every bit pattern valid, so the
/// little-endian payload is already the in-memory representation on a
/// little-endian target. `scalar_size` is the width of a single scalar
/// (`T::ELEM_SIZE`), used only for the big-endian byte-swap.
fn read_le_payload<E, R: Read>(reader: &mut R, len: usize, scalar_size: usize) -> Result<Vec<E>> {
    let mut out: Vec<E> = Vec::new();
    if len == 0 {
        return Ok(out);
    }
    let elem = core::mem::size_of::<E>();
    // Validate the total fits in memory arithmetic before touching the reader.
    len.checked_mul(elem).ok_or(Error::InvalidSize)?;

    // Grow in capped steps so a bogus `len` can't force a huge up-front allocation;
    // each step's elements are committed (`set_len`) only after their bytes arrive.
    const STEP_BYTES: usize = 8 * 1024 * 1024;
    let step_elems = (STEP_BYTES / elem.max(1)).max(1);
    let mut filled = 0usize;
    while filled < len {
        let take = step_elems.min(len - filled);
        out.try_reserve(take)
            .map_err(|_| Error::Message("allocation failed for slice"))?;
        // SAFETY: `take` elements were just reserved past `filled`; zero-initialize
        // their bytes (a valid bit pattern for every supported scalar, so a valid
        // `E`), read exactly those bytes from the reader, then commit them. On a
        // read error `set_len` is not reached, so the reserved-but-unwritten tail
        // is never observed as initialized.
        unsafe {
            let start = out.as_mut_ptr().add(filled) as *mut u8;
            let nbytes = take * elem;
            core::ptr::write_bytes(start, 0, nbytes);
            let dst = core::slice::from_raw_parts_mut(start, nbytes);
            // Preserve the real reader error (a genuine IO failure is not an EOF);
            // the header/size reads above use `Eof` since a failure there is a
            // truncated array header.
            reader.read_exact(dst).map_err(Error::from)?;
            out.set_len(filled + take);
        }
        filled += take;
    }

    #[cfg(not(target_endian = "little"))]
    {
        // Big-endian: the wire is little-endian, so reverse each scalar in place.
        let payload = len * elem;
        let bytes =
            unsafe { core::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, payload) };
        for s in bytes.chunks_exact_mut(scalar_size) {
            s.reverse();
        }
    }
    #[cfg(target_endian = "little")]
    let _ = scalar_size;

    Ok(out)
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
