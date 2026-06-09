//! Aligned typed numeric arrays — the BEVE wire type built for zero-copy access.
//!
//! A regular typed array ([`crate::write_typed_slice`]) packs its element payload
//! immediately after a 1-byte tag and a variable-width SIZE, so the payload is
//! almost never aligned to `align_of::<T>()` on the wire and a reader must copy it
//! into an owned, aligned `Vec<T>`. An *aligned* typed array instead inserts an
//! explicit padding run so the payload lands at `align_of::<T>()` relative to the
//! start of the message buffer, which lets a reader borrow it as `&[T]` directly.
//!
//! Wire layout (BEVE spec §4, "Aligned Typed Arrays"):
//!
//! ```text
//! HEADER(0x5C) | NUMERIC_HEADER | SIZE | PADDING_LENGTH | PADDING | DATA
//! ```
//!
//! * `HEADER` — `0x5C`: a typed array (type 4) with sub-type 3 (the
//!   bool/string/aligned category) and the high-bit discriminator `2` = aligned.
//! * `NUMERIC_HEADER` — the ordinary numeric typed-array header for `T` (the same
//!   byte [`crate::write_typed_slice`] emits, e.g. `0x64` for `f64`).
//! * `SIZE` — compressed element count.
//! * `PADDING_LENGTH` — 1 byte, count of padding bytes that follow (`0 ..
//!   align`).
//! * `PADDING` — that many bytes (written as zeros; contents are ignored on read).
//! * `DATA` — the raw little-endian element block, at an offset that is a multiple
//!   of `align_of::<T>()` from the buffer start.
//!
//! ## Offsets and the buffer base
//!
//! The padding is computed from the absolute byte offset within the *whole*
//! message buffer, so [`write_aligned_typed_slice`] uses `out.len()` as that
//! offset: append it into the actual destination buffer (whatever fixed prefix is
//! already there) and the payload is padded correctly in place. [`crate::to_vec_aligned_slice`]
//! is the offset-0 convenience.
//!
//! Padding only makes the payload aligned *relative to the buffer base*. A
//! zero-copy borrow additionally needs the buffer's base address aligned to
//! `align_of::<T>()` at runtime; [`read_aligned_typed_slice_ref`] verifies the
//! actual payload pointer and refuses (rather than risk unsoundness) when the
//! caller's buffer is not aligned, in which case [`read_aligned_typed_slice`]
//! (one bulk copy, always works) is the fallback.

use crate::error::{Error, Result};
use crate::fast::BeveTypedSlice;
use crate::header::*;
use crate::size::{read_size, size_encoded_len, write_size};
#[cfg(target_endian = "little")]
use core::ptr;

/// High-bit discriminator distinguishing aligned arrays from boolean (0) and
/// string (1) arrays within the bool/string/aligned typed-array category.
const ALIGNED_DISCRIMINATOR: u8 = 2;

/// The marker header byte (`0x5C`) that opens an aligned typed array.
#[inline]
fn aligned_marker_header() -> u8 {
    make_header(
        TYPE_TYPED_ARRAY,
        ARRAY_BOOL_OR_STRING,
        ALIGNED_DISCRIMINATOR,
    )
}

/// Number of padding bytes that must follow a `PADDING_LENGTH` byte sitting at
/// absolute buffer offset `padding_length_offset`, so that the `DATA` that
/// follows the padding is aligned to `align`.
#[inline]
fn padding_for(padding_length_offset: usize, align: usize) -> usize {
    // `DATA` begins at `padding_length_offset + 1 + padding`; choose `padding` in
    // `0..align` so that offset is `0 (mod align)`.
    let offset_after_padding_length = padding_length_offset + 1;
    (align - (offset_after_padding_length % align)) % align
}

/// Append an aligned typed numeric array to `out`, padding the payload to
/// `align_of::<T>()` relative to the start of `out`.
///
/// Because the padding is derived from `out.len()`, this composes with any fixed
/// prefix already in `out`: write the message header/query first, then call this,
/// and the payload is aligned within the final buffer. The bytes are *not*
/// interchangeable with the regular typed array ([`crate::write_typed_slice`]);
/// decode them with [`read_aligned_typed_slice`] / [`read_aligned_typed_slice_ref`].
pub fn write_aligned_typed_slice<T: BeveTypedSlice>(out: &mut Vec<u8>, slice: &[T]) {
    let align = core::mem::align_of::<T>();
    let payload = core::mem::size_of_val(slice);
    // header + numeric header + SIZE (<=9) + PADDING_LENGTH + worst-case padding + payload.
    out.reserve(2 + 9 + 1 + align + payload);

    out.push(aligned_marker_header());
    out.push(make_header(TYPE_TYPED_ARRAY, T::CLASS, T::BYTE_CODE));
    write_size(slice.len() as u64, out);

    let padding = padding_for(out.len(), align);
    out.push(padding as u8);
    out.resize(out.len() + padding, 0);

    if slice.is_empty() {
        return;
    }
    #[cfg(target_endian = "little")]
    {
        // Sound for the same reasons as `write_typed_slice`: a `BeveTypedSlice`
        // scalar is fixed-width, no padding, every bit pattern valid, so the slice
        // reinterprets to exactly `payload` bytes.
        let bytes = unsafe { core::slice::from_raw_parts(slice.as_ptr() as *const u8, payload) };
        out.extend_from_slice(bytes);
    }
    #[cfg(not(target_endian = "little"))]
    {
        for v in slice {
            T::write_one_le(v, out);
        }
    }
}

/// Encode an aligned typed numeric array to a fresh `Vec<u8>` (offset 0).
///
/// Convenience over [`write_aligned_typed_slice`] for a standalone value. Note
/// that a zero-copy borrow of the result requires the `Vec`'s own allocation to
/// be aligned; the in-place [`write_aligned_typed_slice`] into an aligned message
/// buffer is the path that guarantees it.
pub fn to_vec_aligned_slice<T: BeveTypedSlice>(slice: &[T]) -> Vec<u8> {
    let mut out = Vec::new();
    write_aligned_typed_slice(&mut out, slice);
    out
}

/// Exact encoded byte length of [`write_aligned_typed_slice`] for `slice` when it
/// is written starting at absolute buffer offset `start_offset`.
///
/// The length depends on `start_offset` because the padding run does. Pass the
/// offset at which the marker `HEADER` byte will land (e.g. `0` for a standalone
/// value, or the body offset within a larger frame). Matches what
/// `write_aligned_typed_slice` appends when `out.len() == start_offset`.
pub fn aligned_typed_slice_size<T: BeveTypedSlice>(slice: &[T], start_offset: usize) -> usize {
    let align = core::mem::align_of::<T>();
    let size_width = size_encoded_len(slice.len() as u64);
    // Layout up to PADDING_LENGTH: marker + numeric header + SIZE.
    let padding_length_offset = start_offset + 2 + size_width;
    let padding = padding_for(padding_length_offset, align);
    2 + size_width + 1 + padding + core::mem::size_of_val(slice)
}

/// Validate an aligned-typed-array header for element type `T` and return the
/// `DATA` byte slice together with the element count.
fn parse_aligned_header<T: BeveTypedSlice>(input: &[u8]) -> Result<(&[u8], usize)> {
    let mut pos = 0usize;

    let marker = *input.get(pos).ok_or(Error::Eof)?;
    pos += 1;
    if parse_type(marker) != TYPE_TYPED_ARRAY
        || parse_subtype(marker) != ARRAY_BOOL_OR_STRING
        || parse_byte_count_code(marker) != ALIGNED_DISCRIMINATOR
    {
        return Err(Error::InvalidType("not an aligned typed array"));
    }

    let numeric = *input.get(pos).ok_or(Error::Eof)?;
    pos += 1;
    if parse_type(numeric) != TYPE_TYPED_ARRAY {
        return Err(Error::InvalidType("aligned array numeric header malformed"));
    }
    if parse_subtype(numeric) != T::CLASS || parse_byte_count_code(numeric) != T::BYTE_CODE {
        return Err(Error::Mismatch(
            "aligned array element type does not match T",
        ));
    }

    let len = read_size(input, &mut pos)? as usize;

    let padding = *input.get(pos).ok_or(Error::Eof)? as usize;
    pos += 1;
    pos = pos.checked_add(padding).ok_or(Error::InvalidSize)?;
    if pos > input.len() {
        return Err(Error::Eof);
    }

    let payload = len.checked_mul(T::ELEM_SIZE).ok_or(Error::InvalidSize)?;
    if input.len() - pos < payload {
        return Err(Error::Eof);
    }
    Ok((&input[pos..pos + payload], len))
}

/// Decode an aligned typed numeric array into an owned `Vec<T>` with one bulk
/// copy — the read counterpart that always works, regardless of how the input
/// buffer is aligned. Mirrors [`crate::read_typed_slice`] but for the aligned
/// wire layout (skipping the padding run).
///
/// Errors if `input` is not an aligned typed array, if the element class/width do
/// not match `T`, or if the payload is truncated.
pub fn read_aligned_typed_slice<T: BeveTypedSlice>(input: &[u8]) -> Result<Vec<T>> {
    let (data, len) = parse_aligned_header::<T>(input)?;
    let mut out: Vec<T> = Vec::with_capacity(len);
    if len != 0 {
        let payload = data.len();
        #[cfg(target_endian = "little")]
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), out.as_mut_ptr() as *mut u8, payload);
            out.set_len(len);
        }
        #[cfg(not(target_endian = "little"))]
        unsafe {
            let dst = out.as_mut_ptr() as *mut u8;
            core::ptr::copy_nonoverlapping(data.as_ptr(), dst, payload);
            let bytes = core::slice::from_raw_parts_mut(dst, payload);
            for elem in bytes.chunks_exact_mut(T::ELEM_SIZE) {
                elem.reverse();
            }
            out.set_len(len);
        }
    }
    Ok(out)
}

/// Borrow an aligned typed numeric array as `&[T]` with no copy.
///
/// The point of the aligned wire type: when `input` lives in a buffer whose base
/// address is aligned to `align_of::<T>()` (so the padded payload is aligned at
/// runtime, not just relative to the buffer start), this returns a view directly
/// into `input` with zero copying and zero allocation.
///
/// Returns [`Error::Unsupported`] — not a panic or unsoundness — when the payload
/// is not aligned in memory (the caller's buffer base was not aligned) or on a
/// big-endian target (where the little-endian wire bytes are not the native
/// representation). In both cases fall back to [`read_aligned_typed_slice`], which
/// copies. Other errors match [`read_aligned_typed_slice`].
pub fn read_aligned_typed_slice_ref<T: BeveTypedSlice>(input: &[u8]) -> Result<&[T]> {
    let (data, len) = parse_aligned_header::<T>(input)?;
    #[cfg(not(target_endian = "little"))]
    {
        let _ = (data, len);
        return Err(Error::Unsupported(
            "zero-copy aligned borrow requires a little-endian target",
        ));
    }
    #[cfg(target_endian = "little")]
    {
        if (data.as_ptr() as usize) % core::mem::align_of::<T>() != 0 {
            return Err(Error::Unsupported(
                "aligned typed array payload is not aligned in this buffer; the buffer base must be aligned to align_of::<T>() for a zero-copy borrow",
            ));
        }
        // SAFETY: `parse_aligned_header` guarantees `data` is exactly `len *
        // ELEM_SIZE` bytes; the alignment check above guarantees `data.as_ptr()`
        // is aligned for `T`; and every `BeveTypedSlice` scalar is fixed-width with
        // no padding and valid for any bit pattern. On little-endian the wire bytes
        // are already the in-memory representation.
        Ok(unsafe { core::slice::from_raw_parts(data.as_ptr() as *const T, len) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 16-aligned byte buffer (backed by a live `Vec<u128>`) holding `src`
    /// copied in at byte offset `offset`. Lets a test place an encoded aligned
    /// array at a chosen absolute offset within a genuinely aligned message
    /// buffer and then borrow from it.
    struct AlignedBytes {
        backing: Vec<u128>,
        len: usize,
    }
    impl AlignedBytes {
        fn with(src: &[u8], offset: usize) -> Self {
            let len = offset + src.len();
            let backing: Vec<u128> = vec![0u128; len / 16 + 1];
            let mut ab = AlignedBytes { backing, len };
            // SAFETY: the `Vec<u128>` allocation is 16-aligned and large enough.
            let view = unsafe {
                core::slice::from_raw_parts_mut(
                    ab.backing.as_mut_ptr() as *mut u8,
                    ab.backing.len() * 16,
                )
            };
            view[offset..offset + src.len()].copy_from_slice(src);
            ab
        }
        fn bytes(&self) -> &[u8] {
            unsafe { core::slice::from_raw_parts(self.backing.as_ptr() as *const u8, self.len) }
        }
    }

    #[test]
    fn roundtrip_owned_offset0() {
        let data: Vec<f64> = (0..100).map(|i| i as f64 * 0.5).collect();
        let bytes = to_vec_aligned_slice(&data);
        assert_eq!(bytes[0], 0x5C, "marker header");
        let back = read_aligned_typed_slice::<f64>(&bytes).unwrap();
        assert_eq!(back, data);
    }

    #[test]
    fn size_matches_written_length_across_offsets_and_types() {
        macro_rules! check {
            ($ty:ty, $mk:expr) => {
                let data: Vec<$ty> = (0..37).map($mk).collect();
                for start in 0..40usize {
                    let mut buf = vec![0xAAu8; start];
                    write_aligned_typed_slice(&mut buf, &data);
                    let written = buf.len() - start;
                    assert_eq!(
                        written,
                        aligned_typed_slice_size(&data, start),
                        "size mismatch for {} at offset {}",
                        stringify!($ty),
                        start
                    );
                }
            };
        }
        check!(f64, |i| i as f64);
        check!(f32, |i| i as f32);
        check!(i64, |i| i as i64);
        check!(u16, |i| i as u16);
        check!(u8, |i| i as u8);
        check!(i128, |i| i as i128);
    }

    #[test]
    fn payload_is_aligned_within_buffer_for_every_prefix() {
        let data: Vec<f64> = (0..16).map(|i| i as f64).collect();
        for prefix in 0..32usize {
            let mut buf = vec![0u8; prefix];
            write_aligned_typed_slice(&mut buf, &data);
            // Re-derive where DATA starts and assert its offset is 8-aligned.
            let (data_slice, _) = parse_aligned_header::<f64>(&buf[prefix..]).unwrap();
            let data_off = data_slice.as_ptr() as usize - buf.as_ptr() as usize;
            assert_eq!(data_off % 8, 0, "DATA not 8-aligned at prefix {prefix}");
        }
    }

    #[test]
    fn zero_copy_borrow_in_aligned_buffer() {
        let data: Vec<f64> = (0..50).map(|i| (i as f64).sqrt()).collect();
        // Build the frame in a normal Vec with a 16-byte (still 8-aligned) prefix,
        // so the payload is padded for that absolute offset, then place it in a
        // genuinely 16-aligned buffer at offset 0. Base aligned + correct padding =>
        // the payload is aligned at runtime and the borrow succeeds.
        let mut framed = vec![0u8; 16];
        write_aligned_typed_slice(&mut framed, &data);
        let ab = AlignedBytes::with(&framed, 0);
        let buf = ab.bytes();
        assert_eq!(buf.as_ptr() as usize % 16, 0, "aligned backing");

        let view: &[f64] = read_aligned_typed_slice_ref::<f64>(&buf[16..]).unwrap();
        assert_eq!(view, data.as_slice());
        // Genuinely borrowed from `buf`, not a copy.
        let base = buf.as_ptr() as usize;
        let vptr = view.as_ptr() as usize;
        assert!(
            vptr >= base && vptr < base + buf.len(),
            "view borrows from buf"
        );
    }

    #[test]
    fn borrow_refuses_unaligned_buffer() {
        let data: Vec<f64> = (0..8).map(|i| i as f64).collect();
        // Array padded for offset 0, then placed at offset 1 in an aligned buffer:
        // the payload's runtime address is now odd, so the pointer check must refuse
        // rather than hand back an unaligned reference.
        let framed = to_vec_aligned_slice(&data);
        let ab = AlignedBytes::with(&framed, 1);
        let sub = &ab.bytes()[1..];
        // Owned decode still works.
        assert_eq!(read_aligned_typed_slice::<f64>(sub).unwrap(), data);
        // Borrow refuses (Unsupported), no unsoundness.
        assert!(read_aligned_typed_slice_ref::<f64>(sub).is_err());
    }

    #[test]
    fn rejects_regular_typed_array_and_wrong_type() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0];
        // A regular (unaligned) typed array must not parse as aligned.
        let regular = crate::to_vec_typed_slice(&data);
        assert!(read_aligned_typed_slice::<f64>(&regular).is_err());
        // Wrong element type.
        let aligned = to_vec_aligned_slice(&data);
        assert!(read_aligned_typed_slice::<f32>(&aligned).is_err());
        assert!(read_aligned_typed_slice::<i64>(&aligned).is_err());
    }

    #[test]
    fn empty_slice_roundtrips() {
        let data: Vec<f64> = Vec::new();
        let bytes = to_vec_aligned_slice(&data);
        assert!(read_aligned_typed_slice::<f64>(&bytes).unwrap().is_empty());
    }

    #[test]
    fn padding_bytes_are_ignored_on_read() {
        const PREFIX: usize = 3;
        let data: Vec<f64> = (0..5).map(|i| i as f64).collect();
        // Writing at a non-8-aligned offset forces a non-zero padding run.
        let mut buf = vec![0u8; PREFIX];
        write_aligned_typed_slice(&mut buf, &data);

        // Locate the padding run exactly: marker + numeric header + SIZE, then the
        // PADDING_LENGTH byte, then `pad` padding bytes.
        let size_w = size_encoded_len(data.len() as u64);
        let pl_offset = PREFIX + 2 + size_w;
        let pad = padding_for(pl_offset, core::mem::align_of::<f64>());
        assert!(pad > 0, "test needs a non-zero padding run");
        for b in &mut buf[pl_offset + 1..pl_offset + 1 + pad] {
            *b = 0xFF; // corrupt only the padding, not PADDING_LENGTH or DATA
        }
        assert_eq!(
            read_aligned_typed_slice::<f64>(&buf[PREFIX..]).unwrap(),
            data
        );
    }
}
