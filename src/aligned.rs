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
//! already there) and the payload is padded correctly in place. [`crate::to_vec_aligned_typed_slice`]
//! is the offset-0 convenience. When the body is built in a *standalone* buffer and
//! concatenated behind a fixed prefix afterward, [`write_aligned_typed_slice_at`]
//! takes the eventual frame offset explicitly so the padding is still correct.
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
///
/// Equivalent to [`write_aligned_typed_slice_at`] with `base_offset == out.len()`.
/// Reach for `_at` when the payload's eventual position in the wire frame differs
/// from its position in `out` (e.g. when building a standalone body buffer that
/// will later be concatenated behind a header and query).
pub fn write_aligned_typed_slice<T: BeveTypedSlice>(out: &mut Vec<u8>, slice: &[T]) {
    write_aligned_typed_slice_at(out, slice, out.len());
}

/// Append an aligned typed numeric array to `out`, padding the payload so it is
/// aligned to `align_of::<T>()` relative to `base_offset` rather than to the start
/// of `out`.
///
/// [`write_aligned_typed_slice`] aligns the payload relative to the start of the
/// buffer it appends into, which is exactly right when `out` *is* the final wire
/// buffer (header and query already written). But some framers build the body in a
/// standalone buffer and concatenate it behind a fixed prefix afterward; there the
/// payload's position in `out` (offset 0) is not its position in the frame
/// (`prefix_len`), so aligning to `out.len()` would pad for the wrong offset.
///
/// `base_offset` is the absolute byte offset at which the marker `HEADER` byte will
/// land in the final frame. The padding run is sized for that offset, so the
/// emitted bytes are correct for the frame even though `out` itself may start at 0:
/// only the byte sequence (specifically the `PADDING_LENGTH` count) matters, not
/// `out`'s own address. Pair with [`aligned_typed_slice_size`] (passing the same
/// value as its `start_offset`) to size the buffer, and decode with
/// [`read_aligned_typed_slice`] / [`read_aligned_typed_slice_ref`].
///
/// ```
/// # use beve::{write_aligned_typed_slice_at, aligned_typed_slice_size, read_aligned_typed_slice};
/// let data = [1.0_f64, 2.0, 3.0];
/// // Build a standalone body whose payload will be aligned once it sits at byte
/// // offset 54 (e.g. a 48-byte header + 6-byte query) in the final frame.
/// let prefix_len = 54;
/// let mut body = Vec::new();
/// write_aligned_typed_slice_at(&mut body, &data, prefix_len);
/// assert_eq!(body.len(), aligned_typed_slice_size(&data, prefix_len));
/// // The owned decode reads the body regardless of where it lives.
/// assert_eq!(read_aligned_typed_slice::<f64>(&body).unwrap(), data);
/// ```
pub fn write_aligned_typed_slice_at<T: BeveTypedSlice>(
    out: &mut Vec<u8>,
    slice: &[T],
    base_offset: usize,
) {
    let start = out.len();
    let align = core::mem::align_of::<T>();
    let payload = core::mem::size_of_val(slice);
    // header + numeric header + SIZE (<=9) + PADDING_LENGTH + worst-case padding + payload.
    out.reserve(2 + 9 + 1 + align + payload);

    out.push(aligned_marker_header());
    out.push(make_header(TYPE_TYPED_ARRAY, T::CLASS, T::BYTE_CODE));
    write_size(slice.len() as u64, out);

    // The PADDING_LENGTH byte's offset *in the frame* is `base_offset` plus however
    // many bytes (marker + numeric header + SIZE) we have written into `out` since
    // `start` — not `out.len()`, which is the offset within `out`. When
    // `base_offset == start` the two coincide, recovering `write_aligned_typed_slice`.
    // Only this value's residue mod `align` reaches `padding_for`, so a pathological
    // near-`usize::MAX` `base_offset` wraps harmlessly here rather than panicking;
    // the emitted count stays correct for the offset the caller declared.
    let padding_length_offset = base_offset.wrapping_add(out.len() - start);
    let padding = padding_for(padding_length_offset, align);
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
pub fn to_vec_aligned_typed_slice<T: BeveTypedSlice>(slice: &[T]) -> Vec<u8> {
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
    // Layout up to PADDING_LENGTH: marker + numeric header + SIZE. Only the residue
    // mod `align` reaches `padding_for`, so this wraps harmlessly for a pathological
    // near-`usize::MAX` `start_offset`, staying in lockstep with the writer.
    let padding_length_offset = start_offset.wrapping_add(2 + size_width);
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
/// Returns [`Error::Unsupported`] — not a panic or unsoundness — when a zero-copy
/// borrow cannot be served: either the payload is not aligned in memory (the
/// caller's buffer base was not aligned) or the target is big-endian (where the
/// little-endian wire bytes are not the native representation). Both are
/// recoverable, and the fallback is the same in both cases: **any `Unsupported`
/// from this function means "retry with [`read_aligned_typed_slice`]"**, which
/// copies and always succeeds. (The two cases carry distinct messages but are not
/// otherwise meant to be told apart — the big-endian case will never succeed as a
/// borrow, the unaligned case might in a differently-aligned buffer, but the
/// caller's recovery is identical.) All other error conditions (not an aligned
/// typed array, element-type mismatch, truncation) match
/// [`read_aligned_typed_slice`].
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
        if !(data.as_ptr() as usize).is_multiple_of(core::mem::align_of::<T>()) {
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
        let bytes = to_vec_aligned_typed_slice(&data);
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
            assert!(
                data_off.is_multiple_of(8),
                "DATA not 8-aligned at prefix {prefix}"
            );
        }
    }

    // Zero-copy aligned borrow is little-endian only — on big-endian the reader
    // correctly returns `Unsupported` (the wire is LE; a borrowed `&[T]` would be
    // byte-swapped wrong), so these success-path tests are LE-gated.
    #[cfg(target_endian = "little")]
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
        assert!(
            (buf.as_ptr() as usize).is_multiple_of(16),
            "aligned backing"
        );

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
    fn write_at_with_buffer_len_matches_plain_writer() {
        // `write_aligned_typed_slice` is defined as `_at` with `base_offset ==
        // out.len()`; pin that equivalence across prefixes and a couple of types.
        macro_rules! check {
            ($ty:ty, $mk:expr) => {
                let data: Vec<$ty> = (0..29).map($mk).collect();
                for prefix in 0..20usize {
                    let mut a = vec![0x11u8; prefix];
                    let mut b = vec![0x11u8; prefix];
                    write_aligned_typed_slice(&mut a, &data);
                    let off = b.len();
                    write_aligned_typed_slice_at(&mut b, &data, off);
                    assert_eq!(a, b, "{} at prefix {}", stringify!($ty), prefix);
                }
            };
        }
        check!(f64, |i| i as f64);
        check!(i32, |i| i - 14);
        check!(u8, |i| i as u8);
    }

    #[test]
    fn write_at_size_matches_aligned_typed_slice_size() {
        // A standalone body (appended into an empty `out`) still reports the size
        // `aligned_typed_slice_size` predicts for the declared frame offset.
        let data: Vec<f32> = (0..23).map(|i| i as f32 * 0.5).collect();
        for base in 0..40usize {
            let mut body = Vec::new();
            write_aligned_typed_slice_at(&mut body, &data, base);
            assert_eq!(
                body.len(),
                aligned_typed_slice_size(&data, base),
                "size mismatch at base {base}"
            );
        }
    }

    #[cfg(target_endian = "little")] // zero-copy aligned borrow is LE-only
    #[test]
    fn write_at_standalone_body_borrows_at_frame_offset() {
        // The motivating case: build the body in its own buffer (offset 0), padded
        // for where it will *land* in the frame, then place it there inside a
        // 16-aligned buffer. The payload is aligned at runtime and the zero-copy
        // borrow succeeds even for prefixes that are not a multiple of `align`.
        let data: Vec<f64> = (0..40).map(|i| i as f64 * 1.5).collect();
        for base_offset in [8usize, 16, 48, 54, 55, 63, 100] {
            let mut body = Vec::new();
            write_aligned_typed_slice_at(&mut body, &data, base_offset);
            assert_eq!(body[0], 0x5C, "marker header at base {base_offset}");

            let ab = AlignedBytes::with(&body, base_offset);
            let frame = ab.bytes();
            let borrowed =
                read_aligned_typed_slice_ref::<f64>(&frame[base_offset..]).expect("borrow");
            assert_eq!(borrowed, data.as_slice(), "base_offset {base_offset}");
        }
    }

    #[test]
    fn write_at_empty_slice_matches_sizer_and_roundtrips() {
        // An empty array still emits a well-formed (length-prefix-only) aligned
        // body whose size the sizer predicts, for any declared offset.
        let data: Vec<f64> = Vec::new();
        for base in [0usize, 7, 48, 54] {
            let mut body = Vec::new();
            write_aligned_typed_slice_at(&mut body, &data, base);
            assert_eq!(
                body.len(),
                aligned_typed_slice_size(&data, base),
                "base {base}"
            );
            assert!(read_aligned_typed_slice::<f64>(&body).unwrap().is_empty());
        }
    }

    #[cfg(target_endian = "little")] // zero-copy aligned borrow is LE-only
    #[test]
    fn write_at_borrows_for_align_gt_8_element() {
        // i128 has align 16, so the padding run is wider; confirm a standalone body
        // padded for its frame offset is borrowable when placed there in a
        // 16-aligned buffer, including non-16-multiple offsets.
        let data: Vec<i128> = (0..12).map(|i| ((i as i128) << 64) | (i as i128)).collect();
        for base_offset in [16usize, 32, 48, 50, 63, 100] {
            let mut body = Vec::new();
            write_aligned_typed_slice_at(&mut body, &data, base_offset);
            assert_eq!(body.len(), aligned_typed_slice_size(&data, base_offset));

            let ab = AlignedBytes::with(&body, base_offset);
            let frame = ab.bytes();
            let borrowed =
                read_aligned_typed_slice_ref::<i128>(&frame[base_offset..]).expect("borrow");
            assert_eq!(borrowed, data.as_slice(), "base_offset {base_offset}");
        }
    }

    #[cfg(target_endian = "little")] // zero-copy aligned borrow is LE-only
    #[test]
    fn write_at_with_prefilled_out_and_independent_base_offset() {
        // `out` already holds unrelated bytes (start != 0) and `base_offset` is a
        // *different* value (the body's eventual frame offset, unrelated to where it
        // sits in this scratch buffer). The appended bytes must still be a
        // self-consistent aligned body and be borrowable once placed at base_offset.
        let data: Vec<f64> = (0..20).map(|i| i as f64 * 2.0).collect();
        let base_offset = 54usize;
        let mut out = vec![0xEEu8; 13]; // arbitrary prefill, start = 13 != base_offset
        let start = out.len();
        write_aligned_typed_slice_at(&mut out, &data, base_offset);
        let body = &out[start..];

        // Owned decode of the appended body always works.
        assert_eq!(read_aligned_typed_slice::<f64>(body).unwrap(), data);
        // And it borrows when placed at its declared frame offset in an aligned buffer.
        let ab = AlignedBytes::with(body, base_offset);
        let frame = ab.bytes();
        let borrowed = read_aligned_typed_slice_ref::<f64>(&frame[base_offset..]).expect("borrow");
        assert_eq!(borrowed, data.as_slice());
    }

    #[test]
    fn borrow_refuses_unaligned_buffer() {
        let data: Vec<f64> = (0..8).map(|i| i as f64).collect();
        // Array padded for offset 0, then placed at offset 1 in an aligned buffer:
        // the payload's runtime address is now odd, so the pointer check must refuse
        // rather than hand back an unaligned reference.
        let framed = to_vec_aligned_typed_slice(&data);
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
        let aligned = to_vec_aligned_typed_slice(&data);
        assert!(read_aligned_typed_slice::<f32>(&aligned).is_err());
        assert!(read_aligned_typed_slice::<i64>(&aligned).is_err());
    }

    #[test]
    fn empty_slice_roundtrips() {
        let data: Vec<f64> = Vec::new();
        let bytes = to_vec_aligned_typed_slice(&data);
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

    #[test]
    fn rejects_truncated_payload() {
        let data: Vec<f64> = (0..6).map(|i| i as f64).collect();
        let full = to_vec_aligned_typed_slice(&data);
        // Lop bytes off the end so the declared payload runs past the input.
        for cut in 1..=8 {
            let truncated = &full[..full.len() - cut];
            assert!(
                read_aligned_typed_slice::<f64>(truncated).is_err(),
                "truncation by {cut} must be rejected, not read past the buffer"
            );
            assert!(read_aligned_typed_slice_ref::<f64>(truncated).is_err());
        }
        // A header-only prefix (cut before SIZE/PADDING_LENGTH) must also error,
        // never index past the end.
        for keep in 0..4 {
            assert!(read_aligned_typed_slice::<f64>(&full[..keep]).is_err());
        }
    }

    #[test]
    fn rejects_padding_length_past_buffer() {
        let data: Vec<f64> = (0..4).map(|i| i as f64).collect();
        let mut buf = to_vec_aligned_typed_slice(&data);
        // PADDING_LENGTH sits right after marker + numeric header + SIZE.
        let size_w = size_encoded_len(data.len() as u64);
        let pl_offset = 2 + size_w;
        // Claim a wildly oversized padding run; the decoder must bounds-check it
        // (checked_add + `pos > input.len()`) and reject rather than skip past the
        // end into the payload or beyond.
        buf[pl_offset] = 0xFF;
        assert!(read_aligned_typed_slice::<f64>(&buf).is_err());
        assert!(read_aligned_typed_slice_ref::<f64>(&buf).is_err());
    }

    #[cfg(target_endian = "little")] // zero-copy aligned borrow is LE-only
    #[test]
    fn zero_copy_borrow_non_f64_alignments() {
        // Exercise the borrow path for alignments other than 8: align-16 (i128)
        // and align-1 (u8, where the pointer check is trivially satisfied).
        let i: Vec<i128> = (0..20).map(|n| ((n as i128) << 80) | (n as i128)).collect();
        let mut framed = vec![0u8; 16];
        write_aligned_typed_slice(&mut framed, &i);
        let ab = AlignedBytes::with(&framed, 0);
        let view: &[i128] = read_aligned_typed_slice_ref::<i128>(&ab.bytes()[16..]).unwrap();
        assert_eq!(view, i.as_slice());
        assert!((view.as_ptr() as usize).is_multiple_of(16));

        let b: Vec<u8> = (0..37u8).collect();
        let framed_b = to_vec_aligned_typed_slice(&b);
        let ab_b = AlignedBytes::with(&framed_b, 0);
        let view_b: &[u8] = read_aligned_typed_slice_ref::<u8>(ab_b.bytes()).unwrap();
        assert_eq!(view_b, b.as_slice());
    }
}
