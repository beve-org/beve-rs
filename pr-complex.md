# Extend Complex Number Support to All Numeric Scalar Types

## Summary

Previously, `Complex<T>` only supported `f32` and `f64`. This PR extends complex number support to all numeric scalar types: `i8`–`i128`, `u8`–`u128`, and the existing `f32`/`f64`. This applies to both single values and typed arrays, across serialization, deserialization, streaming, and the matrix extension.

## Changes

### Serialization (`ser.rs`, `ext.rs`, `fast.rs`)
- Unified the two separate `NT_COMPLEX32`/`NT_COMPLEX64` newtype markers into a single `NT_COMPLEX` that carries a `class` (float/signed/unsigned) and `byte_code` in its payload.
- Replaced hand-written `Serialize` impls for `Complex<f32>` and `Complex<f64>` with a `impl_complex_serialize!` macro covering all 12 scalar types.
- Generalized `to_vec_complex_slice` and `write_complex_slice` using the `BeveTypedSlice` trait, removing the old type-specific `to_vec_complex32`, `to_vec_complex64`, `to_vec_complex32_slice`, and `to_vec_complex64_slice` functions.
- `write_complex_header` now takes a `class` parameter instead of hardcoding `NUM_FLOAT`.

### Deserialization (`de.rs`, `streaming_de.rs`)
- Removed the `ty != NUM_FLOAT` error guard — the deserializer now dispatches on `class` (float → `parse_f32`/`parse_f64`, signed → `parse_signed`, unsigned → `parse_unsigned`).
- Threaded `class` through `ComplexPairAccess`, `SeqAccessComplexArray`, and `ComplexElemDe`.
- Streaming deserializer updated with the same class-aware dispatch.

### Validation (`field.rs`)
- `skip_value` now validates that the complex class byte is ≤ 2 (float, signed, unsigned), rejecting unknown class values.

### Matrix extension (`ext.rs`, `mat.rs`)
- `try_encode_matrix_extension` expanded from 2 complex type checks (f32/f64) to all 12 scalar types via a `write_complex_value!` macro.
- MAT file writer updated to handle the `class` field in complex headers.

### Foreign type helpers (`ext.rs`)
- New `beve::complex` module with `serialize_with` helpers (`f32_array`, `i16_array`, etc.) for third-party complex types like `num_complex::Complex<T>` that are layout-compatible with `beve::Complex<T>`.

### Public API (`lib.rs`)
- Exports `to_vec_complex_slice`, `write_complex_slice`, and `beve::complex` module.
- Removed `to_vec_complex32`, `to_vec_complex64`, `to_vec_complex32_slice`, and `to_vec_complex64_slice` — use `to_vec_complex_slice` (or `beve::to_vec` for single values) instead.

### Tests
- `tests/basic.rs`: Added tests for integer complex single/vec roundtrips, generic slice encoding, and complex fields embedded in structs.
- `tests/streaming.rs`: Added streaming roundtrip tests for float, integer, and unsigned complex types, including struct embedding.
- `tests/spec.rs`: Updated malformed input test — `0x28` is now a valid signed i16 complex header; added a test for truly invalid class=3.

### Docs
- Updated `README.md` and `docs/complex-and-matrices.md` with integer complex examples, foreign type helper docs, and corrected supported-type lists.

## Test Plan

- [ ] `cargo test` passes all existing and new tests
- [ ] Verify roundtrip for each scalar type (i8–i128, u8–u128, f32, f64) as single values and arrays
- [ ] Verify streaming serialization/deserialization for integer complex types
- [ ] Verify complex fields work correctly inside structs alongside other types
- [ ] Confirm old type-specific functions (`to_vec_complex32_slice`, etc.) are fully removed with no dangling references
