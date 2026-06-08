# Design: unify the bulk-slice LE/BE primitive

## Status

Proposed. No code written. This is the holistic follow-up referenced in the
`read_typed_slice` review (PR #24): the little-/big-endian bulk-copy logic is now
duplicated across the typed and complex readers (and inverted in the writers),
and the question is whether to collapse it behind one audited primitive. This
document scopes that refactor so it can be accepted or declined on its merits,
separately from any feature PR.

## Summary

The contiguous-payload encode/decode in `src/fast.rs` repeats the same
little-endian bulk copy and big-endian per-element byte-reverse in five places.
Replace the duplicated `unsafe` blocks with one (or two) `unsafe` primitive(s) —
a decode side and an encode side — that every public `*_typed_slice` /
`*_complex_slice` function calls, so the `unsafe` reasoning lives in one
documented, exhaustively tested location instead of being re-derived per call
site.

This is a **maintainability / auditability** change. It must be **zero
behavior change** and **byte-for-byte identical** on the wire and in decoded
values, on both endiannesses.

## Motivation

`fast.rs` is the crate's only `unsafe` surface that reinterprets raw bytes as
fixed-width scalars. Today the same shape-dependent reasoning ("every
`BeveTypedSlice` type is a fixed-width, no-padding, all-bit-patterns-valid
scalar, so `N` of them occupy exactly `N * ELEM_SIZE` contiguous little-endian
bytes") is restated at each site. The risk that motivates consolidation: a
future element type, a `Complex`-like 2-scalar aggregate, or a new SIZE/stride
calculation can be added correctly at one site and incorrectly at another,
because nothing forces the five sites to agree.

The review of `read_typed_slice` pushed back on a *read-only* extraction for a
real reason: deduping just the two readers leaves the writers duplicating the
mirror-image logic, so it does not actually achieve "one site," and it pays the
cost of editing a shipped `unsafe` block (`read_complex_slice`) for half the
benefit. The only version worth doing is the holistic one that covers **both**
directions. Hence this doc.

## Current duplication (confirmed against the code)

The same LE-copy / BE-reverse pattern, in `src/fast.rs`:

### Decode (raw bytes -> `Vec<scalar-or-aggregate>`)

- `read_typed_slice` (`src/fast.rs:249`): payload `len * T::ELEM_SIZE` ->
  `Vec<T>`. LE: one `copy_nonoverlapping` + `set_len`. BE: copy, then
  `chunks_exact_mut(T::ELEM_SIZE).reverse()`, then `set_len`.
- `read_complex_slice` (`src/fast.rs:441`): payload `len * 2 * T::ELEM_SIZE` ->
  `Vec<Complex<T>>`. **Identical** body, except the payload stride is doubled and
  the BE reverse still operates on `T::ELEM_SIZE` (per-scalar) chunks across the
  whole payload — i.e. the reversal stride is the *scalar* width, not the element
  (`Complex`) width.

The two decoders differ only in (a) the per-element byte width of the output
`Vec` and (b) the payload-length multiplier (`1` vs `2`). The BE reversal stride
is the scalar width in both.

### Encode (`&[scalar-or-aggregate]` -> bytes)

- `write_typed_slice` (`src/fast.rs:112`): `Vec<u8>` sink. LE:
  `copy_nonoverlapping(slice as bytes -> out)`. BE: per-element `write_one_le`.
- `to_writer_typed_slice` (`src/fast.rs:168`): `W: Write` sink. LE:
  `from_raw_parts(slice as bytes)` + one `write_all`. BE: reused scratch +
  per-element `write_one_le` + `write_all`.
- `write_complex_slice` (`src/fast.rs:380`): `Vec<u8>` sink. LE:
  `copy_nonoverlapping(slice as bytes -> out)` over the doubled payload. BE:
  per-element.

The encoders share the "reinterpret `&[T]` as `&[u8]` of length
`size_of_val(slice)` on LE, fall back to per-element `to_le_bytes` on BE"
reasoning. `to_vec_typed_slice` / `to_vec_complex_slice` are thin allocating
wrappers over the `Vec<u8>` writers and contain no `unsafe`.

## The shared invariant

Every site relies on one property of `T: BeveTypedSlice`:

> `T` is a fixed-width scalar with no padding and every bit pattern valid, so a
> `&[T]` of length `n` is exactly `n * size_of::<T>()` contiguous bytes whose
> little-endian form is the BEVE payload, and a `Vec<T>` allocated with capacity
> `n` has space for exactly that many bytes at `T`'s alignment.

`Complex<T>` is `#[repr(C)]` over two `T`, so it inherits the same property with
`size_of::<Complex<T>>() == 2 * size_of::<T>()` and no padding. That is the one
fact the primitive must document and that the type system should enforce as far
as it can (see Risks).

## Proposed design

Two private `unsafe` primitives in `fast.rs`, one per direction, generic over the
*output/input element* `E` (which is `T` for typed slices and `Complex<T>` for
complex slices), parameterized by the **scalar reversal stride** for the BE path.

### Decode primitive

```rust
/// Bulk-decode `len` elements of type `E` from `src`, which MUST be exactly
/// `len * size_of::<E>()` bytes of little-endian BEVE payload. `scalar_stride`
/// is the width of one fixed-width scalar within `E` (== `size_of::<E>()` for a
/// plain numeric element, `size_of::<E>() / 2` for a `Complex`), used only on
/// big-endian targets to byte-reverse each scalar in place.
///
/// # Safety
/// - `E` must be a fixed-width, no-padding type for which every bit pattern is
///   valid (all `BeveTypedSlice` scalars and `Complex` of them qualify).
/// - `src.len() == len * size_of::<E>()`.
/// - `size_of::<E>() % scalar_stride == 0` and `scalar_stride` is the on-wire
///   scalar width.
unsafe fn bulk_decode_le<E>(src: &[u8], len: usize, scalar_stride: usize) -> Vec<E> {
    let mut out: Vec<E> = Vec::with_capacity(len);
    if len != 0 {
        let dst = out.as_mut_ptr() as *mut u8;
        ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
        #[cfg(not(target_endian = "little"))]
        {
            let bytes = core::slice::from_raw_parts_mut(dst, src.len());
            for s in bytes.chunks_exact_mut(scalar_stride) {
                s.reverse();
            }
        }
        out.set_len(len);
    }
    out
}
```

Callers keep their own header parse, type check, SIZE read, and bounds check
(the cheap, safe, site-specific part) and end with:

```rust
// read_typed_slice
let out = unsafe { bulk_decode_le::<T>(src, len, T::ELEM_SIZE) };
// read_complex_slice
let out = unsafe { bulk_decode_le::<Complex<T>>(src, len, T::ELEM_SIZE) };
```

### Encode primitive(s)

The encoders split by sink (`Vec<u8>` vs `W: Write`), so the cleanest
consolidation is one helper per sink that takes the already-written header and
the element slice:

```rust
/// Append the LE byte image of `slice` to `out` (LE targets: one bulk copy;
/// BE targets: per-element `to_le_bytes`). The header + SIZE prefix must already
/// be written. `# Safety` mirrors `bulk_decode_le`.
unsafe fn bulk_encode_le_vec<E: ???>(out: &mut Vec<u8>, slice: &[E]) { ... }

/// Same, to a `W: Write`, reusing one scratch buffer on BE.
fn bulk_encode_le_writer<W: Write, E: ???>(w: &mut W, slice: &[E]) -> Result<()> { ... }
```

The encode side has a wrinkle the decode side does not: the BE fallback calls
`write_one_le`, which is a `BeveTypedSlice` method defined on the *scalar*, not on
`Complex<T>`. Options, in preference order:

1. Keep the encoders LE-bulk-only in the shared helper and leave each caller's
   BE per-element loop in place (the BE path is already small and rarely hit).
   This consolidates the LE `unsafe` reinterpretation — the only `unsafe` in the
   encoders — and leaves the safe BE loops alone. **Lowest risk, recommended.**
2. Add a scalar-granular `write_scalars_le` to the trait and express the BE path
   over scalars for both typed and complex. More unified, but grows the trait and
   touches more shipped code.

Recommendation: **option 1** for the encode side. The `unsafe` is the LE
reinterpretation; that is what is worth centralizing. The BE per-element loops
are safe code and not where a soundness bug would hide.

## Non-goals

- Changing the wire format or any decoded value. Byte-for-byte identical output
  on both endiannesses is a hard requirement.
- Touching `read_size` / header parsing / bounds checks. Those stay inline at
  each call site; they are the safe, site-specific part and moving them would
  reduce local clarity.
- Public API change. All new helpers are private; the `*_typed_slice` /
  `*_complex_slice` signatures and `lib.rs` re-exports are unchanged.
- Bool/string arrays (`write_bool_slice`, `write_str_slice`, ...). Those use
  bit-packed / length-prefixed layouts, not a contiguous fixed-width block, and
  are not part of this primitive.

## Why this is a judgment call, not an obvious win

Stated plainly so a reviewer can weigh it:

- **For:** one documented `# Safety` contract and one tested `unsafe` site per
  direction instead of five; future element types and stride changes are forced
  through it.
- **Against:** a generic helper has a *contract* each caller must uphold
  (`src.len() == len * size_of::<E>()`, correct `scalar_stride`, `E` is POD). Two
  concrete blocks whose correctness is obvious from the bounds check three lines
  above are arguably easier to audit than one helper reached through a contract
  you re-verify per call. The win is real only if the helper's contract is
  *simpler to check at the call site* than the inlined block it replaces — which
  is true for the decode side (the call sites already compute `len` and `payload`
  and bounds-check them) and weaker for the encode side (hence option 1 there).

If the decision is to proceed, do the **decode** consolidation (clear win, two
near-identical shipped blocks -> one) and the **encode LE** consolidation
(option 1), and stop there.

## Risk and the strongest mitigation

The contract is currently upheld by comments. The single highest-value
improvement this refactor can bake in: make the primitive **assert the length
relationship** in debug builds, so a future caller that gets the stride or
multiplier wrong trips a `debug_assert!` instead of silently reading/writing the
wrong number of bytes.

```rust
debug_assert_eq!(src.len(), len * core::mem::size_of::<E>());
debug_assert_eq!(core::mem::size_of::<E>() % scalar_stride, 0);
```

This converts the most likely future mistake (a stride/multiplier mismatch when
adding a type) from undefined behavior into a test failure, and is itself an
argument *for* centralizing: you cannot put one assert in five places and expect
them to stay consistent.

A `bytemuck`-style `Pod` bound on `E` would let the type system enforce the POD
half of the contract, but `Complex<T>` would need the derive and the crate would
take a (default-off, optional) dependency; out of scope unless `bytemuck` is
wanted for other reasons.

## Testing strategy

The refactor is correct iff it changes nothing observable, so the test bar is
*differential*, not additive:

- The existing `tests/fast.rs` round-trip + rejection suites (now covering
  `u8..u128` / `i8..i128` / `f32` / `f64` / `f16` / `bf16`, empty, complex
  `i128`/`u128`, and the bf16-vs-f16 / wrong-width / wrong-class / truncation
  rejections) must pass unchanged.
- Add a byte-for-byte assertion that each refactored encoder produces output
  identical to a frozen pre-refactor fixture (golden bytes), so an accidental
  layout change cannot pass by being internally self-consistent.
- Run the full suite under Miri for the typed and complex round-trips to validate
  the `unsafe` (alignment, provenance, `set_len` initialization) at both the old
  and new sites.
- A big-endian target in CI (cross or emulated) exercises the reverse path; if BE
  CI is not available, keep the BE logic a literal move of the current code and
  note that it is unverified on hardware, as today.

## Implementation checklist

1. Add `bulk_decode_le<E>` with the `# Safety` doc + `debug_assert`s.
2. Route `read_typed_slice` and `read_complex_slice` through it; delete their
   inline `unsafe` blocks. Confirm `tests/fast.rs` green + Miri clean.
3. Add `bulk_encode_le_vec` / `bulk_encode_le_writer` (option 1: LE bulk only).
4. Route `write_typed_slice`, `to_writer_typed_slice`, `write_complex_slice`
   through them; keep each BE per-element loop. Confirm golden-byte equality.
5. Re-run benches (`numeric_arrays_f64`, `complex_arrays_f64`) to confirm no
   regression — the LE path should be the same instructions, just behind a call
   that inlines.

## Decision

Deferred pending a "yes, worth it" from the maintainer. This is a pure
maintainability refactor on shipped `unsafe` code; it earns its place only if the
team values the single audited site over the current locally-obvious blocks. The
`debug_assert` mitigation is the part that would still be worth doing even if the
rest is declined — it could be added to the existing five sites independently.
