# Design: bulk-slice fast path for the streaming serializer

## Status

Implemented (all three layers). See "Implementation notes" below for where the
final code diverged from this proposal.

## Summary

Give contiguous numeric slices a single-`write_all` encoding path on the streaming serializer (`W: Write`), so that encoding *and* measuring a typed numeric body (`&[f64]`, `&[u32]`, …) is O(1) in the element count instead of O(N) per-element.

Today the zero-copy bulk path (`fast::write_typed_slice`) exists only for `Vec<u8>` sinks and is not wired into serde. The serde serializers — both buffered (`src/ser.rs`) and streaming (`src/streaming_ser.rs`) — walk every numeric `Vec<T>` element-by-element. This is the encoder limitation called out in [`serialized-size.md`](serialized-size.md): because the encoder iterates, neither `to_writer_streaming` nor the proposed `serialized_size` can be O(1) for a `Vec<f64>` body; only a `serialize_bytes` (`&[u8]`) body is.

## Motivation

### Current state, confirmed against the code

- **`fast::write_typed_slice<T: BeveTypedSlice>`** (`src/fast.rs:108`) is a true zero-copy bulk write: header + SIZE + one `ptr::copy_nonoverlapping` of the whole payload on little-endian targets, with a per-element fallback on big-endian. But its sink is `&mut Vec<u8>`, and it is reachable only by calling the `fast` free functions directly or via the matrix extension (`src/ext.rs:418`). It is not a `W: Write` path and is not driven by serde.
- **The buffered serde serializer** appends each element separately (`src/ser.rs:1482`, `:1525`, `:1538`: `extend_from_slice(&bytes_le[..bytes])` per element). It does not call `fast::write_typed_slice`.
- **The streaming serde serializer** does the same, one `write_all` per element (`src/streaming_ser.rs:517` `emit_unsigned`, `:543` `emit_float`, …), and does not import `fast` at all.
- **The only single-`write_all` leaf today is `serialize_bytes`** (`src/streaming_ser.rs:248` → `write_bytes_typed_array` `:156`), and it is hardcoded to the `u8` unsigned typed-array header. It cannot represent an `f64`/`i32` slice.
- **`ComplexSlice`** (`src/ext.rs:113`) — the one existing slice *wrapper* — is not a fast path either; it routes through `serialize_seq` and emits elements one at a time.

So a numeric `Vec<T>` (T ≠ u8) is always O(N): N conversions plus N writer calls. The counting sink proposed in `serialized-size.md` removes the byte movement but not the per-element traversal, so the measure is O(N) too.

### Why O(1) is wanted

REPE's headline case is a large binary/numeric body streamed to a socket. With the measure-then-stream plan from `serialized-size.md`, the body is traversed twice (once to size, once to write). If the body is a typed numeric slice, both traversals are O(N) today. A bulk path collapses each to a header write plus one `write_all`, and — combined with an analytic size — makes the *measure* a closed-form O(1) computation with no traversal at all. That is the difference between "cheap to frame a 100 MB `Vec<f64>`" and "two full walks of 12.5M elements."

This is also the deferred optimization that `serialized-size.md` names in "If leaf-level O(1) for generic structures is ever needed" — scoped here to the one leaf shape that matters in practice (a contiguous numeric slice).

## Goals

- A `W: Write` bulk write for any `T: BeveTypedSlice`: header + SIZE + a single `write_all` of the payload on little-endian targets.
- An exact, closed-form O(1) size for such a slice, reusing the SIZE codec so it cannot drift from the bytes actually written.
- A way to reach the bulk path *through serde* — so a typed slice nested inside a `#[derive(Serialize)]` struct, serialized via `to_writer_streaming` / `serialized_size`, also gets it — **without copying the payload** (the copy would defeat both zero-copy streaming and the O(1) measure).
- Byte-for-byte identical output to the existing element-by-element path, so the bulk path is a drop-in encoding of the same BEVE bytes.
- No new allocation on the streaming bulk path.

## Non-goals

- **Auto-detecting a contiguous `Vec<T>` inside a generic `serialize_seq`.** serde hands the serializer one element at a time; it never exposes the backing slice, so the serializer cannot recover "these N elements are a contiguous `&[f64]`." The bulk path is therefore necessarily *opt-in*, the same way `serde_bytes` is opt-in for `&[u8]`. This is a serde data-model limitation, not a beve choice.
- Changing the on-wire format. The bytes are identical to what the per-element path already produces; only the production path changes.
- Big-endian zero-copy. On BE targets the in-memory bytes are not little-endian, so the bulk path falls back to per-element conversion (correct, O(N)). BE is already the slow path in `fast.rs`.

## Design

### Layer 1 — streaming primitive (low risk, unambiguous)

Add a `W: Write` analog of `fast::write_typed_slice`, reusing the existing `BeveTypedSlice` trait (`src/fast.rs:33`) for the class/byte-code/element-size constants:

```rust
/// Write a BEVE typed numeric array straight to a writer in one bulk write
/// (little-endian targets); falls back to per-element on big-endian.
pub fn to_writer_typed_slice<W: Write, T: BeveTypedSlice>(
    mut w: W,
    slice: &[T],
) -> Result<()> {
    // header byte + SIZE prefix
    w.write_all(&[typed_array_header(T::CLASS, T::BYTE_CODE)])?;
    write_size_to_writer(&mut w, slice.len() as u64)?;
    if slice.is_empty() {
        return Ok(());
    }
    #[cfg(target_endian = "little")]
    {
        // Sound: every BeveTypedSlice type is a fixed-width scalar with no
        // padding and all-bit-patterns-valid; this mirrors fast.rs's existing
        // copy_nonoverlapping. size_of_val gives the exact payload length.
        let bytes = unsafe {
            core::slice::from_raw_parts(
                slice.as_ptr() as *const u8,
                core::mem::size_of_val(slice),
            )
        };
        w.write_all(bytes)?;
    }
    #[cfg(not(target_endian = "little"))]
    {
        for v in slice {
            // small stack buffer per element, no allocation
            let mut buf = [0u8; 16];
            let n = write_one_le(v, &mut buf);
            w.write_all(&buf[..n])?;
        }
    }
    Ok(())
}
```

`fast.rs`'s `Vec<u8>` version stays as-is (its `set_len` + single `reserve` avoids the `Write` trait hop for the common buffered case); the two share the header logic. Whether the `Vec` specialization still beats routing `Vec<u8>` through `to_writer_typed_slice` is a benchmark question, not a correctness one — keep both only if the bench shows a delta, otherwise collapse to the `Write` form (consistent with the "optimize with evidence" stance in `serialized-size.md`).

### Layer 2 — analytic size (truly O(1), no traversal)

```rust
/// Exact streaming-encoded length of `to_writer_typed_slice(_, slice)`.
pub fn typed_slice_size<T: BeveTypedSlice>(slice: &[T]) -> u64 {
    1                                   // header byte
        + size_encoded_len(slice.len() as u64) as u64   // SIZE prefix width
        + (core::mem::size_of_val(slice) as u64)         // payload
}
```

`size_encoded_len(n)` is the width the SIZE codec would emit for `n`; extract it from / colocate it with `src/size.rs` so it shares the threshold logic with `encode_size_to_array` and cannot drift. This is the analytic fast path `serialized-size.md` deferred — safe here because it is one leaf with a trivially checkable formula, and it is pinned to the primitive by the equivalence tests below.

For REPE's common case (the body *is* a typed slice), `typed_slice_size` plus `to_writer_typed_slice` is the whole measure-then-stream story with zero traversal on the measure side. This is the simplest, highest-value slice of the feature and could ship alone as Phase 1.

### Layer 3 — serde integration (composition into nested structs)

To reach the bulk path through `to_writer_streaming` / `serialized_size` for a slice that lives inside a derived struct, an opt-in wrapper is needed, mirroring `serde_bytes` / the existing `ComplexSlice`:

```rust
pub struct TypedSlice<'a, T>(pub &'a [T]);   // name bikesheddable
```

The hard constraint: **the existing `RawBytes` + `BytesExtractor` mechanism must not be reused for this.** `BytesExtractor::serialize_bytes` does `v.to_vec()` (`src/ser.rs:187`), so dispatching a typed slice through the `NT_COMPLEX`-style plumbing would copy the entire payload into a temporary `Vec` before writing — re-introducing exactly the O(N) allocation+copy the bulk path exists to remove, and making the measure O(N) again. serde also cannot return a *borrowed* `&[u8]` out of a `Serializer::Ok` (the `serialize_bytes` slice lifetime is not nameable in `Ok`), so "extract a borrowed view and write it later" is not expressible.

The way that *is* expressible: a **writing extractor** — the serializer hands the wrapper a sink that performs the bulk write inside `serialize_bytes` and returns `()`, so the payload is consumed in place and never materialized.

The (class, byte-code) pair travels in the **newtype name**, one name per `BeveTypedSlice` type — `__beve_typed_array_f64`, `__beve_typed_array_u32`, … The name *is* the type tag, so the value the wrapper passes is nothing but the borrowed `&[u8]` (zero copy on LE):

- A macro — the same one that defines the `BeveTypedSlice` impls (`src/fast.rs:44`) — generates, per type: the `NT_TYPED_ARRAY_<T>` constant, the `impl Serialize for TypedSlice<'_, T>`, and an arm of a shared `typed_array_tag(name: &str) -> Option<(u8 /*class*/, u8 /*byte_code*/)>` lookup in `src/ext.rs`. The lookup is the single source of truth for the dispatch, so neither serializer hand-maintains 14 arms.
- `TypedSlice<T>::serialize` (on LE) calls `serialize_newtype_struct(NT_TYPED_ARRAY_<T>, &RawBytes(cast))`, where `cast` is the slice reinterpreted as `&[u8]` — the existing `RawBytes` wrapper (`src/ext.rs:32`) is fine here because it only borrows; it is `BytesExtractor`, not `RawBytes`, that copies.
- The streaming serializer's `serialize_newtype_struct` (`src/streaming_ser.rs:282`) and the buffered one (`src/ser.rs:805`) test the name against `typed_array_tag`. On a hit they dispatch the inner value to a writing-sink serializer whose `serialize_bytes(bytes)` does, in place: write `header(class, byte_code)`, write `SIZE(bytes.len() / elem_size)`, then one `write_all(bytes)` — returning `()`. With `CountingWriter` that `write_all` is a single `len += bytes.len()`, so the measure stays O(1).

Note the element count the SIZE prefix needs is recovered as `bytes.len() / elem_size` (with `elem_size` derived from `byte_code`); the division is exact by construction, so the borrowed byte payload alone carries everything the header needs — no extra fields, no per-call state.

On big-endian targets `TypedSlice` cannot hand over a borrowed little-endian view, so a *non-empty* slice falls back to emitting through `serialize_seq` (the existing per-element path) — correct, O(N), and rare. An empty slice has no payload bytes to byte-swap, so it takes the same typed-array newtype path as little-endian, keeping the empty encoding identical on every target.

The cost of this choice is ~14 generated name constants and lookup arms. That is mechanical (one macro), keeps each serializer's dispatch a single `typed_array_tag` call with no per-call state, and needs no new extractor variant beyond the writing sink — the right trade for a path on the hot loop.

`ComplexSlice` (`src/ext.rs:113`) can later adopt the same mechanism, since `fast.rs` already has a zero-copy `write_complex_slice`; out of scope here but the design should leave room for it.

### Public API

```rust
pub use crate::fast::{ /* existing */ };
pub use crate::streaming_ser::{ to_writer_typed_slice, typed_slice_size };
// Layer 3:
pub use crate::ext::TypedSlice;
```

Returning/treating sizes as `u64` matches `serialized_size` and the wire SIZE domain (see `serialized-size.md`).

## Performance

- **Layer 1 (`to_writer_typed_slice`)** — one header write, one SIZE write, one `write_all` of `N·elem_size` bytes on LE. No per-element dispatch, no allocation. The single `write_all` is whatever the sink costs (a `memcpy` into a `Vec`, a `len +=` into a counter, a syscall-batched socket write).
- **Layer 2 (`typed_slice_size`)** — pure arithmetic, O(1), no traversal, no sink. This is what makes a typed-slice `serialized_size` genuinely O(1) rather than "O(N) without the memcpy."
- **Layer 3 (`TypedSlice` via serde)** — same single `write_all` as Layer 1; the measure is O(1) via the counting sink. The win over a bare `Vec<f64>` field is the whole point: N writer calls → 1.

## Alternatives considered

### 1. Wire `fast::write_typed_slice` into the generic `serialize_seq` path

Detect, inside the seq serializer, that the elements form a contiguous slice and bulk-write. **Infeasible:** serde delivers elements one at a time via `serialize_element`; the backing `&[T]` is never visible to the serializer. The matrix code in `ext.rs` only gets the whole slice because it is a *non-serde* API. Rejected as impossible, not merely undesirable.

### 2. Reuse the `NT_COMPLEX` / `BytesExtractor` plumbing for `TypedSlice`

Simplest to write, but `BytesExtractor` copies (`v.to_vec()`, `src/ser.rs:187`), so it allocates and walks the full payload — O(N) and a heap allocation on every typed-slice field, and the measure stops being O(1). Rejected; the writing-sink extractor (Layer 3) is the cost of staying zero-copy.

### 3. Free functions only, no serde wrapper

Ship Layers 1–2 and stop. Covers REPE bodies that *are* a top-level typed slice (the targeted case) with the least code and clearest O(1) story, but does not compose into nested structs. Reasonable as Phase 1; Layer 3 is the follow-on when a nested typed slice needs the path.

## Testing strategy

1. **Byte-for-byte equivalence.** For every `BeveTypedSlice` type and a range of lengths, assert `to_writer_typed_slice` produces exactly the same bytes as the element-by-element path (`to_writer_streaming(&mut buf, &slice)`), and that both round-trip through `from_slice`.
2. **Analytic-size exactness.** `typed_slice_size(&slice) == bytes.len()` for the same range, including empty slices and lengths straddling the 1/2/4/8-byte SIZE thresholds (63/64, 16383/16384).
3. **serde-path equivalence (Layer 3).** A `TypedSlice<T>` field inside a derived struct serialized via `to_writer_streaming` equals the same struct with a plain `Vec<T>` field, and `serialized_size` of the `TypedSlice` form equals the written length — confirming the bulk path is a pure encoding swap.
4. **No-copy / O(1) guard.** Assert (e.g. via a counting allocator in the test, or by construction review) that the `TypedSlice` streaming path performs no heap allocation and that `serialized_size` of a large `TypedSlice` does not scale with N — pinning the property that distinguishes this from Alternative 2.
5. **Big-endian fallback.** Behind `#[cfg(target_endian = "big")]` (or a byte-swap-simulated unit test of the fallback branch), assert identical bytes.
6. **Cross-check with `fast::to_vec_typed_slice`.** The new `Write` primitive and the existing `Vec` primitive must agree byte-for-byte.

## Risks and mitigations

- **`unsafe` slice→bytes cast.** Confined to LE, mirrors the existing `fast.rs` `copy_nonoverlapping`, and justified by `BeveTypedSlice` being fixed-width no-padding scalars. Covered by the equivalence + BE-fallback tests.
- **Drift between the analytic size and the bytes written.** Mitigated by sharing `size_encoded_len` with the SIZE codec and by test (2) asserting equality against the real primitive. This is the same drift risk `serialized-size.md` flags for any analytic path, gated the same way.
- **`BytesExtractor` copy trap.** The obvious Layer-3 shortcut silently copies; documented above as Alternative 2 and avoided by the writing-sink design.
- **API surface growth.** Three names (`to_writer_typed_slice`, `typed_slice_size`, `TypedSlice`) plus newtype constants. Phaseable: Layers 1–2 stand alone and deliver REPE's hot path before Layer 3 lands.

## Implementation checklist

- [x] Extract/define `size_encoded_len(n: u64) -> usize` in `src/size.rs`.
- [x] Add `to_writer_typed_slice` and `typed_slice_size` (Layers 1–2) in `src/fast.rs`, sharing header logic with the existing `Vec` primitive (both build the byte via the shared `make_header` constructor).
- [x] Re-export both from `src/lib.rs`.
- [x] Tests 1, 2, 6 (and a simulated 5) for Layers 1–2 in `tests/typed_slice.rs`.
- [x] (Layer 3) Macro in `src/ext.rs` generating, per `BeveTypedSlice` type, the `NT_TYPED_ARRAY_<T>` constant, the `Serialize for TypedSlice<'_, T>` impl (LE bulk via borrowed `RawBytes` / BE per-element), and an arm of the shared `typed_array_tag(name) -> Option<(class, byte_code, elem_size)>` lookup.
- [x] (Layer 3) In `serialize_newtype_struct` of both `src/streaming_ser.rs` and `src/ser.rs` (top-level *and* element serializers), dispatch via `typed_array_tag` to a writing-sink serializer that writes header + SIZE(`bytes.len() / elem_size`) + one `write_all` (no payload copy).
- [x] Rustdoc: the opt-in rationale (serde cannot auto-detect contiguous slices), the LE/BE behavior, and the empty-slice divergence.
- [x] Update `serialized-size.md`'s Performance section: a `TypedSlice` body measures in O(1), closing the gap noted there for numeric `Vec<T>`.

## Implementation notes

Where the shipped code refined this proposal:

- **Location.** Layers 1–2 (`to_writer_typed_slice`, `typed_slice_size`) live in
  `src/fast.rs` (next to `write_typed_slice`/`BeveTypedSlice`), not
  `streaming_ser.rs`, and are re-exported from `lib.rs`. The typed-array header
  byte has one definition: every path (the `Vec<u8>` and `W: Write` primitives and
  the serde `write_typed_array_bytes` dispatch) builds it via the shared
  `make_header` constructor in `src/header.rs`.
- **`typed_array_tag` returns `elem_size`, not just `(class, byte_code)`.** The
  element width is **not** always `1 << byte_code`: `bf16` uses `byte_code` 0
  (the brain-float special case) but is 2 bytes wide. The lookup therefore carries
  the true `ELEM_SIZE` so the writing sink recovers the element count as
  `payload.len() / elem_size` correctly for every type.
- **One generic writing sink.** Rather than two bespoke sink serializers, there is
  a single `TypedArrayWriteSink<F>(F)` (in `src/ser.rs`, `pub(crate)`) whose
  `serialize_bytes` forwards the borrowed payload to a closure `F: FnOnce(&[u8])
  -> Result<()>`. Each of the four dispatch sites passes a closure that writes
  `header + SIZE + write_all(payload)` into its own destination (`Serializer`'s
  `Vec` or `StreamingSerializer`'s `W`). No payload copy on any path; with a
  counting sink the body is a single `len +=`.
- **Element-path dispatch too.** Beyond the top-level `serialize_newtype_struct`,
  the *element* serializers (`StreamingElemSer`, `SeqElemSer`) also dispatch via
  `typed_array_tag` (emitting the typed array as a generic-array element, the way
  `NT_RAW_VALUE` is handled), so a `TypedSlice` nested inside a tuple/sequence
  encodes correctly instead of silently mis-encoding as a `u8` array.
- **Empty-slice divergence (refines the byte-for-byte goal).** For a non-empty
  slice the bulk path is byte-identical to the per-element path, as designed. For
  an *empty* slice they differ: `TypedSlice`/`to_writer_typed_slice` emit a typed
  array of the element type (length 0), whereas a bare empty `Vec<T>` has no
  element from which the streaming serializer can detect the type and so emits a
  generic empty array. Both decode back to an empty `Vec<T>`. This is the single
  length at which "byte-for-byte identical to the element-by-element path" does
  not hold, and it is documented on `TypedSlice` and pinned by a test. It holds on
  every target: a big-endian empty `TypedSlice` also takes the typed-array path
  (an empty payload needs no endianness conversion), so it never degrades to a
  generic array.
