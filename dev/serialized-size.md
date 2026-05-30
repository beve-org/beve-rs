# Design: `serialized_size` — a measuring serializer for BEVE

## Status

Implemented (beve side). The remaining repe-side adoption (last checklist item) is
tracked separately.

## Summary

Add a way to compute the exact number of bytes a value will occupy when encoded
with the streaming serializer, **without** producing those bytes:

```rust
pub fn serialized_size<T: Serialize>(value: &T) -> Result<u64>;
pub fn serialized_size_with_options<T: Serialize>(value: &T, opts: SerializerOptions) -> Result<u64>;
```

The returned length is, by construction, exactly what `to_writer_streaming`
would write for the same value and options. This unlocks a true single-encode,
zero-intermediate-buffer streaming path for length-prefixed framing protocols —
most immediately REPE — over non-seekable transports.

## Motivation

### The framing problem

REPE's `write_message_streaming` (`repe-rs/src/io.rs`) writes a frame header that
contains `body_length` *before* it invokes the body-writer closure, and it
documents that the closure must emit *exactly* `body_len` bytes or the frame on
the wire is corrupt (a short body blocks the reader; a long body desyncs every
subsequent frame).

To know the body length before writing the body, over a non-seekable transport
(a socket), there are only three options:

1. **Measure** the body up front, then stream it. Requires a measuring
   serializer — which does not exist today.
2. **Buffer** the whole body first (`to_vec`, then write). This is exactly the
   intermediate `Vec<u8>` the streaming path was created to avoid, and it forces
   the entire body to reside in memory at once.
3. **Reserve then back-patch** the length field after writing the body. Requires
   `Seek`, which a socket does not provide.

Today only option 2 is available, so the advertised "zero intermediate `Vec<u8>`"
streaming path is unreachable for its headline use case: a large binary body sent
to a socket.

### Current state in beve

- `to_writer` / `to_writer_with_options` (`src/lib.rs`) serialize into an
  intermediate `Vec<u8>` and then copy the bytes to the writer. They are a
  convenience wrapper, not zero-copy.
- `to_writer_streaming` / `to_writer_streaming_with_options` (`src/streaming_ser.rs`)
  encode directly into a `W: Write` in a single pass with no intermediate buffer.
  They require every container to have a known length.
- `src/size.rs` is **not** a measuring facility. It is the variable-length SIZE
  integer codec (the varint that prefixes containers): `write_size`,
  `encode_size_to_array`, `read_size`. It encodes a count into bytes; it does not
  compute how many bytes a value will become.

There is no function that answers "how many bytes will this value encode to?"
This document proposes that function.

### Why measuring is the right tool here

An earlier idea — "stream into the real writer, and on failure fall back to
buffering" — was rejected because it is unsafe (an un-rewindable writer can be
left with partial bytes) and, in its safe sink-probe form, does a full **encode**
twice. A measuring serializer replaces that double-encode with one measure pass
plus one real encode. Note the measure pass is not intrinsically cheaper than the
sink-probe pass it replaces — a counting sink and `io::sink()` both discard bytes,
so for a generic structure both do the same full traversal. The win is that the
measure pass *yields the length* instead of throwing the pass away (see
Alternative 3), and that for a `serialize_bytes` body the measure is genuinely
O(1) while the real encode is O(payload) (see Performance). The result is a
single-encode with zero buffering.

## Goals

- Compute the exact streaming-encoded byte length of any value the streaming
  serializer can encode, for both default and custom `SerializerOptions`.
- Guarantee the measured length always equals what `to_writer_streaming` writes,
  with no risk of the two drifting apart as the encoder evolves.
- Reject the same values `to_writer_streaming` rejects (unknown-length
  containers), with the same errors, so callers can rely on a measured value
  being streamable.
- Add no allocation on the measuring path.

## Non-goals

- Measuring the **buffered** (`to_vec`) encoding. For values containing
  unknown-length containers, `to_vec` may differ from the streaming encoding
  (the streaming serializer rejects those outright). `serialized_size` measures
  the *streaming* encoding only; it is the dual of `to_writer_streaming`, not of
  `to_vec`. This keeps a single, well-defined contract.
- Seek-based back-patching of length fields. Out of scope; a separate technique
  for seekable sinks.
- A guaranteed-O(1) size for arbitrary nested structures. Sizing a generic
  structure is inherently O(payload) (see Performance).

## Design

### Core idea: count, don't encode

The streaming serializer is already parameterized over `W: Write`. The encoding
logic lives entirely in `impl<W: Write> ser::Serializer for &mut StreamingSerializer<W>`
and its helpers (`write_byte`, `write_size`, `write_str_value`,
`write_bytes_typed_array`, the typed-array detection in `SeqMode`, the
complex/raw-value extensions, enum-tag encoding, bf16/f16 handling, etc.).

Rather than reimplement all of that to *count* bytes — which would create a second
encoding path that must be kept byte-for-byte in sync with the first, forever — we
reuse the existing encoder and simply give it a writer that discards everything
and counts.

```rust
/// A `Write` that discards all bytes and counts how many it was given.
#[derive(Default)]
pub(crate) struct CountingWriter {
    len: u64,
}

impl std::io::Write for CountingWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.len += buf.len() as u64;
        Ok(buf.len())
    }
    fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()> {
        self.len += buf.len() as u64;
        Ok(())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
```

The streaming serializer routes every write through `write_all` (`write_byte`,
`write_size`, and the `write_all` helper all call `writer.write_all`), so the
`write` override above is never exercised on this path. It is kept only so
`CountingWriter` is a correct `Write` in its own right; the default `write_all`
would also work given `write` always reports the full length.

`serialized_size` then becomes a thin wrapper:

```rust
pub fn serialized_size<T: Serialize>(value: &T) -> Result<u64> {
    serialized_size_with_options(value, SerializerOptions::default())
}

pub fn serialized_size_with_options<T: Serialize>(
    value: &T,
    opts: SerializerOptions,
) -> Result<u64> {
    let mut counter = StreamingSerializer::with_options(CountingWriter::default(), opts);
    value.serialize(&mut counter)?;
    Ok(counter.into_inner().len)
}
```

### Why this is correct by construction

The measured length is produced by the *same code* that writes the real bytes.
There is exactly one encoding path. It is impossible for the measure to disagree
with `to_writer_streaming`, because the measure *is* `to_writer_streaming` with a
counting sink instead of a real writer. Every encoder change — a new extension, a
tweak to typed-array detection, a different enum tag layout — is automatically
reflected in both, with no separate code to update.

This also means the measure inherits the streaming serializer's validation: a
value containing an unknown-length sequence or map produces the identical
`Error::Unsupported` it would on a real write, so a successful `serialized_size`
guarantees the subsequent `to_writer_streaming` will not *reject* the value
(no `Unsupported`/`Mismatch`) — a property REPE relies on. It does **not**
guarantee the write itself succeeds: an I/O failure mid-body still errors after
emitting fewer than `body_len` bytes, corrupting the frame exactly as the framing
problem above describes. That remains the framing layer's responsibility (tear
down the connection on a mid-body write error); the measure removes the
*serialization-divergence* failure mode, not the *I/O* one.

Two further conditions for the count to match the real write, both trivially held
by the REPE call site: the value's `Serialize` impl must be deterministic across
the two calls, and the caller must not mutate the value between measuring and
streaming (REPE borrows `&value` across both). The trailing `flush()` that
`to_writer_streaming` performs emits no bytes (and `Vec`'s flush is a no-op), so
its absence on the measuring path does not affect the count.

### Public API

Exported from `src/lib.rs` alongside the existing streaming functions:

```rust
pub use crate::streaming_ser::{
    StreamingSerializer, serialized_size, serialized_size_with_options,
    to_writer_streaming, to_writer_streaming_with_options,
};
```

Returning `u64` (not `usize`) matches REPE's `body_len: u64` and the on-wire SIZE
domain, and keeps the value meaningful on 32-bit targets. The name and signature
also follow the established precedent of `bincode::serialized_size`, which is
likewise `fn serialized_size<T: Serialize>(&T) -> Result<u64>` — so the API will
read as familiar to anyone who has used that crate.

### REPE integration

The call site the feature is designed for becomes a clean single-encode:

```rust
let body_len = beve::serialized_size(&value)?;
write_message_streaming(w, header, query, body_len, |w| {
    beve::to_writer_streaming(w, &value).map_err(RepeError::from)
})?;
```

Because `serialized_size` and `to_writer_streaming` share one encoder, the
closure is guaranteed to emit exactly `body_len` bytes, satisfying
`write_message_streaming`'s contract. This belongs in repe (and requires repe to
bump its `beve` dependency to a version that includes this feature); it is shown
here only to validate the API shape.

## Performance

The cost of `serialized_size` is one traversal of the value through the encoder,
writing into a counter instead of a buffer. There is no allocation and no byte
movement (the counting `write` is an integer add).

Crucially, the cost depends on the value's shape — and on *which encoder path* a
given leaf takes, which is not always obvious from the Rust type:

- **A `serialize_bytes` body is O(1).** `serialize_bytes` →
  `write_bytes_typed_array` (`streaming_ser.rs`) issues a single
  `write_all(bytes)`. With a counting writer that is one `len += bytes.len()`; the
  element bytes are never iterated. This is the case the feature targets, and it
  requires the body to serialize *through* `serialize_bytes`: a `&[u8]`,
  `serde_bytes::Bytes`, or `ByteBuf`.
- **A numeric `Vec<T>` is O(N), even though it encodes as a compact typed array.**
  serde's generic `Vec` impl drives `serialize_seq` + per-element
  `serialize_element`, landing in `emit_unsigned`/`emit_float`, each of which calls
  `write_all` **once per element**. The counting writer discards the bytes but the
  encoder still performs N conversions and N write calls. So measuring a bare
  `Vec<u8>`/`Vec<f64>` is the same traversal order as encoding it — it merely skips
  the memcpy and the allocation. It does **not** collapse to a constant.
- **Generic structures are O(payload).** A struct with string fields, a map, or a
  heterogeneous/generic array must be walked element-by-element to sum sizes
  (string lengths vary, nesting varies). For such values the measure is a second
  traversal whose cost is proportional to the data. This is the same tradeoff
  family as any "measure then write" scheme and is acceptable: the alternative
  (buffering) also traverses once and additionally allocates and holds the whole
  body.

So the honest characterization is: **O(1) for a `serialize_bytes` body or a
`TypedSlice<T>` body; O(N)/O(payload) for a bare numeric `Vec<T>` and general
nested structure.** The practical consequence for REPE: to get the O(1) measure on
a large numeric body, that body must serialize through a single bulk write — a
`&[u8]`/`serde_bytes` blob (`serialize_bytes`), or a `beve::TypedSlice<T>` for a
contiguous numeric slice. A bare `Vec<u8>`/`Vec<f64>` body is still measured in
O(N) — correct, but not O(1) — because serde drives it element-by-element and
never exposes the backing slice.

This used to be a hard encoder limitation. It no longer is: the **bulk-slice fast
path** (see [`bulk-slice-fast-path.md`](bulk-slice-fast-path.md), implemented)
adds `to_writer_typed_slice` (one `write_all` for a contiguous numeric slice),
the analytic O(1) `typed_slice_size`, and the opt-in `TypedSlice<T>` wrapper that
reaches the bulk write through serde. With it, both *encoding* and *measuring* a
`TypedSlice<T>` body are O(1) (the counting sink turns the body's single
`write_all` into one `len +=`), and `typed_slice_size` is a closed-form measure
with no traversal at all. The wrapper is opt-in for the reason in the non-goals:
serde cannot auto-detect that N elements are a contiguous `&[f64]`.

### If leaf-level O(1) for generic structures is ever needed

The counting-sink design walks the encoder, so even blob sizing technically runs
through `serialize_bytes` once. If profiling ever shows the traversal of a large
*structure* (many fields, deep nesting) to be a bottleneck, analytic fast paths
could be layered on top — e.g. a specialized size for `&[u8]` / typed slices that
returns `header + size_of(len) + n` arithmetically. This is deliberately deferred:
it reintroduces a second sizing path that must stay in sync with the encoder, and
should only be added where measured benefit justifies that maintenance cost. Start
correct-by-construction; optimize with evidence.

## Alternatives considered

### 1. Hand-written analytic `serialized_size` (no shared encoder)

A standalone function that computes sizes arithmetically from the start.

- **Pro:** O(1) for blobs and typed arrays without traversing element bytes.
- **Con:** A second full encoding model that must reproduce, byte-for-byte,
  everything `to_writer_streaming` does (typed-array detection, complex/raw
  extensions, enum tags, bf16/f16). The two **will** drift, and when they do every
  REPE frame desyncs silently. Rejected as the *starting* design; viable only as a
  later, evidence-driven optimization on top of the correct-by-construction base.

### 2. Reusable scratch buffer (`to_vec_into` into a caller-owned `Vec`)

If literal zero-buffering is not a hard requirement, REPE could keep a `Vec<u8>`
and reuse it across messages via the existing `to_vec_into`:

```rust
scratch.clear();
beve::to_vec_into(&mut scratch, &value)?;
write_message(w, /* body = */ &scratch)?;
```

- **Pro:** Single traversal, single encode, amortized-zero allocation, needs no
  new beve API.
- **Con:** Holds the entire body in memory at once. Fine for small/medium bodies,
  not for very large ones, and not "zero buffer" on principle.

This is a legitimately good option when bodies are bounded; `serialized_size` wins
specifically when the body must not be fully resident in memory or when true
zero-buffering is required. The two are complementary, not competing.

### 3. Sink-probe then stream (previously rejected)

Serialize once to `std::io::sink()` to detect streamability, then stream for real.
Does a full **encode** twice (the sink pass does all the byte work). `serialized_size`
supersedes this: the measure is the useful product of that first pass, and the
caller gets a length out of it rather than throwing the pass away.

## Testing strategy

The central invariant is exactness, so the primary tests assert
`serialized_size == to_writer_streaming output length`:

1. **Equivalence (property + table).** For a wide range of values — scalars,
   strings (incl. multi-byte UTF-8), empty and non-empty typed arrays of every
   numeric type, bool arrays (bit-packed), string arrays, generic/heterogeneous
   arrays, tuples, nested structs, maps with string and integer keys, options,
   all enum variant kinds, bf16/f16, complex, raw values — assert:

   ```rust
   let mut buf = Vec::new();
   beve::to_writer_streaming(&mut buf, &value).unwrap();
   assert_eq!(beve::serialized_size(&value).unwrap(), buf.len() as u64);
   ```

   A `proptest` generator over a representative `Value`-like type gives broad
   coverage cheaply.

2. **Options parity.** Same equivalence under `EnumEncoding::String` vs the default
   `EnumEncoding::Number`, since enum tag width affects size.

3. **Rejection parity.** Values with unknown-length containers
   (`serialize_seq(None)` / `serialize_map(None)`) must return the same
   `Error::Unsupported` from `serialized_size` as from `to_writer_streaming`.

4. **SIZE-prefix boundaries.** Lengths straddling the 1/2/4/8-byte SIZE encoding
   thresholds (e.g. 63 vs 64, 16383 vs 16384 elements) to confirm the prefix width
   is counted correctly.

5. **Path-distinction guard.** A `serde_bytes` body and a bare `Vec<u8>` of the
   same contents both measure to the same byte length (assert it), with a comment
   recording that they reach that length via different encoder paths
   (`serialize_bytes` → O(1) vs `serialize_seq` → O(N)). The test pins the size
   equivalence so nobody "optimizes" by swapping the body type and silently
   regresses the measure from O(1) to O(N) without the equality failing.

6. **REPE integration (in repe, once the dep is bumped).** A round-trip asserting
   the streamed frame's `body_length` equals the bytes the closure actually wrote,
   and that the frame reads back correctly.

## Risks and mitigations

- **Drift between measure and encode.** Eliminated by construction — there is one
  encoder. Any future analytic fast path (Alternative 1) reintroduces this risk and
  must be gated behind the equivalence tests above.
- **`u64` vs `usize` mismatch.** API returns `u64` to match the wire/REPE domain
  and 32-bit targets; conversion happens once at the boundary.
- **Misuse as a `to_vec` size oracle.** Documented clearly: `serialized_size` is
  the dual of `to_writer_streaming`, not of `to_vec`, and rejects unknown-length
  containers accordingly.

## Implementation checklist

- [x] Add `CountingWriter` (private) to `src/streaming_ser.rs`.
- [x] Add `serialized_size` / `serialized_size_with_options` in `src/streaming_ser.rs`.
- [x] Re-export both from `src/lib.rs`.
- [x] Tests per the strategy above (new `tests/serialized_size.rs`).
- [x] Rustdoc on both functions: exact-dual-of-`to_writer_streaming` contract,
      the leaf-O(1)/structure-O(payload) cost note, and the REPE usage example.
- [ ] (repe, separate change) bump the `beve` dependency and adopt the
      measure-then-stream call site.
```
