# Streaming I/O

The streaming APIs read and write BEVE data directly from `std::io::Read` /
`std::io::Write` with zero internal buffering. No intermediate allocations
are made beyond the output values themselves.

## When to Use

| Scenario | Recommended API |
|---|---|
| Small/medium payloads, in-memory | `to_vec` / `from_slice` |
| Large payloads, file or network I/O | `to_writer_streaming` / `from_reader_streaming` |

Use the streaming APIs when the payload is large enough that buffering it
entirely in memory is undesirable (e.g. multi-GB scientific recordings).

## Serialization

```rust
use serde::Serialize;
use std::io::BufWriter;
use std::fs::File;

#[derive(Serialize)]
struct Frame {
    timestamp: f64,
    channels: Vec<f64>,
}

let frame = Frame {
    timestamp: 1.0,
    channels: vec![0.1, 0.2, 0.3],
};

let file = BufWriter::new(File::create("frame.beve")?);
beve::to_writer_streaming(file, &frame)?;
```

Wrap the writer in `BufWriter` for file/network targets to batch syscalls.

### Typed Array Detection

Homogeneous sequences like `Vec<f64>`, `Vec<u32>`, `Vec<bool>`, and
`Vec<String>` are automatically encoded as compact typed arrays -- the same
encoding that `to_vec` produces. The type is detected from the first
element and committed immediately.

This means `to_writer_streaming` produces **byte-for-byte identical output**
to `to_vec` for all standard Rust types.

Supported typed array element types:

| Type | Encoding |
|---|---|
| `u8`--`u128`, `i8`--`i128` | Typed numeric array |
| `f32`, `f64`, `bf16`, `f16` | Typed float array |
| `bool` | Bit-packed boolean array |
| `String`, `&str` | Typed string array |
| `Complex<f32>`, `Complex<f64>` | Typed complex array (extension) |

Heterogeneous sequences (e.g. `Vec<Value>`, tuples) are encoded as generic
arrays where each element carries its own type tag.

### Custom Options

```rust
use beve::{SerializerOptions, EnumEncoding};

let opts = SerializerOptions {
    enum_encoding: EnumEncoding::String,
};

let mut buf = Vec::new();
beve::to_writer_streaming_with_options(&mut buf, &frame, opts)?;
```

### Serialization Constraints

**Known-length containers only.** The streaming serializer requires all
containers to provide their length upfront. This covers all standard Rust
types (`Vec`, `HashMap`, structs, tuples, etc.). Custom `Serialize` impls
that pass `None` for sequence/map length will get an error.

**Type mismatch errors.** The streaming serializer commits to a typed array
encoding after the first element. If a later element has a different type,
serialization returns an error — the message names the expected element type
and suggests using `to_vec` instead. The buffered serializer (`to_vec`)
handles this by converting back to a generic array; the streaming serializer
cannot because the typed header has already been written. In practice this
affects `Vec<Value>` and similar dynamic types.

**`Vec<Option<T>>` is not supported.** Serde serializes `Some(v)` by
delegating directly to `v`, so the first `Some(42u32)` commits to a typed
`u32` array. A subsequent `None` emits a null, which triggers a type mismatch
error. This happens even when all elements are `Some`. Use `to_vec` for
sequences of optional values.

## Deserialization

```rust
use serde::Deserialize;
use std::io::BufReader;
use std::fs::File;

#[derive(Deserialize)]
struct Frame {
    timestamp: f64,
    channels: Vec<f64>,
}

let file = BufReader::new(File::open("frame.beve")?);
let frame: Frame = beve::from_reader_streaming(file)?;
```

Wrap the reader in `BufReader` for file/network targets to batch syscalls.

Each value is read incrementally from the reader -- typed array elements
are read directly into the output with no temporary buffer. Strings are
allocated as owned `String` values (zero-copy borrowing is not possible
when reading from a stream; use `from_slice` for zero-copy).

## Direct Access

For advanced use cases (e.g. multiple values in the same stream), use the
serializer/deserializer structs directly:

```rust
use beve::{StreamingSerializer, StreamingDeserializer};
use serde::{Serialize, Deserialize};

// Write two values
let mut file = std::fs::File::create("multi.beve")?;
let mut ser = StreamingSerializer::new(&mut file);
value1.serialize(&mut ser)?;
value2.serialize(&mut ser)?;

// Read them back
let mut file = std::fs::File::open("multi.beve")?;
let mut de = StreamingDeserializer::new(&mut file);
let v1 = Type1::deserialize(&mut de)?;
let v2 = Type2::deserialize(&mut de)?;
```

## Comparison

| | `to_vec` / `from_slice` | Streaming |
|---|---|---|
| Memory | O(payload size) | No auxiliary buffers (output values like `String` and `Vec` are still allocated) |
| Typed arrays | Yes | Yes |
| Zero-copy strings | Yes (`&str` from buffer) | No (allocates `String`) |
| Heterogeneous fallback | Converts to generic | Error (serialization) |
| Unknown-length containers | Yes (backpatch) | Error (serialization) |

## Error Handling

I/O errors from the underlying reader/writer are propagated as
`beve::Error`. The streaming serializer calls `flush()` on the writer at
the end of `to_writer_streaming`.
