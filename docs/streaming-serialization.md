# Streaming Serialization

The streaming serializer writes BEVE data directly to any `std::io::Write`
with zero internal buffering. Every byte flows straight to the writer as it
is produced — no intermediate `Vec<u8>` is allocated.

## When to Use

| Scenario | Recommended API |
|---|---|
| Small/medium payloads, in-memory | `to_vec` / `to_writer` |
| Large payloads, file or network output | `to_writer_streaming` |

Use `to_writer_streaming` when the serialized output is large enough that
buffering it in memory is undesirable (e.g. multi-GB scientific recordings).

## Basic Usage

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

// Write directly to a file
let file = BufWriter::new(File::create("frame.beve")?);
beve::to_writer_streaming(file, &frame)?;
```

Wrap the writer in `BufWriter` for file/network targets to batch syscalls.
Writing to a `Vec<u8>` works too (for testing), but `to_vec` is faster for
that case.

## Typed Array Detection

Homogeneous sequences like `Vec<f64>`, `Vec<u32>`, `Vec<bool>`, and
`Vec<String>` are automatically encoded as compact typed arrays — the same
encoding that `to_vec` produces. The type is detected from the first
element and committed immediately. If a subsequent element has a different
type, serialization returns an error.

This means `to_writer_streaming` produces **byte-for-byte identical output**
to `to_vec` for all standard Rust types.

Supported typed array element types:

| Type | Encoding |
|---|---|
| `u8`–`u128`, `i8`–`i128` | Typed numeric array |
| `f32`, `f64`, `bf16`, `f16` | Typed float array |
| `bool` | Bit-packed boolean array |
| `String`, `&str` | Typed string array |
| `Complex<f32>`, `Complex<f64>` | Typed complex array (extension) |

Heterogeneous sequences (e.g. `Vec<Value>`, tuples) are encoded as generic
arrays where each element carries its own type tag.

## Custom Options

String enum encoding and other serializer options are supported:

```rust
use beve::{SerializerOptions, EnumEncoding};

let opts = SerializerOptions {
    enum_encoding: EnumEncoding::String,
};

let mut buf = Vec::new();
beve::to_writer_streaming_with_options(&mut buf, &frame, opts)?;
```

## Direct Serializer Access

For advanced use cases (e.g. serializing multiple values into the same
writer), use `StreamingSerializer` directly:

```rust
use beve::StreamingSerializer;
use serde::Serialize;

let mut file = std::fs::File::create("multi.beve")?;
let mut ser = StreamingSerializer::new(&mut file);
value1.serialize(&mut ser)?;
value2.serialize(&mut ser)?;
```

## Constraints

### Known-Length Containers Only

The streaming serializer requires all containers to provide their length
upfront. This covers all standard Rust types:

- Structs (field count is compile-time known)
- `Vec<T>`, `[T; N]`, `HashSet<T>`, `BTreeSet<T>` (serde provides `len()`)
- `HashMap<K,V>`, `BTreeMap<K,V>` (serde provides `len()`)
- Fixed-size tuples (arity is compile-time known)

If a custom `Serialize` impl passes `None` for a sequence or map length
(`serialize_seq(None)` or `serialize_map(None)`), serialization returns
an error. This is rare in practice — it only affects custom iterators
that don't know their size.

### Type Mismatch in Sequences

The streaming serializer commits to a typed array encoding after seeing
the first element. If a later element has a different type (e.g. first
element is `u32`, second is `String`), serialization returns a type
mismatch error.

The buffered serializer (`to_vec`) handles this by converting the typed
array back to a generic array. The streaming serializer cannot do this
because the typed header has already been written to the output.

In practice, this only affects heterogeneous `Vec<Value>` or similar
dynamic types. Standard `Vec<T>` for any concrete `T` is always
homogeneous.

## Comparison

| | `to_vec` / `to_writer` | `to_writer_streaming` |
|---|---|---|
| Memory | O(output size) | O(1) |
| Typed arrays | Yes | Yes |
| Heterogeneous fallback | Converts to generic | Error |
| Unknown-length containers | Yes (backpatch) | Error |
| Small payloads | Faster (fewer syscalls) | Slower (per-write syscalls) |
| Large payloads | Slower (allocation) | Faster (no allocation) |

## Error Handling

I/O errors from the underlying writer are propagated as `beve::Error`.
The streaming serializer calls `flush()` on the writer at the end of
`to_writer_streaming`.
