# Typed Arrays

BEVE encodes homogeneous sequences as *typed arrays* -- a compact
representation that stores one header and a contiguous block of
fixed-width elements, rather than repeating a header per element.

## When Typed Arrays Are Used

The serde serializer automatically uses typed arrays when a `Vec<T>` (or
slice) contains a supported element type:

| Element type | BEVE category | Storage |
|---|---|---|
| `u8`, `u16`, `u32`, `u64` | Unsigned | `N * byte_width` contiguous bytes |
| `i8`, `i16`, `i32`, `i64` | Signed | `N * byte_width` contiguous bytes |
| `f32`, `f64` | Float | `N * byte_width` contiguous bytes |
| `bf16`, `f16` | Float (half) | `N * 2` contiguous bytes |
| `bool` | Boolean | `ceil(N / 8)` packed bytes (LSB-first) |
| `String`, `&str` | String | Length-prefixed UTF-8 per element |

All other element types (structs, enums, nested containers) produce
*generic arrays*, which store each element with its own header.

## Fast-Path Helpers

For maximum throughput, bypass serde entirely with the dedicated helpers:

```rust
// Numeric slice -- no serde overhead
let floats = [1.0f32, 3.5, -2.25];
let bytes = beve::to_vec_typed_slice(&floats);

// Boolean slice -- bit-packed
let flags = [true, false, true, true];
let bytes = beve::to_vec_bool_slice(&flags);

// String slice
let names = ["alpha", "beta", "gamma"];
let bytes = beve::to_vec_str_slice(&names);
```

The resulting payloads are byte-identical to what serde produces, so
`beve::from_slice::<Vec<T>>` works on both.

For the decode side, `read_typed_slice` is the bulk counterpart of the
numeric writers -- a single bounds-checked read that moves the whole
contiguous block into a `Vec<T>` in one `copy_nonoverlapping` (on
little-endian targets), the mirror of `to_vec_typed_slice`:

```rust
let bytes = beve::to_vec_typed_slice(&[1.0f64, -2.5, 3.25]);
let back = beve::read_typed_slice::<f64>(&bytes).unwrap();
assert_eq!(back, vec![1.0, -2.5, 3.25]);
```

Like the writer, it is opt-in and requires naming the element type `T`:
serde hands the deserializer one element at a time and never sees the
contiguous block, so the generic `from_slice` path cannot take it
automatically. Use it when the whole body is a numeric array and you want
to skip per-element deserialization; `from_slice::<Vec<T>>` remains the
right choice for an array nested inside a larger structure. The complex
counterpart is `read_complex_slice`.

## Slicing with `from_field_slice`

Typed numeric arrays support efficient sub-range extraction via
`from_field_slice`. Because elements are fixed-width and contiguous, the
library computes a byte offset and reads only the requested elements:

```rust
let data: Vec<f64> = (0..1_000_000).map(|i| i as f64).collect();
let bytes = beve::to_vec(&data).unwrap();

// Read elements [999_000..999_010) -- skips ~8 MB in O(1)
let slice: Vec<f64> = beve::from_field_slice(&bytes, "", 999_000, 10).unwrap();
```

See [selective-field-loading.md](selective-field-loading.md) for full details.

## Wire Format

A typed array is encoded as:

```
[header: 1 byte] [count: 1-8 bytes (SIZE encoding)] [data: count * elem_bytes]
```

The header byte layout is `[byte_count_code:3][subtype:2][type:3]` where
`type = 4` (typed array) and `subtype` indicates the element category
(float, signed, unsigned, or bool/string).

Boolean arrays pack 8 values per byte, LSB-first. String arrays store
each string as a SIZE-prefixed UTF-8 sequence.
