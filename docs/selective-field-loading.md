# Selective Field Loading

Load individual fields from a BEVE payload without deserializing the entire document, using [JSON Pointer (RFC 6901)](https://datatracker.ietf.org/doc/html/rfc6901) paths.

## Overview

When working with large BEVE payloads it is often wasteful to deserialize the
entire document just to read one or two fields. The selective field loading API
navigates directly to the target value, skipping over intermediate data without
allocation:

| Function | Purpose |
|---|---|
| `from_field` | Navigate to a JSON Pointer path and deserialize the value. |
| `from_field_slice` | Navigate to a typed numeric array and deserialize a sub-range. |
| `skip_value` | Advance past a single BEVE value without deserializing it. |

## JSON Pointer Syntax

A JSON Pointer is either the empty string (the entire document) or a `/`-separated
sequence of reference tokens:

| Pointer | Meaning |
|---|---|
| `""` | The entire document |
| `"/name"` | Top-level object field `"name"` |
| `"/a/b/c"` | Nested field lookup: `a` -> `b` -> `c` |
| `"/0"` | Index 0 of a generic array (0-based) |
| `"/2/data"` | Index 2 of a generic array, then field `"data"` |
| `"/5"` | Integer key `5` in an integer-keyed object |

### Escape sequences

Literal `~` and `/` inside a key are escaped as `~0` and `~1` respectively.
Per RFC 6901, `~1` is unescaped before `~0`:

```text
Key        Pointer token
--------   -------------
a/b        a~1b
c~d        c~0d
a~b/c~d    a~0b~1c~0d
```

## Navigable Types

The following BEVE types can appear as intermediate segments in a pointer path:

- **Objects** with string, signed-integer, or unsigned-integer keys
- **Generic arrays** by 0-based element index

Typed arrays, scalars, strings, and extension types (complex, matrix, enum)
can appear as the *leaf* target but cannot be navigated into with further
pointer segments.

## `from_field` -- Extract a Single Value

```rust
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct Sensor {
    id: u32,
    readings: Vec<f64>,
}

#[derive(Serialize, Deserialize)]
struct Payload {
    timestamp: u64,
    sensor: Sensor,
}

let payload = Payload {
    timestamp: 1700000000,
    sensor: Sensor {
        id: 42,
        readings: vec![1.5, 2.7, 3.3],
    },
};
let bytes = beve::to_vec(&payload).unwrap();

// Read just the sensor ID
let id: u32 = beve::from_field(&bytes, "/sensor/id").unwrap();
assert_eq!(id, 42);

// Read the whole readings array
let readings: Vec<f64> = beve::from_field(&bytes, "/sensor/readings").unwrap();
assert_eq!(readings, vec![1.5, 2.7, 3.3]);
```

### Zero-copy borrowing

`from_field` supports the same zero-copy borrowing as `from_slice`. Types
containing `&str` fields borrow directly from the input buffer:

```rust
let name: &str = beve::from_field(&bytes, "/sensor_name").unwrap();
```

### Error conditions

`from_field` returns an error when:

- The pointer string is malformed (not empty and not starting with `/`).
- A pointer segment navigates into a non-container type (number, string, bool, etc.).
- An object key is not found.
- A generic-array index is out of bounds or not a valid non-negative integer.
- The value at the target cannot be deserialized into the requested type.

## `from_field_slice` -- Extract a Sub-Range of a Typed Array

For large typed numeric arrays, you can read only a contiguous slice without
loading the entire array. Elements before the requested range are skipped in
constant time using a byte-offset calculation.

```rust
use serde::Serialize;

#[derive(Serialize)]
struct Recording {
    samples: Vec<f32>,
}

let rec = Recording {
    samples: (0..1_000_000).map(|i| i as f32).collect(),
};
let bytes = beve::to_vec(&rec).unwrap();

// Read only samples [500_000..500_003)
let slice: Vec<f32> = beve::from_field_slice(
    &bytes, "/samples", 500_000, 3
).unwrap();
assert_eq!(slice, vec![500_000.0, 500_001.0, 500_002.0]);
```

### Supported element types

All fixed-width numeric typed arrays: `u8`, `u16`, `u32`, `u64`, `i8`, `i16`,
`i32`, `i64`, `f32`, `f64`.

Boolean and string typed arrays are **not** supported for slicing.

### Parameters

| Parameter | Description |
|---|---|
| `bytes` | The BEVE payload. |
| `pointer` | JSON Pointer to the target typed array. |
| `start` | 0-based index of the first element to read. |
| `count` | Number of elements to read. |

### Error conditions

Returns an error when:

- Navigation to the pointer fails (same rules as `from_field`).
- The target value is not a typed numeric array.
- `start + count` exceeds the array length.

### Note on borrowing

Because `from_field_slice` constructs a temporary buffer internally, the target
type must implement `DeserializeOwned` (it cannot borrow from the input). In
practice this means you'll typically deserialize into `Vec<T>`.

## `skip_value` -- Low-Level Value Skipping

`skip_value` advances a position cursor past one complete BEVE value. It handles
all BEVE types:

- Null and booleans (header-only, zero extra bytes)
- Numbers (1-16 bytes depending on width)
- Strings (size prefix + UTF-8 bytes)
- Objects with string or integer keys
- Typed arrays (numeric, boolean, string)
- Generic arrays
- Extensions: enums, matrices, complex numbers

```rust
let mut buf = beve::to_vec(&42u32).unwrap();
let boundary = buf.len();
buf.extend_from_slice(&beve::to_vec(&"hello").unwrap());

let mut pos = 0;
beve::skip_value(&buf, &mut pos).unwrap();
assert_eq!(pos, boundary); // skipped exactly past the first value
```

This is the building block used internally by `from_field` during navigation.
It can also be used directly when scanning or filtering BEVE payloads.

## Performance Characteristics

- **No allocation** during navigation. `skip_value` advances a byte offset
  without copying or deserializing data.
- **O(1) skip** for typed numeric arrays and complex arrays (computed from
  element count and byte width).
- **O(n) skip** for objects, generic arrays, string typed arrays, and boolean
  typed arrays (must walk each entry to find its end).
- **O(1) element seek** in `from_field_slice` once the array header is parsed
  (byte offset computed directly from `start * elem_size`).
