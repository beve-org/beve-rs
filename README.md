# beve-rs
Rust implementation of the BEVE (Binary Efficient Versatile Encoding) specification with serde support. Fast, robust, and easy to use.

Features
- Direct struct serialization/deserialization via `serde`
- Little-endian, spec-compliant encoding
- Typed arrays for numeric, booleans (bit-packed), and strings when possible
- String and integer keyed objects
- Enum support using BEVE type-tag extension

Fast Paths
- Numeric slices without serde: `beve::to_vec_typed_slice(&[T])` where `T` ∈ `{i8,i16,i32,i64,i128,u8,u16,u32,u64,u128,f32,f64}`
- Boolean slices: `beve::to_vec_bool_slice(&[bool])`
- String slices: `beve::to_vec_str_slice(&[&str])`, `beve::to_vec_string_slice(&[String])`

These functions write BEVE typed arrays directly to the output buffer with no intermediate serde overhead.

Example
```rust
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Point { x: f64, y: f64 }

let p = Point { x: 1.0, y: -2.0 };
let bytes = beve::to_vec(&p).unwrap();
let p2: Point = beve::from_slice(&bytes).unwrap();
assert_eq!(p, p2);
```

Fast path examples
```rust
// Numeric typed array (u32)
let data = [1u32, 2, 3, 4];
let bytes = beve::to_vec_typed_slice(&data);

// Boolean typed array (bit-packed)
let flags = [true, false, true, true];
let bytes = beve::to_vec_bool_slice(&flags);

// String typed array
let words = ["alpha", "beta", "gamma"];
let bytes = beve::to_vec_str_slice(&words);
```

Spec & References
- Specification: reference/beve/README.md
- C++ reference: reference/glaze
- Julia reference: reference/BEVE.jl

Notes
- Float16, bfloat16, and float128 scalar numbers are recognized but not yet mapped to Rust numeric types (use f32/f64). Typed arrays support f32/f64, signed/unsigned up to 128-bit, booleans, and strings.
- Unknown-length sequences and maps are buffered to determine size, then emitted with a single header and compact body.
- The implementation targets correctness and performance without external dependencies beyond `serde`.
