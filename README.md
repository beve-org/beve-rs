# beve-rs
Rust implementation of the BEVE (Binary Efficient Versatile Encoding) specification with serde support. Fast, robust, and easy to use.

Features
- Direct struct serialization/deserialization via `serde`
- Little-endian, spec-compliant encoding
- Typed arrays for numeric, booleans (bit-packed), and strings when possible
- String and integer keyed objects
- Enum support using BEVE type-tag extension

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

Spec & References
- Specification: reference/beve/README.md
- C++ reference: reference/glaze
- Julia reference: reference/BEVE.jl

Notes
- Float16, bfloat16, and float128 scalar numbers are recognized but not yet mapped to Rust numeric types (use f32/f64). Typed arrays support f32/f64, signed/unsigned up to 128-bit, booleans, and strings.
- Unknown-length sequences and maps are buffered to determine size, then emitted with a single header and compact body.
- The implementation targets correctness and performance without external dependencies beyond `serde`.
