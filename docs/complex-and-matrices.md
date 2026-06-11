# Complex Numbers and Matrices

BEVE includes extension types for complex numbers and dense matrices,
common in scientific computing workflows.

## Complex Numbers

The `Complex<T>` type represents a complex number with real and imaginary
parts. All numeric scalar types are supported: `f32`, `f64`, `i8`–`i128`,
and `u8`–`u128`.

```rust
use beve::Complex;

let z = Complex { re: 1.0f64, im: -0.5 };
let bytes = beve::to_vec(&z).unwrap();
let back: Complex<f64> = beve::from_slice(&bytes).unwrap();
assert_eq!(z, back);

// Integer complex types work the same way
let zi = Complex { re: -3i32, im: 4 };
let bytes = beve::to_vec(&zi).unwrap();
let back: Complex<i32> = beve::from_slice(&bytes).unwrap();
assert_eq!(zi, back);
```

Complex fields work naturally inside structs alongside other types:

```rust
use serde::{Serialize, Deserialize};
use beve::Complex;

#[derive(Serialize, Deserialize, PartialEq, Debug)]
struct Signal {
    label: String,
    sample: Complex<f32>,
    coefficients: Vec<Complex<i16>>,
    gain: f64,
}
```

### Complex Arrays

Vectors of `beve::Complex<T>` automatically serialize as typed complex
arrays via serde. For direct encoding without serde, use the fast-path
helpers:

```rust
use beve::Complex;

let arr = [
    Complex { re: 1.0f64, im: 2.0 },
    Complex { re: 3.0, im: 4.0 },
];
let bytes = beve::to_vec_complex_slice(&arr);
let back: Vec<Complex<f64>> = beve::from_slice(&bytes).unwrap();
assert_eq!(back, arr);
```

`from_slice` decodes a complex array element-by-element through serde. When
the element type is known up front, `read_complex_slice` is the bulk decode
counterpart of `to_vec_complex_slice`: it validates the header and then reads
the whole contiguous little-endian block in a single bounds-checked copy,
which is dramatically faster for large arrays (see `benches/serialization.rs`):

```rust
use beve::Complex;

let arr = [
    Complex { re: 1.0f64, im: 2.0 },
    Complex { re: 3.0, im: 4.0 },
];
let bytes = beve::to_vec_complex_slice(&arr);
let back = beve::read_complex_slice::<f64>(&bytes).unwrap();
assert_eq!(back, arr);
```

Like the writer, it is opt-in and requires you to name the element type,
because serde's `Deserialize` for `Vec<Complex<T>>` never exposes the backing
block to the deserializer. It errors if the bytes are not a complex array, if
the on-wire element type does not match `T`, or if the payload is truncated.

`read_complex_slice` decodes a stream whose *whole body* is a complex array.
For a complex array that is a **field inside a struct**, use the
`#[serde(with = "beve::complex_array::<scalar>")]` helpers (see
[Foreign Complex Types](#foreign-complex-types)) — they apply the same bulk
decode through serde, which the plain `Vec<Complex<T>>` field path cannot.

`to_vec_complex_slice` / `read_complex_slice` work for any supported scalar type:

```rust
use beve::Complex;

let arr = [Complex { re: 1i32, im: -2 }, Complex { re: 3, im: 4 }];
let bytes = beve::to_vec_complex_slice(&arr);
let back: Vec<Complex<i32>> = beve::from_slice(&bytes).unwrap();
assert_eq!(back, arr);
```

### Foreign Complex Types

If you use a third-party complex type like `num_complex::Complex<T>`,
Rust's orphan rules prevent beve from providing serde impls directly.
The `beve::complex_array::<scalar>` modules are `#[serde(with = ...)]`
helpers that give a struct field a compact BEVE complex array **both
directions** — and, crucially, bulk (memcpy) **decode** straight into
the field, which the per-element serde path cannot do for a nested
`Vec<Complex<T>>`:

```rust
use num_complex::Complex32;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct Signal {
    #[serde(with = "beve::complex_array::f32")]
    pub buffer: Vec<Complex32>,
}
```

Available modules: `complex_array::{f32, f64, i8, i16, i32, i64, i128,
u8, u16, u32, u64, u128}`.

The element type must be layout-compatible with `beve::Complex<T>`
(two contiguous `T` fields: re then im) **and** implement
[`bytemuck::AnyBitPattern`] — the bound that makes the bulk byte
reinterpret sound. `num_complex::Complex<T>` satisfies both once you
enable its `bytemuck` feature:

```toml
num-complex = { version = "0.4", features = ["bytemuck"] }
```

These helpers are beve-specific on the binary wire, but
**format-agnostic**: a human-readable serializer (JSON, ...) gets the
portable element-wise form, so a field using them still round-trips
through JSON as long as the element type itself has a portable serde
representation (`num_complex::Complex` does).

[`bytemuck::AnyBitPattern`]: https://docs.rs/bytemuck/latest/bytemuck/trait.AnyBitPattern.html

#### Encode-only helpers (legacy)

The original `#[serde(serialize_with = "beve::complex::f32_array")]`
helpers (`f32_array`, `f64_array`, `i8_array`, …, `u128_array`) encode a
field as a BEVE complex array but do **not** accelerate decode. Prefer
the `complex_array::*` `with` modules above for new code; the
`serialize_with` form remains for encode-only use.

### Wire Format

A scalar complex value uses the BEVE complex extension (`EXT_COMPLEX = 3`):

```
[extension header: 1 byte] [complex header: 1 byte] [re: N bytes] [im: N bytes]
```

The complex header encodes the number class (float, signed, unsigned)
and byte width, following the same scheme as BEVE number scalars.

A complex array adds a SIZE-encoded count:

```
[extension header] [complex header] [count: SIZE] [re0 im0 re1 im1 ...]
```

## Matrices

The `Matrix` type encodes a dense N-dimensional array with layout metadata:

```rust
use beve::{Matrix, MatrixLayout};

let matrix = Matrix {
    layout: MatrixLayout::Right, // row-major
    extents: &[2, 3],
    data: &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0],
};
let bytes = beve::to_vec(&matrix).unwrap();
```

### MatrixLayout

| Variant | Meaning | C/Rust equivalent |
|---|---|---|
| `MatrixLayout::Right` | Row-major | C arrays, NumPy default |
| `MatrixLayout::Left` | Column-major | Fortran, MATLAB, Julia |

### Decoding Matrices

Use `MatrixOwned<T>` to deserialize a matrix with owned data:

```rust
use beve::{MatrixOwned, MatrixLayout};

let m: MatrixOwned<f64> = beve::from_slice(&bytes).unwrap();
assert_eq!(m.layout, MatrixLayout::Right);
assert_eq!(m.extents, vec![2, 3]);
assert_eq!(m.data.len(), 6);
```

### Supported Element Types

The BEVE matrix extension supports:
- Numeric scalars (`u8`..`u128`, `i8`..`i128`, `f32`, `f64`)
- Booleans
- `Complex<T>` for all supported numeric scalar types

For unsupported element types, serialization falls back to a
`{ layout, extents, value }` map representation.
