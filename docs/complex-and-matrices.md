# Complex Numbers and Matrices

BEVE includes extension types for complex numbers and dense matrices,
common in scientific computing workflows.

## Complex Numbers

The `Complex<T>` type represents a complex number with real and imaginary
parts:

```rust
use beve::Complex;

let z = Complex { re: 1.0f64, im: -0.5 };
let bytes = beve::to_vec(&z).unwrap();
let back: Complex<f64> = beve::from_slice(&bytes).unwrap();
assert_eq!(z, back);
```

### Complex Arrays

Use the fast-path helpers for contiguous complex arrays:

```rust
use beve::Complex;

let arr = [
    Complex { re: 1.0f64, im: 2.0 },
    Complex { re: 3.0, im: 4.0 },
];
let bytes = beve::to_vec_complex64_slice(&arr);
let back: Vec<Complex<f64>> = beve::from_slice(&bytes).unwrap();
assert_eq!(back, arr);
```

Both `f32` and `f64` complex types are supported:

- `to_vec_complex32` / `to_vec_complex32_slice` for `Complex<f32>`
- `to_vec_complex64` / `to_vec_complex64_slice` for `Complex<f64>`

### Wire Format

A scalar complex value uses the BEVE complex extension (`EXT_COMPLEX = 3`):

```
[extension header: 1 byte] [complex header: 1 byte] [re: N bytes] [im: N bytes]
```

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
- Numeric scalars (`u8`..`u64`, `i8`..`i64`, `f32`, `f64`)
- Booleans
- `Complex<f32>` and `Complex<f64>`

For unsupported element types, serialization falls back to a
`{ layout, extents, value }` map representation.
