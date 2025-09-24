# beve-rs
Rust implementation of the BEVE (Binary Efficient Versatile Encoding) specification with serde support. The crate targets cross-language interoperability, predictable layout, and zero-copy fast paths for scientific and analytics workloads.

## Getting Started
Add the crate to your project with `cargo add beve` or by editing `Cargo.toml`:
```toml
[dependencies]
beve = "0.1"
```
The library only depends on `serde` and works on stable Rust 1.72+.

## Encode & Decode with Serde
Use `beve::to_vec` and `beve::from_slice` for idiomatic serde round-trips:
```rust
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Point { x: f64, y: f64 }

fn main() -> beve::Result<()> {
    let p = Point { x: 1.0, y: -2.0 };
    let bytes = beve::to_vec(&p)?;
    let p2: Point = beve::from_slice(&bytes)?;
    assert_eq!(p, p2);
    Ok(())
}
```
You can stream to files or sockets with `beve::to_writer` and read everything back using `beve::from_reader`:
```rust
fn write_point(p: &Point) -> beve::Result<()> {
    beve::to_writer(std::fs::File::create("out.beve")?, p)?;
    let decoded: Point = beve::from_reader(std::fs::File::open("out.beve")?)?;
    assert_eq!(*p, decoded);
    Ok(())
}
```

## Typed Array Fast Paths
BEVE bakes in typed arrays for numeric, boolean, and string sequences. Skip serde overhead by calling the dedicated helpers:
```rust
let floats = [1.0f32, 3.5, -2.25];
let bytes = beve::to_vec_typed_slice(&floats);

let flags = [true, false, true, true];
let packed = beve::to_vec_bool_slice(&flags);

let names = ["alpha", "beta", "gamma"];
let encoded = beve::to_vec_str_slice(&names);
```
The resulting payloads match serde output, so `beve::from_slice::<Vec<T>>` continues to work.

## Complex Numbers and Matrices
The crate exposes helpers for common scientific types:
```rust
use beve::{Complex, Matrix, MatrixLayout};

fn encode_science() -> beve::Result<()> {
    let complex = [
        Complex { re: 1.0f64, im: -0.5 },
        Complex { re: 0.0, im: 2.0 },
    ];
    let dense = beve::to_vec_complex64_slice(&complex);
    let roundtrip: Vec<Complex<f64>> = beve::from_slice(&dense)?;
    assert_eq!(roundtrip, complex);

    let matrix = Matrix {
        layout: MatrixLayout::Right,
        extents: &[3, 3],
        data: &[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    };
    let bytes = beve::to_vec(&matrix)?;
    assert!(!bytes.is_empty());
    Ok(())
}
```
Matrices serialize as `{ layout, extents, value }` maps for easy consumption by other languages.

## Examples
- `cargo run --example emit_bool` writes a short boolean stream to stdout so you can inspect the raw bytes.
- `cargo run --example emit_color` demonstrates encoding a struct with enums and typed arrays.

## Enum Configuration
By default enums emit numeric discriminants for compatibility with the reference C++ encoder. Switch to string variants when coordinating with serde-first consumers:
```rust
use serde::Serialize;
use beve::{SerializerOptions, EnumEncoding};

#[derive(Serialize)]
enum MyEnum { Struct { a: i32, b: u32 } }

let opts = SerializerOptions { enum_encoding: EnumEncoding::String };
let bytes = beve::to_vec_with_options(&MyEnum::Struct { a: 1, b: 2 }, opts)?;
```

## Supported Data Model
- Scalars: signed/unsigned integers up to 128-bit, f32/f64, null, bool, and UTF-8 strings
- Collections: typed arrays (numeric, bool, string), generic sequences, maps with string or integer keys, and nested structs/enums
- Streaming: `to_writer`, `to_writer_with_options`, and `from_reader` for IO-bound workflows
- Interop: payloads align with `reference/glaze` and `reference/BEVE.jl`; spec resides in `reference/beve/README.md`

## Checking Your Work
Run the usual cargo commands before sending a change:
```bash
cargo fmt
cargo clippy --all-targets --all-features
cargo test
```
