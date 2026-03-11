# beve-rs
Rust implementation of the BEVE (Binary Efficient Versatile Encoding) specification with serde support. The crate targets cross-language interoperability, predictable layout, and zero-copy fast paths for scientific and analytics workloads.

## Getting Started
Grab the crate from [crates.io](https://crates.io/crates/beve) and add it to your project with `cargo add beve` or by editing `Cargo.toml`:
```toml
[dependencies]
beve = "0.1"
```
The library only depends on `serde` and requires Rust 1.87 or newer. Half-precision floats via `half::f16` are supported alongside the standard numeric types.

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

For hot paths that reuse a `Vec<u8>`, encode directly into an existing buffer:
```rust
let mut buf = Vec::with_capacity(4096);
beve::to_vec_into(&mut buf, &Point { x: 1.0, y: 2.0 })?;
```

## Validate Without Deserializing
Use `validate_slice` or `validate_reader` when you only need to check that input is valid BEVE, without parsing into a Rust type.

Validation is strict: the payload must contain exactly one well-formed BEVE value with no trailing bytes.

```rust
use std::io::Cursor;

fn validate_payload(bytes: &[u8]) -> beve::Result<()> {
    beve::validate_slice(bytes)?;
    beve::validate_reader(Cursor::new(bytes))?;
    Ok(())
}
```

## JSON Interoperability
Convert between JSON payloads and BEVE without allocating an intermediate `serde_json::Value`. These helpers stream bytes on both sides, so large documents never build an in-memory tree and typed arrays stay in their native BEVE representation.
```rust
let json = r#"{"name":"delta","values":[1,2,3]}"#;
let beve_bytes = beve::json_str_to_beve(json)?;

let json_back = beve::beve_slice_to_json_string(&beve_bytes)?;
assert_eq!(
    serde_json::from_str::<serde_json::Value>(json)?,
    serde_json::from_str(&json_back)?
);
```
JSON arrays are always encoded as BEVE generic arrays (we do not attempt to detect homogeneous typed arrays), which avoids backtracking mid-stream. Non-finite floating-point literals (`NaN`, `Infinity`) are rejected because standard JSON cannot express them.

## Dynamic Value Type
When you need to deserialize BEVE data without knowing the schema at compile time, use `beve::Value`:
```rust
use beve::Value;

fn dynamic_data() -> beve::Result<()> {
    let bytes = beve::json_str_to_beve(r#"{"name":"test","count":42}"#)?;
    let value: Value = beve::from_slice(&bytes)?;

    // Access fields dynamically
    assert_eq!(value["name"].as_str(), Some("test"));
    assert_eq!(value["count"].as_i64(), Some(42));
    Ok(())
}
```

`Value` supports all BEVE types: `Null`, `Bool`, `Number`, `String`, `Array`, and `Object`. Numbers preserve their original representation (signed, unsigned, or float) at full precision (up to 128-bit integers).

### Converting Value to Concrete Types
Once you have a `Value`, convert it directly to a concrete type without re-encoding:
```rust
use serde::Deserialize;
use beve::{Value, from_value};

#[derive(Deserialize, Debug, PartialEq)]
struct Config {
    name: String,
    count: i32,
}

fn parse_config(value: Value) -> beve::Result<Config> {
    // Consumes the Value, avoiding clones where possible
    from_value(value)
}
```

Use `from_value_ref` when you need to keep the original `Value` around:
```rust
use beve::from_value_ref;

fn inspect_then_parse(value: &Value) -> beve::Result<Config> {
    println!("Parsing: {}", value);
    from_value_ref(value)
}
```

### Object Keys
BEVE objects support string and integer keys. The `Key` enum handles both:
```rust
use beve::{Value, Key, Object};
use std::collections::BTreeMap;

fn build_object() -> Value {
    let mut obj: Object = BTreeMap::new();
    obj.insert(Key::String("name".into()), Value::String("example".into()));
    obj.insert(Key::Unsigned(1), Value::Bool(true));
    Value::Object(obj)
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

### Typed Arrays Inside Structs
Struct fields automatically use the packed typed-array fast paths, so you get compact encodings without custom serializers:
```rust
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Frame {
    ticks: Vec<u64>,
    flags: Vec<bool>,
}

fn frame_roundtrip() -> beve::Result<()> {
    let frame = Frame {
        ticks: vec![1, 2, 4, 8],
        flags: vec![true, false, true, true],
    };
    let bytes = beve::to_vec(&frame)?;
    let back: Frame = beve::from_slice(&bytes)?;
    assert_eq!(frame, back);
    Ok(())
}
```

### Integer Map Keys
Maps with integer keys serialize deterministically and read back into ordered maps:
```rust
use std::collections::BTreeMap;

fn integer_keys() -> beve::Result<()> {
    let mut m = BTreeMap::new();
    m.insert(1u32, -1i32);
    m.insert(2u32, 4i32);
    let bytes = beve::to_vec(&m)?;
    let back: BTreeMap<u32, i32> = beve::from_slice(&bytes)?;
    assert_eq!(m, back);
    Ok(())
}
```

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
`Matrix` and `MatrixOwned<T>` use the BEVE matrix extension for supported element types (`bool`, numeric scalars, and `Complex<f32/f64>`). For unsupported element types, serialization falls back to a `{ layout, extents, value }` map.

## MATLAB / `.mat` Export
Enable the optional `mat` feature to convert BEVE payloads directly into MATLAB v7.3 MAT files:
```toml
[dependencies]
beve = { version = "0.1", features = ["mat"] }
```

The MAT feature depends on HDF5. On Linux that typically means installing `libhdf5-dev` so the build can find headers and libraries via `pkg-config` or `HDF5_DIR`.

Use `RootBinding::NamedVariable` when one BEVE value should become one MATLAB variable, or `RootBinding::WorkspaceObject` when a string-keyed BEVE object should expand into multiple top-level workspace variables:
```rust
use beve::{MatV73Options, RootBinding};

fn write_mat() -> beve::Result<()> {
    let bytes = beve::to_vec(&vec![1.0f64, 2.0, 3.0])?;
    beve::beve_slice_to_mat_v73_file(
        &bytes,
        "values.mat",
        RootBinding::NamedVariable("values"),
        &MatV73Options::default(),
    )?;
    Ok(())
}
```

Current mappings:
- numeric, logical, and complex scalars/arrays
- UTF-8 strings as MATLAB char arrays
- generic BEVE arrays as MATLAB cell arrays
- string-keyed BEVE objects as MATLAB structs
- BEVE matrix extensions, including row-major to column-major reorder when needed
- `null` as `struct([])` by default

Important limits:
- only MATLAB v7.3 is supported
- MATLAB `string` objects are not emitted yet; strings become char arrays and string arrays become cell arrays of char arrays
- non-string object keys cannot map to MATLAB structs/workspace variables
- `i128`, `u128`, `bf16`, and `f16` require explicit fallback policies when MATLAB has no direct native representation

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
- Interop: payloads align with `reference/glaze` and `reference/BEVE.jl`; spec resides in `reference/beve/README.md` and the upstream [BEVE specification](https://github.com/beve-org/beve)

### Half & BFloat16 Scalars
Half-precision (`f16`) and bfloat16 (`bf16`) values round-trip like any other scalar:
```rust
use half::{bf16, f16};

fn store_halves() -> beve::Result<()> {
    let h = f16::from_f32(-3.5);
    let bytes = beve::to_vec(&h)?;
    let back: f16 = beve::from_slice(&bytes)?;
    assert_eq!(h, back);

    let brain = bf16::from_f32(1.0);
    let brain_bytes = beve::to_vec(&brain)?;
    let brain_back: bf16 = beve::from_slice(&brain_bytes)?;
    assert_eq!(brain, brain_back);
    Ok(())
}
```

## Checking Your Work
Run the usual cargo commands before sending a change:
```bash
cargo fmt
cargo clippy --all-targets --all-features
cargo test
```
