# beve-rs
Rust implementation of the BEVE (Binary Efficient Versatile Encoding) specification with serde support. The crate targets cross-language interoperability, predictable layout, and zero-copy fast paths for scientific and analytics workloads.

## Getting Started
Grab the crate from [crates.io](https://crates.io/crates/beve) and add it to your project with `cargo add beve` or by editing `Cargo.toml`:
```toml
[dependencies]
beve = "6"
```
By default the crate is lean: it depends only on `serde`, `half`, and `simdutf8`, and requires Rust 1.89 or newer. Half-precision floats via `half::f16` are supported alongside the standard numeric types. The MATLAB/HDF5 export path is gated behind the opt-in [`mat` feature](#matlab--mat-export), so the HDF5 dependency stack is never pulled into a default build.

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
You can write to files or sockets with `beve::to_writer` and read everything back using `beve::from_reader` (for zero-buffered streaming I/O, see [Streaming](#streaming)):
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

## Zero-Copy Deserialization
`from_slice` supports borrowing directly from the input buffer for string types. Structs with `&str` fields avoid allocation entirely — the deserialized strings point straight into the BEVE byte slice:
```rust
use serde::Deserialize;

#[derive(Deserialize)]
struct Record<'a> {
    name: &'a str,
    tag: &'a str,
    score: f64,
}

fn parse_record(bytes: &[u8]) -> beve::Result<()> {
    let record: Record = beve::from_slice(bytes)?;
    assert_eq!(record.name, "alice");
    Ok(())
}
```
This works for `&str` fields, `Vec<&str>`, `BTreeMap<&str, V>` keys, and `&[u8]` fields (with `#[serde(borrow)]`). BEVE typed `u8` arrays are contiguous bytes with alignment 1, so `&[u8]` borrows directly from the buffer without copying. Zero-copy borrowing is not available for wider numeric arrays (e.g. `&[f64]`) since BEVE does not guarantee alignment.

`from_reader` continues to require `DeserializeOwned` since it reads into an internal buffer that cannot outlive the call.

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
The resulting payloads match serde output, so `beve::from_slice::<Vec<T>>` continues to work. For the decode side, `read_typed_slice` is the bulk counterpart of `to_vec_typed_slice` (and `read_complex_slice` of `to_vec_complex_slice`): one bounds-checked contiguous read into a `Vec<T>` instead of element-by-element serde, when the whole body is a numeric array.
```rust
let floats = [1.0f32, 3.5, -2.25];
let bytes = beve::to_vec_typed_slice(&floats);
let back = beve::read_typed_slice::<f32>(&bytes).unwrap();
assert_eq!(back, floats);
```

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

The encoding is compact, but **decoding a struct field still goes element-by-element through serde** — the bulk readers (`read_typed_slice` / `read_complex_slice`) operate on a whole-body array and can't reach a field nested inside a struct. For a large numeric or complex field where decode speed matters, annotate it with the `#[serde(with = ...)]` bulk helpers. They decode the field straight into its `Vec` at memcpy speed (and still encode/decode through JSON via the portable element form):

```rust
use beve::Complex;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Capture {
    #[serde(with = "beve::typed::f64")]
    samples: Vec<f64>,
    #[serde(with = "beve::complex_array::f32")]
    iq: Vec<Complex<f32>>,
}
```

`beve::typed::*` covers the numeric scalars (`i8`–`i128`, `u8`–`u128`, `f32`, `f64`); `beve::complex_array::*` covers complex arrays whose element is layout-compatible with `Complex<scalar>` (including `num_complex::Complex` with its `bytemuck` feature). The on-wire bytes are identical to the unannotated `Vec<T>`, so an annotated and a plain field interoperate. See [Complex Numbers and Matrices](#complex-numbers-and-matrices) for the foreign-type details.

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
`Complex<T>` supports all numeric scalar types (`f32`, `f64`, `i8`–`i128`, `u8`–`u128`) and works naturally in structs alongside other fields:
```rust
use beve::{Complex, Matrix, MatrixLayout};

fn encode_science() -> beve::Result<()> {
    // Float complex
    let complex = [
        Complex { re: 1.0f64, im: -0.5 },
        Complex { re: 0.0, im: 2.0 },
    ];
    let dense = beve::to_vec_complex_slice(&complex);
    // `read_complex_slice` is the bulk decode counterpart of
    // `to_vec_complex_slice`: one bounds-checked contiguous read instead of
    // element-by-element serde. `from_slice` still works if you prefer serde.
    let roundtrip = beve::read_complex_slice::<f64>(&dense)?;
    assert_eq!(roundtrip, complex);

    // To frame a complex array straight into a writer without a body buffer, use
    // the streaming counterpart `to_writer_complex_slice` (mirroring
    // `to_writer_typed_slice`), with `complex_slice_size` for the O(1) length.
    let mut framed = Vec::new();
    beve::to_writer_complex_slice(&mut framed, &complex)?;
    assert_eq!(framed.len() as u64, beve::complex_slice_size(&complex));
    assert_eq!(framed, dense);

    // Integer complex
    let iq = [Complex { re: 1i16, im: -2 }, Complex { re: 3, im: 4 }];
    let bytes = beve::to_vec_complex_slice(&iq);
    let back: Vec<Complex<i16>> = beve::from_slice(&bytes)?;
    assert_eq!(back, iq);

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
For foreign complex types (e.g. `num_complex::Complex`), annotate the field with `#[serde(with = "beve::complex_array::f32")]` (and the per-scalar siblings): this gives a compact BEVE complex array on the wire **and** bulk (memcpy) decode straight into the field. The element type must be `bytemuck::AnyBitPattern` and layout-compatible with `Complex<scalar>` — for `num_complex`, enable its `bytemuck` feature. (The older `#[serde(serialize_with = "beve::complex::f32_array")]` helpers encode only.) See the [complex docs](docs/complex-and-matrices.md) for details.

`Matrix` and `MatrixOwned<T>` use the BEVE matrix extension for supported element types (`bool`, numeric scalars, and `Complex<T>`). For unsupported element types, serialization falls back to a `{ layout, extents, value }` map.

## MATLAB / `.mat` Export
The `mat` feature is **off by default** (it pulls in `hdf5-pure` and its compression stack, which the core ser/de does not need). Enable it to convert BEVE payloads directly into MATLAB v7.3 MAT files:
```toml
[dependencies]
beve = { version = "6", features = ["mat"] }
```

The MAT feature uses a pure-Rust HDF5 writer (`hdf5-pure`) and requires no system libraries. The CLI's `to-mat` command is likewise only present when the binary is built with `--features mat`.

### Versioning of the `hdf5-pure` dependency
`beve::mat` re-exports hdf5-pure's option enums (`Compression`, `NullPolicy`, ...), so moving to a new hdf5-pure release ships as a beve **minor** version rather than a major one — hdf5-pure is pre-1.0, and every release of it is a new `0.x`.

This only matters if you depend on hdf5-pure directly as well: match the version named in the [changelog](CHANGELOG.md) and move in step with it, or pin `beve = "~6.1"` to take that move deliberately. Everything else follows ordinary semver.

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

For in-memory conversion (useful in WASM or when you already have the bytes), use `beve_slice_to_mat_v73_bytes`:
```rust
let mat_bytes = beve::beve_slice_to_mat_v73_bytes(
    &bytes,
    RootBinding::NamedVariable("values"),
    &MatV73Options::default(),
)?;
```

To send the result somewhere without holding it, use `beve_slice_to_mat_v73_writer`. It produces byte-for-byte what `beve_slice_to_mat_v73_bytes` returns, but writes the file front-to-back and never seeks, so the sink can be a socket as readily as a file and the output size no longer bounds what can be converted:
```rust
beve::beve_slice_to_mat_v73_writer(
    &bytes,
    &mut sink,
    RootBinding::NamedVariable("values"),
    &MatV73Options::default(),
)?;
```
`beve_slice_to_mat_v73_file` streams this way too, into a temp file it renames over the target on success.

Current mappings:
- numeric, logical, and complex scalars/arrays
- UTF-8 strings and typed string arrays as MATLAB `string` objects
- generic BEVE arrays as MATLAB cell arrays
- string-keyed BEVE objects as MATLAB structs
- BEVE matrix extensions, including row-major to column-major reorder when needed
- `null` as `struct([])` by default

### On-disk format and compression
MAT files are written in the **HDF5 1.8** format (`MatV73Options::libver`, defaulting to `LibVer::V18`), which is what MATLAB's `load` accepts. A version 3 superblock is an HDF5 1.10 addition, and MATLAB's MAT v7.3 loader refuses one even on releases whose own libhdf5 reads it without difficulty, so a 1.10 file fails to `load` outright.

`Compression` is the one thing that format cannot carry: compression needs chunked storage, whose chunk indices arrived in 1.10. The pair is refused rather than silently resolved, since dropping the compression loses what you asked for and raising the format produces a file MATLAB cannot read. Set `libver` yourself to take that trade:
```rust
let options = MatV73Options {
    compression: Compression::Deflate { level: 4, shuffle: false },
    // Required for compression, and no longer loadable by MATLAB.
    // Still readable by h5py, `h5dump`, and hdf5-pure.
    libver: LibVer::V110,
    ..Default::default()
};
```

Complex values keep their element type. MATLAB has first-class complex integer
arrays, so a complex `i16` payload becomes a complex `int16` MATLAB array rather
than being widened to `single`; the same holds for the other integer widths
(`i8`/`i16`/`i32`/`i64` and `u8`/`u16`/`u32`/`u64`). Nothing on this path
silently promotes an integer to a float, so the file always reports the width
the data was actually stored at.

Important limits:
- only MATLAB v7.3 is supported
- `i128`, `u128`, `bf16`, and `f16` require explicit fallback policies when MATLAB has no direct native representation
- `bf16` and `f16` complex arrays follow the same rule: they widen to `single` under `UnsupportedPolicy::LossyNumericWidening` and error otherwise
- 128-bit complex (`i128`/`u128`) is always rejected. MATLAB has no 128-bit class, and unlike a real 128-bit *scalar* there is no `StringFallback` representation for a complex one
- non-string object keys are converted to their string representation (e.g. integer key `48000` becomes field name `"x48000"` with `InvalidNamePolicy::Sanitize`)
- the MATIO-based oracle used in tests does not decode MATLAB `string` objects semantically, so string coverage is validated structurally against MATLAB-generated fixtures

## CLI

The crate includes `beve-cli`, a command-line tool for converting BEVE files:

```bash
cargo install beve --bin beve-cli
```

```
Usage: beve-cli <command> [options] <input> [output]

Commands:
  to-json    Convert BEVE to JSON
  to-mat     Convert BEVE to MATLAB v7.3 MAT
  from-json  Convert JSON to BEVE
```

Examples:
```bash
beve-cli to-json data.beve              # writes data.json
beve-cli to-mat data.beve               # writes data.mat
beve-cli to-mat data.beve output.mat    # explicit output path
beve-cli from-json data.json            # writes data.beve
```

The `to-mat` command supports `--name <var>` to set the MATLAB variable name (default: `data`) and `--workspace` to expand top-level object keys into separate workspace variables.

## Examples
- `cargo run --example emit_bool` writes a short boolean stream to stdout so you can inspect the raw bytes.
- `cargo run --example emit_color` demonstrates encoding a struct with enums and typed arrays.

## Enums and variants

This crate targets **BEVE Version 2**, which has no variant encoding of its own: a variant is an ordinary value, chosen exactly as it would be for JSON. There is nothing to configure, and the crate writes what `serde_json` writes.

That means serde's own attributes select the shape, per type, and all four representations work without any variant-specific support:

```rust
use serde::Serialize;

#[derive(Serialize)]
enum Shape {                       // externally tagged (serde's default)
    Empty,                         //   "Empty"
    Circle { radius: f64 },        //   { "Circle": { "radius": 5.25 } }
}

#[derive(Serialize)]
#[serde(tag = "kind")]             // internally tagged
enum Tagged {                      //   { "kind": "Circle", "radius": 5.25 }
    Circle { radius: f64 },
}

#[derive(Serialize)]
#[serde(untagged)]                 // no discriminator; the bare value
enum Either { Num(u32), Text(String) }
```

The internally tagged form is the shape a Glaze `std::variant` declaring `tag`/`ids` produces, so cross-language interop is a plain `#[serde(tag = "...")]` rather than a crate option.

### Reading Version 1 documents

Version 1 encoded a variant as the **type tag extension** (id `1`, header byte `0x0E`) followed by a positional index, and this crate additionally offered a numeric encoding that wrote a bare index for a unit variant. Extension 1 is deprecated and reserved in Version 2; **this crate never emits it**, but it still reads both legacy forms, so existing documents load unchanged.

The reverse does not hold: a peer pinned to a pre-Version-2 decoder cannot read variants written here. Upgrade the peer, or pin this side, until both ends move.

## Streaming
For large payloads where you don't want to buffer the entire input or output in memory, use the streaming APIs. They read and write directly from `std::io::Read` / `std::io::Write` with zero internal buffering:
```rust
use serde::{Serialize, Deserialize};
use std::io::BufWriter;
use std::fs::File;

#[derive(Serialize, Deserialize)]
struct Recording {
    name: String,
    samples: Vec<f64>,
}

fn write_large_recording(rec: &Recording) -> beve::Result<()> {
    let file = BufWriter::new(File::create("recording.beve")?);
    beve::to_writer_streaming(file, rec)?;
    Ok(())
}

fn read_large_recording() -> beve::Result<Recording> {
    let file = std::io::BufReader::new(File::open("recording.beve")?);
    beve::from_reader_streaming(file)
}
```

Both directions process data incrementally with no intermediate allocations beyond the output values themselves. Homogeneous *sequences* (`Vec<f64>`, `Vec<u32>`, `Vec<bool>`, `Vec<String>`, etc.) are automatically encoded as compact typed arrays, producing byte-for-byte identical output to `to_vec`.

Two constraints follow from writing front-to-back, with no ability to revise a header already emitted:

- **Containers must have known lengths** (structs, `Vec`, `HashMap`, etc.), which covers virtually all standard Rust types. A custom `Serialize` that passes `None` for a length is an error here, and `#[serde(flatten)]` is rejected for the same reason.
- **Tuples and fixed-size arrays encode as generic arrays**, where `to_vec` coalesces a homogeneous one into a typed array. Serde routes Rust arrays through the same hook as tuples, so `[u8; 4]` cannot be told apart from `(u8, bool)`; detecting from the first element would commit this writer to a header it cannot revise, and every mixed tuple would then fail. Both encodings are valid and read back identically through either reader, so this affects bytes and size, not meaning. Use `to_vec`, or a `Vec<T>`, when the compact form matters.

### Data Delimiters
When writing multiple values to the same stream, use `beve::write_delimiter` to insert the BEVE data delimiter byte (`0x06`) between entries — analogous to `\n` in NDJSON. `from_slice` and `from_reader_streaming` skip delimiters transparently during deserialization (note: `validate_slice` expects a single value and will reject delimiter-separated streams):
```rust
use serde::{Serialize, Deserialize};
use std::io::Cursor;

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Record { id: u32, value: f64 }

let mut buf = Vec::new();
let r1 = Record { id: 1, value: 1.5 };
let r2 = Record { id: 2, value: 2.5 };

beve::to_writer_streaming(&mut buf, &r1)?;
beve::write_delimiter(&mut buf)?;
beve::to_writer_streaming(&mut buf, &r2)?;

// Read back — delimiters are skipped automatically
let mut cursor = Cursor::new(&buf);
let back1: Record = beve::from_reader_streaming(&mut cursor)?;
let back2: Record = beve::from_reader_streaming(&mut cursor)?;
assert_eq!(back1, r1);
assert_eq!(back2, r2);
```

## Supported Data Model
- Scalars: signed/unsigned integers up to 128-bit, f32/f64, null, bool, and UTF-8 strings
- Complex numbers: `Complex<T>` for all numeric scalar types, with typed complex arrays
- Collections: typed arrays (numeric, bool, string), generic sequences, maps with string or integer keys, and nested structs/enums
- Streaming: `to_writer_streaming` / `from_reader_streaming` for zero-buffered I/O; `to_writer` and `from_reader` for buffered workflows
- Interop: payloads align with `reference/glaze` and `reference/BEVE.jl`; targets **Version 2** of the upstream [BEVE specification](https://github.com/beve-org/beve)

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
