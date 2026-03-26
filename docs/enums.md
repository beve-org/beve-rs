# Enum Encoding

BEVE supports Rust enums via the type-tag extension. By default, variants
are encoded as numeric discriminants for interoperability with the
reference C++ encoder.

## Default: Numeric Encoding

```rust
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, PartialEq, Debug)]
enum Direction {
    North,  // variant index 0
    South,  // variant index 1
    East,   // variant index 2
    West,   // variant index 3
}

let bytes = beve::to_vec(&Direction::East).unwrap();
let back: Direction = beve::from_slice(&bytes).unwrap();
assert_eq!(back, Direction::East);
```

## String Encoding

Switch to string variant names when coordinating with serde-first
consumers or when human-readable format inspection is more important
than compactness:

```rust
use beve::{SerializerOptions, EnumEncoding};

let opts = SerializerOptions {
    enum_encoding: EnumEncoding::String,
};
let bytes = beve::to_vec_with_options(&Direction::East, opts).unwrap();
```

## Data-Carrying Variants

All four serde enum representations are supported:

```rust
#[derive(Serialize, Deserialize)]
enum Shape {
    Circle(f64),                      // newtype variant
    Rect { w: f64, h: f64 },         // struct variant
    Triangle(f64, f64, f64),          // tuple variant
    Point,                            // unit variant
}
```

Data-carrying variants use the BEVE type-tag extension: a tag
(numeric index or string name) followed by the variant's payload.

## Selective Field Loading with Enums

Enum values can be skipped over during `from_field` navigation and can
appear as the leaf target:

```rust
#[derive(Serialize, Deserialize, PartialEq, Debug)]
enum Status { Active, Inactive }

#[derive(Serialize, Deserialize)]
struct Record {
    status: Status,
    value: u32,
}

let rec = Record { status: Status::Active, value: 42 };
let bytes = beve::to_vec(&rec).unwrap();

// Skip past the enum to read the next field
let v: u32 = beve::from_field(&bytes, "/value").unwrap();
assert_eq!(v, 42);

// Or read the enum itself
let s: Status = beve::from_field(&bytes, "/status").unwrap();
assert_eq!(s, Status::Active);
```
