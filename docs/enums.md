# Enums and variants

This crate targets **BEVE Version 2**, which has no variant encoding of its own. A variant is an ordinary value, chosen exactly as it would be for JSON, so the crate writes what `serde_json` writes and there is nothing to configure.

That property is the point: `beve_to_json(x)` equals `write_json(x)` for every enum shape, by construction rather than case by case.

## The default: externally tagged

Serde's default representation is externally tagged, which means a unit variant is its name as a plain string and a data-carrying variant is a single-key object.

```rust
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, PartialEq, Debug)]
enum Direction {
    North,
    South,
    East,
    West,
}

let bytes = beve::to_vec(&Direction::East).unwrap();
// The document is the string "East": no tag, no index, no extension.
let back: Direction = beve::from_slice(&bytes).unwrap();
assert_eq!(back, Direction::East);
```

All four variant kinds follow the same rule:

```rust
#[derive(Serialize, Deserialize)]
enum Shape {
    Point,                           // "Point"
    Circle(f64),                     // { "Circle": 1.5 }
    Triangle(f64, f64, f64),         // { "Triangle": [3.0, 4.0, 5.0] }
    Rect { w: f64, h: f64 },         // { "Rect": { "w": 2.0, "h": 3.0 } }
}
```

## Choosing another shape

Because the crate defers to serde, the shape is selected per type with serde's own attributes rather than with a crate-wide option. All four serde representations work with no variant-specific support:

```rust
use serde::Serialize;

#[derive(Serialize)]
#[serde(tag = "kind")]              // internally tagged
enum Tagged {                       //   { "kind": "Circle", "radius": 5.25 }
    Circle { radius: f64 },
}

#[derive(Serialize)]
#[serde(tag = "t", content = "c")]  // adjacently tagged
enum Adjacent {                     //   { "t": "Circle", "c": { "radius": 5.25 } }
    Circle { radius: f64 },
}

#[derive(Serialize)]
#[serde(untagged)]                  // no discriminator; the bare value
enum Either {
    Num(u32),
    Text(String),
}
```

Serde's derive lowers the latter three into ordinary maps and bare values before the serializer ever sees them, which is why they need nothing from this crate.

The internally tagged form is the shape a Glaze `std::variant` declaring `tag`/`ids` produces, so cross-language interop is a plain `#[serde(tag = "...")]`.

## Reading Version 1 documents

Version 1 encoded a variant as the **type tag extension** (id `1`, header byte `0x0E`) followed by a positional index or a string name, and this crate additionally offered a numeric encoding that wrote a bare index for a unit variant. Extension 1 is deprecated and reserved in Version 2. **This crate never emits it**, but both legacy forms still decode, so existing documents load unchanged.

Two limits are worth knowing. Legacy leniency lives in the enum path, so a Version 1 document read through `#[serde(untagged)]` or into `beve::Value` is not covered; and a Version 1 unit variant that carries a payload cannot be distinguished from one that does not, because the extension records no count, so the payload is left for the caller and surfaces as a decode error rather than being silently consumed.

The reverse direction does not hold either: a peer pinned to a pre-Version-2 decoder cannot read variants written here. Upgrade the peer, or pin this side, until both ends move.

## Selective field loading with enums

Enum values can be skipped over during `from_field` navigation and can appear as the leaf target:

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
