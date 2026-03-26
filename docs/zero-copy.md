# Zero-Copy Deserialization

BEVE supports zero-copy deserialization: string and byte-slice fields can
borrow directly from the input buffer, avoiding allocation entirely.

## How It Works

When you call `beve::from_slice`, the returned value can contain references
back into the input `&[u8]`. The Rust borrow checker enforces that the
input slice outlives the deserialized value.

BEVE strings are stored as a length prefix followed by contiguous UTF-8
bytes. During deserialization the library validates the UTF-8 (using
SIMD-accelerated `simdutf8`) and returns a `&str` pointing directly into
the buffer -- no copy, no allocation.

## Supported Field Types

| Field type | Borrows? | Notes |
|---|---|---|
| `&str` | Yes | Points into buffer; UTF-8 validated |
| `String` | No | Allocates a new `String` |
| `&[u8]` | Yes | Requires `#[serde(borrow)]`; only for `u8` typed arrays |
| `Vec<u8>` | No | Allocates |
| `Cow<'a, str>` | Yes | Borrows when possible |
| `Vec<&str>` | Yes | Each element borrows from buffer |
| `BTreeMap<&str, V>` | Yes | Keys borrow from buffer |

## Examples

### Borrowed strings in a struct

```rust
use serde::Deserialize;

#[derive(Deserialize)]
struct Record<'a> {
    name: &'a str,
    tag: &'a str,
    score: f64,
}

let bytes = beve::to_vec(&("alice", "admin", 99.5)).unwrap();
// `record.name` and `record.tag` point into `bytes`
let record: Record = beve::from_slice(&bytes).unwrap();
```

### Borrowed byte slices

```rust
use serde::Deserialize;

#[derive(Deserialize)]
struct Packet<'a> {
    #[serde(borrow)]
    payload: &'a [u8],
}

let raw = vec![0xDE, 0xAD, 0xBE, 0xEF];
let bytes = beve::to_vec(&raw).unwrap();
let packet: Packet = beve::from_slice(&bytes).unwrap();
assert_eq!(packet.payload, &[0xDE, 0xAD, 0xBE, 0xEF]);
```

### With `from_field`

Zero-copy borrowing works with selective field loading too:

```rust
use serde::{Serialize, Deserialize};

#[derive(Serialize)]
struct Doc {
    title: String,
    body: String,
}

let doc = Doc {
    title: "Hello".into(),
    body: "World".into(),
};
let bytes = beve::to_vec(&doc).unwrap();

// Borrows directly from `bytes`
let title: &str = beve::from_field(&bytes, "/title").unwrap();
assert_eq!(title, "Hello");
```

## Limitations

- **`from_reader` cannot borrow.** It reads into an internal buffer that is
  dropped when the function returns, so the target type must implement
  `DeserializeOwned`.
- **`from_field_slice` cannot borrow.** It constructs a synthetic buffer
  internally, so the target type must implement `DeserializeOwned`.
- **Wider numeric arrays cannot be borrowed as slices.** BEVE does not
  guarantee alignment, so `&[f64]` etc. cannot safely alias the buffer.
  Use `Vec<f64>` instead.
