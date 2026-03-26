# Dynamic Value Type

When you need to work with BEVE data without knowing the schema at compile
time, use `beve::Value`.

## Overview

`Value` is an enum that can represent any BEVE value:

```rust
pub enum Value {
    Null,
    Bool(bool),
    Number(Number),
    String(String),
    Array(Vec<Value>),
    Object(Object),
}
```

`Number` preserves the original representation at full precision:

```rust
pub enum Number {
    I64(i64),
    U64(u64),
    F64(f64),
    Big(Box<BigInt>), // i128 / u128
}
```

## Deserializing into Value

```rust
use beve::Value;

let bytes = beve::json_str_to_beve(r#"{"name":"test","count":42}"#).unwrap();
let value: Value = beve::from_slice(&bytes).unwrap();

assert_eq!(value["name"].as_str(), Some("test"));
assert_eq!(value["count"].as_i64(), Some(42));
```

## Converting Value to a Concrete Type

Once you have a `Value`, convert it to a concrete Rust type without
re-encoding:

```rust
use serde::Deserialize;
use beve::{Value, from_value, from_value_ref};

#[derive(Deserialize, PartialEq, Debug)]
struct Config {
    name: String,
    count: i32,
}

let value: Value = beve::from_slice(&bytes).unwrap();

// Consuming conversion (avoids clones where possible)
let cfg: Config = from_value(value.clone()).unwrap();

// Borrowing conversion (keeps the original Value)
let cfg2: Config = from_value_ref(&value).unwrap();
assert_eq!(cfg, cfg2);
```

## Object Keys

BEVE objects support both string and integer keys. The `Key` enum handles
both:

```rust
use beve::{Value, Key, Object};
use std::collections::BTreeMap;

let mut obj: Object = BTreeMap::new();
obj.insert(Key::String("name".into()), Value::String("example".into()));
obj.insert(Key::Unsigned(1), Value::Bool(true));
let value = Value::Object(obj);
```

## Indexing

`Value` supports `Index` for convenient field access:

```rust
let name = &value["name"]; // by string key
let first = &value[0];     // by array index
```

Indexing into a missing key or out-of-bounds index returns `Value::Null`
rather than panicking.
