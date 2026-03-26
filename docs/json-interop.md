# JSON Interoperability

BEVE provides streaming conversion between JSON and BEVE without
allocating an intermediate DOM tree.

## JSON to BEVE

```rust
// From a &str
let json = r#"{"name":"delta","values":[1,2,3]}"#;
let beve_bytes = beve::json_str_to_beve(json).unwrap();

// From &[u8]
let beve_bytes = beve::json_slice_to_beve(json.as_bytes()).unwrap();
```

JSON arrays are always encoded as BEVE generic arrays (the converter does
not attempt to detect homogeneous typed arrays), which avoids backtracking
mid-stream.

## BEVE to JSON

```rust
// To a String
let json_string = beve::beve_slice_to_json_string(&beve_bytes).unwrap();

// To a Vec<u8>
let json_bytes = beve::beve_slice_to_json(&beve_bytes).unwrap();
```

## Limitations

- Non-finite floating-point values (`NaN`, `Infinity`, `-Infinity`) are
  rejected during BEVE-to-JSON conversion because standard JSON cannot
  represent them.
- JSON has no typed-array concept, so BEVE typed arrays become JSON arrays.
- 128-bit integers may lose precision when round-tripped through JSON
  (JavaScript's `Number` type is an IEEE 754 double).
- BEVE extensions (complex numbers, matrices) are serialized as their
  structural representation (arrays/objects), not as special JSON types.
