# Changelog

This crate follows [Semantic Versioning](https://semver.org/), with one exemption: the `mat` feature's [hdf5-pure version](README.md#versioning-of-the-hdf5-pure-dependency) moves in minor releases. Dates are the crates.io publication date.

Entries for 4.0.0 and earlier were written after the fact, from the tagged releases and their merged pull requests, so they summarize each release rather than enumerate it. 5.0.0 onward is written as part of the change.

## Unreleased

### Fixed

- A build of the published crate no longer warns `Failed to build Glaze interop helper` when no C++23 compiler is present. The C++ test fixtures under `tests/cpp/` were shipping in the `.crate` while the Glaze headers they `#include` were not, so `build.rs` tried to compile them and failed on every such downstream build rather than skipping them, as it already does when the sources are absent. `tests/cpp/` is now excluded from the package.

## 6.1.0 - 2026-08-03

### Changed

- The `mat` feature moves to **hdf5-pure 0.33** (was 0.31). The conversion API, the re-exported option enums, and the files beve writes are all unchanged. What you gain is better error text: 0.32 gave HDF5's descriptive types (`Datatype`, `Filter`, `Layout`, ...) a `Display` that reads as HDF5 rather than as a Rust value.

  If you depend on hdf5-pure directly as well, move to 0.33 in step with this release — beve's public API is typed by the re-exported enums, so the versions must agree.

### Added

- A documented [versioning policy](README.md#versioning-of-the-hdf5-pure-dependency) for the `mat` feature's hdf5-pure dependency: an hdf5-pure move ships as a beve minor version, not a major.

## 6.0.0 - 2026-07-30

### Changed

- **Breaking:** the `mat` feature moves to **hdf5-pure 0.31** (was 0.30). The conversion API is unchanged, and so are the option enums this crate re-exports (`Compression`, `NullPolicy`, ...), but beve's public API is typed by them, so a caller who enables `mat` and also depends on hdf5-pure directly must move to 0.31 for a single major version to resolve. 0.31 fixes attribute round-tripping: an attribute reads back as the variant it was written from, `repack` no longer re-encodes the attributes it copies, and an empty-string attribute is written with a datatype libhdf5 accepts (a single one previously made every attribute on that object unreadable to the C library). Files beve wrote with earlier versions still read.

## 5.0.2 - 2026-07-30

### Fixed

- A map whose body does not match the header written ahead of it is now caught at `end`, closing the gap 5.0.1 left open when it added the same guard for structs. A known-length map writes its entry count into the header on the first key and can never revise it, so a `len` that disagrees with the body leaves a header promising entries the reader never finds. Serde's own container impls always declare an honest `len`; a hand-written `Serialize` typically gets here by declaring `Some(collection.len())` and then filtering entries out of the loop.

  Note one case that changes from valid output to an error: declaring entries and then writing none previously emitted a well-formed empty object, because the no-keys-written fallback writes its own zero-count header and never consulted the declared length. That was the same mistake getting a pass on an accident of the empty-map path, and it corrupts as soon as one entry survives the filter.

- A key with no value following it is also reported. `serialize_key` has already written key bytes, so a missing `serialize_value` leaves half an entry on the wire. This one corrupts unknown-length maps (`serialize_map(None)`) too, where the count is patched afterwards from the number of *values* and so cannot account for the orphan (#36).

## 5.0.1 - 2026-07-30

### Fixed

- **`#[serde(skip_serializing_if = ...)]` works again.** 5.0.0 made `SerializeStruct::skip_field` return `Unsupported` on both struct serializers, so serializing a struct failed whenever such a field was actually skipped. (A value whose predicate was false still encoded, which is what let the regression through: the attribute alone is harmless, only the skip fails.) The premise was wrong: serde's contract is that the `len` given to `serialize_struct` is "the number of data fields that will be serialized", excluding skipped ones, and `serde_derive` honors it, emitting that count as `... + if skip_serializing_if(&field) { 0 } else { 1 }` per field. The object header is therefore already correct by the time a skip is reported, so `skip_field` is a no-op, as it is in every other format. (The 5.0.0 note claiming serde's derive never calls `skip_field` was also wrong; it does.)

  Struct *variants* were never affected, because neither variant serializer overrode `skip_field`. 5.0.0 accepted a skipped field inside `Enum::Variant { .. }` while rejecting the identical field on a plain struct.

- A struct whose body does not match the `len` it declared is now caught at `end`, with a message naming both counts, instead of writing an object header that promises a field the reader never finds. This is the corruption 5.0.0 was reaching for; verifying the tally catches over- and under-declaring alike without breaking the derive, and only a hand-written `Serialize` can trip it. Maps are not covered by this guard: a hand-written `serialize_map(Some(n))` that writes a different number of entries still produces a document `validate_slice` rejects (#35).

## 5.0.0 - 2026-07-30

BEVE **Version 2** compliance. Variants are now ordinary values, and the crate writes exactly what `serde_json` writes.

### Breaking: the wire format for enums changed

This affects every caller, including those who never touched serializer options.

| Variant kind | 4.x wrote | 5.0 writes |
| --- | --- | --- |
| unit | type-tag extension, or a bare index | the name as a string, `"Beta"` |
| newtype / tuple / struct | type-tag extension + tag + payload | `{ "VariantName": payload }` |

Version 2 deprecates and reserves the type tag extension (id `1`, header byte `0x0E`). This crate no longer emits it. Both Version 1 forms still **decode**, so existing documents load unchanged, but a peer pinned to a pre-Version-2 decoder cannot read what this version writes.

Consequence for MAT output: a unit variant now lands as a MATLAB `string` where it was a `uint32` scalar, so compare it with `strcmp` rather than `==`.

### Breaking: the serializer options API is removed

Variant shape is a per-type concern, which serde already models with attributes; a serializer-global switch could not express three of serde's four representations. Select the shape with serde instead:

| Removed | Replacement |
| --- | --- |
| `EnumEncoding`, `SerializerOptions` | serde attributes on the type (see below) |
| `to_vec_with_options`, `to_vec_into_with_options` | `to_vec`, `to_vec_into` |
| `to_writer_with_options` | `to_writer` |
| `to_writer_streaming_with_options` | `to_writer_streaming` |
| `serialized_size_with_options` | `serialized_size` |
| `Serializer::with_options`, `Serializer::with_capacity_and_options` | `Serializer::new`, `Serializer::with_capacity` |
| `StreamingSerializer::with_options` | `StreamingSerializer::new` |

`EnumEncoding::String` was the closest thing to a default, and it is now simply the behavior. Callers who passed options only for that can drop the argument. For any other shape, annotate the type: `#[serde(tag = "...")]` for internally tagged (the shape a Glaze `std::variant` with `tag`/`ids` produces), `#[serde(tag = "...", content = "...")]` for adjacently tagged, `#[serde(untagged)]` for a bare value.

### Added: streaming MAT v7.3 conversion

`beve_slice_to_mat_v73_writer` converts a BEVE payload into a MATLAB v7.3 file written front-to-back onto a `Write` sink. It produces byte-for-byte what `beve_slice_to_mat_v73_bytes` returns, but never holds the assembled file, so the output size no longer bounds what can be converted. The sink is never seeked, so it may be a socket as readily as a file.

The `mat` feature now requires **hdf5-pure 0.30** (was 0.28), which brings `NullPolicy::Omit`.

### Fixed

- An unknown-length sequence whose first element was a newtype variant wrote an array header counting one element too few, so the tail decoded as trailing data and `from_slice` silently returned a short sequence.
- A unit-variant target reading a variant that carries a payload behaved differently in the two readers: the streaming reader substituted the discarded payload for the next element, while the buffered reader guessed from how many bytes were left in the whole document and could consume a sibling. Both now discard exactly one value and report a malformed one instead of swallowing the error. This is not only schema drift: version 4 wrote a unit variant leading a sequence as the type-tag extension plus an explicit `null`, so a 4.x-written `Vec<SomeUnitEnum>` depends on that value being consumed.
- A truncated single-key variant object, whose header promises a value that is not present, is now rejected rather than decoding as a unit variant.
- `is_human_readable` was missing on the buffered reader and the sequence-element serializer, both of which then defaulted to `true` where every other impl returns `false`. `from_slice::<Ipv4Addr>` could not read this crate's own output, and a `Vec<Ipv4Addr>` encoded as strings through one writer and as octets through the other.
- `SerializeStruct::skip_field` wrote a document whose object header counted a field that was not on the wire, so `validate_slice` reported EOF. Both struct serializers now refuse it. Serde's derive never calls it; a hand-written `Serialize` can.
- A null cell-array element under `NullPolicy::Omit` left the element referencing an object that was never written, which MATLAB could not dereference. A cell array's element count is fixed by its dims, so `Omit` has nothing to omit there; such an element now takes the empty-struct-array marker.

### Notes

- `docs/enums.md` is rewritten for Version 2.
- Interop against a pre-Version-2 Glaze cannot exercise variants; those cases are gated behind `GLAZE_INTEROP_V2=1` pending `stephenberry/glaze#2707`.

## 4.0.0 - 2026-07-28

**Breaking:** the `mat` feature moves to hdf5-pure 0.28 (was 0.21). The crate re-exports that crate's option enums (`Compression`, `InvalidNamePolicy`, `NullPolicy`, ...) and `MatV73Options` has public fields typed by them, so the bump is incompatible even though the conversion API itself is unchanged. If you enable `mat` and also depend on hdf5-pure directly, move your own dependency to 0.28.

**Breaking:** five error strings no longer occur, so code matching on them needs updating: `only floating-point complex supported in MAT conversion at {path}`, the same for `MAT matrix conversion`, `unsupported complex element width at {path}`, `unsupported complex scalar width at {path}`, and `unsupported complex matrix element width at {path}`.

MAT conversion now keys on the element type carried in the BEVE complex header, so an integer complex payload keeps its width instead of being rejected: `i8`–`i64` and `u8`–`u64` become the matching MATLAB integer complex class, `f32`/`f64` are unchanged, `f16`/`bf16` widen to `single` only under `UnsupportedPolicy::LossyNumericWidening`, and `i128`/`u128` are an error because MATLAB has no 128-bit class. Nothing on this path promotes an integer to a float: a file reporting `single` for `int16` samples misstates its own provenance undetectably. Scalars, arrays, and matrix-extension payloads all take the same path. The bulk readers in the MAT codec also bound a wire-supplied length before allocating (#33).

## 3.0.0 - 2026-07-13

**Breaking:** the `mat` feature moves to hdf5-pure 0.21 (was 0.5), incompatible for the same public-re-export reason as 4.0.0, and the declared MSRV moves to **1.89** (was 1.88) because hdf5-pure 0.21 requires it. The conversion API keeps its names and signatures, and a default build is unaffected.

## 2.5.0 - 2026-06-11

`beve::typed::<scalar>` and `beve::complex_array::<scalar>` are `#[serde(with = ...)]` helpers that decode a numeric or complex `Vec<T>` struct field at memcpy speed, the counterpart to the bulk encode the crate already did (~43 GiB/s against ~0.85 GiB/s element-wise). The complex path is bounded on `bytemuck::AnyBitPattern` for soundness, and both stay format-agnostic, so a JSON round-trip still works through the portable element form. MSRV is now declared (`rust-version = "1.88"`), and CI gained a big-endian (s390x) job that exercises the byte-swap branches (#32).

## 2.4.0 - 2026-06-11

- `read_typed_slice_from_reader` and `read_complex_slice_from_reader` are the `Read`-based counterparts of the in-memory bulk readers, decoding a large numeric or complex buffer straight from a stream without first materializing the encoded bytes. The payload is read in capped allocation steps, so a corrupt or hostile length fails gracefully rather than forcing one huge up-front allocation (#30).
- Per-element integer typed-array decode converts at native width instead of widening every element through a 16-byte buffer and a `u128`/`i128` round-trip. 8- and 16-byte widths keep the previous path, which was already optimal, so their output is byte-identical (#31).

## 2.3.0 - 2026-06-09

`write_aligned_typed_slice_at(out, slice, base_offset)` sizes the padding run for the marker's eventual frame offset rather than its position within `out`, so a body built in a standalone buffer stays borrowable once concatenated behind a fixed prefix. `write_aligned_typed_slice` is now defined in terms of it and is byte-for-byte unchanged (#29).

## 2.2.0 - 2026-06-09

The BEVE spec's aligned typed array wire type (`0x5C`): write, owned read, zero-copy borrow, and offset-aware size, in a new `aligned` module (#28).

## 2.1.0 - 2026-06-09

`to_writer_complex_slice` is the streaming counterpart of `to_vec_complex_slice`, writing the complex header, the SIZE prefix, and then the interleaved payload with a single bulk `write_all` on little-endian targets. `complex_slice_size` gives the O(1) encoded length from the same SIZE codec the writer uses, so it cannot drift. Together they allow zero-body-buffer framing of complex arrays (#27).

## 2.0.0 - 2026-06-08

**Breaking:** `mat` is no longer a default feature. The default build is lean (serde, half, simdutf8) and no longer pulls in hdf5-pure and its compression stack for callers who only use core BEVE ser/de. Migration is one line: `beve = { version = "2", features = ["mat"] }`. Removing a feature from the default set is breaking under Cargo's semver rules, hence the major bump (#26).

## 1.6.0 - 2026-06-08

- `read_typed_slice` is the bulk decode counterpart of `to_vec_typed_slice` for plain numeric arrays, which previously fell back to per-element serde. This completes the typed-slice read/write family (#24).
- `Complex<f16>` and `Complex<bf16>` work through serde. They had neither a `Serialize` impl nor a deserializer arm for the half-float byte codes, though the bulk path already handled them (#25).

## 1.5.0 - 2026-06-01

`read_complex_slice` decodes a complex array in bulk after validating the extension header, replacing the per-element visitor dispatch that dominated the cost of `from_slice::<Vec<Complex<T>>>` (#23).

## 1.4.0 - 2026-05-30

- `serialized_size` computes the exact byte length `to_writer_streaming` will emit, without producing the bytes, by driving the same encoder through a counting sink — so it cannot drift from what is actually written. This enables single-pass length-prefixed framing over a non-seekable transport: measure the body, write the prefix, stream the body once. It is O(1) for `serialize_bytes` and `TypedSlice<T>` bodies, O(N) for a bare numeric `Vec<T>`, and rejects unknown-length containers exactly as the streaming writer does (#22).
- `TypedSlice<T>`, `to_writer_typed_slice`, and the closed-form `typed_slice_size` give O(1) encoding and measuring of a contiguous numeric slice: header, SIZE prefix, and one bulk `write_all` on little-endian targets, with a per-element fallback on big-endian (#21).
- The `field` module is now public, surfacing its JSON Pointer (RFC 6901) reference documentation. Additive: `from_field`, `from_field_slice`, and `skip_value` remain re-exported at the crate root.

## 1.3.0 - 2026-04-30

The `mat` feature moves to hdf5-pure 0.5 (was 0.4).

## 1.2.0 - 2026-04-27

MAT v7.3 conversion is driven through hdf5-pure 0.4's `MatBuilder`, which owns the MATLAB conventions (`MATLAB_class`, empty markers, the lazy `#refs#` group, the `#subsystem#`/MCOS subsystem). `src/mat.rs` is now a thin BEVE wire-format walker rather than a second implementation of that machinery (#19).

## 1.1.0 - 2026-04-09

`DATA_DELIMITER` and `write_delimiter` frame consecutive BEVE values in one stream, the role `\n` plays in NDJSON. The deserializers skip delimiter bytes transparently; `validate_slice` still expects exactly one value with no trailing bytes, so it rejects a delimited stream (#18).

## 1.0.0 - 2026-04-07

**Breaking:** the fixed-width complex helpers `to_vec_complex32`, `to_vec_complex64`, `to_vec_complex32_slice`, and `to_vec_complex64_slice` are replaced by the generic `ext::complex` module, which covers every complex element type across the buffered and streaming paths (#17).

## 0.8.0 - 2026-03-27

- Streaming serialization and deserialization: `to_writer_streaming` and `from_reader_streaming` encode and decode against `Write`/`Read` without holding the whole document (#16).
- Selective field loading: `from_field` and `from_field_slice` read one field out of a document by JSON Pointer (RFC 6901) instead of decoding all of it, and `skip_value` steps over one encoded value. Adds the `docs/` set covering enums, JSON interop, typed arrays, zero-copy, and this path (#15).

## 0.7.0 - 2026-03-16

`beve-cli`, a command-line converter with `to-json`, `from-json`, and `to-mat` subcommands (#14).

## 0.6.1 - 2026-03-15

Documentation only.

## 0.6.0 - 2026-03-14

- MATLAB `.mat` support is backed by the pure-Rust `hdf5-pure` instead of `hdf5_metno`, so the export path needs no system HDF5 library (#13).
- Zero-copy strings and typed `u8` arrays on the read path (#12).

## 0.5.0 - 2026-03-13

MATLAB v7.3 `.mat` conversion from BEVE, with the output cross-checked against matio (#11).

## 0.4.3 - 2026-03-05

Fixed: an empty hash map did not round-trip (#10).

## 0.4.2 - 2026-02-27

- Matrix support gains owned and raw forms alongside the borrowed one (`MatrixOwned<T>`, `RawMatrix`, `DecodedMatrix<T>`) and `decode_matrix_slice`, whose `MatrixDecodeMode` chooses whether the payload is decoded or left raw (#9).
- `to_vec_into` encodes into a caller-supplied `Vec`, so a repeated encode can reuse one buffer (#9).

## 0.4.1 - 2026-02-24

`validate_slice` and `validate_reader` check that a document is well formed without decoding it into a type (#8).

## 0.4.0 - 2025-12-02

- `beve::Value` and `from_value` for dynamic documents, where the type is not known at compile time (#5).
- 128-bit numbers move behind a boxed `BigInt` (and `BigIntKey` for object keys), so the common `Number` and `Key` cases no longer pay for the width of the largest variant (#6).
- Value conversion failures report through a `ValueError` enum rather than a string (#7).

## 0.3.0 - 2025-10-16

Complex number support and boolean typed arrays (#4).

## 0.2.0 - 2025-10-16

JSON round-trip conversion (#3).

## 0.1.2 - 2025-09-24

Crate metadata and documentation links.

## 0.1.1 - 2025-09-24

Fixed the repository link in the crate metadata.

## 0.1.0 - 2025-09-24

First release: BEVE serialization and deserialization with serde, half-precision float support, property tests, and cross-language interop tests against Glaze.
