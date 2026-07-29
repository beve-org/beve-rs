# Changelog

## 5.0.0

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

`EnumEncoding::String` was the closest thing to a default, and it is now simply the behavior. Callers who passed options only for that can drop the argument. For any other shape, annotate the type: `#[serde(tag = "...")]` for internally tagged (the shape a Glaze `std::variant` with `tag`/`ids` produces), `#[serde(tag, content)]` for adjacently tagged, `#[serde(untagged)]` for a bare value.

### Added: streaming MAT v7.3 conversion

`beve_slice_to_mat_v73_writer` converts a BEVE payload into a MATLAB v7.3 file written front-to-back onto a `Write` sink. It produces byte-for-byte what `beve_slice_to_mat_v73_bytes` returns, but never holds the assembled file, so the output size no longer bounds what can be converted. The sink is never seeked, so it may be a socket as readily as a file.

The `mat` feature now requires **hdf5-pure 0.30** (was 0.28), which brings `NullPolicy::Omit`.

### Fixed

- An unknown-length sequence whose first element was a newtype variant wrote an array header counting one element too few, so the tail decoded as trailing data and `from_slice` silently returned a short sequence.
- A unit-variant target reading a variant that carries a payload (ordinary schema drift) behaved differently in the two readers: the streaming reader substituted the discarded payload for the next element, while the buffered reader guessed from how many bytes were left in the document and could consume a sibling. Both now discard exactly the value the header declares, and report a malformed one instead of swallowing the error.
- A truncated single-key variant object, whose header promises a value that is not present, is now rejected rather than decoding as a unit variant.

### Notes

- `docs/enums.md` is rewritten for Version 2.
- Interop against a pre-Version-2 Glaze cannot exercise variants; those cases are gated behind `GLAZE_INTEROP_V2=1` pending `stephenberry/glaze#2707`.
