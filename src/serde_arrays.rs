//! `#[serde(with = ...)]` helpers that decode a numeric or complex array field
//! at bulk (memcpy) speed, the symmetric counterpart to the bulk *encode* that
//! beve already does.
//!
//! beve's serializer coalesces a homogeneous `Vec<f64>` / `Vec<Complex<f32>>`
//! into a bulk typed/complex array on the wire, but serde's `Vec<T>::deserialize`
//! decodes element by element, and there is no hook to bulk-fill a nested field.
//! These helpers close that gap: annotate the field and decoding routes through a
//! newtype marker that beve's deserializer recognizes and bulk-copies straight
//! into the result `Vec`.
//!
//! ```ignore
//! use serde::{Serialize, Deserialize};
//!
//! #[derive(Serialize, Deserialize)]
//! struct Frame {
//!     #[serde(with = "beve::typed::f64")]
//!     samples: Vec<f64>,
//!     #[serde(with = "beve::complex_array::f32")]
//!     iq: Vec<beve::Complex<f32>>,   // or any layout-compatible complex type
//! }
//! ```
//!
//! Both `from_slice` and `from_reader_streaming` hit the bulk path. The bulk path
//! needs the annotation because serde exposes no way to bulk-fill an
//! *un-annotated* `Vec<T>` field — the same reason `serde_bytes` exists for
//! `&[u8]`.
//!
//! **Format-agnostic.** The bulk byte path is used only for non-human-readable
//! serde formats (beve). Human-readable formats (e.g. JSON) get the portable
//! element-wise form — each element via its own `Serialize`/`Deserialize` — so a
//! field using these helpers still round-trips through JSON, as long as the
//! element type itself has a portable serde representation (every scalar does, as
//! does `num_complex::Complex`; note `beve::Complex`'s own representation is
//! beve-specific). Non-beve *binary* formats are not special-cased and will
//! mis-read the bulk form.
//!
//! `complex_array::*` decodes the bulk (beve) form by reinterpreting wire bytes as
//! the element type, so the element `T` must be [`bytemuck::AnyBitPattern`] (every
//! bit pattern is a valid value) and layout-compatible with `Complex<scalar>`.
//! `beve::Complex` qualifies, as does `num_complex::Complex` with its `bytemuck`
//! feature enabled.

use bytemuck::AnyBitPattern;
use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::ext::{Complex, ComplexSlice, serialize_typed_slice};
use crate::fast::BeveTypedSlice;

// ---------------------------------------------------------------------------
// Decode markers for complex arrays (the typed-array markers reuse
// `crate::ext::typed_array_tag` / `NT_TYPED_ARRAY_*`).
// ---------------------------------------------------------------------------

/// Map a complex-array newtype marker name to `(class, byte_code, scalar_size)`
/// of its element scalar. `None` for any other name.
pub(crate) fn complex_array_tag(name: &str) -> Option<(u8, u8, usize)> {
    macro_rules! arm {
        ($($scalar:ty => $name:literal),* $(,)?) => {
            match name {
                $(
                    $name => Some((
                        <$scalar as BeveTypedSlice>::CLASS,
                        <$scalar as BeveTypedSlice>::BYTE_CODE,
                        <$scalar as BeveTypedSlice>::ELEM_SIZE,
                    )),
                )*
                _ => None,
            }
        };
    }
    arm! {
        f32 => "__beve_complex_array_f32",
        f64 => "__beve_complex_array_f64",
        i8 => "__beve_complex_array_i8",
        i16 => "__beve_complex_array_i16",
        i32 => "__beve_complex_array_i32",
        i64 => "__beve_complex_array_i64",
        i128 => "__beve_complex_array_i128",
        u8 => "__beve_complex_array_u8",
        u16 => "__beve_complex_array_u16",
        u32 => "__beve_complex_array_u32",
        u64 => "__beve_complex_array_u64",
        u128 => "__beve_complex_array_u128",
    }
}

// ---------------------------------------------------------------------------
// Shared bulk-copy core
// ---------------------------------------------------------------------------

/// Copy a little-endian payload straight into a `Vec<E>`. The `E: AnyBitPattern`
/// bound guarantees any byte sequence is a valid `E`, which is what makes the
/// memcpy sound; `payload.len()` must be a whole multiple of `size_of::<E>()`
/// (the decoders guarantee this). `scalar_size` is the byte-swap granularity on
/// big-endian targets (the element's *scalar* width).
fn bytes_to_vec<E: AnyBitPattern>(payload: &[u8], scalar_size: usize) -> Vec<E> {
    let elem = core::mem::size_of::<E>();
    debug_assert!(elem != 0 && payload.len().is_multiple_of(elem));
    let n = payload.len() / elem.max(1);
    let mut out: Vec<E> = Vec::with_capacity(n);
    if n != 0 {
        let nbytes = n * elem;
        // SAFETY: `out` has capacity for `n` `E`s = `nbytes` bytes; `payload` is
        // exactly that long; `E: AnyBitPattern` accepts any bit pattern.
        unsafe {
            core::ptr::copy_nonoverlapping(payload.as_ptr(), out.as_mut_ptr() as *mut u8, nbytes);
            out.set_len(n);
        }
        #[cfg(not(target_endian = "little"))]
        {
            // Wire is little-endian; reverse each scalar in place on big-endian.
            let bytes =
                unsafe { core::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, nbytes) };
            for s in bytes.chunks_exact_mut(scalar_size) {
                s.reverse();
            }
        }
    }
    #[cfg(target_endian = "little")]
    let _ = scalar_size;
    out
}

// ---------------------------------------------------------------------------
// Visitors
// ---------------------------------------------------------------------------

/// Visitor for a typed numeric array decoded into `Vec<S>`.
struct TypedArrayVisitor<S>(core::marker::PhantomData<S>);

impl<'de, S> Visitor<'de> for TypedArrayVisitor<S>
where
    S: BeveTypedSlice + AnyBitPattern + Deserialize<'de>,
{
    type Value = Vec<S>;

    fn expecting(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "a beve typed numeric array")
    }

    fn visit_borrowed_bytes<E: de::Error>(self, v: &'de [u8]) -> Result<Self::Value, E> {
        Ok(bytes_to_vec::<S>(v, S::ELEM_SIZE))
    }
    fn visit_bytes<E: de::Error>(self, v: &[u8]) -> Result<Self::Value, E> {
        Ok(bytes_to_vec::<S>(v, S::ELEM_SIZE))
    }
    fn visit_byte_buf<E: de::Error>(self, v: Vec<u8>) -> Result<Self::Value, E> {
        Ok(bytes_to_vec::<S>(&v, S::ELEM_SIZE))
    }
    fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let mut out = Vec::with_capacity(seq.size_hint().unwrap_or(0));
        while let Some(x) = seq.next_element::<S>()? {
            out.push(x);
        }
        Ok(out)
    }
    fn visit_newtype_struct<D: Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
        Vec::<S>::deserialize(d)
    }
}

/// Visitor for a complex array decoded into `Vec<T>`, where `T` is layout-
/// compatible with `Complex<S>` (two contiguous `S` scalars: re then im).
struct ComplexArrayVisitor<S, T>(core::marker::PhantomData<(S, T)>);

impl<'de, S, T> Visitor<'de> for ComplexArrayVisitor<S, T>
where
    S: BeveTypedSlice,
    T: AnyBitPattern + Deserialize<'de>,
{
    type Value = Vec<T>;

    fn expecting(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "a beve complex array")
    }

    fn visit_borrowed_bytes<E: de::Error>(self, v: &'de [u8]) -> Result<Self::Value, E> {
        Ok(bytes_to_vec::<T>(v, S::ELEM_SIZE))
    }
    fn visit_bytes<E: de::Error>(self, v: &[u8]) -> Result<Self::Value, E> {
        Ok(bytes_to_vec::<T>(v, S::ELEM_SIZE))
    }
    fn visit_byte_buf<E: de::Error>(self, v: Vec<u8>) -> Result<Self::Value, E> {
        Ok(bytes_to_vec::<T>(&v, S::ELEM_SIZE))
    }
    fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let mut out = Vec::with_capacity(seq.size_hint().unwrap_or(0));
        while let Some(x) = seq.next_element::<T>()? {
            out.push(x);
        }
        Ok(out)
    }
    fn visit_newtype_struct<D: Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
        Vec::<T>::deserialize(d)
    }
}

/// Assert `T` is layout-compatible with `Complex<S>` (same size and alignment),
/// the safety contract the bulk copy relies on. Field order is the caller's.
fn assert_complex_layout<S, T>() {
    assert_eq!(
        core::mem::size_of::<T>(),
        core::mem::size_of::<Complex<S>>(),
        "beve complex array: element type size mismatch"
    );
    assert_eq!(
        core::mem::align_of::<T>(),
        core::mem::align_of::<Complex<S>>(),
        "beve complex array: element type alignment mismatch"
    );
}

// ---------------------------------------------------------------------------
// Public `with` modules
// ---------------------------------------------------------------------------

/// `#[serde(with = "beve::typed::<scalar>")]` for a `Vec<scalar>` field: bulk
/// encode and bulk (memcpy) decode of a typed numeric array.
pub mod typed {
    use super::*;

    macro_rules! typed_with {
        ($module:ident, $scalar:ty, $nt:expr) => {
            #[doc = concat!("Bulk `Vec<", stringify!($scalar), ">` codec for `#[serde(with = ...)]`.")]
            pub mod $module {
                use super::*;

                pub fn serialize<S: Serializer>(
                    value: &[$scalar],
                    serializer: S,
                ) -> Result<S::Ok, S::Error> {
                    serialize_typed_slice(value, $nt, serializer)
                }

                pub fn deserialize<'de, D: Deserializer<'de>>(
                    deserializer: D,
                ) -> Result<Vec<$scalar>, D::Error> {
                    deserializer.deserialize_newtype_struct(
                        $nt,
                        TypedArrayVisitor::<$scalar>(core::marker::PhantomData),
                    )
                }
            }
        };
    }

    typed_with!(i8, i8, crate::ext::NT_TYPED_ARRAY_I8);
    typed_with!(i16, i16, crate::ext::NT_TYPED_ARRAY_I16);
    typed_with!(i32, i32, crate::ext::NT_TYPED_ARRAY_I32);
    typed_with!(i64, i64, crate::ext::NT_TYPED_ARRAY_I64);
    typed_with!(i128, i128, crate::ext::NT_TYPED_ARRAY_I128);
    typed_with!(u8, u8, crate::ext::NT_TYPED_ARRAY_U8);
    typed_with!(u16, u16, crate::ext::NT_TYPED_ARRAY_U16);
    typed_with!(u32, u32, crate::ext::NT_TYPED_ARRAY_U32);
    typed_with!(u64, u64, crate::ext::NT_TYPED_ARRAY_U64);
    typed_with!(u128, u128, crate::ext::NT_TYPED_ARRAY_U128);
    typed_with!(f32, f32, crate::ext::NT_TYPED_ARRAY_F32);
    typed_with!(f64, f64, crate::ext::NT_TYPED_ARRAY_F64);
}

/// `#[serde(with = "beve::complex_array::<scalar>")]` for a `Vec<T>` field where
/// `T` is layout-compatible with `Complex<scalar>`: bulk encode and bulk (memcpy)
/// decode of a complex array.
pub mod complex_array {
    use super::*;

    macro_rules! complex_with {
        ($module:ident, $scalar:ty, $nt:expr) => {
            #[doc = concat!("Bulk `Vec<Complex<", stringify!($scalar), ">>` codec for `#[serde(with = ...)]`.")]
            pub mod $module {
                use super::*;

                pub fn serialize<S, T>(value: &[T], serializer: S) -> Result<S::Ok, S::Error>
                where
                    S: Serializer,
                    T: Serialize,
                {
                    // Human-readable formats get the portable element-wise form
                    // (each `T` via its own `Serialize`), so the field round-trips
                    // through e.g. JSON for a portable element type.
                    if serializer.is_human_readable() {
                        use serde::ser::SerializeSeq;
                        let mut seq = serializer.serialize_seq(Some(value.len()))?;
                        for v in value {
                            seq.serialize_element(v)?;
                        }
                        return seq.end();
                    }
                    assert_complex_layout::<$scalar, T>();
                    // SAFETY: `T` is layout-compatible with `Complex<$scalar>`
                    // (asserted above), so a `&[T]` is a valid `&[Complex<$scalar>]`.
                    let slice: &[Complex<$scalar>] = unsafe {
                        core::slice::from_raw_parts(
                            value.as_ptr() as *const Complex<$scalar>,
                            value.len(),
                        )
                    };
                    ComplexSlice(slice).serialize(serializer)
                }

                pub fn deserialize<'de, D, T>(deserializer: D) -> Result<Vec<T>, D::Error>
                where
                    D: Deserializer<'de>,
                    T: AnyBitPattern + Deserialize<'de>,
                {
                    assert_complex_layout::<$scalar, T>();
                    deserializer.deserialize_newtype_struct(
                        $nt,
                        ComplexArrayVisitor::<$scalar, T>(core::marker::PhantomData),
                    )
                }
            }
        };
    }

    complex_with!(f32, f32, "__beve_complex_array_f32");
    complex_with!(f64, f64, "__beve_complex_array_f64");
    complex_with!(i8, i8, "__beve_complex_array_i8");
    complex_with!(i16, i16, "__beve_complex_array_i16");
    complex_with!(i32, i32, "__beve_complex_array_i32");
    complex_with!(i64, i64, "__beve_complex_array_i64");
    complex_with!(i128, i128, "__beve_complex_array_i128");
    complex_with!(u8, u8, "__beve_complex_array_u8");
    complex_with!(u16, u16, "__beve_complex_array_u16");
    complex_with!(u32, u32, "__beve_complex_array_u32");
    complex_with!(u64, u64, "__beve_complex_array_u64");
    complex_with!(u128, u128, "__beve_complex_array_u128");
}
