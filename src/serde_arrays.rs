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
//!     iq: Vec<beve::Complex<f32>>,   // or any `ComplexElement` of `f32`
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
//! `complex_array::*` reinterprets wire bytes as the element type in both
//! directions, so the element `T` must be [`bytemuck::AnyBitPattern`] (every bit
//! pattern is a valid value) and must carry an `unsafe impl` of
//! [`crate::ComplexElement`] naming the scalar as its `Component`. That
//! associated type appears in each module's bound, so `complex_array::f32`
//! accepts only pairs of `f32` — a same-width class such as `i32` is a compile
//! error rather than a silent misread. `beve::Complex` qualifies for every
//! scalar; `num_complex::Complex` does under this crate's `num-complex`
//! feature, which also needs `num-complex`'s own `bytemuck` feature for the
//! decode half.
//!
//! **A different component class on the wire still decodes.** A complex `i16`
//! array read into a `Vec<Complex<f32>>` field converts in one pass rather than
//! one element at a time; read into an integer field of another width, it falls
//! back to the element-wise path, which range-checks each component. Either way
//! the field decodes to exactly what it would without the annotation: these
//! helpers are a throughput change and never a semantic one.

use bytemuck::AnyBitPattern;
use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::de::reserve_from_hint;
use crate::ext::{Complex, ComplexElement, ComplexSlice, serialize_typed_slice};
use crate::fast::BeveTypedSlice;

// ---------------------------------------------------------------------------
// The markers for both array kinds live in `crate::ext`, next to the wrapper
// types that write them: `typed_array_tag` / `NT_TYPED_ARRAY_*` and
// `complex_array_tag` / `NT_COMPLEX_ARRAY_*`. The `#[serde(with = ...)]`
// helpers below name those constants rather than repeating the string
// literals, so an encode marker and its decode marker cannot drift apart --
// a typo in a repeated literal would not fail to compile, it would silently
// drop the field to the element-wise path.
// ---------------------------------------------------------------------------

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
        let mut out = Vec::with_capacity(reserve_from_hint::<S>(seq.size_hint()));
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
        let mut out = Vec::with_capacity(reserve_from_hint::<T>(seq.size_hint()));
        while let Some(x) = seq.next_element::<T>()? {
            out.push(x);
        }
        Ok(out)
    }
    fn visit_newtype_struct<D: Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
        Vec::<T>::deserialize(d)
    }
}

/// Backstops against a wrong `unsafe impl ComplexElement`. Alignment matters
/// because `#[repr(packed)]` meets the rest of the contract yet drops to align
/// 1, and the serialize half builds a `&[Complex<S>]` over the caller's slice,
/// which must be aligned even when empty.
fn assert_complex_layout<S, T>() {
    assert_eq!(
        core::mem::align_of::<T>(),
        core::mem::align_of::<Complex<S>>(),
        "beve complex array: ComplexElement impl has the wrong element alignment"
    );
    assert_eq!(
        core::mem::size_of::<T>(),
        core::mem::size_of::<Complex<S>>(),
        "beve complex array: ComplexElement impl has the wrong element size"
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

/// `#[serde(with = "beve::complex_array::<scalar>")]` for a `Vec<T>` field whose
/// `T` is a [`ComplexElement`] of that scalar: bulk encode and bulk (memcpy)
/// decode of a complex array.
///
/// The component class is part of the bound in both directions, so a same-width
/// class cannot cross: pairs of `i32` into the `f32` module would otherwise
/// encode the payload unchanged under an `f32` class tag.
///
/// ```compile_fail
/// use serde::{Serialize, Deserialize};
///
/// #[derive(Clone, Copy, Serialize, Deserialize)]
/// #[repr(C)]
/// struct IntPair {
///     re: i32,
///     im: i32,
/// }
///
/// // SAFETY: `#[repr(C)]` over two `i32`, real first.
/// unsafe impl beve::ComplexElement for IntPair {
///     type Component = i32;
/// }
/// unsafe impl bytemuck::Zeroable for IntPair {}
/// unsafe impl bytemuck::Pod for IntPair {}
///
/// #[derive(Serialize, Deserialize)]
/// struct Frame {
///     #[serde(with = "beve::complex_array::f32")]
///     iq: Vec<IntPair>,
/// }
/// ```
///
/// The same field compiles against its own class:
///
/// ```
/// use serde::{Serialize, Deserialize};
///
/// #[derive(Clone, Copy, Serialize, Deserialize)]
/// #[repr(C)]
/// struct IntPair {
///     re: i32,
///     im: i32,
/// }
///
/// // SAFETY: `#[repr(C)]` over two `i32`, real first.
/// unsafe impl beve::ComplexElement for IntPair {
///     type Component = i32;
/// }
/// unsafe impl bytemuck::Zeroable for IntPair {}
/// unsafe impl bytemuck::Pod for IntPair {}
///
/// #[derive(Serialize, Deserialize)]
/// struct Frame {
///     #[serde(with = "beve::complex_array::i32")]
///     iq: Vec<IntPair>,
/// }
/// ```
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
                    T: ComplexElement<Component = $scalar> + Serialize,
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
                    // SAFETY: `T: ComplexElement<Component = $scalar>` promises
                    // `T` is two initialized `$scalar`, real first — the layout
                    // of `Complex<$scalar>`, whose width and alignment the
                    // assert above confirms.
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
                    T: ComplexElement<Component = $scalar> + AnyBitPattern + Deserialize<'de>,
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

    complex_with!(f32, f32, crate::ext::NT_COMPLEX_ARRAY_F32);
    complex_with!(f64, f64, crate::ext::NT_COMPLEX_ARRAY_F64);
    complex_with!(i8, i8, crate::ext::NT_COMPLEX_ARRAY_I8);
    complex_with!(i16, i16, crate::ext::NT_COMPLEX_ARRAY_I16);
    complex_with!(i32, i32, crate::ext::NT_COMPLEX_ARRAY_I32);
    complex_with!(i64, i64, crate::ext::NT_COMPLEX_ARRAY_I64);
    complex_with!(i128, i128, crate::ext::NT_COMPLEX_ARRAY_I128);
    complex_with!(u8, u8, crate::ext::NT_COMPLEX_ARRAY_U8);
    complex_with!(u16, u16, crate::ext::NT_COMPLEX_ARRAY_U16);
    complex_with!(u32, u32, crate::ext::NT_COMPLEX_ARRAY_U32);
    complex_with!(u64, u64, crate::ext::NT_COMPLEX_ARRAY_U64);
    complex_with!(u128, u128, crate::ext::NT_COMPLEX_ARRAY_U128);
}

#[cfg(test)]
mod marker_tests {
    use super::*;
    use core::cell::Cell;
    use core::fmt;

    /// A `Deserializer` that records the newtype-struct name it is asked for,
    /// then refuses.
    ///
    /// A wrong marker is invisible to a round-trip test: beve falls through to
    /// the element-wise path and returns identical values, just slowly. Only
    /// watching the name catches it.
    struct NameSpy<'a>(&'a Cell<&'static str>);

    #[derive(Debug)]
    struct Stop;

    impl fmt::Display for Stop {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("stop")
        }
    }
    impl std::error::Error for Stop {}
    impl de::Error for Stop {
        fn custom<T: fmt::Display>(_: T) -> Self {
            Stop
        }
    }

    impl<'de> Deserializer<'de> for NameSpy<'_> {
        type Error = Stop;

        fn deserialize_newtype_struct<V: Visitor<'de>>(
            self,
            name: &'static str,
            _visitor: V,
        ) -> Result<V::Value, Stop> {
            self.0.set(name);
            Err(Stop)
        }

        fn deserialize_any<V: Visitor<'de>>(self, _visitor: V) -> Result<V::Value, Stop> {
            Err(Stop)
        }

        serde::forward_to_deserialize_any! {
            bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
            bytes byte_buf option unit unit_struct seq tuple tuple_struct map
            struct enum identifier ignored_any
        }
    }

    /// Every `complex_array::*` module must ask for its OWN marker. A typo, or a
    /// decode that skips the visitor entirely, leaves the name unset or wrong.
    #[test]
    fn each_complex_array_module_asks_for_its_own_marker() {
        macro_rules! check {
            ($module:ident, $scalar:ty, $nt:expr) => {{
                let seen = Cell::new("");
                let _ = complex_array::$module::deserialize::<_, Complex<$scalar>>(NameSpy(&seen));
                assert_eq!(
                    seen.get(),
                    $nt,
                    concat!(
                        "complex_array::",
                        stringify!($module),
                        " must request its own marker"
                    )
                );
            }};
        }

        check!(f32, f32, crate::ext::NT_COMPLEX_ARRAY_F32);
        check!(f64, f64, crate::ext::NT_COMPLEX_ARRAY_F64);
        check!(i8, i8, crate::ext::NT_COMPLEX_ARRAY_I8);
        check!(i16, i16, crate::ext::NT_COMPLEX_ARRAY_I16);
        check!(i32, i32, crate::ext::NT_COMPLEX_ARRAY_I32);
        check!(i64, i64, crate::ext::NT_COMPLEX_ARRAY_I64);
        check!(i128, i128, crate::ext::NT_COMPLEX_ARRAY_I128);
        check!(u8, u8, crate::ext::NT_COMPLEX_ARRAY_U8);
        check!(u16, u16, crate::ext::NT_COMPLEX_ARRAY_U16);
        check!(u32, u32, crate::ext::NT_COMPLEX_ARRAY_U32);
        check!(u64, u64, crate::ext::NT_COMPLEX_ARRAY_U64);
        check!(u128, u128, crate::ext::NT_COMPLEX_ARRAY_U128);
    }

    /// The twelve markers must be distinct, or two classes share a decode path.
    #[test]
    fn complex_array_markers_are_all_distinct() {
        let names = [
            crate::ext::NT_COMPLEX_ARRAY_F32,
            crate::ext::NT_COMPLEX_ARRAY_F64,
            crate::ext::NT_COMPLEX_ARRAY_I8,
            crate::ext::NT_COMPLEX_ARRAY_I16,
            crate::ext::NT_COMPLEX_ARRAY_I32,
            crate::ext::NT_COMPLEX_ARRAY_I64,
            crate::ext::NT_COMPLEX_ARRAY_I128,
            crate::ext::NT_COMPLEX_ARRAY_U8,
            crate::ext::NT_COMPLEX_ARRAY_U16,
            crate::ext::NT_COMPLEX_ARRAY_U32,
            crate::ext::NT_COMPLEX_ARRAY_U64,
            crate::ext::NT_COMPLEX_ARRAY_U128,
        ];
        let mut sorted = names.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len(), "markers must be distinct");
    }
}
