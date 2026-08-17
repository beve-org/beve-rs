use crate::de::from_slice;
use crate::fast::{BeveTypedSlice, to_vec_complex_slice, write_bool_slice, write_typed_slice};
use crate::header::{ARRAY_UNSIGNED, EXT_MATRICES, TYPE_TYPED_ARRAY, make_extension_header};
use crate::size::write_size;
use crate::value::Value;
use core::any::TypeId;
use half::{bf16, f16};
use serde::{Deserialize, Serialize, de, ser};

pub(crate) const NT_RAW_VALUE: &str = "__beve_raw_value";
pub(crate) const NT_COMPLEX: &str = "__beve_complex";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixDecodeMode {
    Materialized,
    Raw,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DecodedMatrix<T> {
    Materialized(MatrixOwned<T>),
    Raw(RawMatrix),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawMatrix {
    pub layout: MatrixLayout,
    pub extents: Vec<usize>,
    pub value: Value,
}

struct RawBytes<'a>(&'a [u8]);

impl Serialize for RawBytes<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> core::result::Result<S::Ok, S::Error> {
        s.serialize_bytes(self.0)
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex<T> {
    pub re: T,
    pub im: T,
}

// SAFETY: `Complex<T>` is `#[repr(C)]` over two `T` fields with no padding, so it
// is zeroable / all-bits-valid exactly when `T` is. These let `Complex<scalar>`
// be decoded through the bulk `complex_array::*` path (which requires
// `AnyBitPattern`); `bytemuck` provides the blanket `Pod: AnyBitPattern`.
unsafe impl<T: bytemuck::Zeroable> bytemuck::Zeroable for Complex<T> {}
unsafe impl<T: bytemuck::Pod> bytemuck::Pod for Complex<T> {}

/// A type the bulk complex-array helpers may read as raw bytes: two contiguous
/// [`Self::Component`] scalars, real part first.
///
/// Implementing this is how a foreign complex type — `num_complex::Complex<T>`,
/// or your own `#[repr(C)]` pair — reaches [`crate::complex`]'s
/// `serialize_with` helpers and [`crate::complex_array`]'s `serde(with)`
/// modules. [`Complex`] implements it for every scalar BEVE encodes, and the
/// `num-complex` feature adds `num_complex::Complex<T>`.
///
/// `Component` is an associated type so the helpers can name it in their
/// bounds: `complex::f32_array` requires `Component = f32`, so a same-width
/// class cannot cross.
///
/// # Safety
///
/// `Self` must have the layout `#[repr(C)] struct { re: Component, im: Component }`:
/// two `Component` values, real first, no padding, every byte initialized. The
/// helpers read `size_of_val(slice)` bytes out of a `&[Self]`, so padding leaks
/// uninitialized memory into the output and a pointer field writes its address.
///
/// `Self` must also share `Component`'s alignment. A plain `#[repr(C)]` pair
/// does; `#[repr(packed)]` satisfies everything above but drops to align 1, and
/// the helpers build a `&[Complex<Component>]` over your slice.
///
/// Size and alignment are checked at run time. Padding and field order cannot
/// be checked and are the implementor's promise.
///
/// # A padded type needs no impl, and gets none
///
/// `#[repr(C)] struct { re: i16, tag: u8 }` is size 4, align 2 — identical to
/// `Complex<i16>` — so no run-time check can reject it. Absent an `unsafe impl`
/// it does not compile:
///
/// ```compile_fail
/// #[derive(Clone, Copy)]
/// #[repr(C)]
/// struct Padded {
///     re: i16,
///     tag: u8,
/// }
///
/// fn encode<S: serde::Serializer>(data: &[Padded], s: S) {
///     let _ = beve::complex::i16_array(data, s);
/// }
/// ```
///
/// # A same-width class is refused
///
/// Pairs of `i32` cannot reach the `f32` helper, which would otherwise emit the
/// payload unchanged under an `f32` class tag:
///
/// ```compile_fail
/// #[derive(Clone, Copy)]
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
///
/// fn encode<S: serde::Serializer>(data: &[IntPair], s: S) {
///     let _ = beve::complex::f32_array(data, s);
/// }
/// ```
///
/// The same type reaches its own class's helper:
///
/// ```
/// #[derive(Clone, Copy)]
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
///
/// fn encode<S: serde::Serializer>(data: &[IntPair], s: S) {
///     let _ = beve::complex::i32_array(data, s);
/// }
/// ```
pub unsafe trait ComplexElement {
    /// The scalar type of the real and imaginary parts.
    type Component;
}

/// One `ComplexElement` impl per scalar with a bulk complex-array helper, for
/// [`Complex`] and (under `num-complex`) `num_complex::Complex`.
///
/// Enumerated, not blanket over `T: BeveTypedSlice`: that trait is public and
/// safe, so a blanket impl would extend this crate's unsafe promise to any
/// downstream type implementing it, padding and all. `f16`/`bf16` are absent
/// because no `complex::f16_array` helper exists to reach.
macro_rules! impl_complex_element {
    ($($scalar:ty),* $(,)?) => {
        $(
            // SAFETY: `#[repr(C)]` over two `$scalar`, real first. Equal-width
            // scalars leave no padding, every bit pattern of a fixed-width
            // scalar is initialized, and the pair's alignment is the scalar's.
            unsafe impl ComplexElement for Complex<$scalar> {
                type Component = $scalar;
            }

            // SAFETY: `num_complex::Complex<T>` is `#[repr(C)]`, `re` then `im`
            // — the same shape and alignment as this crate's `Complex<T>`.
            #[cfg(feature = "num-complex")]
            #[cfg_attr(docsrs, doc(cfg(feature = "num-complex")))]
            unsafe impl ComplexElement for num_complex::Complex<$scalar> {
                type Component = $scalar;
            }

            // The run-time asserts only catch a wrong `Component` where the
            // widths differ. Pin the layout of both impls here instead.
            const _: () = {
                assert!(
                    core::mem::size_of::<Complex<$scalar>>() == 2 * core::mem::size_of::<$scalar>()
                );
                assert!(
                    core::mem::align_of::<Complex<$scalar>>() == core::mem::align_of::<$scalar>()
                );
            };
            #[cfg(feature = "num-complex")]
            const _: () = {
                assert!(
                    core::mem::size_of::<num_complex::Complex<$scalar>>()
                        == 2 * core::mem::size_of::<$scalar>()
                );
                assert!(
                    core::mem::align_of::<num_complex::Complex<$scalar>>()
                        == core::mem::align_of::<$scalar>()
                );
            };
        )*
    };
}

impl_complex_element!(f32, f64, i8, i16, i32, i64, i128, u8, u16, u32, u64, u128);

/// Generates `Serialize` for `Complex<$scalar>` using a single `NT_COMPLEX` newtype.
/// Payload layout: `[class: u8, byte_code: u8, re_le_bytes..., im_le_bytes...]`
macro_rules! impl_complex_serialize {
    ($scalar:ty) => {
        impl Serialize for Complex<$scalar> {
            fn serialize<S: serde::Serializer>(
                &self,
                s: S,
            ) -> core::result::Result<S::Ok, S::Error> {
                const ELEM: usize = core::mem::size_of::<$scalar>();
                const TOTAL: usize = 2 + ELEM * 2;
                let mut bytes = [0u8; TOTAL];
                bytes[0] = <$scalar as BeveTypedSlice>::CLASS;
                bytes[1] = <$scalar as BeveTypedSlice>::BYTE_CODE;
                bytes[2..2 + ELEM].copy_from_slice(&self.re.to_le_bytes());
                bytes[2 + ELEM..TOTAL].copy_from_slice(&self.im.to_le_bytes());
                s.serialize_newtype_struct(NT_COMPLEX, &RawBytes(&bytes))
            }
        }
    };
}

impl_complex_serialize!(f16);
impl_complex_serialize!(bf16);
impl_complex_serialize!(f32);
impl_complex_serialize!(f64);
impl_complex_serialize!(i8);
impl_complex_serialize!(i16);
impl_complex_serialize!(i32);
impl_complex_serialize!(i64);
impl_complex_serialize!(i128);
impl_complex_serialize!(u8);
impl_complex_serialize!(u16);
impl_complex_serialize!(u32);
impl_complex_serialize!(u64);
impl_complex_serialize!(u128);

impl<'de, T> serde::Deserialize<'de> for Complex<T>
where
    T: serde::Deserialize<'de>,
{
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> core::result::Result<Self, D::Error> {
        struct V<T>(core::marker::PhantomData<T>);
        impl<'de, T> serde::de::Visitor<'de> for V<T>
        where
            T: serde::Deserialize<'de>,
        {
            type Value = Complex<T>;
            fn expecting(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                write!(f, "complex number as [re, im]")
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut a: A,
            ) -> core::result::Result<Self::Value, A::Error> {
                let re: T = a
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::custom("missing real"))?;
                let im: T = a
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::custom("missing imag"))?;
                Ok(Complex { re, im })
            }
        }
        d.deserialize_seq(V(core::marker::PhantomData))
    }
}

/// A contiguous slice of complex values, serialized as a BEVE complex array
/// with a single bulk write of the borrowed payload — the complex counterpart
/// of [`TypedSlice`].
///
/// Both families of complex-array helpers route through this type, so a field
/// using either gets the bulk path with no further opt-in: the [`complex`]
/// `serialize_with` helpers, and the encode half of the
/// [`complex_array`](crate::complex_array) `serde(with)` helpers.
///
/// The encoded bytes are identical to serializing the same values one at a
/// time, so this is a throughput change and not a format change; the
/// `complex_slice_bulk_matches_element_wise` tests pin that.
///
/// An **empty** slice keeps the element-wise encoding (a generic empty array,
/// not a zero-length complex array), which is what this wrapper has always
/// written. That differs from [`TypedSlice`], whose empty form is a typed empty
/// array, and from [`crate::to_writer_complex_slice`], whose empty form is a
/// zero-length complex array. All three decode back to an empty `Vec`, but the
/// distinction is visible to consumers that read the element type off the
/// header — notably the MATLAB export, where a typed empty array becomes a cell
/// and a generic one becomes `[]`.
pub struct ComplexSlice<'a, T>(pub &'a [Complex<T>]);

/// Shared `Serialize` body for every `ComplexSlice<T>`: hand the interleaved
/// `(re, im)` payload to the serializer as borrowed bytes tagged by `name`, so
/// beve emits it with one `write_all` instead of a `serialize_element` per
/// sample. The tagged form is beve-private, which costs nothing in portability
/// that this wrapper had to begin with: `Complex<T>`'s own `Serialize` already
/// emits a beve-private newtype, so the element-wise path was never portable to
/// another binary format either.
///
/// Three cases keep the element-wise path, and each is load-bearing:
/// human-readable formats (JSON and friends must still see a sequence),
/// big-endian targets (the in-memory bytes are not little-endian), and the
/// empty slice (see [`ComplexSlice`]'s note on empty encoding).
#[inline]
pub(crate) fn serialize_complex_slice<S, T>(
    slice: &[Complex<T>],
    name: &'static str,
    s: S,
) -> core::result::Result<S::Ok, S::Error>
where
    S: serde::Serializer,
    // Not decoration: `BeveTypedSlice` is what makes the reinterpret below
    // sound, by restricting `T` to the fixed-width scalars whose every bit
    // pattern is valid. Without it the bound would be `Complex<T>: Serialize`,
    // which a padded or niche-carrying `T` could satisfy.
    T: BeveTypedSlice,
    Complex<T>: Serialize,
{
    // `cfg!` rather than `#[cfg]` so the bulk arm is type-checked on every
    // target; it folds to a constant, so big-endian still compiles it out.
    if cfg!(target_endian = "little") && !s.is_human_readable() && !slice.is_empty() {
        // SAFETY: `Complex<T>` is `#[repr(C)]` over two `T`, and `T: BeveTypedSlice`
        // is a fixed-width scalar with every bit pattern valid, so there is no
        // padding and the slice's bytes are exactly the interleaved `(re, im)`
        // payload. Mirrors `fast::to_writer_complex_slice`. The borrowed view is
        // only read inside `serialize_newtype_struct`, during which `slice` stays
        // alive.
        let payload: &[u8] = unsafe {
            core::slice::from_raw_parts(slice.as_ptr() as *const u8, core::mem::size_of_val(slice))
        };
        return s.serialize_newtype_struct(name, &RawBytes(payload));
    }
    let mut seq = s.serialize_seq(Some(slice.len()))?;
    for c in slice {
        ser::SerializeSeq::serialize_element(&mut seq, c)?;
    }
    ser::SerializeSeq::end(seq)
}

/// Generates, per scalar type: the complex-array newtype-name constant, the
/// `Serialize for ComplexSlice<'_, T>` impl, and an arm of the shared
/// [`complex_array_tag`] lookup — the same shape as
/// [`impl_typed_slice_serialize`], and for the same reason: both serializers
/// consult one table instead of hand-maintaining a per-type dispatch.
macro_rules! impl_complex_slice_serialize {
    ($( ($scalar:ty, $nt_const:ident, $nt_name:literal) ),* $(,)?) => {
        $(
            pub(crate) const $nt_const: &str = $nt_name;

            impl<'a> Serialize for ComplexSlice<'a, $scalar> {
                fn serialize<S: serde::Serializer>(
                    &self,
                    s: S,
                ) -> core::result::Result<S::Ok, S::Error> {
                    serialize_complex_slice(self.0, $nt_const, s)
                }
            }
        )*

        /// Map a beve complex-array newtype name to its
        /// `(class, byte_code, scalar_size)`, where `scalar_size` is the width of
        /// one *component* — the same shape [`typed_array_tag`] returns, and the
        /// same units, so the two cannot be confused for each other.
        ///
        /// This is the one table for the complex markers, read by both directions:
        /// [`ComplexSlice`] and the [`crate::complex_array`] helpers set these
        /// names on the way out, and both deserializers match them on the way in.
        /// A writer needing the width of a whole complex VALUE doubles it through
        /// `ser::complex_elem_bytes` rather than carrying a second, differently
        /// scaled copy of the width here.
        pub(crate) fn complex_array_tag(name: &str) -> Option<(u8, u8, usize)> {
            match name {
                $(
                    $nt_const => Some((
                        <$scalar as BeveTypedSlice>::CLASS,
                        <$scalar as BeveTypedSlice>::BYTE_CODE,
                        <$scalar as BeveTypedSlice>::ELEM_SIZE,
                    )),
                )*
                _ => None,
            }
        }
    };
}

impl_complex_slice_serialize! {
    (f16,  NT_COMPLEX_ARRAY_F16,  "__beve_complex_array_f16"),
    (bf16, NT_COMPLEX_ARRAY_BF16, "__beve_complex_array_bf16"),
    (f32,  NT_COMPLEX_ARRAY_F32,  "__beve_complex_array_f32"),
    (f64,  NT_COMPLEX_ARRAY_F64,  "__beve_complex_array_f64"),
    (i8,   NT_COMPLEX_ARRAY_I8,   "__beve_complex_array_i8"),
    (i16,  NT_COMPLEX_ARRAY_I16,  "__beve_complex_array_i16"),
    (i32,  NT_COMPLEX_ARRAY_I32,  "__beve_complex_array_i32"),
    (i64,  NT_COMPLEX_ARRAY_I64,  "__beve_complex_array_i64"),
    (i128, NT_COMPLEX_ARRAY_I128, "__beve_complex_array_i128"),
    (u8,   NT_COMPLEX_ARRAY_U8,   "__beve_complex_array_u8"),
    (u16,  NT_COMPLEX_ARRAY_U16,  "__beve_complex_array_u16"),
    (u32,  NT_COMPLEX_ARRAY_U32,  "__beve_complex_array_u32"),
    (u64,  NT_COMPLEX_ARRAY_U64,  "__beve_complex_array_u64"),
    (u128, NT_COMPLEX_ARRAY_U128, "__beve_complex_array_u128"),
}

// -------- Typed numeric slices (opt-in zero-copy serde bulk path) --------

/// An opt-in wrapper that serializes a contiguous numeric slice `&[T]` as a BEVE
/// typed array via a single bulk write, rather than element-by-element.
///
/// serde delivers sequence elements one at a time and never exposes the backing
/// slice, so a plain `Vec<T>` / `&[T]` field cannot be bulk-written
/// automatically; like `serde_bytes` for `&[u8]`, this wrapper is the opt-in.
/// Use it as a derived-struct field to get the bulk path through
/// [`crate::to_writer_streaming`] / [`crate::to_vec`]:
///
/// ```rust
/// use serde::Serialize;
///
/// #[derive(Serialize)]
/// struct Frame<'a> {
///     samples: beve::TypedSlice<'a, f64>,
/// }
///
/// let data = vec![1.0f64, 2.0, 3.0];
/// let mut buf = Vec::new();
/// beve::to_writer_streaming(&mut buf, &Frame { samples: beve::TypedSlice(&data) }).unwrap();
/// ```
///
/// On little-endian targets the payload is handed to the serializer as borrowed
/// bytes and written with one `write_all` (no copy, no allocation); on big-endian
/// targets a non-empty slice falls back to the per-element sequence path. For a
/// non-empty slice the encoded bytes are identical to serializing the equivalent
/// `Vec<T>` on either target. For an *empty* slice they differ, on every target:
/// `TypedSlice` still emits a typed array of the element's type, whereas a bare
/// empty `Vec<T>` has no element from which to detect the type and encodes as a
/// generic empty array. Both decode back to an empty `Vec<T>`.
pub struct TypedSlice<'a, T>(pub &'a [T]);

/// Shared `Serialize` body for every `TypedSlice<T>`: on little-endian, borrow the
/// slice as bytes and tag it by `name`; on big-endian, fall back to the
/// per-element sequence path (except an empty slice, which has no bytes to convert
/// and so takes the little-endian typed-array path on every target).
#[inline]
pub(crate) fn serialize_typed_slice<S, T>(
    slice: &[T],
    name: &'static str,
    s: S,
) -> core::result::Result<S::Ok, S::Error>
where
    S: serde::Serializer,
    T: Serialize,
{
    // Human-readable formats (JSON, ...) get the portable element-wise sequence,
    // so a field using these helpers still round-trips through them. The bulk
    // newtype/raw-bytes form is meaningful only to beve (and other binary
    // formats that don't special-case the marker would mis-read it).
    if s.is_human_readable() {
        use serde::ser::SerializeSeq;
        let mut seq = s.serialize_seq(Some(slice.len()))?;
        for v in slice {
            seq.serialize_element(v)?;
        }
        return seq.end();
    }
    #[cfg(target_endian = "little")]
    {
        // Sound: `BeveTypedSlice` types are fixed-width no-padding scalars with
        // every bit pattern valid; this mirrors the `fast` module's
        // reinterpret-as-bytes. The borrowed view is only read inside
        // `serialize_newtype_struct`, during which `slice` stays alive.
        let payload: &[u8] = unsafe {
            core::slice::from_raw_parts(slice.as_ptr() as *const u8, core::mem::size_of_val(slice))
        };
        s.serialize_newtype_struct(name, &RawBytes(payload))
    }
    #[cfg(not(target_endian = "little"))]
    {
        // On big-endian the in-memory bytes are not little-endian, so a non-empty
        // slice is converted element-by-element through the sequence path. An
        // *empty* slice has no payload bytes to convert, so it takes the same
        // typed-array newtype path as little-endian, keeping `TypedSlice`
        // byte-identical to `to_writer_typed_slice` (a typed empty array, never a
        // generic one) on every target.
        if slice.is_empty() {
            return s.serialize_newtype_struct(name, &RawBytes(&[]));
        }
        use serde::ser::SerializeSeq;
        let mut seq = s.serialize_seq(Some(slice.len()))?;
        for v in slice {
            seq.serialize_element(v)?;
        }
        seq.end()
    }
}

/// Generates, per scalar type: the typed-array newtype-name constant, the
/// `Serialize for TypedSlice<'_, T>` impl, and an arm of the shared
/// [`typed_array_tag`] lookup. The lookup is the single source of truth that both
/// serializers consult, so neither hand-maintains a per-type dispatch table.
macro_rules! impl_typed_slice_serialize {
    ($( ($scalar:ty, $nt_const:ident, $nt_name:literal) ),* $(,)?) => {
        $(
            pub(crate) const $nt_const: &str = $nt_name;

            impl<'a> Serialize for TypedSlice<'a, $scalar> {
                fn serialize<S: serde::Serializer>(
                    &self,
                    s: S,
                ) -> core::result::Result<S::Ok, S::Error> {
                    serialize_typed_slice(self.0, $nt_const, s)
                }
            }
        )*

        /// Map a beve typed-array newtype name (the tag [`TypedSlice`] sets via
        /// `serialize_newtype_struct`) to its `(class, byte_code, elem_size)`.
        ///
        /// `elem_size` is the true element width, which is **not** always
        /// `1 << byte_code` (e.g. `bf16` uses `byte_code` 0 but is 2 bytes), so
        /// the writing sink divides the borrowed payload length by it to recover
        /// the element count for the SIZE prefix.
        pub(crate) fn typed_array_tag(name: &str) -> Option<(u8, u8, usize)> {
            match name {
                $(
                    $nt_const => Some((
                        <$scalar as BeveTypedSlice>::CLASS,
                        <$scalar as BeveTypedSlice>::BYTE_CODE,
                        <$scalar as BeveTypedSlice>::ELEM_SIZE,
                    )),
                )*
                _ => None,
            }
        }
    };
}

impl_typed_slice_serialize! {
    (i8,   NT_TYPED_ARRAY_I8,   "__beve_typed_array_i8"),
    (i16,  NT_TYPED_ARRAY_I16,  "__beve_typed_array_i16"),
    (i32,  NT_TYPED_ARRAY_I32,  "__beve_typed_array_i32"),
    (i64,  NT_TYPED_ARRAY_I64,  "__beve_typed_array_i64"),
    (i128, NT_TYPED_ARRAY_I128, "__beve_typed_array_i128"),
    (u8,   NT_TYPED_ARRAY_U8,   "__beve_typed_array_u8"),
    (u16,  NT_TYPED_ARRAY_U16,  "__beve_typed_array_u16"),
    (u32,  NT_TYPED_ARRAY_U32,  "__beve_typed_array_u32"),
    (u64,  NT_TYPED_ARRAY_U64,  "__beve_typed_array_u64"),
    (u128, NT_TYPED_ARRAY_U128, "__beve_typed_array_u128"),
    (f32,  NT_TYPED_ARRAY_F32,  "__beve_typed_array_f32"),
    (f64,  NT_TYPED_ARRAY_F64,  "__beve_typed_array_f64"),
    (f16,  NT_TYPED_ARRAY_F16,  "__beve_typed_array_f16"),
    (bf16, NT_TYPED_ARRAY_BF16, "__beve_typed_array_bf16"),
}

/// Serde `serialize_with` helpers for foreign complex types (e.g. `num_complex::Complex`)
/// that implement [`ComplexElement`] (two contiguous `Component` fields: re then im).
///
/// These are only needed for foreign types. `beve::Complex<T>` serializes correctly
/// without any annotation.
///
/// Available helpers: `f32_array`, `f64_array`, `i8_array`, `i16_array`, `i32_array`,
/// `i64_array`, `i128_array`, `u8_array`, `u16_array`, `u32_array`, `u64_array`, `u128_array`.
///
/// Each takes a [`ComplexElement`] whose `Component` is the named scalar. A
/// foreign type needs an `unsafe impl` of that trait, which the `num-complex`
/// feature provides for `num_complex::Complex<T>`.
///
/// Encode only. To bulk-*decode* the same field as well, annotate it with
/// [`crate::complex_array`]'s `serde(with)` module instead of this one.
///
/// # Example
/// ```ignore
/// #[serde(serialize_with = "beve::complex::f32_array")]
/// pub buffer: Vec<num_complex::Complex<f32>>,
/// ```
pub mod complex {
    use super::*;

    macro_rules! complex_array_fn {
        ($name:ident, $scalar:ty) => {
            #[doc = concat!(
                                        "Serialize a slice of complex `", stringify!($scalar),
                                        "` as one BEVE complex array."
                                    )]
            ///
            /// Takes any [`ComplexElement`] whose `Component` is this scalar —
            /// [`Complex`], `num_complex::Complex` under the `num-complex`
            /// feature, or your own type. The component is part of the bound,
            /// so a same-width class cannot slip through.
            pub fn $name<S: serde::Serializer, T>(
                data: &[T],
                serializer: S,
            ) -> core::result::Result<S::Ok, S::Error>
            where
                T: ComplexElement<Component = $scalar>,
            {
                // Backstops against a wrong `unsafe impl`. Alignment matters
                // because `#[repr(packed)]` meets the rest of the contract yet
                // drops to align 1, and the reference below must be aligned
                // even when `data` is empty (`as_ptr` is then a dangling
                // pointer aligned only for `T`).
                assert_eq!(
                    core::mem::size_of::<T>(),
                    core::mem::size_of::<Complex<$scalar>>(),
                    concat!(
                        "beve::complex::",
                        stringify!($name),
                        ": ComplexElement impl has the wrong element size",
                    )
                );
                assert_eq!(
                    core::mem::align_of::<T>(),
                    core::mem::align_of::<Complex<$scalar>>(),
                    concat!(
                        "beve::complex::",
                        stringify!($name),
                        ": ComplexElement impl has the wrong element alignment",
                    )
                );
                // SAFETY: `T: ComplexElement<Component = $scalar>` promises `T`
                // is two initialized `$scalar`, real first — the layout of
                // `Complex<$scalar>`, whose width and alignment the asserts
                // above confirm. The lifetime is the caller's `data`.
                let slice: &[Complex<$scalar>] = unsafe {
                    core::slice::from_raw_parts(
                        data.as_ptr() as *const Complex<$scalar>,
                        data.len(),
                    )
                };
                ComplexSlice(slice).serialize(serializer)
            }
        };
    }

    complex_array_fn!(f32_array, f32);
    complex_array_fn!(f64_array, f64);
    complex_array_fn!(i8_array, i8);
    complex_array_fn!(i16_array, i16);
    complex_array_fn!(i32_array, i32);
    complex_array_fn!(i64_array, i64);
    complex_array_fn!(i128_array, i128);
    complex_array_fn!(u8_array, u8);
    complex_array_fn!(u16_array, u16);
    complex_array_fn!(u32_array, u32);
    complex_array_fn!(u64_array, u64);
    complex_array_fn!(u128_array, u128);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixLayout {
    Right,
    Left,
}

impl Serialize for MatrixLayout {
    fn serialize<S: serde::Serializer>(&self, s: S) -> core::result::Result<S::Ok, S::Error> {
        let layout = match self {
            MatrixLayout::Right => "layout_right",
            MatrixLayout::Left => "layout_left",
        };
        s.serialize_str(layout)
    }
}

impl<'de> Deserialize<'de> for MatrixLayout {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> core::result::Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = MatrixLayout;
            fn expecting(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                write!(f, "matrix layout string")
            }
            fn visit_str<E: serde::de::Error>(
                self,
                v: &str,
            ) -> core::result::Result<Self::Value, E> {
                match v {
                    "layout_right" | "row_major" | "right" => Ok(MatrixLayout::Right),
                    "layout_left" | "column_major" | "left" => Ok(MatrixLayout::Left),
                    _ => Err(E::custom("invalid matrix layout")),
                }
            }
        }
        d.deserialize_str(V)
    }
}

pub struct Matrix<'a, T> {
    pub layout: MatrixLayout,
    pub extents: &'a [usize],
    pub data: &'a [T],
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatrixOwned<T> {
    pub layout: MatrixLayout,
    pub extents: Vec<usize>,
    pub data: Vec<T>,
}

impl<T> MatrixOwned<T> {
    pub fn as_borrowed(&self) -> Matrix<'_, T> {
        Matrix {
            layout: self.layout,
            extents: &self.extents,
            data: &self.data,
        }
    }
}

impl<'a, T: Clone> From<Matrix<'a, T>> for MatrixOwned<T> {
    fn from(value: Matrix<'a, T>) -> Self {
        Self {
            layout: value.layout,
            extents: value.extents.to_vec(),
            data: value.data.to_vec(),
        }
    }
}

impl<'de, T> Deserialize<'de> for MatrixOwned<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> core::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct MatrixRepr<T> {
            layout: MatrixLayout,
            extents: Vec<usize>,
            value: Vec<T>,
        }

        let repr = MatrixRepr::deserialize(deserializer)?;
        validate_matrix_shape(&repr.extents, repr.value.len()).map_err(de::Error::custom)?;
        Ok(Self {
            layout: repr.layout,
            extents: repr.extents,
            data: repr.value,
        })
    }
}

impl<T> Serialize for MatrixOwned<T>
where
    T: Serialize + 'static,
{
    fn serialize<S: serde::Serializer>(&self, s: S) -> core::result::Result<S::Ok, S::Error> {
        self.as_borrowed().serialize(s)
    }
}

impl<'a, T> Serialize for Matrix<'a, T>
where
    T: Serialize + 'static,
{
    fn serialize<S: serde::Serializer>(&self, s: S) -> core::result::Result<S::Ok, S::Error> {
        if let Some(bytes) = try_encode_matrix_extension(self.layout, self.extents, self.data)
            .map_err(ser::Error::custom)?
        {
            return s.serialize_newtype_struct(NT_RAW_VALUE, &RawBytes(&bytes));
        }
        serialize_matrix_as_map(self.layout, self.extents, self.data, s)
    }
}

fn serialize_matrix_as_map<S, T: Serialize>(
    layout: MatrixLayout,
    extents: &[usize],
    data: &[T],
    s: S,
) -> core::result::Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let mut map = s.serialize_map(Some(3))?;
    ser::SerializeMap::serialize_entry(&mut map, "layout", &layout)?;
    ser::SerializeMap::serialize_entry(&mut map, "extents", &extents)?;
    ser::SerializeMap::serialize_entry(&mut map, "value", &data)?;
    ser::SerializeMap::end(map)
}

fn validate_matrix_shape(extents: &[usize], data_len: usize) -> core::result::Result<(), String> {
    if extents.is_empty() {
        return Err("matrix extents cannot be empty".into());
    }
    if extents.contains(&0) {
        return Err("matrix dimensions cannot be zero".into());
    }
    let expected = extents.iter().try_fold(1usize, |acc, &extent| {
        acc.checked_mul(extent)
            .ok_or_else(|| "matrix extents overflow".to_string())
    })?;
    if expected != data_len {
        return Err(format!(
            "matrix data length {} does not match product of extents {}",
            data_len, expected
        ));
    }
    Ok(())
}

fn write_unsigned_typed_header(out: &mut Vec<u8>, byte_code: u8, len: usize) {
    let header = ((byte_code & 0b111) << 5) | ((ARRAY_UNSIGNED & 0b11) << 3) | TYPE_TYPED_ARRAY;
    out.push(header);
    write_size(len as u64, out);
}

fn write_matrix_extents(out: &mut Vec<u8>, extents: &[usize]) {
    let max_extent = extents.iter().copied().max().unwrap_or(0);
    if u8::try_from(max_extent).is_ok() {
        write_unsigned_typed_header(out, 0, extents.len());
        for &extent in extents {
            out.push(extent as u8);
        }
    } else if u16::try_from(max_extent).is_ok() {
        write_unsigned_typed_header(out, 1, extents.len());
        for &extent in extents {
            out.extend_from_slice(&(extent as u16).to_le_bytes());
        }
    } else if u32::try_from(max_extent).is_ok() {
        write_unsigned_typed_header(out, 2, extents.len());
        for &extent in extents {
            out.extend_from_slice(&(extent as u32).to_le_bytes());
        }
    } else {
        write_unsigned_typed_header(out, 3, extents.len());
        for &extent in extents {
            out.extend_from_slice(&(extent as u64).to_le_bytes());
        }
    }
}

fn try_encode_matrix_extension<T: Serialize + 'static>(
    layout: MatrixLayout,
    extents: &[usize],
    data: &[T],
) -> core::result::Result<Option<Vec<u8>>, String> {
    validate_matrix_shape(extents, data.len())?;

    let mut out = Vec::new();
    out.push(make_extension_header(EXT_MATRICES));
    out.push(match layout {
        MatrixLayout::Right => 0u8,
        MatrixLayout::Left => 1u8,
    });
    write_matrix_extents(&mut out, extents);

    macro_rules! write_typed_value {
        ($ty:ty) => {
            if TypeId::of::<T>() == TypeId::of::<$ty>() {
                let typed = unsafe { &*(data as *const [T] as *const [$ty]) };
                write_typed_slice(&mut out, typed);
                return Ok(Some(out));
            }
        };
    }

    write_typed_value!(f64);
    write_typed_value!(f32);
    write_typed_value!(f16);
    write_typed_value!(bf16);
    write_typed_value!(i8);
    write_typed_value!(i16);
    write_typed_value!(i32);
    write_typed_value!(i64);
    write_typed_value!(i128);
    write_typed_value!(u8);
    write_typed_value!(u16);
    write_typed_value!(u32);
    write_typed_value!(u64);
    write_typed_value!(u128);

    if TypeId::of::<T>() == TypeId::of::<bool>() {
        let typed = unsafe { &*(data as *const [T] as *const [bool]) };
        write_bool_slice(&mut out, typed);
        return Ok(Some(out));
    }

    macro_rules! write_complex_value {
        ($scalar:ty) => {
            if TypeId::of::<T>() == TypeId::of::<Complex<$scalar>>() {
                let typed = unsafe { &*(data as *const [T] as *const [Complex<$scalar>]) };
                out.extend_from_slice(&to_vec_complex_slice(typed));
                return Ok(Some(out));
            }
        };
    }

    write_complex_value!(f32);
    write_complex_value!(f64);
    write_complex_value!(i8);
    write_complex_value!(i16);
    write_complex_value!(i32);
    write_complex_value!(i64);
    write_complex_value!(i128);
    write_complex_value!(u8);
    write_complex_value!(u16);
    write_complex_value!(u32);
    write_complex_value!(u64);
    write_complex_value!(u128);

    Ok(None)
}

pub fn decode_matrix_slice<T: serde::de::DeserializeOwned>(
    bytes: &[u8],
    mode: MatrixDecodeMode,
) -> crate::Result<DecodedMatrix<T>> {
    match mode {
        MatrixDecodeMode::Materialized => {
            let matrix: MatrixOwned<T> = from_slice(bytes)?;
            Ok(DecodedMatrix::Materialized(matrix))
        }
        MatrixDecodeMode::Raw => {
            let matrix: RawMatrix = from_slice(bytes)?;
            Ok(DecodedMatrix::Raw(matrix))
        }
    }
}
