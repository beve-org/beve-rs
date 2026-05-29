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

pub struct ComplexSlice<'a, T>(pub &'a [Complex<T>]);

macro_rules! impl_complex_slice_serialize {
    ($scalar:ty) => {
        impl<'a> Serialize for ComplexSlice<'a, $scalar> {
            fn serialize<S: serde::Serializer>(
                &self,
                s: S,
            ) -> core::result::Result<S::Ok, S::Error> {
                let mut seq = s.serialize_seq(Some(self.0.len()))?;
                for c in self.0 {
                    ser::SerializeSeq::serialize_element(&mut seq, c)?;
                }
                ser::SerializeSeq::end(seq)
            }
        }
    };
}

impl_complex_slice_serialize!(f32);
impl_complex_slice_serialize!(f64);
impl_complex_slice_serialize!(i8);
impl_complex_slice_serialize!(i16);
impl_complex_slice_serialize!(i32);
impl_complex_slice_serialize!(i64);
impl_complex_slice_serialize!(i128);
impl_complex_slice_serialize!(u8);
impl_complex_slice_serialize!(u16);
impl_complex_slice_serialize!(u32);
impl_complex_slice_serialize!(u64);
impl_complex_slice_serialize!(u128);

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
/// bytes and written with one `write_all` (no copy, no allocation); on
/// big-endian targets it falls back to the per-element sequence path. For a
/// non-empty slice the encoded bytes are identical to serializing the equivalent
/// `Vec<T>`. For an *empty* slice they differ: `TypedSlice` still emits a typed
/// array of the element's type, whereas a bare empty `Vec<T>` has no element from
/// which to detect the type and encodes as a generic empty array. Both decode
/// back to an empty `Vec<T>`.
pub struct TypedSlice<'a, T>(pub &'a [T]);

/// Shared `Serialize` body for every `TypedSlice<T>`: on little-endian, borrow the
/// slice as bytes and tag it by `name`; on big-endian, fall back to the
/// per-element sequence path.
#[inline]
fn serialize_typed_slice<S, T>(
    slice: &[T],
    name: &'static str,
    s: S,
) -> core::result::Result<S::Ok, S::Error>
where
    S: serde::Serializer,
    T: Serialize,
{
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
        let _ = name;
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
/// that are layout-compatible with `beve::Complex<T>` (two contiguous `T` fields: re then im).
///
/// These are only needed for foreign types. `beve::Complex<T>` serializes correctly
/// without any annotation.
///
/// Available helpers: `f32_array`, `f64_array`, `i8_array`, `i16_array`, `i32_array`,
/// `i64_array`, `i128_array`, `u8_array`, `u16_array`, `u32_array`, `u64_array`, `u128_array`.
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
            /// Serialize a slice of layout-compatible complex values as a BEVE complex array.
            ///
            /// # Safety contract
            /// The caller's type `T` must have the same layout as `beve::Complex<$scalar>`
            /// (two contiguous `$scalar` fields in re, im order). Size and alignment are
            /// checked at runtime; field order is the caller's responsibility.
            pub fn $name<S: serde::Serializer, T>(
                data: &[T],
                serializer: S,
            ) -> core::result::Result<S::Ok, S::Error> {
                assert_eq!(
                    core::mem::size_of::<T>(),
                    core::mem::size_of::<Complex<$scalar>>(),
                    concat!("beve::complex::", stringify!($name), ": type size mismatch")
                );
                assert_eq!(
                    core::mem::align_of::<T>(),
                    core::mem::align_of::<Complex<$scalar>>(),
                    concat!(
                        "beve::complex::",
                        stringify!($name),
                        ": type alignment mismatch"
                    )
                );
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
