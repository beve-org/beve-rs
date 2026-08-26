//! BEVE -> MATLAB v7.3 conversion.
//!
//! Requires the `mat` feature, which is off by default:
//! `beve = { version = "8", features = ["mat"] }`.
//!
//! Walks the BEVE wire format and emits MATLAB-equivalent values directly
//! through `hdf5_pure::mat::MatBuilder`. No intermediate value tree is
//! materialized: the BEVE reader and the MAT builder advance in lockstep.

use std::ffi::OsString;
use std::fmt::Write as FmtWrite;
use std::fs::OpenOptions;
use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use half::{bf16, f16};
use hdf5_pure::mat::{self as mat_pure, CellWriter, MatBuilder};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::ext::MatrixLayout;
use crate::header::*;
use crate::raw::{ComplexHeader, ObjectHeader, Reader, TypedArrayClass};

/// Byte codes for the two 2-byte encodings in the BEVE float class. Named so
/// the half-precision paths key on the encoding rather than on a bare literal.
const ARRAY_FLOAT_BF16_CODE: u8 = 0;
const ARRAY_FLOAT_F16_CODE: u8 = 1;

// Re-export shared option enums from hdf5-pure so beve users keep using them
// under the same names.
pub use hdf5_pure::LibVer;
pub use hdf5_pure::mat::{
    Compression, EmptyMarkerEncoding, InvalidNamePolicy, NullPolicy, OneDimensionalMode,
    RowMajorPolicy, StringClass, UnsupportedPolicy,
};

/// Controls how the BEVE root value is bound into MATLAB workspace variables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RootBinding<'a> {
    /// Convert the BEVE root into a single MATLAB variable with the provided name.
    NamedVariable(&'a str),
    /// Require a string-keyed BEVE object and expand each top-level entry into a
    /// separate MATLAB workspace variable.
    WorkspaceObject,
}

/// Options for BEVE -> MATLAB v7.3 conversion.
///
/// Convenience wrapper over [`hdf5_pure::mat::Options`] that pins
/// `string_class = String` (the BEVE historical default, and what real
/// MATLAB's `save -v7.3` produces). The remaining policies mirror upstream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatV73Options {
    pub compression: Compression,
    pub invalid_name_policy: InvalidNamePolicy,
    pub null_policy: NullPolicy,
    pub unsupported_policy: UnsupportedPolicy,
    pub one_dimensional_mode: OneDimensionalMode,
    pub row_major_policy: RowMajorPolicy,
    /// The newest HDF5 on-disk format the MAT file may use.
    ///
    /// Defaults to [`LibVer::V18`], the newest format every MATLAB release can
    /// open: a version 3 superblock is an HDF5 1.10 addition, and MATLAB's MAT
    /// v7.3 loader refuses one even on releases whose own libhdf5 reads it
    /// without difficulty.
    ///
    /// [`LibVer::V110`] is what [`Compression`] needs, since compression
    /// requires chunked storage and the chunk indices hdf5-pure writes arrived
    /// in 1.10. Compression against a lower bound is refused rather than
    /// resolved either way, so raise this deliberately and accept that MATLAB
    /// will not `load` the result.
    pub libver: LibVer,
}

impl Default for MatV73Options {
    fn default() -> Self {
        Self {
            compression: Compression::None,
            invalid_name_policy: InvalidNamePolicy::Error,
            null_policy: NullPolicy::EmptyStructArray,
            unsupported_policy: UnsupportedPolicy::Error,
            one_dimensional_mode: OneDimensionalMode::ColumnVector,
            row_major_policy: RowMajorPolicy::ReorderToColumnMajor,
            libver: LibVer::V18,
        }
    }
}

impl MatV73Options {
    /// Convert to the underlying hdf5-pure options. Always pins
    /// `string_class = String`.
    ///
    /// `unit_variant_encoding` and `empty_sequence_policy` are deliberately not
    /// mirrored: hdf5-pure consults them only from its serde writer, to recover
    /// type information that serde withholds. BEVE carries that information in
    /// the document, so this walker reads the answer rather than choosing it.
    fn to_pure(&self) -> mat_pure::Options {
        let mut opts = mat_pure::Options::with_modern_strings();
        opts.compression = self.compression;
        opts.invalid_name_policy = self.invalid_name_policy;
        opts.null_policy = self.null_policy;
        opts.unsupported_policy = self.unsupported_policy;
        opts.one_dimensional_mode = self.one_dimensional_mode;
        opts.row_major_policy = self.row_major_policy;
        opts.libver = self.libver;
        opts
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Convert a BEVE payload into a MATLAB v7.3 MAT file. Stages output in a
/// temp file next to the destination and only replaces the target after the
/// MAT file is fully written.
///
/// The MAT file is streamed into the temp file rather than assembled in
/// memory first, so converting costs the payload plus one dataset rather than
/// the payload plus the whole output.
pub fn beve_slice_to_mat_v73_file<P: AsRef<Path>>(
    beve: &[u8],
    output_path: P,
    root: RootBinding<'_>,
    options: &MatV73Options,
) -> Result<()> {
    let output_path = output_path.as_ref();
    let temp_output = TempOutputFile::new(output_path)?;
    {
        let file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(temp_output.path())
            .map_err(|e| Error::msg(e.to_string()))?;
        let mut sink = std::io::BufWriter::new(file);
        beve_slice_to_mat_v73_writer(beve, &mut sink, root, options)?;
        sink.flush().map_err(|e| Error::msg(e.to_string()))?;
    }
    sync_regular_file(temp_output.path())?;
    temp_output.persist(output_path)
}

/// Convert a BEVE payload to MATLAB v7.3 MAT file bytes in memory.
///
/// Prefer [`beve_slice_to_mat_v73_writer`] when the output is headed for a
/// file or a socket: it produces the same bytes without ever holding the
/// assembled file.
pub fn beve_slice_to_mat_v73_bytes(
    beve: &[u8],
    root: RootBinding<'_>,
    options: &MatV73Options,
) -> Result<Vec<u8>> {
    build_mat_v73(beve, root, options)?
        .finish()
        .map_err(map_mat_error)
}

/// Convert a BEVE payload into a MATLAB v7.3 MAT file written front-to-back
/// onto `w`.
///
/// Produces byte-for-byte what [`beve_slice_to_mat_v73_bytes`] returns, but
/// never holds the assembled file, so the output size stops bounding what can
/// be converted. The sink is never seeked and so may be a socket as readily as
/// a file.
///
/// A failure partway leaves whatever was already written on `w`. A caller that
/// needs all-or-nothing should write to a temporary path and rename on success,
/// which is what [`beve_slice_to_mat_v73_file`] does.
pub fn beve_slice_to_mat_v73_writer<W: std::io::Write>(
    beve: &[u8],
    w: W,
    root: RootBinding<'_>,
    options: &MatV73Options,
) -> Result<()> {
    build_mat_v73(beve, root, options)?
        .finish_to(w)
        .map_err(map_mat_error)
}

/// Walk `beve` into a builder, leaving the choice of exit to the caller. The
/// two public conversions differ only in how they finish, so the walk lives
/// here rather than being duplicated and left to drift.
fn build_mat_v73(
    beve: &[u8],
    root: RootBinding<'_>,
    options: &MatV73Options,
) -> Result<MatBuilder> {
    // Refuse over-nested input before the walk starts, not during it. The walk
    // runs inside hdf5-pure's `struct_`/`cell` closures, whose error type is
    // `MatError`, so every beve error raised below the root round-trips through
    // `MatError::Custom(String)` and comes back as an untyped message -- which
    // would leave this the one entry point that reports the depth ceiling
    // without `Error::RecursionLimitExceeded`, the exact split the shared
    // constant exists to prevent.
    //
    // `skip_value` is the crate's own bounded walker and charges a level
    // everywhere this one recurses, plus a few places it does not (tag and
    // matrix payloads), so it refuses everything the walk could not survive.
    // The walker keeps its own check regardless: the stack guarantee should not
    // rest on two walkers agreeing.
    //
    // Only the depth verdict is taken from the pre-pass. Every other malformed
    // input is left to the walk, whose errors name the path at which they were
    // found -- `unsupported complex element class at $.a[3]` is worth more than
    // the bare header byte a skipper can report.
    if matches!(
        crate::skip_value(beve, &mut 0),
        Err(Error::RecursionLimitExceeded)
    ) {
        return Err(Error::RecursionLimitExceeded);
    }

    let mut mb = MatBuilder::new(options.to_pure());
    let mut reader = Reader::new(beve);
    walk_root(&mut reader, &mut mb, options, root)?;
    if !reader.is_finished() {
        return Err(Error::InvalidType("unexpected trailing BEVE data"));
    }
    Ok(mb)
}

/// Read a `.beve` file from disk and write a MATLAB v7.3 MAT file.
pub fn beve_file_to_mat_v73_file<I: AsRef<Path>, O: AsRef<Path>>(
    input_path: I,
    output_path: O,
    root: RootBinding<'_>,
    options: &MatV73Options,
) -> Result<()> {
    let bytes = std::fs::read(input_path).map_err(|e| Error::msg(e.to_string()))?;
    beve_slice_to_mat_v73_file(&bytes, output_path, root, options)
}

// ---------------------------------------------------------------------------
// Bulk typed-array readers
//
// BEVE typed numeric arrays are LE-packed. Reading the whole region as `&[u8]`
// once and chunking is significantly faster than the per-element reader calls,
// which each re-check bounds and update `pos`. Conversion is host-endian
// independent via `from_le_bytes`.
// ---------------------------------------------------------------------------

#[inline]
fn read_f32_array(reader: &mut Reader<'_>, len: usize) -> Result<Vec<f32>> {
    let bytes = reader.read_exact(len * 4)?;
    let mut out = Vec::with_capacity(len);
    for chunk in bytes.as_chunks::<4>().0 {
        out.push(f32::from_le_bytes(*chunk));
    }
    Ok(out)
}

#[inline]
fn read_f64_array(reader: &mut Reader<'_>, len: usize) -> Result<Vec<f64>> {
    let bytes = reader.read_exact(len * 8)?;
    let mut out = Vec::with_capacity(len);
    for chunk in bytes.as_chunks::<8>().0 {
        out.push(f64::from_le_bytes(*chunk));
    }
    Ok(out)
}

#[inline]
fn read_i8_array(reader: &mut Reader<'_>, len: usize) -> Result<Vec<i8>> {
    let bytes = reader.read_exact(len)?;
    Ok(bytes.iter().map(|&b| b as i8).collect())
}

#[inline]
fn read_i16_array(reader: &mut Reader<'_>, len: usize) -> Result<Vec<i16>> {
    let bytes = reader.read_exact(len * 2)?;
    let mut out = Vec::with_capacity(len);
    for chunk in bytes.as_chunks::<2>().0 {
        out.push(i16::from_le_bytes(*chunk));
    }
    Ok(out)
}

#[inline]
fn read_i32_array(reader: &mut Reader<'_>, len: usize) -> Result<Vec<i32>> {
    let bytes = reader.read_exact(len * 4)?;
    let mut out = Vec::with_capacity(len);
    for chunk in bytes.as_chunks::<4>().0 {
        out.push(i32::from_le_bytes(*chunk));
    }
    Ok(out)
}

#[inline]
fn read_i64_array(reader: &mut Reader<'_>, len: usize) -> Result<Vec<i64>> {
    let bytes = reader.read_exact(len * 8)?;
    let mut out = Vec::with_capacity(len);
    for chunk in bytes.as_chunks::<8>().0 {
        out.push(i64::from_le_bytes(*chunk));
    }
    Ok(out)
}

#[inline]
fn read_u8_array(reader: &mut Reader<'_>, len: usize) -> Result<Vec<u8>> {
    Ok(reader.read_exact(len)?.to_vec())
}

#[inline]
fn read_u16_array(reader: &mut Reader<'_>, len: usize) -> Result<Vec<u16>> {
    let bytes = reader.read_exact(len * 2)?;
    let mut out = Vec::with_capacity(len);
    for chunk in bytes.as_chunks::<2>().0 {
        out.push(u16::from_le_bytes(*chunk));
    }
    Ok(out)
}

#[inline]
fn read_u32_array(reader: &mut Reader<'_>, len: usize) -> Result<Vec<u32>> {
    let bytes = reader.read_exact(len * 4)?;
    let mut out = Vec::with_capacity(len);
    for chunk in bytes.as_chunks::<4>().0 {
        out.push(u32::from_le_bytes(*chunk));
    }
    Ok(out)
}

#[inline]
fn read_u64_array(reader: &mut Reader<'_>, len: usize) -> Result<Vec<u64>> {
    let bytes = reader.read_exact(len * 8)?;
    let mut out = Vec::with_capacity(len);
    for chunk in bytes.as_chunks::<8>().0 {
        out.push(u64::from_le_bytes(*chunk));
    }
    Ok(out)
}

/// A BEVE complex array is LE-packed interleaved re/im pairs of the element
/// scalar, so one bulk read plus chunking covers every width. The element type
/// is fixed by the complex header's (class, byte_code), not by the array.
macro_rules! read_complex_array_fn {
    ($name:ident, $ty:ty) => {
        #[inline]
        fn $name(reader: &mut Reader<'_>, len: usize) -> Result<Vec<($ty, $ty)>> {
            const ELEM: usize = core::mem::size_of::<$ty>();
            const PAIR: usize = ELEM * 2;
            let byte_len = len.checked_mul(PAIR).ok_or(Error::InvalidSize)?;
            let bytes = reader.read_exact(byte_len)?;
            let mut out = Vec::with_capacity(len);
            for chunk in bytes.as_chunks::<PAIR>().0 {
                let (re_bytes, im_bytes) = chunk.split_at(ELEM);
                let re = <$ty>::from_le_bytes(re_bytes.try_into().unwrap());
                let im = <$ty>::from_le_bytes(im_bytes.try_into().unwrap());
                out.push((re, im));
            }
            Ok(out)
        }
    };
}

read_complex_array_fn!(read_complex_f32_array, f32);
read_complex_array_fn!(read_complex_f64_array, f64);
read_complex_array_fn!(read_complex_i8_array, i8);
read_complex_array_fn!(read_complex_i16_array, i16);
read_complex_array_fn!(read_complex_i32_array, i32);
read_complex_array_fn!(read_complex_i64_array, i64);
read_complex_array_fn!(read_complex_u8_array, u8);
read_complex_array_fn!(read_complex_u16_array, u16);
read_complex_array_fn!(read_complex_u32_array, u32);
read_complex_array_fn!(read_complex_u64_array, u64);

/// Half-precision complex arrays widen to `f32` pairs, matching how the real
/// f16/bf16 typed arrays are handled. Gated by the caller on
/// [`UnsupportedPolicy::LossyNumericWidening`].
///
/// Reads the whole region up front like the fixed-width readers above, so a
/// wire-supplied `len` is bounded by the remaining input before anything is
/// allocated. Reserving first would let a hostile SIZE field request an
/// arbitrary allocation from a payload only a few bytes long.
#[inline]
fn read_complex_half_array(
    reader: &mut Reader<'_>,
    len: usize,
    byte_code: u8,
) -> Result<Vec<(f32, f32)>> {
    // Both f16 and bf16 are 2 bytes per component, 4 per complex element.
    let byte_len = len.checked_mul(4).ok_or(Error::InvalidSize)?;
    let bytes = reader.read_exact(byte_len)?;
    let widen: fn(u16) -> f32 = if byte_code == ARRAY_FLOAT_BF16_CODE {
        |bits| bf16::from_bits(bits).to_f32()
    } else {
        |bits| f16::from_bits(bits).to_f32()
    };
    let mut out = Vec::with_capacity(len);
    for chunk in bytes.as_chunks::<4>().0 {
        let (re_bytes, im_bytes) = chunk.split_at(2);
        let re = widen(u16::from_le_bytes(re_bytes.try_into().unwrap()));
        let im = widen(u16::from_le_bytes(im_bytes.try_into().unwrap()));
        out.push((re, im));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Walker
//
// Routes through `&mut MatBuilder`. The builder's internal scope stack
// (open structs + cell-element pending refs) ensures `mb.write_*` /
// `mb.struct_` / `mb.cell` calls land in the right place regardless of how
// deep the recursion is.
//
// `path` is a single mutable `String` reused for the whole walk: each step
// pushes its segment, recurses, and truncates back to the prefix. Errors
// observed at any depth capture the path in their message at that moment, so
// the unwinding path doesn't need to restore.
// ---------------------------------------------------------------------------

fn walk_root(
    reader: &mut Reader<'_>,
    mb: &mut MatBuilder,
    options: &MatV73Options,
    root: RootBinding<'_>,
) -> Result<()> {
    let mut path = String::with_capacity(64);
    path.push('$');
    match root {
        RootBinding::NamedVariable(name) => walk_value(reader, mb, options, name, &mut path, 0),
        RootBinding::WorkspaceObject => {
            let header = reader.read_header()?;
            let object = reader.read_object_header(header)?;
            if object.key_type != KEY_STRING {
                return Err(Error::Unsupported(
                    "workspace root objects must use string keys",
                ));
            }
            for _ in 0..object.len {
                let key = reader.read_string()?.to_owned();
                let prefix_len = path.len();
                path.push('.');
                path.push_str(&key);
                // The root object this branch just read is itself the first
                // level, so its entries start one deeper.
                walk_value(reader, mb, options, &key, &mut path, 1)?;
                path.truncate(prefix_len);
            }
            Ok(())
        }
    }
}

/// Walk one BEVE value into the current MatBuilder scope.
fn walk_value(
    reader: &mut Reader<'_>,
    mb: &mut MatBuilder,
    options: &MatV73Options,
    name: &str,
    path: &mut String,
    depth: usize,
) -> Result<()> {
    let header = reader.read_header()?;
    walk_value_with_header(reader, mb, options, name, header, path, depth)
}

fn walk_value_with_header(
    reader: &mut Reader<'_>,
    mb: &mut MatBuilder,
    options: &MatV73Options,
    name: &str,
    header: u8,
    path: &mut String,
    depth: usize,
) -> Result<()> {
    // The single funnel for every nested value: `walk_value` and
    // `walk_value_at_cell_element` both route through here, so one check bounds
    // the whole walk. Counted on entry from zero, matching `skip_value`, which
    // admits `MAX_RECURSION_DEPTH` nested values and refuses the next.
    if depth >= crate::MAX_RECURSION_DEPTH {
        return Err(Error::RecursionLimitExceeded);
    }
    match parse_type(header) {
        TYPE_NULL_BOOL => {
            if header == 0 {
                handle_null(mb, options, name, path)
            } else {
                let v = bool_value(header)?;
                mb.write_scalar_logical(name, v).map_err(map_mat_error)?;
                Ok(())
            }
        }
        TYPE_NUMBER => walk_number(reader, mb, options, name, header, path),
        TYPE_STRING => {
            let value = reader.read_string()?.to_owned();
            mb.write_string_object(name, &[value], &[1, 1])
                .map_err(map_mat_error)?;
            Ok(())
        }
        TYPE_OBJECT => walk_object(reader, mb, options, name, header, path, depth),
        TYPE_TYPED_ARRAY => walk_typed_array(reader, mb, options, name, header, path),
        TYPE_GENERIC_ARRAY => walk_generic_array(reader, mb, options, name, path, depth),
        TYPE_EXTENSION => walk_extension(reader, mb, options, name, header, path),
        _ => Err(Error::InvalidHeader(header)),
    }
}

fn handle_null(mb: &mut MatBuilder, options: &MatV73Options, name: &str, path: &str) -> Result<()> {
    match options.null_policy {
        NullPolicy::EmptyStructArray => mb
            .write_empty_struct_array(name)
            .map(|_| ())
            .map_err(map_mat_error),
        // Write nothing at all: `walk_object` builds the struct's field list
        // from what the builder actually received, so skipping the write is
        // what drops the field.
        NullPolicy::Omit => Ok(()),
        NullPolicy::Error => Err(Error::msg(format!("unsupported null value at {path}"))),
        _ => Err(Error::Unsupported("unrecognized null policy")),
    }
}

fn walk_number(
    reader: &mut Reader<'_>,
    mb: &mut MatBuilder,
    options: &MatV73Options,
    name: &str,
    header: u8,
    path: &str,
) -> Result<()> {
    let subtype = parse_subtype(header);
    let byte_code = parse_byte_count_code(header);
    match subtype {
        NUM_SIGNED => match byte_code {
            0 => {
                let v = reader.read_signed(0)? as i8;
                mb.write_scalar_i8(name, v).map_err(map_mat_error)?;
                Ok(())
            }
            1 => {
                let v = reader.read_signed(1)? as i16;
                mb.write_scalar_i16(name, v).map_err(map_mat_error)?;
                Ok(())
            }
            2 => {
                let v = reader.read_signed(2)? as i32;
                mb.write_scalar_i32(name, v).map_err(map_mat_error)?;
                Ok(())
            }
            3 => {
                let v = reader.read_signed(3)? as i64;
                mb.write_scalar_i64(name, v).map_err(map_mat_error)?;
                Ok(())
            }
            4 => handle_128_bit(mb, options, name, path, reader.read_signed(4)?.to_string()),
            _ => Err(Error::InvalidHeader(header)),
        },
        NUM_UNSIGNED => match byte_code {
            0 => {
                let v = reader.read_unsigned(0)? as u8;
                mb.write_scalar_u8(name, v).map_err(map_mat_error)?;
                Ok(())
            }
            1 => {
                let v = reader.read_unsigned(1)? as u16;
                mb.write_scalar_u16(name, v).map_err(map_mat_error)?;
                Ok(())
            }
            2 => {
                let v = reader.read_unsigned(2)? as u32;
                mb.write_scalar_u32(name, v).map_err(map_mat_error)?;
                Ok(())
            }
            3 => {
                let v = reader.read_unsigned(3)? as u64;
                mb.write_scalar_u64(name, v).map_err(map_mat_error)?;
                Ok(())
            }
            4 => handle_128_bit(
                mb,
                options,
                name,
                path,
                reader.read_unsigned(4)?.to_string(),
            ),
            _ => Err(Error::InvalidHeader(header)),
        },
        NUM_FLOAT => match byte_code {
            0 => {
                check_low_precision_float(options, "bf16 scalar", path)?;
                let v = reader.read_bf16_as_f32()?;
                mb.write_scalar_f32(name, v).map_err(map_mat_error)?;
                Ok(())
            }
            1 => {
                check_low_precision_float(options, "f16 scalar", path)?;
                let v = reader.read_f16_as_f32()?;
                mb.write_scalar_f32(name, v).map_err(map_mat_error)?;
                Ok(())
            }
            2 => {
                let v = reader.read_f32()?;
                mb.write_scalar_f32(name, v).map_err(map_mat_error)?;
                Ok(())
            }
            3 => {
                let v = reader.read_f64()?;
                mb.write_scalar_f64(name, v).map_err(map_mat_error)?;
                Ok(())
            }
            _ => Err(Error::Unsupported("float128 is not supported")),
        },
        _ => Err(Error::InvalidHeader(header)),
    }
}

fn handle_128_bit(
    mb: &mut MatBuilder,
    options: &MatV73Options,
    name: &str,
    path: &str,
    value: String,
) -> Result<()> {
    match options.unsupported_policy {
        UnsupportedPolicy::StringFallback => mb
            .write_string_object(name, &[value], &[1, 1])
            .map(|_| ())
            .map_err(map_mat_error),
        _ => Err(Error::msg(format!(
            "unsupported 128-bit integer scalar at {path}"
        ))),
    }
}

fn walk_object(
    reader: &mut Reader<'_>,
    mb: &mut MatBuilder,
    options: &MatV73Options,
    name: &str,
    header: u8,
    path: &mut String,
    depth: usize,
) -> Result<()> {
    let ObjectHeader {
        key_type,
        byte_code,
        len,
    } = reader.read_object_header(header)?;
    // No walker-side dedup: MatBuilder enforces uniqueness inside the open
    // struct group, so adding our own BTreeSet/HashSet here just doubles the
    // work. The builder also handles validation/sanitization per the
    // configured policy.
    mb.struct_(name, |sw| {
        let inner_mb = sw.builder();
        for _ in 0..len {
            let key: String = match key_type {
                KEY_STRING => reader.read_string().map_err(map_beve_error)?.to_owned(),
                KEY_SIGNED => reader
                    .read_signed(byte_code)
                    .map_err(map_beve_error)?
                    .to_string(),
                KEY_UNSIGNED => reader
                    .read_unsigned(byte_code)
                    .map_err(map_beve_error)?
                    .to_string(),
                _ => return Err(map_beve_error(Error::InvalidHeader(header))),
            };
            let prefix_len = path.len();
            path.push('.');
            path.push_str(&key);
            walk_value(reader, inner_mb, options, &key, path, depth + 1).map_err(map_beve_error)?;
            path.truncate(prefix_len);
        }
        Ok(())
    })
    .map_err(map_mat_error)?;
    Ok(())
}

fn walk_typed_array(
    reader: &mut Reader<'_>,
    mb: &mut MatBuilder,
    options: &MatV73Options,
    name: &str,
    header: u8,
    path: &str,
) -> Result<()> {
    let info = reader.read_typed_array_header(header)?;
    let dims = mb.vector_dims(info.len);
    match info.class {
        TypedArrayClass::Float => write_float_array(
            reader,
            mb,
            options,
            name,
            &dims,
            info.byte_code,
            info.len,
            path,
        ),
        TypedArrayClass::Signed => {
            write_signed_array(reader, mb, name, &dims, info.byte_code, info.len, path)
        }
        TypedArrayClass::Unsigned => {
            write_unsigned_array(reader, mb, name, &dims, info.byte_code, info.len, path)
        }
        TypedArrayClass::Bool => {
            let bytes = read_bool_array(reader, info.len)?;
            mb.write_logical(name, &dims, &bytes)
                .map_err(map_mat_error)?;
            Ok(())
        }
        TypedArrayClass::String => {
            let values = read_string_array(reader, info.len)?;
            mb.write_string_object(name, &values, &dims)
                .map_err(map_mat_error)?;
            Ok(())
        }
    }
}

fn walk_generic_array(
    reader: &mut Reader<'_>,
    mb: &mut MatBuilder,
    options: &MatV73Options,
    name: &str,
    path: &mut String,
    depth: usize,
) -> Result<()> {
    let len = reader.read_size()?;
    let dims = mb.vector_dims(len);
    if len == 0 {
        mb.write_empty(name, mat_pure::MatClass::Cell, &dims)
            .map_err(map_mat_error)?;
        return Ok(());
    }
    mb.cell(name, &dims, |cw| {
        for idx in 0..len {
            let prefix_len = path.len();
            // String's std::fmt::Write impl is infallible.
            let _ = write!(path, "[{idx}]");
            walk_value_at_cell_element(reader, cw, options, path, depth + 1)
                .map_err(map_beve_error)?;
            path.truncate(prefix_len);
        }
        Ok(())
    })
    .map_err(map_mat_error)?;
    Ok(())
}

fn walk_extension(
    reader: &mut Reader<'_>,
    mb: &mut MatBuilder,
    options: &MatV73Options,
    name: &str,
    header: u8,
    path: &str,
) -> Result<()> {
    match parse_extension_id(header) {
        EXT_COMPLEX => walk_complex(reader, mb, options, name, path),
        EXT_MATRICES => walk_matrix(reader, mb, options, name, path),
        _ => Err(Error::msg(format!(
            "unsupported BEVE extension {:#x} at {path}",
            parse_extension_id(header)
        ))),
    }
}

fn walk_complex(
    reader: &mut Reader<'_>,
    mb: &mut MatBuilder,
    options: &MatV73Options,
    name: &str,
    path: &str,
) -> Result<()> {
    let header = reader.read_complex_header()?;
    if header.is_array {
        let len = reader.read_size()?;
        let dims = mb.vector_dims(len);
        write_complex_array(reader, mb, options, name, &dims, header, len, None, path)
    } else {
        // A complex scalar's payload is the same LE-packed re/im pair a
        // one-element array carries, so the array path covers it directly.
        write_complex_array(reader, mb, options, name, &[1, 1], header, 1, None, path)
    }
}

/// Read a BEVE complex payload of `len` elements and write it as a MATLAB
/// complex array. The element class comes from the complex header rather than
/// the surrounding context, so MATLAB receives an `int16` complex array for an
/// `int16` complex payload and a `single` one for an `f32` payload. Nothing
/// here widens an integer to a float: MATLAB has first-class complex integer
/// arrays, and silently promoting would make the file misreport what was
/// captured.
///
/// `reorder` is `Some((layout, extents))` for matrix payloads, which may need
/// row-major to column-major reordering, and `None` for plain arrays and
/// scalars, which are already in the order MATLAB expects.
#[allow(clippy::too_many_arguments)]
fn write_complex_array(
    reader: &mut Reader<'_>,
    mb: &mut MatBuilder,
    options: &MatV73Options,
    name: &str,
    dims: &[usize],
    header: ComplexHeader,
    len: usize,
    reorder: Option<(MatrixLayout, &[usize])>,
    path: &str,
) -> Result<()> {
    macro_rules! emit {
        ($read:ident, $write:ident) => {{
            let data = $read(reader, len)?;
            let data = maybe_reorder_complex(data, reorder, options, path)?;
            mb.$write(name, dims, &data).map_err(map_mat_error)?;
            Ok(())
        }};
    }

    match (header.class, header.byte_code) {
        (ARRAY_FLOAT, ARRAY_FLOAT_BF16_CODE | ARRAY_FLOAT_F16_CODE) => {
            // Match the split labels the real f16/bf16 paths use, so the error
            // names the encoding rather than just "half-precision".
            let label = if header.byte_code == ARRAY_FLOAT_BF16_CODE {
                "bf16 complex"
            } else {
                "f16 complex"
            };
            check_low_precision_float(options, label, path)?;
            let data = read_complex_half_array(reader, len, header.byte_code)?;
            let data = maybe_reorder_complex(data, reorder, options, path)?;
            mb.write_complex_f32(name, dims, &data)
                .map_err(map_mat_error)?;
            Ok(())
        }
        (ARRAY_FLOAT, 2) => emit!(read_complex_f32_array, write_complex_f32),
        (ARRAY_FLOAT, 3) => emit!(read_complex_f64_array, write_complex_f64),
        (ARRAY_SIGNED, 0) => emit!(read_complex_i8_array, write_complex_i8),
        (ARRAY_SIGNED, 1) => emit!(read_complex_i16_array, write_complex_i16),
        (ARRAY_SIGNED, 2) => emit!(read_complex_i32_array, write_complex_i32),
        (ARRAY_SIGNED, 3) => emit!(read_complex_i64_array, write_complex_i64),
        (ARRAY_UNSIGNED, 0) => emit!(read_complex_u8_array, write_complex_u8),
        (ARRAY_UNSIGNED, 1) => emit!(read_complex_u16_array, write_complex_u16),
        (ARRAY_UNSIGNED, 2) => emit!(read_complex_u32_array, write_complex_u32),
        (ARRAY_UNSIGNED, 3) => emit!(read_complex_u64_array, write_complex_u64),
        (ARRAY_FLOAT, 4) => Err(Error::msg(format!(
            "unsupported 128-bit float complex at {path}"
        ))),
        (ARRAY_SIGNED | ARRAY_UNSIGNED, 4) => Err(Error::msg(format!(
            "unsupported 128-bit integer complex at {path}; MATLAB has no 128-bit class"
        ))),
        _ => Err(Error::msg(format!(
            "unsupported complex element type (class {}, width code {}) at {path}",
            header.class, header.byte_code
        ))),
    }
}

fn maybe_reorder_complex<T: Clone>(
    data: Vec<T>,
    reorder: Option<(MatrixLayout, &[usize])>,
    options: &MatV73Options,
    path: &str,
) -> Result<Vec<T>> {
    match reorder {
        Some((layout, extents)) => maybe_reorder(data, layout, extents, options, path),
        None => Ok(data),
    }
}

fn walk_matrix(
    reader: &mut Reader<'_>,
    mb: &mut MatBuilder,
    options: &MatV73Options,
    name: &str,
    path: &str,
) -> Result<()> {
    let layout = if (reader.read_header()? & 0x01) == 0 {
        MatrixLayout::Right
    } else {
        MatrixLayout::Left
    };
    let extents = reader.read_matrix_extents()?;
    validate_matrix_extents(&extents, path)?;
    let header = reader.read_header()?;
    let dims = mb.matrix_dims(&extents);
    match parse_type(header) {
        TYPE_TYPED_ARRAY => walk_matrix_typed_array(
            reader, mb, options, name, &dims, &extents, layout, header, path,
        ),
        TYPE_EXTENSION if parse_extension_id(header) == EXT_COMPLEX => {
            walk_matrix_complex(reader, mb, options, name, &dims, &extents, layout, path)
        }
        _ => Err(Error::msg(format!("unsupported matrix payload at {path}"))),
    }
}

// ---------------------------------------------------------------------------
// Cell-element walker
// ---------------------------------------------------------------------------

fn walk_value_at_cell_element(
    reader: &mut Reader<'_>,
    cw: &mut CellWriter<'_>,
    options: &MatV73Options,
    path: &mut String,
    depth: usize,
) -> Result<()> {
    let header = reader.read_header()?;

    // `NullPolicy::Omit` cannot be honored here. A cell array's element count is
    // fixed by its dims, so the slot exists whether or not anything is written
    // into it: omitting a field is expressible, omitting an element is not.
    // Writing nothing still consumes a `#refs#` slot and leaves the element
    // pointing at an object that was never created, which MATLAB cannot
    // dereference. So a null in this position takes the empty-struct-array
    // marker regardless, matching what hdf5-pure's own serde emitter does for
    // the same case. The other two policies need no special handling: one
    // writes that marker anyway, and `Error` fails the whole conversion.
    if header == 0 && matches!(options.null_policy, NullPolicy::Omit) {
        cw.push_empty_struct_array().map_err(map_mat_error)?;
        return Ok(());
    }

    cw.push_with(|mb| {
        // The closure runs with a fresh `next_target` armed; the first
        // MatBuilder call routes to `#refs#/ref_NNNN`. After that first call
        // the arm is consumed and any deeper writes (e.g. inside a struct or
        // cell that was just opened) target the right scope automatically.
        walk_value_with_header(reader, mb, options, "", header, path, depth).map_err(map_beve_error)
    })
    .map_err(map_mat_error)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Numeric typed-array writers (use bulk readers above)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn write_float_array(
    reader: &mut Reader<'_>,
    mb: &mut MatBuilder,
    options: &MatV73Options,
    name: &str,
    dims: &[usize],
    byte_code: u8,
    len: usize,
    path: &str,
) -> Result<()> {
    match byte_code {
        0 => {
            check_low_precision_float(options, "bf16 array", path)?;
            let mut values = Vec::with_capacity(len);
            for _ in 0..len {
                values.push(reader.read_bf16_as_f32()?);
            }
            mb.write_f32(name, dims, &values).map_err(map_mat_error)?;
            Ok(())
        }
        1 => {
            check_low_precision_float(options, "f16 array", path)?;
            let mut values = Vec::with_capacity(len);
            for _ in 0..len {
                values.push(reader.read_f16_as_f32()?);
            }
            mb.write_f32(name, dims, &values).map_err(map_mat_error)?;
            Ok(())
        }
        2 => {
            let values = read_f32_array(reader, len)?;
            mb.write_f32(name, dims, &values).map_err(map_mat_error)?;
            Ok(())
        }
        3 => {
            let values = read_f64_array(reader, len)?;
            mb.write_f64(name, dims, &values).map_err(map_mat_error)?;
            Ok(())
        }
        _ => Err(Error::Unsupported("float128 arrays are not supported")),
    }
}

fn write_signed_array(
    reader: &mut Reader<'_>,
    mb: &mut MatBuilder,
    name: &str,
    dims: &[usize],
    byte_code: u8,
    len: usize,
    path: &str,
) -> Result<()> {
    match byte_code {
        0 => {
            let values = read_i8_array(reader, len)?;
            mb.write_i8(name, dims, &values).map_err(map_mat_error)?;
            Ok(())
        }
        1 => {
            let values = read_i16_array(reader, len)?;
            mb.write_i16(name, dims, &values).map_err(map_mat_error)?;
            Ok(())
        }
        2 => {
            let values = read_i32_array(reader, len)?;
            mb.write_i32(name, dims, &values).map_err(map_mat_error)?;
            Ok(())
        }
        3 => {
            let values = read_i64_array(reader, len)?;
            mb.write_i64(name, dims, &values).map_err(map_mat_error)?;
            Ok(())
        }
        4 => Err(Error::msg(format!("unsupported i128 array at {path}"))),
        _ => Err(Error::InvalidSize),
    }
}

fn write_unsigned_array(
    reader: &mut Reader<'_>,
    mb: &mut MatBuilder,
    name: &str,
    dims: &[usize],
    byte_code: u8,
    len: usize,
    path: &str,
) -> Result<()> {
    match byte_code {
        0 => {
            let values = read_u8_array(reader, len)?;
            mb.write_u8(name, dims, &values).map_err(map_mat_error)?;
            Ok(())
        }
        1 => {
            let values = read_u16_array(reader, len)?;
            mb.write_u16(name, dims, &values).map_err(map_mat_error)?;
            Ok(())
        }
        2 => {
            let values = read_u32_array(reader, len)?;
            mb.write_u32(name, dims, &values).map_err(map_mat_error)?;
            Ok(())
        }
        3 => {
            let values = read_u64_array(reader, len)?;
            mb.write_u64(name, dims, &values).map_err(map_mat_error)?;
            Ok(())
        }
        4 => Err(Error::msg(format!("unsupported u128 array at {path}"))),
        _ => Err(Error::InvalidSize),
    }
}

// ---------------------------------------------------------------------------
// Matrix walkers
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn walk_matrix_typed_array(
    reader: &mut Reader<'_>,
    mb: &mut MatBuilder,
    options: &MatV73Options,
    name: &str,
    dims: &[usize],
    extents: &[usize],
    layout: MatrixLayout,
    header: u8,
    path: &str,
) -> Result<()> {
    let info = reader.read_typed_array_header(header)?;
    let expected_len = product(extents)?;
    if info.len != expected_len {
        return Err(Error::msg(format!(
            "matrix payload length {} does not match extents product {} at {path}",
            info.len, expected_len
        )));
    }
    match info.class {
        TypedArrayClass::Float => walk_matrix_float(
            reader,
            mb,
            options,
            name,
            dims,
            extents,
            layout,
            info.byte_code,
            path,
        ),
        TypedArrayClass::Signed => walk_matrix_signed(
            reader,
            mb,
            options,
            name,
            dims,
            extents,
            layout,
            info.byte_code,
            path,
        ),
        TypedArrayClass::Unsigned => walk_matrix_unsigned(
            reader,
            mb,
            options,
            name,
            dims,
            extents,
            layout,
            info.byte_code,
            path,
        ),
        TypedArrayClass::Bool => {
            let bytes = read_bool_array(reader, info.len)?;
            let reordered = maybe_reorder(bytes, layout, extents, options, path)?;
            mb.write_logical(name, dims, &reordered)
                .map_err(map_mat_error)?;
            Ok(())
        }
        TypedArrayClass::String => {
            let values = read_string_array(reader, info.len)?;
            let reordered = maybe_reorder(values, layout, extents, options, path)?;
            mb.write_string_object(name, &reordered, dims)
                .map_err(map_mat_error)?;
            Ok(())
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn walk_matrix_float(
    reader: &mut Reader<'_>,
    mb: &mut MatBuilder,
    options: &MatV73Options,
    name: &str,
    dims: &[usize],
    extents: &[usize],
    layout: MatrixLayout,
    byte_code: u8,
    path: &str,
) -> Result<()> {
    let len = product(extents)?;
    match byte_code {
        0 => {
            check_low_precision_float(options, "bf16 matrix", path)?;
            let mut values = Vec::with_capacity(len);
            for _ in 0..len {
                values.push(reader.read_bf16_as_f32()?);
            }
            let v = maybe_reorder(values, layout, extents, options, path)?;
            mb.write_f32(name, dims, &v).map_err(map_mat_error)?;
            Ok(())
        }
        1 => {
            check_low_precision_float(options, "f16 matrix", path)?;
            let mut values = Vec::with_capacity(len);
            for _ in 0..len {
                values.push(reader.read_f16_as_f32()?);
            }
            let v = maybe_reorder(values, layout, extents, options, path)?;
            mb.write_f32(name, dims, &v).map_err(map_mat_error)?;
            Ok(())
        }
        2 => {
            let values = read_f32_array(reader, len)?;
            let v = maybe_reorder(values, layout, extents, options, path)?;
            mb.write_f32(name, dims, &v).map_err(map_mat_error)?;
            Ok(())
        }
        3 => {
            let values = read_f64_array(reader, len)?;
            let v = maybe_reorder(values, layout, extents, options, path)?;
            mb.write_f64(name, dims, &v).map_err(map_mat_error)?;
            Ok(())
        }
        _ => Err(Error::msg(format!(
            "unsupported float matrix width at {path}"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn walk_matrix_signed(
    reader: &mut Reader<'_>,
    mb: &mut MatBuilder,
    options: &MatV73Options,
    name: &str,
    dims: &[usize],
    extents: &[usize],
    layout: MatrixLayout,
    byte_code: u8,
    path: &str,
) -> Result<()> {
    let len = product(extents)?;
    match byte_code {
        0 => {
            let values = read_i8_array(reader, len)?;
            let v = maybe_reorder(values, layout, extents, options, path)?;
            mb.write_i8(name, dims, &v).map_err(map_mat_error)?;
            Ok(())
        }
        1 => {
            let values = read_i16_array(reader, len)?;
            let v = maybe_reorder(values, layout, extents, options, path)?;
            mb.write_i16(name, dims, &v).map_err(map_mat_error)?;
            Ok(())
        }
        2 => {
            let values = read_i32_array(reader, len)?;
            let v = maybe_reorder(values, layout, extents, options, path)?;
            mb.write_i32(name, dims, &v).map_err(map_mat_error)?;
            Ok(())
        }
        3 => {
            let values = read_i64_array(reader, len)?;
            let v = maybe_reorder(values, layout, extents, options, path)?;
            mb.write_i64(name, dims, &v).map_err(map_mat_error)?;
            Ok(())
        }
        _ => Err(Error::msg(format!("unsupported i128 matrix at {path}"))),
    }
}

#[allow(clippy::too_many_arguments)]
fn walk_matrix_unsigned(
    reader: &mut Reader<'_>,
    mb: &mut MatBuilder,
    options: &MatV73Options,
    name: &str,
    dims: &[usize],
    extents: &[usize],
    layout: MatrixLayout,
    byte_code: u8,
    path: &str,
) -> Result<()> {
    let len = product(extents)?;
    match byte_code {
        0 => {
            let values = read_u8_array(reader, len)?;
            let v = maybe_reorder(values, layout, extents, options, path)?;
            mb.write_u8(name, dims, &v).map_err(map_mat_error)?;
            Ok(())
        }
        1 => {
            let values = read_u16_array(reader, len)?;
            let v = maybe_reorder(values, layout, extents, options, path)?;
            mb.write_u16(name, dims, &v).map_err(map_mat_error)?;
            Ok(())
        }
        2 => {
            let values = read_u32_array(reader, len)?;
            let v = maybe_reorder(values, layout, extents, options, path)?;
            mb.write_u32(name, dims, &v).map_err(map_mat_error)?;
            Ok(())
        }
        3 => {
            let values = read_u64_array(reader, len)?;
            let v = maybe_reorder(values, layout, extents, options, path)?;
            mb.write_u64(name, dims, &v).map_err(map_mat_error)?;
            Ok(())
        }
        _ => Err(Error::msg(format!("unsupported u128 matrix at {path}"))),
    }
}

#[allow(clippy::too_many_arguments)]
fn walk_matrix_complex(
    reader: &mut Reader<'_>,
    mb: &mut MatBuilder,
    options: &MatV73Options,
    name: &str,
    dims: &[usize],
    extents: &[usize],
    layout: MatrixLayout,
    path: &str,
) -> Result<()> {
    let info = reader.read_complex_header()?;
    if !info.is_array {
        return Err(Error::msg(format!(
            "matrix complex payload must be an array at {path}"
        )));
    }
    let len = reader.read_size()?;
    let expected = product(extents)?;
    if len != expected {
        return Err(Error::msg(format!(
            "matrix complex payload length {} does not match extents product {} at {path}",
            len, expected
        )));
    }
    write_complex_array(
        reader,
        mb,
        options,
        name,
        dims,
        info,
        len,
        Some((layout, extents)),
        path,
    )
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn read_bool_array(reader: &mut Reader<'_>, len: usize) -> Result<Vec<u8>> {
    let packed = reader.read_exact(len.div_ceil(8))?;
    Ok(unpack_bools(packed, len))
}

fn read_string_array(reader: &mut Reader<'_>, len: usize) -> Result<Vec<String>> {
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        values.push(reader.read_string()?.to_owned());
    }
    Ok(values)
}

fn unpack_bools(packed: &[u8], len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    for idx in 0..len {
        let byte = packed[idx / 8];
        let bit = (byte >> (idx % 8)) & 1;
        out.push(bit);
    }
    out
}

fn check_low_precision_float(options: &MatV73Options, label: &str, path: &str) -> Result<()> {
    match options.unsupported_policy {
        UnsupportedPolicy::LossyNumericWidening => Ok(()),
        _ => Err(Error::msg(format!(
            "unsupported {label} at {path}; enable LossyNumericWidening to map it to MATLAB single"
        ))),
    }
}

fn maybe_reorder<T: Clone>(
    data: Vec<T>,
    layout: MatrixLayout,
    extents: &[usize],
    options: &MatV73Options,
    path: &str,
) -> Result<Vec<T>> {
    match layout {
        MatrixLayout::Left => Ok(data),
        MatrixLayout::Right => match options.row_major_policy {
            RowMajorPolicy::ReorderToColumnMajor => {
                Ok(reorder_row_major_to_column_major(&data, extents))
            }
            RowMajorPolicy::Error => Err(Error::msg(format!(
                "row-major matrix at {path} requires reordering to MATLAB column-major layout"
            ))),
            _ => Err(Error::Unsupported("unrecognized row-major policy")),
        },
    }
}

fn reorder_row_major_to_column_major<T: Clone>(data: &[T], extents: &[usize]) -> Vec<T> {
    if extents.len() <= 1 || data.len() <= 1 {
        return data.to_vec();
    }
    if extents.len() == 2 {
        // 2-D fast path: avoid the modulo/divide per element of the N-D path.
        let rows = extents[0];
        let cols = extents[1];
        let mut out = Vec::with_capacity(rows * cols);
        for c in 0..cols {
            for r in 0..rows {
                out.push(data[r * cols + c].clone());
            }
        }
        return out;
    }
    let mut row_strides = vec![1usize; extents.len()];
    for idx in (0..extents.len().saturating_sub(1)).rev() {
        row_strides[idx] = row_strides[idx + 1] * extents[idx + 1];
    }
    let mut out = Vec::with_capacity(data.len());
    for column_major_pos in 0..data.len() {
        let mut remaining = column_major_pos;
        let mut row_index = 0usize;
        for (axis, &extent) in extents.iter().enumerate() {
            let coord = remaining % extent;
            remaining /= extent;
            row_index += coord * row_strides[axis];
        }
        out.push(data[row_index].clone());
    }
    out
}

fn product(extents: &[usize]) -> Result<usize> {
    extents.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or(Error::InvalidSize)
    })
}

fn validate_matrix_extents(extents: &[usize], path: &str) -> Result<()> {
    if extents.is_empty() {
        return Err(Error::msg(format!(
            "matrix extents cannot be empty at {path}"
        )));
    }
    if extents.contains(&0) {
        return Err(Error::msg(format!(
            "matrix dimensions cannot be zero at {path}"
        )));
    }
    Ok(())
}

fn map_mat_error(e: hdf5_pure::mat::MatError) -> Error {
    Error::msg(e.to_string())
}

fn map_beve_error(e: Error) -> hdf5_pure::mat::MatError {
    hdf5_pure::mat::MatError::Custom(e.to_string())
}

// ---------------------------------------------------------------------------
// Atomic temp-file write helpers (beve-specific)
// ---------------------------------------------------------------------------

struct TempOutputFile {
    path: PathBuf,
    persisted: bool,
}

impl TempOutputFile {
    fn new(output_path: &Path) -> Result<Self> {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);

        let directory = output_path
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let base_name = output_path
            .file_name()
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| OsString::from("beve.mat"));

        for _ in 0..1024 {
            let sequence = NEXT_ID.fetch_add(1, Ordering::Relaxed);
            let mut temp_name = OsString::from(".");
            temp_name.push(&base_name);
            temp_name.push(format!(".beve-tmp-{:x}-{:x}", std::process::id(), sequence));
            let path = directory.join(temp_name);
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(file) => {
                    drop(file);
                    return Ok(Self {
                        path,
                        persisted: false,
                    });
                }
                Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(err) => return Err(Error::msg(err.to_string())),
            }
        }

        Err(Error::msg(format!(
            "failed to reserve a temporary MAT path near {}",
            output_path.display()
        )))
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn persist(mut self, output_path: &Path) -> Result<()> {
        replace_file(&self.path, output_path).map_err(|e| Error::msg(e.to_string()))?;
        self.persisted = true;
        Ok(())
    }
}

impl Drop for TempOutputFile {
    fn drop(&mut self) {
        if !self.persisted {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

fn sync_regular_file(path: &Path) -> Result<()> {
    OpenOptions::new()
        .read(true)
        .open(path)
        .and_then(|file| file.sync_all())
        .map_err(|e| Error::msg(e.to_string()))
}

#[cfg(not(windows))]
fn replace_file(from: &Path, to: &Path) -> std::io::Result<()> {
    std::fs::rename(from, to)
}

#[cfg(windows)]
fn replace_file(from: &Path, to: &Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;

    const MOVEFILE_REPLACE_EXISTING: u32 = 0x1;
    const MOVEFILE_WRITE_THROUGH: u32 = 0x8;

    unsafe extern "system" {
        fn MoveFileExW(
            existing_file_name: *const u16,
            new_file_name: *const u16,
            flags: u32,
        ) -> i32;
    }

    fn to_wide(path: &Path) -> Vec<u16> {
        path.as_os_str().encode_wide().chain(Some(0)).collect()
    }

    let from_wide = to_wide(from);
    let to_wide = to_wide(to);
    let result = unsafe {
        MoveFileExW(
            from_wide.as_ptr(),
            to_wide.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if result != 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}
