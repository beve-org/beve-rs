use std::collections::BTreeSet;
use std::ffi::OsString;
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use hdf5_pure::FileBuilder;
use hdf5_pure::type_builders::{AttrValue, DatasetBuilder, GroupBuilder};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::ext::MatrixLayout;
use crate::header::*;
use crate::raw::{ComplexHeader, ObjectHeader, Reader, TypedArrayClass};

const MATLAB_USERBLOCK_SIZE: u64 = 512;
const MATLAB_NAME_MAX: usize = 2048;
const MCOS_MAGIC_NUMBER: u32 = 0xDD00_0000;
const FILEWRAPPER_VERSION: u32 = 4;
const MATLAB_STRING_SAVEOBJ_VERSION: u64 = 1;
const MATLAB_OBJECT_DECODE_OPAQUE: i32 = 3;
const MATLAB_CLASS_STRING: &str = "string";
const MATLAB_CLASS_FILEWRAPPER: &str = "FileWrapper__";
const MATLAB_KEYWORDS: &[&str] = &[
    "break",
    "case",
    "catch",
    "classdef",
    "continue",
    "else",
    "elseif",
    "end",
    "events",
    "for",
    "function",
    "global",
    "if",
    "methods",
    "otherwise",
    "parfor",
    "persistent",
    "properties",
    "return",
    "spmd",
    "switch",
    "try",
    "while",
];

/// Controls how the BEVE root value is bound into MATLAB workspace variables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RootBinding<'a> {
    /// Convert the BEVE root into a single MATLAB variable with the provided name.
    NamedVariable(&'a str),
    /// Require a string-keyed BEVE object and expand each top-level entry into a
    /// separate MATLAB workspace variable.
    WorkspaceObject,
}

/// Compression settings for MATLAB datasets written through HDF5.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Compression {
    /// Write datasets without chunking or compression.
    None,
    /// Apply HDF5 deflate compression.
    Deflate { level: u8, shuffle: bool },
}

/// Policy for invalid MATLAB variable names and struct field names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InvalidNamePolicy {
    /// Return an error when a name is not a valid MATLAB identifier.
    Error,
    /// Rewrite invalid names into valid MATLAB identifiers and deduplicate them.
    Sanitize,
}

/// Policy for BEVE values that do not have a direct MATLAB representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnsupportedPolicy {
    /// Return an error for unsupported values.
    Error,
    /// Convert unsupported scalar values to MATLAB string objects using their string form.
    StringFallback,
    /// Widen unsupported numeric values to the nearest MATLAB numeric class.
    LossyNumericWidening,
}

/// Policy for BEVE `null` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NullPolicy {
    /// Return an error for `null`.
    Error,
    /// Map `null` to an empty MATLAB struct array (`struct([])`).
    EmptyStructArray,
}

/// Policy for one-dimensional arrays and vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OneDimensionalMode {
    /// Encode 1D values as `Nx1` MATLAB column vectors.
    ColumnVector,
    /// Encode 1D values as `1xN` MATLAB row vectors.
    RowVector,
}

/// Policy for BEVE row-major matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RowMajorPolicy {
    /// Reorder row-major matrix payloads into MATLAB's column-major layout.
    ReorderToColumnMajor,
    /// Return an error instead of reordering row-major matrices.
    Error,
}

/// Options for BEVE to MATLAB v7.3 conversion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatV73Options {
    /// HDF5 dataset compression settings.
    pub compression: Compression,
    /// How invalid MATLAB variable and field names are handled.
    pub invalid_name_policy: InvalidNamePolicy,
    /// How BEVE `null` is handled.
    pub null_policy: NullPolicy,
    /// How unsupported BEVE types are handled.
    pub unsupported_policy: UnsupportedPolicy,
    /// Whether 1D values become row or column vectors.
    pub one_dimensional_mode: OneDimensionalMode,
    /// Whether row-major matrices are reordered or rejected.
    pub row_major_policy: RowMajorPolicy,
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
        }
    }
}

// ---------------------------------------------------------------------------
// Intermediate representation
// ---------------------------------------------------------------------------

enum MatNode {
    Dataset(MatDataset),
    Group(MatGroup),
}

struct MatDataset {
    name: String,
    content: DatasetContent,
    attrs: Vec<(&'static str, AttrValue)>,
}

enum DatasetContent {
    U8(Vec<u64>, Vec<u8>),
    U16(Vec<u64>, Vec<u16>),
    U32(Vec<u64>, Vec<u32>),
    U64(Vec<u64>, Vec<u64>),
    I8(Vec<u64>, Vec<i8>),
    I16(Vec<u64>, Vec<i16>),
    I32(Vec<u64>, Vec<i32>),
    I64(Vec<u64>, Vec<i64>),
    F32(Vec<u64>, Vec<f32>),
    F64(Vec<u64>, Vec<f64>),
    Complex32(Vec<u64>, Vec<(f32, f32)>),
    Complex64(Vec<u64>, Vec<(f64, f64)>),
    References(Vec<u64>, Vec<String>),
}

struct MatGroup {
    name: String,
    children: Vec<MatNode>,
    attrs: Vec<(&'static str, AttrValue)>,
}

struct McosData {
    shape: Vec<u64>,
    ref_paths: Vec<String>,
}

// ---------------------------------------------------------------------------
// MatWriter — phase 1: accumulates IR
// ---------------------------------------------------------------------------

struct MatWriter {
    root_nodes: Vec<MatNode>,
    refs_children: Vec<MatNode>,
    options: MatV73Options,
    next_ref_id: u64,
    string_objects: StringObjectState,
}

struct StringObjectState {
    payload_paths: Vec<String>,
}

impl StringObjectState {
    fn new() -> Self {
        Self {
            payload_paths: Vec::new(),
        }
    }

    fn register(&mut self, payload_path: String) -> u32 {
        self.payload_paths.push(payload_path);
        self.payload_paths.len() as u32
    }

    fn is_empty(&self) -> bool {
        self.payload_paths.is_empty()
    }
}

#[derive(Clone, Copy)]
struct MatrixWriteContext<'a> {
    path: &'a str,
    layout: MatrixLayout,
    extents: &'a [usize],
}

/// Convert a BEVE payload into a MATLAB v7.3 MAT file.
///
/// This API writes directly from the BEVE wire format without materializing an
/// intermediate `Value` tree. Output is staged in a temporary file next to the
/// destination and only replaces the target path after the MAT file and MATLAB
/// userblock are fully written.
///
/// # Errors
///
/// Returns an error if the BEVE payload is malformed, if a value cannot be
/// mapped to MATLAB under the selected policies, or if HDF5 / filesystem IO
/// fails.
///
/// # Example
///
/// ```no_run
/// use beve::{MatV73Options, RootBinding};
///
/// let bytes = beve::to_vec(&vec![1.0f64, 2.0, 3.0]).unwrap();
/// beve::beve_slice_to_mat_v73_file(
///     &bytes,
///     "values.mat",
///     RootBinding::NamedVariable("values"),
///     &MatV73Options::default(),
/// )?;
/// # Ok::<(), beve::Error>(())
/// ```
pub fn beve_slice_to_mat_v73_file<P: AsRef<Path>>(
    beve: &[u8],
    output_path: P,
    root: RootBinding<'_>,
    options: &MatV73Options,
) -> Result<()> {
    let output_path = output_path.as_ref();
    let temp_output = TempOutputFile::new(output_path)?;
    let bytes = beve_slice_to_mat_v73_bytes(beve, root, options)?;
    std::fs::write(temp_output.path(), &bytes).map_err(|e| Error::msg(e.to_string()))?;
    sync_regular_file(temp_output.path())?;
    temp_output.persist(output_path)
}

/// Convert a BEVE payload to MATLAB v7.3 MAT file bytes in memory.
///
/// This is the WASM-compatible entry point — no filesystem access required.
pub fn beve_slice_to_mat_v73_bytes(
    beve: &[u8],
    root: RootBinding<'_>,
    options: &MatV73Options,
) -> Result<Vec<u8>> {
    let mut writer = MatWriter {
        root_nodes: Vec::new(),
        refs_children: Vec::new(),
        options: options.clone(),
        next_ref_id: 0,
        string_objects: StringObjectState::new(),
    };
    let mut reader = Reader::new(beve);
    writer.write_root(&mut reader, root)?;
    if !reader.is_finished() {
        return Err(Error::InvalidType("unexpected trailing BEVE data"));
    }
    writer.build_hdf5_bytes()
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
// Phase 1: BEVE streaming → IR nodes
// ---------------------------------------------------------------------------

impl MatWriter {
    fn write_root(&mut self, reader: &mut Reader<'_>, root: RootBinding<'_>) -> Result<()> {
        match root {
            RootBinding::NamedVariable(name) => {
                let variable = self.normalize_name(name, &mut BTreeSet::new())?;
                let node = self.write_named_value(variable, reader, "$")?;
                self.root_nodes.push(node);
                Ok(())
            }
            RootBinding::WorkspaceObject => {
                let header = reader.read_header()?;
                let object = reader.read_object_header(header)?;
                if object.key_type != KEY_STRING {
                    return Err(Error::Unsupported(
                        "workspace root objects must use string keys",
                    ));
                }
                let mut used = BTreeSet::new();
                for _ in 0..object.len {
                    let key = reader.read_string()?;
                    let variable = self.normalize_name(key, &mut used)?;
                    let path = format!("$.{variable}");
                    let node = self.write_named_value(variable, reader, &path)?;
                    self.root_nodes.push(node);
                }
                Ok(())
            }
        }
    }

    fn write_named_value(
        &mut self,
        name: String,
        reader: &mut Reader<'_>,
        path: &str,
    ) -> Result<MatNode> {
        let header = reader.read_header()?;
        self.write_named_value_with_header(name, reader, header, path)
    }

    fn write_named_value_with_header(
        &mut self,
        name: String,
        reader: &mut Reader<'_>,
        header: u8,
        path: &str,
    ) -> Result<MatNode> {
        match parse_type(header) {
            TYPE_NULL_BOOL => {
                if header == 0 {
                    self.handle_null(name, path)
                } else {
                    let value = bool_value(header)?;
                    Ok(self.make_logical(name, vec![1, 1], vec![u8::from(value)]))
                }
            }
            TYPE_NUMBER => self.write_number(name, reader, header, path),
            TYPE_STRING => {
                let value = reader.read_string()?.to_owned();
                self.write_string_scalar(name, &value)
            }
            TYPE_OBJECT => self.write_object(name, reader, header, path),
            TYPE_TYPED_ARRAY => self.write_typed_array(name, reader, header, path),
            TYPE_GENERIC_ARRAY => self.write_generic_array(name, reader, path),
            TYPE_EXTENSION => self.write_extension(name, reader, header, path),
            _ => Err(Error::InvalidHeader(header)),
        }
    }

    fn handle_null(&mut self, name: String, path: &str) -> Result<MatNode> {
        match self.options.null_policy {
            NullPolicy::EmptyStructArray => Ok(self.make_empty_struct_array(name)),
            NullPolicy::Error => Err(Error::msg(format!("unsupported null value at {path}"))),
        }
    }

    fn write_number(
        &mut self,
        name: String,
        reader: &mut Reader<'_>,
        header: u8,
        path: &str,
    ) -> Result<MatNode> {
        let subtype = parse_subtype(header);
        let byte_code = parse_byte_count_code(header);
        match subtype {
            NUM_SIGNED => match byte_code {
                0 => Ok(self.make_numeric_i8(name, vec![1, 1], vec![reader.read_signed(0)? as i8])),
                1 => Ok(self.make_numeric_i16(
                    name,
                    vec![1, 1],
                    vec![reader.read_signed(1)? as i16],
                )),
                2 => Ok(self.make_numeric_i32(
                    name,
                    vec![1, 1],
                    vec![reader.read_signed(2)? as i32],
                )),
                3 => Ok(self.make_numeric_i64(
                    name,
                    vec![1, 1],
                    vec![reader.read_signed(3)? as i64],
                )),
                4 => self.handle_128_bit_scalar(name, path, reader.read_signed(4)?.to_string()),
                _ => Err(Error::InvalidHeader(header)),
            },
            NUM_UNSIGNED => match byte_code {
                0 => Ok(self.make_numeric_u8(
                    name,
                    vec![1, 1],
                    vec![reader.read_unsigned(0)? as u8],
                )),
                1 => Ok(self.make_numeric_u16(
                    name,
                    vec![1, 1],
                    vec![reader.read_unsigned(1)? as u16],
                )),
                2 => Ok(self.make_numeric_u32(
                    name,
                    vec![1, 1],
                    vec![reader.read_unsigned(2)? as u32],
                )),
                3 => Ok(self.make_numeric_u64(
                    name,
                    vec![1, 1],
                    vec![reader.read_unsigned(3)? as u64],
                )),
                4 => self.handle_128_bit_scalar(name, path, reader.read_unsigned(4)?.to_string()),
                _ => Err(Error::InvalidHeader(header)),
            },
            NUM_FLOAT => match byte_code {
                0 => match self.options.unsupported_policy {
                    UnsupportedPolicy::LossyNumericWidening => Ok(self.make_numeric_f32(
                        name,
                        vec![1, 1],
                        vec![reader.read_bf16_as_f32()?],
                    )),
                    _ => Err(Error::msg(format!(
                        "unsupported bf16 scalar at {path}; enable LossyNumericWidening to map it to MATLAB single"
                    ))),
                },
                1 => match self.options.unsupported_policy {
                    UnsupportedPolicy::LossyNumericWidening => Ok(self.make_numeric_f32(
                        name,
                        vec![1, 1],
                        vec![reader.read_f16_as_f32()?],
                    )),
                    _ => Err(Error::msg(format!(
                        "unsupported f16 scalar at {path}; enable LossyNumericWidening to map it to MATLAB single"
                    ))),
                },
                2 => Ok(self.make_numeric_f32(name, vec![1, 1], vec![reader.read_f32()?])),
                3 => Ok(self.make_numeric_f64(name, vec![1, 1], vec![reader.read_f64()?])),
                _ => Err(Error::Unsupported("float128 is not supported")),
            },
            _ => Err(Error::InvalidHeader(header)),
        }
    }

    fn write_object(
        &mut self,
        name: String,
        reader: &mut Reader<'_>,
        header: u8,
        path: &str,
    ) -> Result<MatNode> {
        let ObjectHeader { key_type, byte_code, len } = reader.read_object_header(header)?;
        let mut used = BTreeSet::new();
        let mut children = Vec::with_capacity(len);
        let mut fields = Vec::with_capacity(len);
        for _ in 0..len {
            let key: String = match key_type {
                KEY_STRING => reader.read_string()?.to_owned(),
                KEY_SIGNED => reader.read_signed(byte_code)?.to_string(),
                KEY_UNSIGNED => reader.read_unsigned(byte_code)?.to_string(),
                _ => return Err(Error::InvalidHeader(header)),
            };
            let field = self.normalize_name(&key, &mut used)?;
            let child_path = format!("{path}.{field}");
            let child = self.write_named_value(field.clone(), reader, &child_path)?;
            children.push(child);
            fields.push(field);
        }
        let mut attrs: Vec<(&'static str, AttrValue)> =
            vec![("MATLAB_class", AttrValue::AsciiString("struct".into()))];
        attrs.push(("MATLAB_fields", AttrValue::AsciiStringArray(fields)));
        Ok(MatNode::Group(MatGroup {
            name,
            children,
            attrs,
        }))
    }

    fn write_typed_array(
        &mut self,
        name: String,
        reader: &mut Reader<'_>,
        header: u8,
        path: &str,
    ) -> Result<MatNode> {
        let info = reader.read_typed_array_header(header)?;
        match info.class {
            TypedArrayClass::Float => {
                self.write_float_array(name, reader, info.byte_code, info.len, path)
            }
            TypedArrayClass::Signed => {
                self.write_signed_array(name, reader, info.byte_code, info.len, path)
            }
            TypedArrayClass::Unsigned => {
                self.write_unsigned_array(name, reader, info.byte_code, info.len, path)
            }
            TypedArrayClass::Bool => self.write_bool_array(name, reader, info.len),
            TypedArrayClass::String => self.write_string_array(name, reader, info.len),
        }
    }

    fn write_generic_array(
        &mut self,
        name: String,
        reader: &mut Reader<'_>,
        path: &str,
    ) -> Result<MatNode> {
        let len = reader.read_size()?;
        let matlab_dims = vector_dims(len, self.options.one_dimensional_mode);
        if len == 0 {
            return Ok(self.make_empty_marker(name, &matlab_dims, "cell", None));
        }
        let mut ref_paths = Vec::with_capacity(len);
        for idx in 0..len {
            let child_path = format!("{path}[{idx}]");
            ref_paths.push(self.write_value_reference(reader, &child_path)?);
        }
        Ok(self.make_reference_array(name, &matlab_dims, ref_paths, "cell"))
    }

    fn write_extension(
        &mut self,
        name: String,
        reader: &mut Reader<'_>,
        header: u8,
        path: &str,
    ) -> Result<MatNode> {
        match parse_extension_id(header) {
            EXT_COMPLEX => self.write_complex_extension(name, reader, path),
            EXT_MATRICES => self.write_matrix_extension(name, reader, path),
            _ => Err(Error::msg(format!(
                "unsupported BEVE extension {:#x} at {path}",
                parse_extension_id(header)
            ))),
        }
    }

    fn write_complex_extension(
        &mut self,
        name: String,
        reader: &mut Reader<'_>,
        path: &str,
    ) -> Result<MatNode> {
        let ComplexHeader {
            is_array,
            byte_code,
        } = reader.read_complex_header()?;
        if is_array {
            let len = reader.read_size()?;
            let matlab_dims = vector_dims(len, self.options.one_dimensional_mode);
            match byte_code {
                2 => {
                    let mut data = Vec::with_capacity(len);
                    for _ in 0..len {
                        data.push((reader.read_f32()?, reader.read_f32()?));
                    }
                    Ok(self.make_complex32(name, matlab_dims, data))
                }
                3 => {
                    let mut data = Vec::with_capacity(len);
                    for _ in 0..len {
                        data.push((reader.read_f64()?, reader.read_f64()?));
                    }
                    Ok(self.make_complex64(name, matlab_dims, data))
                }
                _ => Err(Error::msg(format!(
                    "unsupported complex element width at {path}"
                ))),
            }
        } else {
            match byte_code {
                2 => Ok(self.make_complex32(
                    name,
                    vec![1, 1],
                    vec![(reader.read_f32()?, reader.read_f32()?)],
                )),
                3 => Ok(self.make_complex64(
                    name,
                    vec![1, 1],
                    vec![(reader.read_f64()?, reader.read_f64()?)],
                )),
                _ => Err(Error::msg(format!(
                    "unsupported complex scalar width at {path}"
                ))),
            }
        }
    }

    fn write_matrix_extension(
        &mut self,
        name: String,
        reader: &mut Reader<'_>,
        path: &str,
    ) -> Result<MatNode> {
        let layout = if (reader.read_header()? & 0x01) == 0 {
            MatrixLayout::Right
        } else {
            MatrixLayout::Left
        };
        let extents = reader.read_matrix_extents()?;
        validate_matrix_extents(&extents, path)?;
        let header = reader.read_header()?;
        let ctx = MatrixWriteContext {
            path,
            layout,
            extents: &extents,
        };
        match parse_type(header) {
            TYPE_TYPED_ARRAY => self.write_matrix_typed_array(name.clone(), reader, header, ctx),
            TYPE_EXTENSION if parse_extension_id(header) == EXT_COMPLEX => {
                self.write_matrix_complex(name, reader, path, layout, &extents)
            }
            _ => Err(Error::msg(format!("unsupported matrix payload at {path}"))),
        }
    }

    fn write_matrix_typed_array(
        &mut self,
        name: String,
        reader: &mut Reader<'_>,
        header: u8,
        ctx: MatrixWriteContext<'_>,
    ) -> Result<MatNode> {
        let info = reader.read_typed_array_header(header)?;
        let expected_len = product(ctx.extents)?;
        if info.len != expected_len {
            return Err(Error::msg(format!(
                "matrix payload length {} does not match extents product {} at {}",
                info.len, expected_len, ctx.path
            )));
        }
        match info.class {
            TypedArrayClass::Float => self.write_matrix_float(name, reader, info.byte_code, ctx),
            TypedArrayClass::Signed => self.write_matrix_signed(name, reader, info.byte_code, ctx),
            TypedArrayClass::Unsigned => {
                self.write_matrix_unsigned(name, reader, info.byte_code, ctx)
            }
            TypedArrayClass::Bool => self.write_matrix_bool(name, reader, ctx.extents, ctx.layout),
            TypedArrayClass::String => self.write_matrix_string(name, reader, ctx),
        }
    }

    fn write_matrix_complex(
        &mut self,
        name: String,
        reader: &mut Reader<'_>,
        path: &str,
        layout: MatrixLayout,
        extents: &[usize],
    ) -> Result<MatNode> {
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
        let matlab_dims = matrix_dims(extents, self.options.one_dimensional_mode);
        match info.byte_code {
            2 => {
                let mut data = Vec::with_capacity(len);
                for _ in 0..len {
                    data.push((reader.read_f32()?, reader.read_f32()?));
                }
                let data = self.maybe_reorder(data, layout, extents, path)?;
                Ok(self.make_complex32(name, matlab_dims, data))
            }
            3 => {
                let mut data = Vec::with_capacity(len);
                for _ in 0..len {
                    data.push((reader.read_f64()?, reader.read_f64()?));
                }
                let data = self.maybe_reorder(data, layout, extents, path)?;
                Ok(self.make_complex64(name, matlab_dims, data))
            }
            _ => Err(Error::msg(format!(
                "unsupported complex matrix element width at {path}"
            ))),
        }
    }

    fn write_float_array(
        &mut self,
        name: String,
        reader: &mut Reader<'_>,
        byte_code: u8,
        len: usize,
        path: &str,
    ) -> Result<MatNode> {
        let matlab_dims = vector_dims(len, self.options.one_dimensional_mode);
        match byte_code {
            0 => match self.options.unsupported_policy {
                UnsupportedPolicy::LossyNumericWidening => {
                    let mut values = Vec::with_capacity(len);
                    for _ in 0..len {
                        values.push(reader.read_bf16_as_f32()?);
                    }
                    if len == 0 {
                        return Ok(self.make_empty_marker(name, &matlab_dims, "single", None));
                    }
                    Ok(self.make_numeric_f32(name, matlab_dims, values))
                }
                _ => Err(Error::msg(format!(
                    "unsupported bf16 array at {path}; enable LossyNumericWidening to map it to MATLAB single"
                ))),
            },
            1 => match self.options.unsupported_policy {
                UnsupportedPolicy::LossyNumericWidening => {
                    let mut values = Vec::with_capacity(len);
                    for _ in 0..len {
                        values.push(reader.read_f16_as_f32()?);
                    }
                    if len == 0 {
                        return Ok(self.make_empty_marker(name, &matlab_dims, "single", None));
                    }
                    Ok(self.make_numeric_f32(name, matlab_dims, values))
                }
                _ => Err(Error::msg(format!(
                    "unsupported f16 array at {path}; enable LossyNumericWidening to map it to MATLAB single"
                ))),
            },
            2 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_f32()?);
                }
                if len == 0 {
                    return Ok(self.make_empty_marker(name, &matlab_dims, "single", None));
                }
                Ok(self.make_numeric_f32(name, matlab_dims, values))
            }
            3 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_f64()?);
                }
                if len == 0 {
                    return Ok(self.make_empty_marker(name, &matlab_dims, "double", None));
                }
                Ok(self.make_numeric_f64(name, matlab_dims, values))
            }
            _ => Err(Error::Unsupported("float128 arrays are not supported")),
        }
    }

    fn write_signed_array(
        &mut self,
        name: String,
        reader: &mut Reader<'_>,
        byte_code: u8,
        len: usize,
        path: &str,
    ) -> Result<MatNode> {
        let matlab_dims = vector_dims(len, self.options.one_dimensional_mode);
        match byte_code {
            0 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_signed(0)? as i8);
                }
                if len == 0 {
                    return Ok(self.make_empty_marker(name, &matlab_dims, "int8", None));
                }
                Ok(self.make_numeric_i8(name, matlab_dims, values))
            }
            1 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_signed(1)? as i16);
                }
                if len == 0 {
                    return Ok(self.make_empty_marker(name, &matlab_dims, "int16", None));
                }
                Ok(self.make_numeric_i16(name, matlab_dims, values))
            }
            2 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_signed(2)? as i32);
                }
                if len == 0 {
                    return Ok(self.make_empty_marker(name, &matlab_dims, "int32", None));
                }
                Ok(self.make_numeric_i32(name, matlab_dims, values))
            }
            3 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_signed(3)? as i64);
                }
                if len == 0 {
                    return Ok(self.make_empty_marker(name, &matlab_dims, "int64", None));
                }
                Ok(self.make_numeric_i64(name, matlab_dims, values))
            }
            4 => Err(Error::msg(format!("unsupported i128 array at {path}"))),
            _ => Err(Error::InvalidSize),
        }
    }

    fn write_unsigned_array(
        &mut self,
        name: String,
        reader: &mut Reader<'_>,
        byte_code: u8,
        len: usize,
        path: &str,
    ) -> Result<MatNode> {
        let matlab_dims = vector_dims(len, self.options.one_dimensional_mode);
        match byte_code {
            0 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_unsigned(0)? as u8);
                }
                if len == 0 {
                    return Ok(self.make_empty_marker(name, &matlab_dims, "uint8", None));
                }
                Ok(self.make_numeric_u8(name, matlab_dims, values))
            }
            1 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_unsigned(1)? as u16);
                }
                if len == 0 {
                    return Ok(self.make_empty_marker(name, &matlab_dims, "uint16", None));
                }
                Ok(self.make_numeric_u16(name, matlab_dims, values))
            }
            2 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_unsigned(2)? as u32);
                }
                if len == 0 {
                    return Ok(self.make_empty_marker(name, &matlab_dims, "uint32", None));
                }
                Ok(self.make_numeric_u32(name, matlab_dims, values))
            }
            3 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_unsigned(3)? as u64);
                }
                if len == 0 {
                    return Ok(self.make_empty_marker(name, &matlab_dims, "uint64", None));
                }
                Ok(self.make_numeric_u64(name, matlab_dims, values))
            }
            4 => Err(Error::msg(format!("unsupported u128 array at {path}"))),
            _ => Err(Error::InvalidSize),
        }
    }

    fn write_bool_array(
        &mut self,
        name: String,
        reader: &mut Reader<'_>,
        len: usize,
    ) -> Result<MatNode> {
        let matlab_dims = vector_dims(len, self.options.one_dimensional_mode);
        if len == 0 {
            return Ok(self.make_empty_marker(name, &matlab_dims, "logical", Some(1)));
        }
        let packed = reader.read_exact(len.div_ceil(8))?;
        let unpacked = unpack_bools(packed, len);
        Ok(self.make_logical(name, matlab_dims, unpacked))
    }

    fn write_string_array(
        &mut self,
        name: String,
        reader: &mut Reader<'_>,
        len: usize,
    ) -> Result<MatNode> {
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push(reader.read_string()?.to_owned());
        }
        let matlab_dims = vector_dims(len, self.options.one_dimensional_mode);
        self.write_string_object(name, &values, &matlab_dims)
    }

    fn write_matrix_string(
        &mut self,
        name: String,
        reader: &mut Reader<'_>,
        ctx: MatrixWriteContext<'_>,
    ) -> Result<MatNode> {
        let len = product(ctx.extents)?;
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push(reader.read_string()?.to_owned());
        }
        let values = self.maybe_reorder(values, ctx.layout, ctx.extents, ctx.path)?;
        let matlab_dims = matrix_dims(ctx.extents, self.options.one_dimensional_mode);
        self.write_string_object(name, &values, &matlab_dims)
    }

    fn write_matrix_float(
        &mut self,
        name: String,
        reader: &mut Reader<'_>,
        byte_code: u8,
        ctx: MatrixWriteContext<'_>,
    ) -> Result<MatNode> {
        let len = product(ctx.extents)?;
        let matlab_dims = matrix_dims(ctx.extents, self.options.one_dimensional_mode);
        match byte_code {
            0 => match self.options.unsupported_policy {
                UnsupportedPolicy::LossyNumericWidening => {
                    let mut values = Vec::with_capacity(len);
                    for _ in 0..len {
                        values.push(reader.read_bf16_as_f32()?);
                    }
                    let values = self.maybe_reorder(values, ctx.layout, ctx.extents, ctx.path)?;
                    Ok(self.make_numeric_f32(name, matlab_dims, values))
                }
                _ => Err(Error::msg(format!(
                    "unsupported bf16 matrix at {}; enable LossyNumericWidening to map it to MATLAB single",
                    ctx.path
                ))),
            },
            1 => match self.options.unsupported_policy {
                UnsupportedPolicy::LossyNumericWidening => {
                    let mut values = Vec::with_capacity(len);
                    for _ in 0..len {
                        values.push(reader.read_f16_as_f32()?);
                    }
                    let values = self.maybe_reorder(values, ctx.layout, ctx.extents, ctx.path)?;
                    Ok(self.make_numeric_f32(name, matlab_dims, values))
                }
                _ => Err(Error::msg(format!(
                    "unsupported f16 matrix at {}; enable LossyNumericWidening to map it to MATLAB single",
                    ctx.path
                ))),
            },
            2 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_f32()?);
                }
                let values = self.maybe_reorder(values, ctx.layout, ctx.extents, ctx.path)?;
                Ok(self.make_numeric_f32(name, matlab_dims, values))
            }
            3 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_f64()?);
                }
                let values = self.maybe_reorder(values, ctx.layout, ctx.extents, ctx.path)?;
                Ok(self.make_numeric_f64(name, matlab_dims, values))
            }
            _ => Err(Error::msg(format!(
                "unsupported float matrix width at {}",
                ctx.path
            ))),
        }
    }

    fn write_matrix_signed(
        &mut self,
        name: String,
        reader: &mut Reader<'_>,
        byte_code: u8,
        ctx: MatrixWriteContext<'_>,
    ) -> Result<MatNode> {
        let len = product(ctx.extents)?;
        let matlab_dims = matrix_dims(ctx.extents, self.options.one_dimensional_mode);
        match byte_code {
            0 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_signed(0)? as i8);
                }
                let values = self.maybe_reorder(values, ctx.layout, ctx.extents, ctx.path)?;
                Ok(self.make_numeric_i8(name, matlab_dims, values))
            }
            1 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_signed(1)? as i16);
                }
                let values = self.maybe_reorder(values, ctx.layout, ctx.extents, ctx.path)?;
                Ok(self.make_numeric_i16(name, matlab_dims, values))
            }
            2 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_signed(2)? as i32);
                }
                let values = self.maybe_reorder(values, ctx.layout, ctx.extents, ctx.path)?;
                Ok(self.make_numeric_i32(name, matlab_dims, values))
            }
            3 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_signed(3)? as i64);
                }
                let values = self.maybe_reorder(values, ctx.layout, ctx.extents, ctx.path)?;
                Ok(self.make_numeric_i64(name, matlab_dims, values))
            }
            _ => Err(Error::msg(format!(
                "unsupported i128 matrix at {}",
                ctx.path
            ))),
        }
    }

    fn write_matrix_unsigned(
        &mut self,
        name: String,
        reader: &mut Reader<'_>,
        byte_code: u8,
        ctx: MatrixWriteContext<'_>,
    ) -> Result<MatNode> {
        let len = product(ctx.extents)?;
        let matlab_dims = matrix_dims(ctx.extents, self.options.one_dimensional_mode);
        match byte_code {
            0 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_unsigned(0)? as u8);
                }
                let values = self.maybe_reorder(values, ctx.layout, ctx.extents, ctx.path)?;
                Ok(self.make_numeric_u8(name, matlab_dims, values))
            }
            1 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_unsigned(1)? as u16);
                }
                let values = self.maybe_reorder(values, ctx.layout, ctx.extents, ctx.path)?;
                Ok(self.make_numeric_u16(name, matlab_dims, values))
            }
            2 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_unsigned(2)? as u32);
                }
                let values = self.maybe_reorder(values, ctx.layout, ctx.extents, ctx.path)?;
                Ok(self.make_numeric_u32(name, matlab_dims, values))
            }
            3 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_unsigned(3)? as u64);
                }
                let values = self.maybe_reorder(values, ctx.layout, ctx.extents, ctx.path)?;
                Ok(self.make_numeric_u64(name, matlab_dims, values))
            }
            _ => Err(Error::msg(format!(
                "unsupported u128 matrix at {}",
                ctx.path
            ))),
        }
    }

    fn write_matrix_bool(
        &mut self,
        name: String,
        reader: &mut Reader<'_>,
        extents: &[usize],
        layout: MatrixLayout,
    ) -> Result<MatNode> {
        let len = product(extents)?;
        let matlab_dims = matrix_dims(extents, self.options.one_dimensional_mode);
        let packed = reader.read_exact(len.div_ceil(8))?;
        let unpacked = unpack_bools(packed, len);
        let values = self.maybe_reorder(unpacked, layout, extents, "$matrix")?;
        Ok(self.make_logical(name, matlab_dims, values))
    }

    fn write_value_reference(&mut self, reader: &mut Reader<'_>, path: &str) -> Result<String> {
        let ref_name = self.next_ref_name();
        let ref_path = format!("#refs#/{ref_name}");
        let node = self.write_named_value(ref_name, reader, path)?;
        self.refs_children.push(node);
        Ok(ref_path)
    }

    fn handle_128_bit_scalar(
        &mut self,
        name: String,
        path: &str,
        value: String,
    ) -> Result<MatNode> {
        match self.options.unsupported_policy {
            UnsupportedPolicy::StringFallback => self.write_string_scalar(name, &value),
            _ => Err(Error::msg(format!(
                "unsupported 128-bit integer scalar at {path}"
            ))),
        }
    }

    fn normalize_name(&self, raw: &str, used: &mut BTreeSet<String>) -> Result<String> {
        let candidate = match self.options.invalid_name_policy {
            InvalidNamePolicy::Error => {
                validate_name(raw)?;
                raw.to_owned()
            }
            InvalidNamePolicy::Sanitize => sanitize_name(raw),
        };
        let unique = dedupe_name(candidate, used);
        used.insert(unique.clone());
        Ok(unique)
    }

    fn next_ref_name(&mut self) -> String {
        let name = format!("ref_{:016x}", self.next_ref_id);
        self.next_ref_id += 1;
        name
    }

    fn write_string_scalar(&mut self, name: String, value: &str) -> Result<MatNode> {
        self.write_string_object(name, &[value.to_owned()], &[1, 1])
    }

    fn write_string_object(
        &mut self,
        name: String,
        values: &[String],
        matlab_dims: &[usize],
    ) -> Result<MatNode> {
        let payload_path = self.write_string_saveobj_payload(values, matlab_dims)?;
        let object_id = self.string_objects.register(payload_path);
        let metadata = create_string_object_metadata(object_id);
        let shape = vec![1, metadata.len() as u64];
        Ok(MatNode::Dataset(MatDataset {
            name,
            content: DatasetContent::U32(shape, metadata.to_vec()),
            attrs: vec![
                (
                    "MATLAB_class",
                    AttrValue::AsciiString(MATLAB_CLASS_STRING.into()),
                ),
                (
                    "MATLAB_object_decode",
                    AttrValue::I32(MATLAB_OBJECT_DECODE_OPAQUE),
                ),
            ],
        }))
    }

    fn write_string_saveobj_payload(
        &mut self,
        values: &[String],
        matlab_dims: &[usize],
    ) -> Result<String> {
        let payload = encode_string_saveobj_payload(values, matlab_dims)?;
        let ref_name = self.next_ref_name();
        let ref_path = format!("#refs#/{ref_name}");
        let shape = vec![1, payload.len() as u64];
        self.refs_children.push(MatNode::Dataset(MatDataset {
            name: ref_name,
            content: DatasetContent::U64(shape, payload),
            attrs: vec![("MATLAB_class", AttrValue::AsciiString("uint64".into()))],
        }));
        Ok(ref_path)
    }

    // ---------------------------------------------------------------------------
    // IR node constructors
    // ---------------------------------------------------------------------------

    fn make_numeric_u8(&self, name: String, dims: Vec<usize>, data: Vec<u8>) -> MatNode {
        let shape = storage_dims_u64(&dims);
        MatNode::Dataset(MatDataset {
            name,
            content: DatasetContent::U8(shape, data),
            attrs: vec![("MATLAB_class", AttrValue::AsciiString("uint8".into()))],
        })
    }

    fn make_numeric_u16(&self, name: String, dims: Vec<usize>, data: Vec<u16>) -> MatNode {
        let shape = storage_dims_u64(&dims);
        MatNode::Dataset(MatDataset {
            name,
            content: DatasetContent::U16(shape, data),
            attrs: vec![("MATLAB_class", AttrValue::AsciiString("uint16".into()))],
        })
    }

    fn make_numeric_u32(&self, name: String, dims: Vec<usize>, data: Vec<u32>) -> MatNode {
        let shape = storage_dims_u64(&dims);
        MatNode::Dataset(MatDataset {
            name,
            content: DatasetContent::U32(shape, data),
            attrs: vec![("MATLAB_class", AttrValue::AsciiString("uint32".into()))],
        })
    }

    fn make_numeric_u64(&self, name: String, dims: Vec<usize>, data: Vec<u64>) -> MatNode {
        let shape = storage_dims_u64(&dims);
        MatNode::Dataset(MatDataset {
            name,
            content: DatasetContent::U64(shape, data),
            attrs: vec![("MATLAB_class", AttrValue::AsciiString("uint64".into()))],
        })
    }

    fn make_numeric_i8(&self, name: String, dims: Vec<usize>, data: Vec<i8>) -> MatNode {
        let shape = storage_dims_u64(&dims);
        MatNode::Dataset(MatDataset {
            name,
            content: DatasetContent::I8(shape, data),
            attrs: vec![("MATLAB_class", AttrValue::AsciiString("int8".into()))],
        })
    }

    fn make_numeric_i16(&self, name: String, dims: Vec<usize>, data: Vec<i16>) -> MatNode {
        let shape = storage_dims_u64(&dims);
        MatNode::Dataset(MatDataset {
            name,
            content: DatasetContent::I16(shape, data),
            attrs: vec![("MATLAB_class", AttrValue::AsciiString("int16".into()))],
        })
    }

    fn make_numeric_i32(&self, name: String, dims: Vec<usize>, data: Vec<i32>) -> MatNode {
        let shape = storage_dims_u64(&dims);
        MatNode::Dataset(MatDataset {
            name,
            content: DatasetContent::I32(shape, data),
            attrs: vec![("MATLAB_class", AttrValue::AsciiString("int32".into()))],
        })
    }

    fn make_numeric_i64(&self, name: String, dims: Vec<usize>, data: Vec<i64>) -> MatNode {
        let shape = storage_dims_u64(&dims);
        MatNode::Dataset(MatDataset {
            name,
            content: DatasetContent::I64(shape, data),
            attrs: vec![("MATLAB_class", AttrValue::AsciiString("int64".into()))],
        })
    }

    fn make_numeric_f32(&self, name: String, dims: Vec<usize>, data: Vec<f32>) -> MatNode {
        let shape = storage_dims_u64(&dims);
        MatNode::Dataset(MatDataset {
            name,
            content: DatasetContent::F32(shape, data),
            attrs: vec![("MATLAB_class", AttrValue::AsciiString("single".into()))],
        })
    }

    fn make_numeric_f64(&self, name: String, dims: Vec<usize>, data: Vec<f64>) -> MatNode {
        let shape = storage_dims_u64(&dims);
        MatNode::Dataset(MatDataset {
            name,
            content: DatasetContent::F64(shape, data),
            attrs: vec![("MATLAB_class", AttrValue::AsciiString("double".into()))],
        })
    }

    fn make_logical(&self, name: String, dims: Vec<usize>, data: Vec<u8>) -> MatNode {
        let shape = storage_dims_u64(&dims);
        MatNode::Dataset(MatDataset {
            name,
            content: DatasetContent::U8(shape, data),
            attrs: vec![
                ("MATLAB_class", AttrValue::AsciiString("logical".into())),
                ("MATLAB_int_decode", AttrValue::I32(1)),
            ],
        })
    }

    fn make_complex32(&self, name: String, dims: Vec<usize>, data: Vec<(f32, f32)>) -> MatNode {
        let shape = storage_dims_u64(&dims);
        MatNode::Dataset(MatDataset {
            name,
            content: DatasetContent::Complex32(shape, data),
            attrs: vec![("MATLAB_class", AttrValue::AsciiString("single".into()))],
        })
    }

    fn make_complex64(&self, name: String, dims: Vec<usize>, data: Vec<(f64, f64)>) -> MatNode {
        let shape = storage_dims_u64(&dims);
        MatNode::Dataset(MatDataset {
            name,
            content: DatasetContent::Complex64(shape, data),
            attrs: vec![("MATLAB_class", AttrValue::AsciiString("double".into()))],
        })
    }

    fn make_empty_marker(
        &self,
        name: String,
        matlab_dims: &[usize],
        class: &str,
        int_decode: Option<u8>,
    ) -> MatNode {
        let shape_data: Vec<u64> = matlab_dims.iter().map(|&dim| dim as u64).collect();
        let mut attrs: Vec<(&'static str, AttrValue)> = vec![
            ("MATLAB_class", AttrValue::AsciiString(class.to_string())),
            ("MATLAB_empty", AttrValue::U32(1)),
        ];
        if let Some(code) = int_decode {
            attrs.push(("MATLAB_int_decode", AttrValue::I32(i32::from(code))));
        }
        MatNode::Dataset(MatDataset {
            name,
            content: DatasetContent::U64(vec![shape_data.len() as u64], shape_data),
            attrs,
        })
    }

    fn make_empty_struct_array(&self, name: String) -> MatNode {
        self.make_empty_marker(name, &[0, 0], "struct", None)
    }

    fn make_reference_array(
        &self,
        name: String,
        matlab_dims: &[usize],
        ref_paths: Vec<String>,
        class: &str,
    ) -> MatNode {
        if ref_paths.is_empty() {
            return self.make_empty_marker(name, matlab_dims, class, None);
        }
        let shape = storage_dims_u64(matlab_dims);
        MatNode::Dataset(MatDataset {
            name,
            content: DatasetContent::References(shape, ref_paths),
            attrs: vec![("MATLAB_class", AttrValue::AsciiString(class.to_string()))],
        })
    }

    fn maybe_reorder<T: Clone>(
        &self,
        data: Vec<T>,
        layout: MatrixLayout,
        extents: &[usize],
        path: &str,
    ) -> Result<Vec<T>> {
        match layout {
            MatrixLayout::Left => Ok(data),
            MatrixLayout::Right => match self.options.row_major_policy {
                RowMajorPolicy::ReorderToColumnMajor => {
                    Ok(reorder_row_major_to_column_major(&data, extents))
                }
                RowMajorPolicy::Error => Err(Error::msg(format!(
                    "row-major matrix at {path} requires reordering to MATLAB column-major layout"
                ))),
            },
        }
    }

    // ---------------------------------------------------------------------------
    // Phase 2: IR → HDF5 bytes
    // ---------------------------------------------------------------------------

    fn build_hdf5_bytes(mut self) -> Result<Vec<u8>> {
        // Build string subsystem refs before constructing the HDF5 tree
        let subsystem_mcos = if !self.string_objects.is_empty() {
            Some(self.build_string_subsystem_mcos())
        } else {
            None
        };

        let mut builder = FileBuilder::new();
        builder.with_userblock(MATLAB_USERBLOCK_SIZE);

        // Place root nodes
        for node in self.root_nodes {
            place_node_on_file(&mut builder, node, &self.options.compression);
        }

        // Place #refs# group
        if !self.refs_children.is_empty() {
            let mut refs_group = builder.create_group("#refs#");
            for node in self.refs_children {
                place_node_on_group(&mut refs_group, node, &self.options.compression);
            }
            builder.add_group(refs_group.finish());
        }

        // Place #subsystem# group
        if let Some(mcos) = subsystem_mcos {
            let mut subsystem_group = builder.create_group("#subsystem#");
            let ds = subsystem_group.create_dataset("MCOS");
            ds.with_shape(&mcos.shape);
            ds.with_path_references(
                &mcos
                    .ref_paths
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>(),
            );
            ds.set_attr(
                "MATLAB_class",
                AttrValue::AsciiString(MATLAB_CLASS_FILEWRAPPER.into()),
            );
            ds.set_attr(
                "MATLAB_object_decode",
                AttrValue::I32(MATLAB_OBJECT_DECODE_OPAQUE),
            );
            builder.add_group(subsystem_group.finish());
        }

        let mut bytes = builder.finish()?;
        write_userblock_header(&mut bytes);
        Ok(bytes)
    }

    fn build_string_subsystem_mcos(&mut self) -> McosData {
        // Build all the helper refs needed for the subsystem metadata
        let metadata_ref_name = self.next_ref_name();
        let metadata_data =
            build_string_filewrapper_metadata(self.string_objects.payload_paths.len());
        self.refs_children.push(MatNode::Dataset(MatDataset {
            name: metadata_ref_name.clone(),
            content: DatasetContent::U8(vec![1, metadata_data.len() as u64], metadata_data),
            attrs: vec![("MATLAB_class", AttrValue::AsciiString("uint8".into()))],
        }));
        let metadata_ref_path = format!("#refs#/{metadata_ref_name}");

        // Canonical empty
        let canonical_empty_name = self.next_ref_name();
        self.refs_children.push(self.make_empty_marker(
            canonical_empty_name.clone(),
            &[0, 0],
            "canonical empty",
            None,
        ));
        let canonical_empty_path = format!("#refs#/{canonical_empty_name}");

        // Unknown template A: cell { empty_struct, empty_struct }
        let empty_struct_a1_name = self.next_ref_name();
        self.refs_children.push(self.make_empty_marker(
            empty_struct_a1_name.clone(),
            &[1, 0],
            "struct",
            None,
        ));
        let empty_struct_a1_path = format!("#refs#/{empty_struct_a1_name}");

        let empty_struct_a2_name = self.next_ref_name();
        self.refs_children.push(self.make_empty_marker(
            empty_struct_a2_name.clone(),
            &[1, 0],
            "struct",
            None,
        ));
        let empty_struct_a2_path = format!("#refs#/{empty_struct_a2_name}");

        let unknown_template_a_name = self.next_ref_name();
        self.refs_children.push(self.make_reference_array(
            unknown_template_a_name.clone(),
            &[2, 1],
            vec![empty_struct_a1_path, empty_struct_a2_path],
            "cell",
        ));
        let unknown_template_a_path = format!("#refs#/{unknown_template_a_name}");

        // Alias metadata ref
        let alias_ref_name = self.next_ref_name();
        self.refs_children.push(MatNode::Dataset(MatDataset {
            name: alias_ref_name.clone(),
            content: DatasetContent::I32(vec![1, 2], vec![0i32, 0]),
            attrs: vec![("MATLAB_class", AttrValue::AsciiString("int32".into()))],
        }));
        let alias_ref_path = format!("#refs#/{alias_ref_name}");

        // Unknown template B: cell { empty_struct, empty_struct }
        let empty_struct_b1_name = self.next_ref_name();
        self.refs_children.push(self.make_empty_marker(
            empty_struct_b1_name.clone(),
            &[1, 0],
            "struct",
            None,
        ));
        let empty_struct_b1_path = format!("#refs#/{empty_struct_b1_name}");

        let empty_struct_b2_name = self.next_ref_name();
        self.refs_children.push(self.make_empty_marker(
            empty_struct_b2_name.clone(),
            &[1, 0],
            "struct",
            None,
        ));
        let empty_struct_b2_path = format!("#refs#/{empty_struct_b2_name}");

        let unknown_template_b_name = self.next_ref_name();
        self.refs_children.push(self.make_reference_array(
            unknown_template_b_name.clone(),
            &[2, 1],
            vec![empty_struct_b1_path, empty_struct_b2_path],
            "cell",
        ));
        let unknown_template_b_path = format!("#refs#/{unknown_template_b_name}");

        // Build the MCOS reference array paths
        let mut mcos_paths = Vec::with_capacity(self.string_objects.payload_paths.len() + 5);
        mcos_paths.push(metadata_ref_path);
        mcos_paths.push(canonical_empty_path);
        mcos_paths.extend(self.string_objects.payload_paths.iter().cloned());
        mcos_paths.push(unknown_template_a_path);
        mcos_paths.push(alias_ref_path);
        mcos_paths.push(unknown_template_b_path);

        let shape = vec![1u64, mcos_paths.len() as u64];
        McosData {
            shape,
            ref_paths: mcos_paths,
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 2 helpers: place IR nodes onto FileBuilder / GroupBuilder
// ---------------------------------------------------------------------------

fn place_node_on_file(builder: &mut FileBuilder, node: MatNode, compression: &Compression) {
    match node {
        MatNode::Dataset(ds) => {
            place_dataset_on_file(builder, ds, compression);
        }
        MatNode::Group(grp) => {
            let mut gb = builder.create_group(&grp.name);
            for (attr_name, attr_value) in grp.attrs {
                gb.set_attr(attr_name, attr_value);
            }
            for child in grp.children {
                place_node_on_group(&mut gb, child, compression);
            }
            builder.add_group(gb.finish());
        }
    }
}

fn place_node_on_group(group: &mut GroupBuilder, node: MatNode, compression: &Compression) {
    match node {
        MatNode::Dataset(ds) => {
            place_dataset_on_group(group, ds, compression);
        }
        MatNode::Group(grp) => {
            let mut child = group.create_group(&grp.name);
            for (attr_name, attr_value) in grp.attrs {
                child.set_attr(attr_name, attr_value);
            }
            for inner in grp.children {
                place_node_on_group(&mut child, inner, compression);
            }
            group.add_group(child.finish());
        }
    }
}

fn configure_dataset(ds: &mut DatasetBuilder, compression: &Compression) {
    if let Compression::Deflate { level, shuffle } = compression {
        if *shuffle {
            ds.with_shuffle();
        }
        ds.with_deflate(*level as u32);
    }
}

fn place_dataset_on_file(builder: &mut FileBuilder, mat_ds: MatDataset, compression: &Compression) {
    let ds = builder.create_dataset(&mat_ds.name);
    apply_dataset_content(ds, mat_ds.content, compression);
    for (attr_name, attr_value) in mat_ds.attrs {
        ds.set_attr(attr_name, attr_value);
    }
}

fn place_dataset_on_group(group: &mut GroupBuilder, mat_ds: MatDataset, compression: &Compression) {
    let ds = group.create_dataset(&mat_ds.name);
    apply_dataset_content(ds, mat_ds.content, compression);
    for (attr_name, attr_value) in mat_ds.attrs {
        ds.set_attr(attr_name, attr_value);
    }
}

fn apply_dataset_content(
    ds: &mut DatasetBuilder,
    content: DatasetContent,
    compression: &Compression,
) {
    match content {
        DatasetContent::U8(shape, data) => {
            ds.with_shape(&shape);
            configure_dataset(ds, compression);
            ds.with_u8_data(&data);
        }
        DatasetContent::U16(shape, data) => {
            ds.with_shape(&shape);
            configure_dataset(ds, compression);
            ds.with_u16_data(&data);
        }
        DatasetContent::U32(shape, data) => {
            ds.with_shape(&shape);
            configure_dataset(ds, compression);
            ds.with_u32_data(&data);
        }
        DatasetContent::U64(shape, data) => {
            ds.with_shape(&shape);
            configure_dataset(ds, compression);
            ds.with_u64_data(&data);
        }
        DatasetContent::I8(shape, data) => {
            ds.with_shape(&shape);
            configure_dataset(ds, compression);
            ds.with_i8_data(&data);
        }
        DatasetContent::I16(shape, data) => {
            ds.with_shape(&shape);
            configure_dataset(ds, compression);
            ds.with_i16_data(&data);
        }
        DatasetContent::I32(shape, data) => {
            ds.with_shape(&shape);
            configure_dataset(ds, compression);
            ds.with_i32_data(&data);
        }
        DatasetContent::I64(shape, data) => {
            ds.with_shape(&shape);
            configure_dataset(ds, compression);
            ds.with_i64_data(&data);
        }
        DatasetContent::F32(shape, data) => {
            ds.with_shape(&shape);
            configure_dataset(ds, compression);
            ds.with_f32_data(&data);
        }
        DatasetContent::F64(shape, data) => {
            ds.with_shape(&shape);
            configure_dataset(ds, compression);
            ds.with_f64_data(&data);
        }
        DatasetContent::Complex32(shape, data) => {
            ds.with_shape(&shape);
            configure_dataset(ds, compression);
            ds.with_complex32_data(&data);
        }
        DatasetContent::Complex64(shape, data) => {
            ds.with_shape(&shape);
            configure_dataset(ds, compression);
            ds.with_complex64_data(&data);
        }
        DatasetContent::References(shape, paths) => {
            ds.with_shape(&shape);
            let path_strs: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
            ds.with_path_references(&path_strs);
        }
    }
}

// ---------------------------------------------------------------------------
// Userblock
// ---------------------------------------------------------------------------

fn write_userblock_header(bytes: &mut [u8]) {
    let text = matlab_header_text();
    bytes[..116].copy_from_slice(&text);
    bytes[124] = 0;
    bytes[125] = 2;
    bytes[126] = b'I';
    bytes[127] = b'M';
}

// ---------------------------------------------------------------------------
// Dimension helpers
// ---------------------------------------------------------------------------

fn vector_dims(len: usize, mode: OneDimensionalMode) -> Vec<usize> {
    match mode {
        OneDimensionalMode::ColumnVector => vec![len, 1],
        OneDimensionalMode::RowVector => vec![1, len],
    }
}

fn matrix_dims(extents: &[usize], mode: OneDimensionalMode) -> Vec<usize> {
    match extents {
        [] => vec![1, 1],
        [len] => vector_dims(*len, mode),
        _ => extents.to_vec(),
    }
}

fn storage_dims(matlab_dims: &[usize]) -> Vec<usize> {
    let mut dims = matlab_dims.to_vec();
    if dims.len() < 2 {
        dims.resize(2, 1);
    }
    dims.reverse();
    dims
}

fn storage_dims_u64(matlab_dims: &[usize]) -> Vec<u64> {
    storage_dims(matlab_dims)
        .iter()
        .map(|&d| d as u64)
        .collect()
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

// ---------------------------------------------------------------------------
// Data reordering
// ---------------------------------------------------------------------------

fn reorder_row_major_to_column_major<T: Clone>(data: &[T], extents: &[usize]) -> Vec<T> {
    if extents.len() <= 1 || data.len() <= 1 {
        return data.to_vec();
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

fn unpack_bools(packed: &[u8], len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    for idx in 0..len {
        let byte = packed[idx / 8];
        let bit = (byte >> (idx % 8)) & 1;
        out.push(bit);
    }
    out
}

// ---------------------------------------------------------------------------
// MATLAB string object helpers
// ---------------------------------------------------------------------------

fn create_string_object_metadata(object_id: u32) -> [u32; 6] {
    [MCOS_MAGIC_NUMBER, 2, 1, 1, object_id, 1]
}

fn encode_string_saveobj_payload(values: &[String], matlab_dims: &[usize]) -> Result<Vec<u64>> {
    let expected = product(matlab_dims)?;
    if expected != values.len() {
        return Err(Error::msg(format!(
            "string payload length {} does not match product of dims {}",
            values.len(),
            expected
        )));
    }

    let mut payload = Vec::with_capacity(2 + matlab_dims.len() + values.len());
    payload.push(MATLAB_STRING_SAVEOBJ_VERSION);
    payload.push(matlab_dims.len() as u64);
    payload.extend(matlab_dims.iter().map(|&dim| dim as u64));

    let mut utf16_bytes = Vec::new();
    for value in values {
        let units: Vec<u16> = value.encode_utf16().collect();
        payload.push(units.len() as u64);
        for unit in units {
            utf16_bytes.extend_from_slice(&unit.to_le_bytes());
        }
    }

    while utf16_bytes.len() % 8 != 0 {
        utf16_bytes.push(0);
    }
    payload.extend(
        utf16_bytes
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap())),
    );
    Ok(payload)
}

fn build_string_filewrapper_metadata(object_count: usize) -> Vec<u8> {
    let mut names_bytes = b"any\0string\0".to_vec();
    while !names_bytes.len().is_multiple_of(8) {
        names_bytes.push(0);
    }

    let mut region_offsets = [0u32; 8];
    let mut regions = Vec::new();

    let version = u32_bytes(&[FILEWRAPPER_VERSION]);
    let num_names = u32_bytes(&[2]);
    let class_id_metadata = u32_bytes(&[0, 0, 0, 0, 0, 2, 0, 0]);

    let mut saveobj_metadata = Vec::with_capacity(2 + object_count * 4);
    saveobj_metadata.extend_from_slice(&[0, 0]);
    for idx in 0..object_count {
        saveobj_metadata.extend_from_slice(&[1, 1, 1, idx as u32]);
    }
    let saveobj_metadata = u32_bytes(&saveobj_metadata);

    let mut object_id_metadata = Vec::with_capacity(6 + object_count * 6);
    object_id_metadata.extend_from_slice(&[0, 0, 0, 0, 0, 0]);
    for id in 1..=object_count as u32 {
        object_id_metadata.extend_from_slice(&[1, 0, 0, id, 0, id]);
    }
    let object_id_metadata = u32_bytes(&object_id_metadata);

    let nobj_metadata = u32_bytes(&[0, 0]);

    let mut dynprop_metadata = Vec::with_capacity(2 + object_count * 2);
    dynprop_metadata.extend_from_slice(&[0, 0]);
    for _ in 0..object_count {
        dynprop_metadata.extend_from_slice(&[0, 0]);
    }
    let dynprop_metadata = u32_bytes(&dynprop_metadata);

    let region6 = Vec::new();
    let region7 = vec![0u8; 8];

    let mut offset = 40u32 + names_bytes.len() as u32;
    region_offsets[0] = offset;
    offset += class_id_metadata.len() as u32;
    region_offsets[1] = offset;
    offset += saveobj_metadata.len() as u32;
    region_offsets[2] = offset;
    offset += object_id_metadata.len() as u32;
    region_offsets[3] = offset;
    offset += nobj_metadata.len() as u32;
    region_offsets[4] = offset;
    offset += dynprop_metadata.len() as u32;
    region_offsets[5] = offset;
    offset += region6.len() as u32;
    region_offsets[6] = offset;
    offset += region7.len() as u32;
    region_offsets[7] = offset;

    regions.extend_from_slice(&version);
    regions.extend_from_slice(&num_names);
    regions.extend_from_slice(&u32_bytes(&region_offsets));
    regions.extend_from_slice(&names_bytes);
    regions.extend_from_slice(&class_id_metadata);
    regions.extend_from_slice(&saveobj_metadata);
    regions.extend_from_slice(&object_id_metadata);
    regions.extend_from_slice(&nobj_metadata);
    regions.extend_from_slice(&dynprop_metadata);
    regions.extend_from_slice(&region6);
    regions.extend_from_slice(&region7);
    regions
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

// ---------------------------------------------------------------------------
// MATLAB name validation / sanitization
// ---------------------------------------------------------------------------

fn validate_name(name: &str) -> Result<()> {
    if !is_valid_name(name) {
        return Err(Error::msg(format!("invalid MATLAB name: {name}")));
    }
    Ok(())
}

fn is_valid_name(name: &str) -> bool {
    if name.is_empty() || name.len() > MATLAB_NAME_MAX || is_keyword(name) {
        return false;
    }
    let mut chars = name.chars();
    match chars.next() {
        Some(ch) if ch.is_ascii_alphabetic() => {}
        _ => return false,
    }
    chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

fn is_keyword(name: &str) -> bool {
    MATLAB_KEYWORDS.binary_search(&name).is_ok()
}

fn sanitize_name(name: &str) -> String {
    let mut out = String::new();
    for (idx, ch) in name.chars().enumerate() {
        let valid = if idx == 0 {
            ch.is_ascii_alphabetic()
        } else {
            ch.is_ascii_alphanumeric() || ch == '_'
        };
        if valid {
            out.push(ch);
        } else if idx == 0 {
            out.push('x');
            if ch.is_ascii_alphanumeric() || ch == '_' {
                out.push(ch);
            } else {
                out.push('_');
            }
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        out.push('x');
    }
    if is_keyword(&out) {
        out.push('_');
    }
    if out.len() > MATLAB_NAME_MAX {
        out.truncate(MATLAB_NAME_MAX);
    }
    out
}

fn dedupe_name(mut candidate: String, used: &BTreeSet<String>) -> String {
    if !used.contains(&candidate) {
        return candidate;
    }
    let base = candidate.clone();
    let mut suffix = 1usize;
    loop {
        let suffix_str = format!("_{suffix}");
        let max_base_len = MATLAB_NAME_MAX.saturating_sub(suffix_str.len());
        if max_base_len == 0 {
            let tail_len = MATLAB_NAME_MAX.saturating_sub(1);
            let start = suffix_str.len().saturating_sub(tail_len);
            candidate.clear();
            candidate.push('x');
            candidate.push_str(&suffix_str[start..]);
        } else {
            let prefix = if base.len() > max_base_len {
                &base[..max_base_len]
            } else {
                &base
            };
            candidate.clear();
            candidate.push_str(prefix);
            candidate.push_str(&suffix_str);
        }
        if !used.contains(&candidate) {
            return candidate;
        }
        suffix += 1;
    }
}

// ---------------------------------------------------------------------------
// File I/O helpers
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

fn matlab_header_text() -> [u8; 116] {
    let mut out = [b' '; 116];
    let text = format!(
        "MATLAB 7.3 MAT-file, Platform: {platform}, Created by: beve-rs, HDF5 schema 1.00 .",
        platform = std::env::consts::OS
    );
    let bytes = text.as_bytes();
    let len = bytes.len().min(out.len());
    out[..len].copy_from_slice(&bytes[..len]);
    out
}
