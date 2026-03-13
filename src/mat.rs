use std::collections::BTreeSet;
use std::ffi::OsString;
use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use hdf5::types::{FixedAscii, VarLenArray};
use hdf5::{Dataset, DatasetBuilder, File, Group, H5Type, ObjectReference1};
use hdf5_metno as hdf5;
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

#[repr(C)]
#[derive(Clone, Copy, H5Type)]
struct MatComplex32 {
    real: f32,
    imag: f32,
}

#[repr(C)]
#[derive(Clone, Copy, H5Type)]
struct MatComplex64 {
    real: f64,
    imag: f64,
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
    let file = File::with_options()
        .with_fcpl(|plist| plist.userblock(MATLAB_USERBLOCK_SIZE))
        .with_fapl(|plist| plist.libver_v18())
        .create(temp_output.path())?;
    let refs = file.create_group("#refs#")?;
    let mut writer = MatWriter {
        file,
        refs,
        options: options.clone(),
        next_ref_id: 0,
        string_objects: StringObjectState::new(),
    };
    let mut reader = Reader::new(beve);
    writer.write_root(&mut reader, root)?;
    if !reader.is_finished() {
        return Err(Error::InvalidType("unexpected trailing BEVE data"));
    }
    writer.finish()?;
    writer.file.flush()?;
    drop(writer);
    write_userblock(temp_output.path())?;
    sync_regular_file(temp_output.path())?;
    temp_output.persist(output_path)
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

struct MatWriter {
    file: File,
    refs: Group,
    options: MatV73Options,
    next_ref_id: u64,
    string_objects: StringObjectState,
}

#[derive(Clone, Copy)]
struct MatrixWriteContext<'a> {
    parent: &'a Group,
    name: &'a str,
    path: &'a str,
    layout: MatrixLayout,
    extents: &'a [usize],
}

struct StringObjectState {
    payload_refs: Vec<ObjectReference1>,
}

impl StringObjectState {
    fn new() -> Self {
        Self {
            payload_refs: Vec::new(),
        }
    }

    fn register(&mut self, payload_ref: ObjectReference1) -> u32 {
        self.payload_refs.push(payload_ref);
        self.payload_refs.len() as u32
    }

    fn is_empty(&self) -> bool {
        self.payload_refs.is_empty()
    }
}

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

impl MatWriter {
    fn finish(&mut self) -> Result<()> {
        if self.string_objects.is_empty() {
            return Ok(());
        }
        self.write_string_subsystem()
    }

    fn write_root(&mut self, reader: &mut Reader<'_>, root: RootBinding<'_>) -> Result<()> {
        match root {
            RootBinding::NamedVariable(name) => {
                let variable = self.normalize_name(name, &mut BTreeSet::new())?;
                // Clone the ref-counted HDF5 handle to avoid borrowing self immutably and mutably at once.
                let file = self.file.clone();
                self.write_named_value(&file, &variable, reader, "$")
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
                    let file = self.file.clone();
                    self.write_named_value(&file, &variable, reader, &path)?;
                }
                Ok(())
            }
        }
    }

    fn write_named_value(
        &mut self,
        parent: &Group,
        name: &str,
        reader: &mut Reader<'_>,
        path: &str,
    ) -> Result<()> {
        let header = reader.read_header()?;
        self.write_named_value_with_header(parent, name, reader, header, path)
    }

    fn write_named_value_with_header(
        &mut self,
        parent: &Group,
        name: &str,
        reader: &mut Reader<'_>,
        header: u8,
        path: &str,
    ) -> Result<()> {
        match parse_type(header) {
            TYPE_NULL_BOOL => {
                if header == 0 {
                    self.handle_null(parent, name, path)
                } else {
                    let value = bool_value(header)?;
                    self.write_logical(parent, name, &[1, 1], &[u8::from(value)])
                }
            }
            TYPE_NUMBER => self.write_number(parent, name, reader, header, path),
            TYPE_STRING => {
                let value = reader.read_string()?.to_owned();
                self.write_string_scalar(parent, name, &value)
            }
            TYPE_OBJECT => self.write_object(parent, name, reader, header, path),
            TYPE_TYPED_ARRAY => self.write_typed_array(parent, name, reader, header, path),
            TYPE_GENERIC_ARRAY => self.write_generic_array(parent, name, reader, path),
            TYPE_EXTENSION => self.write_extension(parent, name, reader, header, path),
            _ => Err(Error::InvalidHeader(header)),
        }
    }

    fn handle_null(&mut self, parent: &Group, name: &str, path: &str) -> Result<()> {
        match self.options.null_policy {
            NullPolicy::EmptyStructArray => self.write_empty_struct_array(parent, name),
            NullPolicy::Error => Err(Error::msg(format!("unsupported null value at {path}"))),
        }
    }

    fn write_number(
        &mut self,
        parent: &Group,
        name: &str,
        reader: &mut Reader<'_>,
        header: u8,
        path: &str,
    ) -> Result<()> {
        let subtype = parse_subtype(header);
        let byte_code = parse_byte_count_code(header);
        match subtype {
            NUM_SIGNED => match byte_code {
                0 => self.write_numeric(parent, name, &[1, 1], &[reader.read_signed(0)? as i8], "int8"),
                1 => self.write_numeric(
                    parent,
                    name,
                    &[1, 1],
                    &[reader.read_signed(1)? as i16],
                    "int16",
                ),
                2 => self.write_numeric(
                    parent,
                    name,
                    &[1, 1],
                    &[reader.read_signed(2)? as i32],
                    "int32",
                ),
                3 => self.write_numeric(
                    parent,
                    name,
                    &[1, 1],
                    &[reader.read_signed(3)? as i64],
                    "int64",
                ),
                4 => self.handle_128_bit_scalar(parent, name, path, reader.read_signed(4)?.to_string()),
                _ => Err(Error::InvalidHeader(header)),
            },
            NUM_UNSIGNED => match byte_code {
                0 => self.write_numeric(
                    parent,
                    name,
                    &[1, 1],
                    &[reader.read_unsigned(0)? as u8],
                    "uint8",
                ),
                1 => self.write_numeric(
                    parent,
                    name,
                    &[1, 1],
                    &[reader.read_unsigned(1)? as u16],
                    "uint16",
                ),
                2 => self.write_numeric(
                    parent,
                    name,
                    &[1, 1],
                    &[reader.read_unsigned(2)? as u32],
                    "uint32",
                ),
                3 => self.write_numeric(
                    parent,
                    name,
                    &[1, 1],
                    &[reader.read_unsigned(3)? as u64],
                    "uint64",
                ),
                4 => self.handle_128_bit_scalar(parent, name, path, reader.read_unsigned(4)?.to_string()),
                _ => Err(Error::InvalidHeader(header)),
            },
            NUM_FLOAT => match byte_code {
                0 => match self.options.unsupported_policy {
                    UnsupportedPolicy::LossyNumericWidening => {
                        self.write_numeric(parent, name, &[1, 1], &[reader.read_bf16_as_f32()?], "single")
                    }
                    _ => Err(Error::msg(format!(
                        "unsupported bf16 scalar at {path}; enable LossyNumericWidening to map it to MATLAB single"
                    ))),
                },
                1 => match self.options.unsupported_policy {
                    UnsupportedPolicy::LossyNumericWidening => {
                        self.write_numeric(parent, name, &[1, 1], &[reader.read_f16_as_f32()?], "single")
                    }
                    _ => Err(Error::msg(format!(
                        "unsupported f16 scalar at {path}; enable LossyNumericWidening to map it to MATLAB single"
                    ))),
                },
                2 => self.write_numeric(parent, name, &[1, 1], &[reader.read_f32()?], "single"),
                3 => self.write_numeric(parent, name, &[1, 1], &[reader.read_f64()?], "double"),
                _ => Err(Error::Unsupported("float128 is not supported")),
            },
            _ => Err(Error::InvalidHeader(header)),
        }
    }

    fn write_object(
        &mut self,
        parent: &Group,
        name: &str,
        reader: &mut Reader<'_>,
        header: u8,
        path: &str,
    ) -> Result<()> {
        let ObjectHeader { key_type, len, .. } = reader.read_object_header(header)?;
        if key_type != KEY_STRING {
            return Err(Error::msg(format!(
                "unsupported non-string object keys at {path}"
            )));
        }
        let group = parent.create_group(name)?;
        write_ascii_attr(&group, "MATLAB_class", "struct")?;
        let mut used = BTreeSet::new();
        let mut fields = Vec::with_capacity(len);
        for _ in 0..len {
            let key = reader.read_string()?;
            let field = self.normalize_name(key, &mut used)?;
            let child_path = format!("{path}.{field}");
            self.write_named_value(&group, &field, reader, &child_path)?;
            fields.push(field);
        }
        write_ascii_list_attr(&group, "MATLAB_fields", &fields)?;
        Ok(())
    }

    fn write_typed_array(
        &mut self,
        parent: &Group,
        name: &str,
        reader: &mut Reader<'_>,
        header: u8,
        path: &str,
    ) -> Result<()> {
        let info = reader.read_typed_array_header(header)?;
        match info.class {
            TypedArrayClass::Float => {
                self.write_float_array(parent, name, reader, info.byte_code, info.len, path)
            }
            TypedArrayClass::Signed => {
                self.write_signed_array(parent, name, reader, info.byte_code, info.len, path)
            }
            TypedArrayClass::Unsigned => {
                self.write_unsigned_array(parent, name, reader, info.byte_code, info.len, path)
            }
            TypedArrayClass::Bool => self.write_bool_array(parent, name, reader, info.len),
            TypedArrayClass::String => self.write_string_array(parent, name, reader, info.len),
        }
    }

    fn write_generic_array(
        &mut self,
        parent: &Group,
        name: &str,
        reader: &mut Reader<'_>,
        path: &str,
    ) -> Result<()> {
        let len = reader.read_size()?;
        let matlab_dims = vector_dims(len, self.options.one_dimensional_mode);
        if len == 0 {
            return self.write_empty_marker(parent, name, &matlab_dims, "cell", None);
        }
        let mut refs = Vec::with_capacity(len);
        for idx in 0..len {
            let child_path = format!("{path}[{idx}]");
            refs.push(self.write_value_reference(reader, &child_path)?);
        }
        self.write_reference_array(parent, name, &matlab_dims, &refs, "cell")
    }

    fn write_extension(
        &mut self,
        parent: &Group,
        name: &str,
        reader: &mut Reader<'_>,
        header: u8,
        path: &str,
    ) -> Result<()> {
        match parse_extension_id(header) {
            EXT_COMPLEX => self.write_complex_extension(parent, name, reader, path),
            EXT_MATRICES => self.write_matrix_extension(parent, name, reader, path),
            _ => Err(Error::msg(format!(
                "unsupported BEVE extension {:#x} at {path}",
                parse_extension_id(header)
            ))),
        }
    }

    fn write_complex_extension(
        &mut self,
        parent: &Group,
        name: &str,
        reader: &mut Reader<'_>,
        path: &str,
    ) -> Result<()> {
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
                        data.push(MatComplex32 {
                            real: reader.read_f32()?,
                            imag: reader.read_f32()?,
                        });
                    }
                    self.write_complex(parent, name, &matlab_dims, &data, "single")
                }
                3 => {
                    let mut data = Vec::with_capacity(len);
                    for _ in 0..len {
                        data.push(MatComplex64 {
                            real: reader.read_f64()?,
                            imag: reader.read_f64()?,
                        });
                    }
                    self.write_complex(parent, name, &matlab_dims, &data, "double")
                }
                _ => Err(Error::msg(format!(
                    "unsupported complex element width at {path}"
                ))),
            }
        } else {
            match byte_code {
                2 => self.write_complex(
                    parent,
                    name,
                    &[1, 1],
                    &[MatComplex32 {
                        real: reader.read_f32()?,
                        imag: reader.read_f32()?,
                    }],
                    "single",
                ),
                3 => self.write_complex(
                    parent,
                    name,
                    &[1, 1],
                    &[MatComplex64 {
                        real: reader.read_f64()?,
                        imag: reader.read_f64()?,
                    }],
                    "double",
                ),
                _ => Err(Error::msg(format!(
                    "unsupported complex scalar width at {path}"
                ))),
            }
        }
    }

    fn write_matrix_extension(
        &mut self,
        parent: &Group,
        name: &str,
        reader: &mut Reader<'_>,
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
        let ctx = MatrixWriteContext {
            parent,
            name,
            path,
            layout,
            extents: &extents,
        };
        match parse_type(header) {
            TYPE_TYPED_ARRAY => self.write_matrix_typed_array(reader, header, ctx),
            TYPE_EXTENSION if parse_extension_id(header) == EXT_COMPLEX => self
                .write_matrix_complex(
                    ctx.parent,
                    ctx.name,
                    reader,
                    ctx.path,
                    ctx.layout,
                    ctx.extents,
                ),
            _ => Err(Error::msg(format!("unsupported matrix payload at {path}"))),
        }
    }

    fn write_matrix_typed_array(
        &mut self,
        reader: &mut Reader<'_>,
        header: u8,
        ctx: MatrixWriteContext<'_>,
    ) -> Result<()> {
        let info = reader.read_typed_array_header(header)?;
        let expected_len = product(ctx.extents)?;
        if info.len != expected_len {
            return Err(Error::msg(format!(
                "matrix payload length {} does not match extents product {} at {}",
                info.len, expected_len, ctx.path
            )));
        }
        match info.class {
            TypedArrayClass::Float => self.write_matrix_float(reader, info.byte_code, ctx),
            TypedArrayClass::Signed => self.write_matrix_signed(reader, info.byte_code, ctx),
            TypedArrayClass::Unsigned => self.write_matrix_unsigned(reader, info.byte_code, ctx),
            TypedArrayClass::Bool => {
                self.write_matrix_bool(ctx.parent, ctx.name, reader, ctx.extents, ctx.layout)
            }
            TypedArrayClass::String => self.write_matrix_string(reader, ctx),
        }
    }

    fn write_matrix_complex(
        &mut self,
        parent: &Group,
        name: &str,
        reader: &mut Reader<'_>,
        path: &str,
        layout: MatrixLayout,
        extents: &[usize],
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
        let matlab_dims = matrix_dims(extents, self.options.one_dimensional_mode);
        match info.byte_code {
            2 => {
                let mut data = Vec::with_capacity(len);
                for _ in 0..len {
                    data.push(MatComplex32 {
                        real: reader.read_f32()?,
                        imag: reader.read_f32()?,
                    });
                }
                let data = self.maybe_reorder(data, layout, extents, path)?;
                self.write_complex(parent, name, &matlab_dims, &data, "single")
            }
            3 => {
                let mut data = Vec::with_capacity(len);
                for _ in 0..len {
                    data.push(MatComplex64 {
                        real: reader.read_f64()?,
                        imag: reader.read_f64()?,
                    });
                }
                let data = self.maybe_reorder(data, layout, extents, path)?;
                self.write_complex(parent, name, &matlab_dims, &data, "double")
            }
            _ => Err(Error::msg(format!(
                "unsupported complex matrix element width at {path}"
            ))),
        }
    }

    fn write_float_array(
        &mut self,
        parent: &Group,
        name: &str,
        reader: &mut Reader<'_>,
        byte_code: u8,
        len: usize,
        path: &str,
    ) -> Result<()> {
        let matlab_dims = vector_dims(len, self.options.one_dimensional_mode);
        match byte_code {
            0 => match self.options.unsupported_policy {
                UnsupportedPolicy::LossyNumericWidening => {
                    let mut values = Vec::with_capacity(len);
                    for _ in 0..len {
                        values.push(reader.read_bf16_as_f32()?);
                    }
                    if len == 0 {
                        return self.write_empty_marker(parent, name, &matlab_dims, "single", None);
                    }
                    self.write_numeric(parent, name, &matlab_dims, &values, "single")
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
                        return self.write_empty_marker(parent, name, &matlab_dims, "single", None);
                    }
                    self.write_numeric(parent, name, &matlab_dims, &values, "single")
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
                    return self.write_empty_marker(parent, name, &matlab_dims, "single", None);
                }
                self.write_numeric(parent, name, &matlab_dims, &values, "single")
            }
            3 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_f64()?);
                }
                if len == 0 {
                    return self.write_empty_marker(parent, name, &matlab_dims, "double", None);
                }
                self.write_numeric(parent, name, &matlab_dims, &values, "double")
            }
            _ => Err(Error::Unsupported("float128 arrays are not supported")),
        }
    }

    fn write_signed_array(
        &mut self,
        parent: &Group,
        name: &str,
        reader: &mut Reader<'_>,
        byte_code: u8,
        len: usize,
        path: &str,
    ) -> Result<()> {
        let matlab_dims = vector_dims(len, self.options.one_dimensional_mode);
        match byte_code {
            0 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_signed(0)? as i8);
                }
                if len == 0 {
                    return self.write_empty_marker(parent, name, &matlab_dims, "int8", None);
                }
                self.write_numeric(parent, name, &matlab_dims, &values, "int8")
            }
            1 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_signed(1)? as i16);
                }
                if len == 0 {
                    return self.write_empty_marker(parent, name, &matlab_dims, "int16", None);
                }
                self.write_numeric(parent, name, &matlab_dims, &values, "int16")
            }
            2 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_signed(2)? as i32);
                }
                if len == 0 {
                    return self.write_empty_marker(parent, name, &matlab_dims, "int32", None);
                }
                self.write_numeric(parent, name, &matlab_dims, &values, "int32")
            }
            3 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_signed(3)? as i64);
                }
                if len == 0 {
                    return self.write_empty_marker(parent, name, &matlab_dims, "int64", None);
                }
                self.write_numeric(parent, name, &matlab_dims, &values, "int64")
            }
            4 => Err(Error::msg(format!("unsupported i128 array at {path}"))),
            _ => Err(Error::InvalidSize),
        }
    }

    fn write_unsigned_array(
        &mut self,
        parent: &Group,
        name: &str,
        reader: &mut Reader<'_>,
        byte_code: u8,
        len: usize,
        path: &str,
    ) -> Result<()> {
        let matlab_dims = vector_dims(len, self.options.one_dimensional_mode);
        match byte_code {
            0 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_unsigned(0)? as u8);
                }
                if len == 0 {
                    return self.write_empty_marker(parent, name, &matlab_dims, "uint8", None);
                }
                self.write_numeric(parent, name, &matlab_dims, &values, "uint8")
            }
            1 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_unsigned(1)? as u16);
                }
                if len == 0 {
                    return self.write_empty_marker(parent, name, &matlab_dims, "uint16", None);
                }
                self.write_numeric(parent, name, &matlab_dims, &values, "uint16")
            }
            2 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_unsigned(2)? as u32);
                }
                if len == 0 {
                    return self.write_empty_marker(parent, name, &matlab_dims, "uint32", None);
                }
                self.write_numeric(parent, name, &matlab_dims, &values, "uint32")
            }
            3 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_unsigned(3)? as u64);
                }
                if len == 0 {
                    return self.write_empty_marker(parent, name, &matlab_dims, "uint64", None);
                }
                self.write_numeric(parent, name, &matlab_dims, &values, "uint64")
            }
            4 => Err(Error::msg(format!("unsupported u128 array at {path}"))),
            _ => Err(Error::InvalidSize),
        }
    }

    fn write_bool_array(
        &mut self,
        parent: &Group,
        name: &str,
        reader: &mut Reader<'_>,
        len: usize,
    ) -> Result<()> {
        let matlab_dims = vector_dims(len, self.options.one_dimensional_mode);
        if len == 0 {
            return self.write_empty_marker(parent, name, &matlab_dims, "logical", Some(1));
        }
        let packed = reader.read_exact(len.div_ceil(8))?;
        let unpacked = unpack_bools(packed, len);
        self.write_logical(parent, name, &matlab_dims, &unpacked)
    }

    fn write_string_array(
        &mut self,
        parent: &Group,
        name: &str,
        reader: &mut Reader<'_>,
        len: usize,
    ) -> Result<()> {
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push(reader.read_string()?.to_owned());
        }
        let matlab_dims = vector_dims(len, self.options.one_dimensional_mode);
        self.write_string_object(parent, name, &values, &matlab_dims)
    }

    fn write_matrix_string(
        &mut self,
        reader: &mut Reader<'_>,
        ctx: MatrixWriteContext<'_>,
    ) -> Result<()> {
        let len = product(ctx.extents)?;
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push(reader.read_string()?.to_owned());
        }
        let values = self.maybe_reorder(values, ctx.layout, ctx.extents, ctx.path)?;
        let matlab_dims = matrix_dims(ctx.extents, self.options.one_dimensional_mode);
        self.write_string_object(ctx.parent, ctx.name, &values, &matlab_dims)
    }

    fn write_matrix_float(
        &mut self,
        reader: &mut Reader<'_>,
        byte_code: u8,
        ctx: MatrixWriteContext<'_>,
    ) -> Result<()> {
        let len = product(ctx.extents)?;
        let matlab_dims = matrix_dims(ctx.extents, self.options.one_dimensional_mode);
        match byte_code {
            0 => match self.options.unsupported_policy {
                UnsupportedPolicy::LossyNumericWidening => {
                    let mut values = Vec::with_capacity(len);
                    for _ in 0..len {
                        values.push(reader.read_bf16_as_f32()?);
                    }
                    let values =
                        self.maybe_reorder(values, ctx.layout, ctx.extents, ctx.path)?;
                    self.write_numeric(ctx.parent, ctx.name, &matlab_dims, &values, "single")
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
                    let values =
                        self.maybe_reorder(values, ctx.layout, ctx.extents, ctx.path)?;
                    self.write_numeric(ctx.parent, ctx.name, &matlab_dims, &values, "single")
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
                self.write_numeric(ctx.parent, ctx.name, &matlab_dims, &values, "single")
            }
            3 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_f64()?);
                }
                let values = self.maybe_reorder(values, ctx.layout, ctx.extents, ctx.path)?;
                self.write_numeric(ctx.parent, ctx.name, &matlab_dims, &values, "double")
            }
            _ => Err(Error::msg(format!(
                "unsupported float matrix width at {}",
                ctx.path
            ))),
        }
    }

    fn write_matrix_signed(
        &mut self,
        reader: &mut Reader<'_>,
        byte_code: u8,
        ctx: MatrixWriteContext<'_>,
    ) -> Result<()> {
        let len = product(ctx.extents)?;
        let matlab_dims = matrix_dims(ctx.extents, self.options.one_dimensional_mode);
        match byte_code {
            0 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_signed(0)? as i8);
                }
                let values = self.maybe_reorder(values, ctx.layout, ctx.extents, ctx.path)?;
                self.write_numeric(ctx.parent, ctx.name, &matlab_dims, &values, "int8")
            }
            1 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_signed(1)? as i16);
                }
                let values = self.maybe_reorder(values, ctx.layout, ctx.extents, ctx.path)?;
                self.write_numeric(ctx.parent, ctx.name, &matlab_dims, &values, "int16")
            }
            2 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_signed(2)? as i32);
                }
                let values = self.maybe_reorder(values, ctx.layout, ctx.extents, ctx.path)?;
                self.write_numeric(ctx.parent, ctx.name, &matlab_dims, &values, "int32")
            }
            3 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_signed(3)? as i64);
                }
                let values = self.maybe_reorder(values, ctx.layout, ctx.extents, ctx.path)?;
                self.write_numeric(ctx.parent, ctx.name, &matlab_dims, &values, "int64")
            }
            _ => Err(Error::msg(format!(
                "unsupported i128 matrix at {}",
                ctx.path
            ))),
        }
    }

    fn write_matrix_unsigned(
        &mut self,
        reader: &mut Reader<'_>,
        byte_code: u8,
        ctx: MatrixWriteContext<'_>,
    ) -> Result<()> {
        let len = product(ctx.extents)?;
        let matlab_dims = matrix_dims(ctx.extents, self.options.one_dimensional_mode);
        match byte_code {
            0 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_unsigned(0)? as u8);
                }
                let values = self.maybe_reorder(values, ctx.layout, ctx.extents, ctx.path)?;
                self.write_numeric(ctx.parent, ctx.name, &matlab_dims, &values, "uint8")
            }
            1 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_unsigned(1)? as u16);
                }
                let values = self.maybe_reorder(values, ctx.layout, ctx.extents, ctx.path)?;
                self.write_numeric(ctx.parent, ctx.name, &matlab_dims, &values, "uint16")
            }
            2 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_unsigned(2)? as u32);
                }
                let values = self.maybe_reorder(values, ctx.layout, ctx.extents, ctx.path)?;
                self.write_numeric(ctx.parent, ctx.name, &matlab_dims, &values, "uint32")
            }
            3 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_unsigned(3)? as u64);
                }
                let values = self.maybe_reorder(values, ctx.layout, ctx.extents, ctx.path)?;
                self.write_numeric(ctx.parent, ctx.name, &matlab_dims, &values, "uint64")
            }
            _ => Err(Error::msg(format!(
                "unsupported u128 matrix at {}",
                ctx.path
            ))),
        }
    }

    fn write_matrix_bool(
        &mut self,
        parent: &Group,
        name: &str,
        reader: &mut Reader<'_>,
        extents: &[usize],
        layout: MatrixLayout,
    ) -> Result<()> {
        let len = product(extents)?;
        let matlab_dims = matrix_dims(extents, self.options.one_dimensional_mode);
        let packed = reader.read_exact(len.div_ceil(8))?;
        let unpacked = unpack_bools(packed, len);
        let values = self.maybe_reorder(unpacked, layout, extents, "$matrix")?;
        self.write_logical(parent, name, &matlab_dims, &values)
    }

    fn write_value_reference(
        &mut self,
        reader: &mut Reader<'_>,
        path: &str,
    ) -> Result<ObjectReference1> {
        let ref_name = self.next_ref_name();
        // Clone the ref-counted HDF5 handle to avoid borrowing self immutably and mutably at once.
        let refs = self.refs.clone();
        self.write_named_value(&refs, &ref_name, reader, path)?;
        self.file
            .reference(&format!("/#refs#/{ref_name}"))
            .map_err(Into::into)
    }

    fn handle_128_bit_scalar(
        &mut self,
        parent: &Group,
        name: &str,
        path: &str,
        value: String,
    ) -> Result<()> {
        match self.options.unsupported_policy {
            UnsupportedPolicy::StringFallback => self.write_string_scalar(parent, name, &value),
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

    fn write_numeric<T: H5Type>(
        &self,
        parent: &Group,
        name: &str,
        matlab_dims: &[usize],
        data: &[T],
        class: &str,
    ) -> Result<()> {
        if data.is_empty() {
            return self.write_empty_marker(parent, name, matlab_dims, class, None);
        }
        let storage_dims = storage_dims(matlab_dims);
        let ds = self.create_dataset::<T>(parent, name, &storage_dims)?;
        ds.write_raw(data)?;
        write_ascii_attr(&ds, "MATLAB_class", class)
    }

    fn write_complex<T: H5Type>(
        &self,
        parent: &Group,
        name: &str,
        matlab_dims: &[usize],
        data: &[T],
        class: &str,
    ) -> Result<()> {
        let storage_dims = storage_dims(matlab_dims);
        let ds = self.create_dataset::<T>(parent, name, &storage_dims)?;
        if !data.is_empty() {
            ds.write_raw(data)?;
        }
        write_ascii_attr(&ds, "MATLAB_class", class)
    }

    fn write_logical(
        &self,
        parent: &Group,
        name: &str,
        matlab_dims: &[usize],
        data: &[u8],
    ) -> Result<()> {
        if data.is_empty() {
            return self.write_empty_marker(parent, name, matlab_dims, "logical", Some(1));
        }
        let storage_dims = storage_dims(matlab_dims);
        let ds = self.create_dataset::<u8>(parent, name, &storage_dims)?;
        ds.write_raw(data)?;
        write_ascii_attr(&ds, "MATLAB_class", "logical")?;
        write_i32_attr(&ds, "MATLAB_int_decode", 1)
    }

    fn write_string_scalar(&mut self, parent: &Group, name: &str, value: &str) -> Result<()> {
        self.write_string_object(parent, name, &[value.to_owned()], &[1, 1])
    }

    fn write_string_object(
        &mut self,
        parent: &Group,
        name: &str,
        values: &[String],
        matlab_dims: &[usize],
    ) -> Result<()> {
        let payload_ref = self.write_string_saveobj_payload(values, matlab_dims)?;
        let object_id = self.string_objects.register(payload_ref);
        let metadata = create_string_object_metadata(object_id);
        let ds = self.create_row_vector_dataset::<u32>(parent, name, &metadata)?;
        write_ascii_attr(&ds, "MATLAB_class", MATLAB_CLASS_STRING)?;
        write_i32_attr(&ds, "MATLAB_object_decode", MATLAB_OBJECT_DECODE_OPAQUE)
    }

    fn write_string_saveobj_payload(
        &mut self,
        values: &[String],
        matlab_dims: &[usize],
    ) -> Result<ObjectReference1> {
        let payload = encode_string_saveobj_payload(values, matlab_dims)?;
        let ref_name = self.next_ref_name();
        self.write_numeric(
            &self.refs,
            &ref_name,
            &[1, payload.len()],
            &payload,
            "uint64",
        )?;
        self.file
            .reference::<ObjectReference1>(&format!("/#refs#/{ref_name}"))
            .map_err(Into::into)
    }

    fn write_string_subsystem(&mut self) -> Result<()> {
        let subsystem = self.file.create_group("#subsystem#")?;

        let metadata_ref = self.write_string_subsystem_metadata()?;
        let canonical_empty_ref = self.write_string_canonical_empty_ref()?;
        let unknown_template_a = self.write_string_unknown_template_ref()?;
        let alias_ref = self.write_string_alias_metadata_ref()?;
        let unknown_template_b = self.write_string_unknown_template_ref()?;

        let mut refs = Vec::with_capacity(self.string_objects.payload_refs.len() + 5);
        refs.push(metadata_ref);
        refs.push(canonical_empty_ref);
        refs.extend(self.string_objects.payload_refs.iter().copied());
        refs.push(unknown_template_a);
        refs.push(alias_ref);
        refs.push(unknown_template_b);

        let mcos = self.create_dataset::<ObjectReference1>(&subsystem, "MCOS", &[1, refs.len()])?;
        mcos.write_raw(&refs)?;
        write_ascii_attr(&mcos, "MATLAB_class", MATLAB_CLASS_FILEWRAPPER)?;
        write_i32_attr(&mcos, "MATLAB_object_decode", MATLAB_OBJECT_DECODE_OPAQUE)
    }

    fn write_string_subsystem_metadata(&mut self) -> Result<ObjectReference1> {
        let data = build_string_filewrapper_metadata(self.string_objects.payload_refs.len());
        let ref_name = self.next_ref_name();
        let ds = self.create_row_vector_dataset::<u8>(&self.refs, &ref_name, &data)?;
        write_ascii_attr(&ds, "MATLAB_class", "uint8")?;
        self.file
            .reference::<ObjectReference1>(&format!("/#refs#/{ref_name}"))
            .map_err(Into::into)
    }

    fn write_string_canonical_empty_ref(&mut self) -> Result<ObjectReference1> {
        let ref_name = self.next_ref_name();
        self.write_empty_marker(&self.refs, &ref_name, &[0, 0], "canonical empty", None)?;
        self.file
            .reference::<ObjectReference1>(&format!("/#refs#/{ref_name}"))
            .map_err(Into::into)
    }

    fn write_string_alias_metadata_ref(&mut self) -> Result<ObjectReference1> {
        let ref_name = self.next_ref_name();
        let aliases = [0i32, 0];
        let ds = self.create_row_vector_dataset::<i32>(&self.refs, &ref_name, &aliases)?;
        write_ascii_attr(&ds, "MATLAB_class", "int32")?;
        self.file
            .reference::<ObjectReference1>(&format!("/#refs#/{ref_name}"))
            .map_err(Into::into)
    }

    fn write_string_unknown_template_ref(&mut self) -> Result<ObjectReference1> {
        let refs = [
            self.write_string_empty_struct_ref()?,
            self.write_string_empty_struct_ref()?,
        ];
        let ref_name = self.next_ref_name();
        self.write_reference_array(&self.refs, &ref_name, &[2, 1], &refs, "cell")?;
        self.file
            .reference::<ObjectReference1>(&format!("/#refs#/{ref_name}"))
            .map_err(Into::into)
    }

    fn write_string_empty_struct_ref(&mut self) -> Result<ObjectReference1> {
        let ref_name = self.next_ref_name();
        self.write_empty_marker(&self.refs, &ref_name, &[1, 0], "struct", None)?;
        self.file
            .reference::<ObjectReference1>(&format!("/#refs#/{ref_name}"))
            .map_err(Into::into)
    }

    fn write_reference_array(
        &self,
        parent: &Group,
        name: &str,
        matlab_dims: &[usize],
        refs: &[ObjectReference1],
        class: &str,
    ) -> Result<()> {
        if refs.is_empty() {
            return self.write_empty_marker(parent, name, matlab_dims, class, None);
        }
        let storage_dims = storage_dims(matlab_dims);
        let ds = self.create_dataset::<ObjectReference1>(parent, name, &storage_dims)?;
        ds.write_raw(refs)?;
        write_ascii_attr(&ds, "MATLAB_class", class)
    }

    fn write_empty_marker(
        &self,
        parent: &Group,
        name: &str,
        matlab_dims: &[usize],
        class: &str,
        int_decode: Option<u8>,
    ) -> Result<()> {
        let shape_data: Vec<u64> = matlab_dims.iter().map(|&dim| dim as u64).collect();
        let ds = self.create_dataset::<u64>(parent, name, &[shape_data.len()])?;
        if !shape_data.is_empty() {
            ds.write_raw(&shape_data)?;
        }
        write_ascii_attr(&ds, "MATLAB_class", class)?;
        write_u32_attr(&ds, "MATLAB_empty", 1)?;
        if let Some(code) = int_decode {
            write_i32_attr(&ds, "MATLAB_int_decode", i32::from(code))?;
        }
        Ok(())
    }

    fn write_empty_struct_array(&self, parent: &Group, name: &str) -> Result<()> {
        self.write_empty_marker(parent, name, &[0, 0], "struct", None)
    }

    fn create_dataset<T: H5Type>(
        &self,
        parent: &Group,
        name: &str,
        shape: &[usize],
    ) -> Result<Dataset> {
        let builder = configure_builder(&self.options.compression, parent.new_dataset_builder());
        builder
            .empty::<T>()
            .shape(shape.to_vec())
            .create(name)
            .map_err(Into::into)
    }

    fn create_row_vector_dataset<T: H5Type>(
        &self,
        parent: &Group,
        name: &str,
        data: &[T],
    ) -> Result<Dataset> {
        let ds = self.create_dataset::<T>(parent, name, &[1, data.len()])?;
        if !data.is_empty() {
            ds.write_raw(data)?;
        }
        Ok(ds)
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
}

fn configure_builder(compression: &Compression, builder: DatasetBuilder) -> DatasetBuilder {
    match compression {
        Compression::None => builder.no_chunk(),
        Compression::Deflate { level, shuffle } => {
            let builder = if *shuffle { builder.shuffle() } else { builder };
            builder.deflate(*level)
        }
    }
}

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

fn write_ascii_attr(target: &hdf5::Location, name: &str, value: &str) -> Result<()> {
    // FixedAscii<N> needs a compile-time width, so this dispatch is the simplest
    // way to keep MATLAB metadata in fixed-width ASCII attributes.
    match value.len() {
        0 => write_fixed_ascii_attr::<1>(target, name, value),
        1 => write_fixed_ascii_attr::<1>(target, name, value),
        2 => write_fixed_ascii_attr::<2>(target, name, value),
        3 => write_fixed_ascii_attr::<3>(target, name, value),
        4 => write_fixed_ascii_attr::<4>(target, name, value),
        5 => write_fixed_ascii_attr::<5>(target, name, value),
        6 => write_fixed_ascii_attr::<6>(target, name, value),
        7 => write_fixed_ascii_attr::<7>(target, name, value),
        8 => write_fixed_ascii_attr::<8>(target, name, value),
        9 => write_fixed_ascii_attr::<9>(target, name, value),
        10 => write_fixed_ascii_attr::<10>(target, name, value),
        11 => write_fixed_ascii_attr::<11>(target, name, value),
        12 => write_fixed_ascii_attr::<12>(target, name, value),
        13 => write_fixed_ascii_attr::<13>(target, name, value),
        14 => write_fixed_ascii_attr::<14>(target, name, value),
        15 => write_fixed_ascii_attr::<15>(target, name, value),
        16 => write_fixed_ascii_attr::<16>(target, name, value),
        len => Err(Error::msg(format!(
            "ASCII attribute `{name}` exceeds supported fixed width: {len}"
        ))),
    }
}

fn write_fixed_ascii_attr<const N: usize>(
    target: &hdf5::Location,
    name: &str,
    value: &str,
) -> Result<()> {
    let ascii =
        FixedAscii::<N>::from_ascii(value.as_bytes()).map_err(|e| Error::msg(e.to_string()))?;
    target
        .new_attr::<FixedAscii<N>>()
        .create(name)?
        .write_scalar(&ascii)?;
    Ok(())
}

fn write_ascii_list_attr(target: &hdf5::Location, name: &str, values: &[String]) -> Result<()> {
    let ascii: Vec<VarLenArray<FixedAscii<1>>> = values
        .iter()
        .map(|value| {
            let chars = value
                .bytes()
                .map(|byte| {
                    FixedAscii::<1>::from_ascii(&[byte]).map_err(|e| Error::msg(e.to_string()))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(VarLenArray::from_slice(&chars))
        })
        .collect::<Result<_>>()?;
    target.new_attr_builder().with_data(&ascii).create(name)?;
    Ok(())
}

fn write_i32_attr(target: &hdf5::Location, name: &str, value: i32) -> Result<()> {
    target
        .new_attr::<i32>()
        .create(name)?
        .write_scalar(&value)?;
    Ok(())
}

fn write_u32_attr(target: &hdf5::Location, name: &str, value: u32) -> Result<()> {
    target
        .new_attr::<u32>()
        .create(name)?
        .write_scalar(&value)?;
    Ok(())
}

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
            // Names are sanitized/validated to ASCII, so byte truncation is safe here.
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

fn write_userblock(path: &Path) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .open(path)
        .map_err(|e| Error::msg(e.to_string()))?;
    file.seek(SeekFrom::Start(0))
        .map_err(|e| Error::msg(e.to_string()))?;

    let mut block = [0u8; MATLAB_USERBLOCK_SIZE as usize];
    let text = matlab_header_text();
    block[..116].copy_from_slice(&text);
    block[124] = 0;
    block[125] = 2;
    block[126] = b'I';
    block[127] = b'M';
    file.write_all(&block)
        .map_err(|e| Error::msg(e.to_string()))
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
