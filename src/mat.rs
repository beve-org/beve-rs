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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RootBinding<'a> {
    NamedVariable(&'a str),
    WorkspaceObject,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Compression {
    None,
    Deflate { level: u8, shuffle: bool },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InvalidNamePolicy {
    Error,
    Sanitize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnsupportedPolicy {
    Error,
    StringFallback,
    LossyNumericWidening,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NullPolicy {
    Error,
    EmptyStructArray,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OneDimensionalMode {
    ColumnVector,
    RowVector,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RowMajorPolicy {
    ReorderToColumnMajor,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatV73Options {
    pub compression: Compression,
    pub invalid_name_policy: InvalidNamePolicy,
    pub null_policy: NullPolicy,
    pub unsupported_policy: UnsupportedPolicy,
    pub one_dimensional_mode: OneDimensionalMode,
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
    };
    let mut reader = Reader::new(beve);
    writer.write_root(&mut reader, root)?;
    if !reader.is_finished() {
        return Err(Error::InvalidType("unexpected trailing BEVE data"));
    }
    writer.file.flush()?;
    drop(writer);
    write_userblock(temp_output.path())?;
    sync_regular_file(temp_output.path())?;
    temp_output.persist(output_path)
}

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
    fn write_root(&mut self, reader: &mut Reader<'_>, root: RootBinding<'_>) -> Result<()> {
        match root {
            RootBinding::NamedVariable(name) => {
                let variable = self.normalize_name(name, &mut BTreeSet::new())?;
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
                let value = reader.read_string()?;
                self.write_char(parent, name, value)
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
            TypedArrayClass::String => self.write_string_cell_array(parent, name, reader, info.len),
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
        let header = reader.read_header()?;
        match parse_type(header) {
            TYPE_TYPED_ARRAY => {
                self.write_matrix_typed_array(parent, name, reader, header, path, layout, &extents)
            }
            TYPE_EXTENSION if parse_extension_id(header) == EXT_COMPLEX => {
                self.write_matrix_complex(parent, name, reader, path, layout, &extents)
            }
            _ => Err(Error::msg(format!("unsupported matrix payload at {path}"))),
        }
    }

    fn write_matrix_typed_array(
        &mut self,
        parent: &Group,
        name: &str,
        reader: &mut Reader<'_>,
        header: u8,
        path: &str,
        layout: MatrixLayout,
        extents: &[usize],
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
            TypedArrayClass::Float => {
                self.write_matrix_float(parent, name, reader, extents, layout, info.byte_code, path)
            }
            TypedArrayClass::Signed => self.write_matrix_signed(
                parent,
                name,
                reader,
                extents,
                layout,
                info.byte_code,
                path,
            ),
            TypedArrayClass::Unsigned => self.write_matrix_unsigned(
                parent,
                name,
                reader,
                extents,
                layout,
                info.byte_code,
                path,
            ),
            TypedArrayClass::Bool => self.write_matrix_bool(parent, name, reader, extents, layout),
            TypedArrayClass::String => Err(Error::msg(format!(
                "string matrices are not supported at {path}"
            ))),
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

    fn write_string_cell_array(
        &mut self,
        parent: &Group,
        name: &str,
        reader: &mut Reader<'_>,
        len: usize,
    ) -> Result<()> {
        let matlab_dims = vector_dims(len, self.options.one_dimensional_mode);
        if len == 0 {
            return self.write_empty_marker(parent, name, &matlab_dims, "cell", None);
        }
        let mut refs = Vec::with_capacity(len);
        for _ in 0..len {
            let value = reader.read_string()?;
            let ref_name = self.next_ref_name();
            self.write_char(&self.refs, &ref_name, value)?;
            refs.push(
                self.file
                    .reference::<ObjectReference1>(&format!("/#refs#/{ref_name}"))?,
            );
        }
        self.write_reference_array(parent, name, &matlab_dims, &refs, "cell")
    }

    fn write_matrix_float(
        &mut self,
        parent: &Group,
        name: &str,
        reader: &mut Reader<'_>,
        extents: &[usize],
        layout: MatrixLayout,
        byte_code: u8,
        path: &str,
    ) -> Result<()> {
        let len = product(extents)?;
        let matlab_dims = matrix_dims(extents, self.options.one_dimensional_mode);
        match byte_code {
            0 => match self.options.unsupported_policy {
                UnsupportedPolicy::LossyNumericWidening => {
                    let mut values = Vec::with_capacity(len);
                    for _ in 0..len {
                        values.push(reader.read_bf16_as_f32()?);
                    }
                    let values = self.maybe_reorder(values, layout, extents, path)?;
                    self.write_numeric(parent, name, &matlab_dims, &values, "single")
                }
                _ => Err(Error::msg(format!(
                    "unsupported bf16 matrix at {path}; enable LossyNumericWidening to map it to MATLAB single"
                ))),
            },
            1 => match self.options.unsupported_policy {
                UnsupportedPolicy::LossyNumericWidening => {
                    let mut values = Vec::with_capacity(len);
                    for _ in 0..len {
                        values.push(reader.read_f16_as_f32()?);
                    }
                    let values = self.maybe_reorder(values, layout, extents, path)?;
                    self.write_numeric(parent, name, &matlab_dims, &values, "single")
                }
                _ => Err(Error::msg(format!(
                    "unsupported f16 matrix at {path}; enable LossyNumericWidening to map it to MATLAB single"
                ))),
            },
            2 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_f32()?);
                }
                let values = self.maybe_reorder(values, layout, extents, path)?;
                self.write_numeric(parent, name, &matlab_dims, &values, "single")
            }
            3 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_f64()?);
                }
                let values = self.maybe_reorder(values, layout, extents, path)?;
                self.write_numeric(parent, name, &matlab_dims, &values, "double")
            }
            _ => Err(Error::msg(format!(
                "unsupported float matrix width at {path}"
            ))),
        }
    }

    fn write_matrix_signed(
        &mut self,
        parent: &Group,
        name: &str,
        reader: &mut Reader<'_>,
        extents: &[usize],
        layout: MatrixLayout,
        byte_code: u8,
        path: &str,
    ) -> Result<()> {
        let len = product(extents)?;
        let matlab_dims = matrix_dims(extents, self.options.one_dimensional_mode);
        match byte_code {
            0 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_signed(0)? as i8);
                }
                let values = self.maybe_reorder(values, layout, extents, path)?;
                self.write_numeric(parent, name, &matlab_dims, &values, "int8")
            }
            1 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_signed(1)? as i16);
                }
                let values = self.maybe_reorder(values, layout, extents, path)?;
                self.write_numeric(parent, name, &matlab_dims, &values, "int16")
            }
            2 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_signed(2)? as i32);
                }
                let values = self.maybe_reorder(values, layout, extents, path)?;
                self.write_numeric(parent, name, &matlab_dims, &values, "int32")
            }
            3 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_signed(3)? as i64);
                }
                let values = self.maybe_reorder(values, layout, extents, path)?;
                self.write_numeric(parent, name, &matlab_dims, &values, "int64")
            }
            _ => Err(Error::msg(format!("unsupported i128 matrix at {path}"))),
        }
    }

    fn write_matrix_unsigned(
        &mut self,
        parent: &Group,
        name: &str,
        reader: &mut Reader<'_>,
        extents: &[usize],
        layout: MatrixLayout,
        byte_code: u8,
        path: &str,
    ) -> Result<()> {
        let len = product(extents)?;
        let matlab_dims = matrix_dims(extents, self.options.one_dimensional_mode);
        match byte_code {
            0 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_unsigned(0)? as u8);
                }
                let values = self.maybe_reorder(values, layout, extents, path)?;
                self.write_numeric(parent, name, &matlab_dims, &values, "uint8")
            }
            1 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_unsigned(1)? as u16);
                }
                let values = self.maybe_reorder(values, layout, extents, path)?;
                self.write_numeric(parent, name, &matlab_dims, &values, "uint16")
            }
            2 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_unsigned(2)? as u32);
                }
                let values = self.maybe_reorder(values, layout, extents, path)?;
                self.write_numeric(parent, name, &matlab_dims, &values, "uint32")
            }
            3 => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(reader.read_unsigned(3)? as u64);
                }
                let values = self.maybe_reorder(values, layout, extents, path)?;
                self.write_numeric(parent, name, &matlab_dims, &values, "uint64")
            }
            _ => Err(Error::msg(format!("unsupported u128 matrix at {path}"))),
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
            UnsupportedPolicy::StringFallback => self.write_char(parent, name, &value),
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

    fn write_char(&self, parent: &Group, name: &str, value: &str) -> Result<()> {
        let code_units: Vec<u16> = value.encode_utf16().collect();
        let matlab_dims = vec![1, code_units.len()];
        if code_units.is_empty() {
            return self.write_empty_marker(parent, name, &matlab_dims, "char", Some(2));
        }
        let storage_dims = storage_dims(&matlab_dims);
        let ds = self.create_dataset::<u16>(parent, name, &storage_dims)?;
        ds.write_raw(&code_units)?;
        write_ascii_attr(&ds, "MATLAB_class", "char")?;
        write_i32_attr(&ds, "MATLAB_int_decode", 2)
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
