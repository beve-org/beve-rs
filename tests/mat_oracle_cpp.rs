#![cfg(feature = "mat")]
#![deny(warnings)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use beve::{
    Complex, EnumEncoding, InvalidNamePolicy, Key, MatV73Options, MatrixLayout, MatrixOwned,
    NullPolicy, Object, OneDimensionalMode, RootBinding, RowMajorPolicy, SerializerOptions,
    UnsupportedPolicy, Value,
};
use half::{bf16, f16};
use serde::{Deserialize, Serialize};

fn matio_oracle_bin() -> Option<String> {
    std::env::var("MATIO_ORACLE_BIN").ok()
}

fn oracle_test_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn lock_oracle_test() -> MutexGuard<'static, ()> {
    oracle_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

struct TempMatFile(PathBuf);

impl TempMatFile {
    fn new(test_name: &str) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        Self(std::env::temp_dir().join(format!("beve-mat-cpp-oracle-{test_name}-{nanos}.mat")))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempMatFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn write_mat<T: Serialize>(
    test_name: &str,
    value: &T,
    root: RootBinding<'_>,
    options: &MatV73Options,
) -> TempMatFile {
    let path = TempMatFile::new(test_name);
    let bytes = beve::to_vec(value).unwrap();
    beve::beve_slice_to_mat_v73_file(&bytes, path.path(), root, options).unwrap();
    path
}

fn write_mat_err<T: Serialize>(
    test_name: &str,
    value: &T,
    root: RootBinding<'_>,
    options: &MatV73Options,
) -> String {
    let path = TempMatFile::new(test_name);
    let bytes = beve::to_vec(value).unwrap();
    beve::beve_slice_to_mat_v73_file(&bytes, path.path(), root, options)
        .unwrap_err()
        .to_string()
}

fn write_mat_bytes(
    test_name: &str,
    bytes: &[u8],
    root: RootBinding<'_>,
    options: &MatV73Options,
) -> TempMatFile {
    let path = TempMatFile::new(test_name);
    beve::beve_slice_to_mat_v73_file(bytes, path.path(), root, options).unwrap();
    path
}

fn write_mat_bytes_err(
    test_name: &str,
    bytes: &[u8],
    root: RootBinding<'_>,
    options: &MatV73Options,
) -> String {
    let path = TempMatFile::new(test_name);
    beve::beve_slice_to_mat_v73_file(bytes, path.path(), root, options)
        .unwrap_err()
        .to_string()
}

fn run_matio_oracle(path: &Path) -> Option<OracleDump> {
    let bin = matio_oracle_bin()?;
    let output = Command::new(bin).arg("dump").arg(path).output().unwrap();
    if !output.status.success() {
        panic!(
            "matio oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let stdout = String::from_utf8(output.stdout).unwrap();
    Some(
        serde_json::from_str(&stdout).unwrap_or_else(|err| {
            panic!("failed to parse MATIO oracle JSON: {err}; json={stdout}")
        }),
    )
}

#[derive(Debug, Deserialize, PartialEq)]
struct OracleDump {
    variables: BTreeMap<String, OracleValue>,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum OracleValue {
    Char {
        dims: Vec<usize>,
        value: String,
    },
    Single {
        dims: Vec<usize>,
        data: Vec<f32>,
    },
    Logical {
        dims: Vec<usize>,
        data: Vec<bool>,
    },
    Int8 {
        dims: Vec<usize>,
        data: Vec<i8>,
    },
    Int16 {
        dims: Vec<usize>,
        data: Vec<i16>,
    },
    Int32 {
        dims: Vec<usize>,
        data: Vec<i32>,
    },
    Int64 {
        dims: Vec<usize>,
        data: Vec<i64>,
    },
    Double {
        dims: Vec<usize>,
        data: Vec<f64>,
    },
    Uint8 {
        dims: Vec<usize>,
        data: Vec<u8>,
    },
    Uint16 {
        dims: Vec<usize>,
        data: Vec<u16>,
    },
    Uint32 {
        dims: Vec<usize>,
        data: Vec<u32>,
    },
    Uint64 {
        dims: Vec<usize>,
        data: Vec<u64>,
    },
    ComplexSingle {
        dims: Vec<usize>,
        real: Vec<f32>,
        imag: Vec<f32>,
    },
    ComplexDouble {
        dims: Vec<usize>,
        real: Vec<f64>,
        imag: Vec<f64>,
    },
    Cell {
        dims: Vec<usize>,
        elements: Vec<OracleValue>,
    },
    Struct {
        dims: Vec<usize>,
        fields: BTreeMap<String, Vec<OracleValue>>,
    },
    Empty {
        dims: Vec<usize>,
    },
}

#[derive(Debug, Serialize)]
struct OracleStruct<'a> {
    answer: u8,
    label: &'a str,
}

#[derive(Debug, Serialize)]
enum TaggedEnum {
    Scalar(u8),
}

fn expect_var<'a>(dump: &'a OracleDump, name: &str) -> &'a OracleValue {
    dump.variables
        .get(name)
        .unwrap_or_else(|| panic!("missing variable `{name}`; dump={dump:#?}"))
}

fn expect_struct_field<'a>(value: &'a OracleValue, field: &str) -> &'a OracleValue {
    match value {
        OracleValue::Struct { fields, .. } => fields
            .get(field)
            .and_then(|values| values.first())
            .unwrap_or_else(|| panic!("missing struct field `{field}` in {value:#?}")),
        _ => panic!("expected struct, got {value:#?}"),
    }
}

#[test]
fn matio_oracle_reads_string_scalar() {
    let _guard = lock_oracle_test();
    let path = write_mat(
        "string",
        &"hello",
        RootBinding::NamedVariable("greeting"),
        &MatV73Options::default(),
    );
    let Some(dump) = run_matio_oracle(path.path()) else {
        return;
    };

    assert_eq!(
        dump.variables.get("greeting"),
        Some(&OracleValue::Char {
            dims: vec![1, 5],
            value: "hello".to_owned(),
        }),
        "dump={dump:#?}"
    );
}

#[test]
fn matio_oracle_reads_logical_array() {
    let _guard = lock_oracle_test();
    let path = write_mat(
        "logical",
        &vec![true, false, true],
        RootBinding::NamedVariable("flags"),
        &MatV73Options::default(),
    );
    let Some(dump) = run_matio_oracle(path.path()) else {
        return;
    };

    assert_eq!(
        dump.variables.get("flags"),
        Some(&OracleValue::Logical {
            dims: vec![3, 1],
            data: vec![true, false, true],
        }),
        "dump={dump:#?}"
    );
}

#[test]
fn matio_oracle_reads_cell_array() {
    let _guard = lock_oracle_test();
    let value = Value::Array(vec![Value::from(1u8), Value::from("hi")]);
    let path = write_mat(
        "cell",
        &value,
        RootBinding::NamedVariable("cells"),
        &MatV73Options::default(),
    );
    let Some(dump) = run_matio_oracle(path.path()) else {
        return;
    };

    assert_eq!(
        dump.variables.get("cells"),
        Some(&OracleValue::Cell {
            dims: vec![2, 1],
            elements: vec![
                OracleValue::Uint8 {
                    dims: vec![1, 1],
                    data: vec![1],
                },
                OracleValue::Char {
                    dims: vec![1, 2],
                    value: "hi".to_owned(),
                },
            ],
        }),
        "dump={dump:#?}"
    );
}

#[test]
fn matio_oracle_reads_struct_value() {
    let _guard = lock_oracle_test();
    let path = write_mat(
        "struct",
        &OracleStruct {
            answer: 7,
            label: "ready",
        },
        RootBinding::NamedVariable("payload"),
        &MatV73Options::default(),
    );
    let Some(dump) = run_matio_oracle(path.path()) else {
        return;
    };

    assert_eq!(
        dump.variables.get("payload"),
        Some(&OracleValue::Struct {
            dims: vec![1, 1],
            fields: BTreeMap::from([
                (
                    "answer".to_owned(),
                    vec![OracleValue::Uint8 {
                        dims: vec![1, 1],
                        data: vec![7],
                    }],
                ),
                (
                    "label".to_owned(),
                    vec![OracleValue::Char {
                        dims: vec![1, 5],
                        value: "ready".to_owned(),
                    }],
                ),
            ]),
        }),
        "dump={dump:#?}"
    );
}

#[test]
fn matio_oracle_reads_null_as_empty_struct_array() {
    let _guard = lock_oracle_test();
    let path = write_mat(
        "null",
        &Value::Null,
        RootBinding::NamedVariable("nothing"),
        &MatV73Options::default(),
    );
    let Some(dump) = run_matio_oracle(path.path()) else {
        return;
    };

    assert_eq!(
        dump.variables.get("nothing"),
        Some(&OracleValue::Struct {
            dims: vec![0, 0],
            fields: BTreeMap::new(),
        }),
        "dump={dump:#?}"
    );
}

#[test]
fn matio_oracle_reads_reordered_matrix_values() {
    let _guard = lock_oracle_test();
    let matrix = beve::MatrixOwned {
        layout: beve::MatrixLayout::Right,
        extents: vec![2, 3],
        data: vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    };
    let path = write_mat(
        "matrix",
        &matrix,
        RootBinding::NamedVariable("a"),
        &MatV73Options::default(),
    );
    let Some(dump) = run_matio_oracle(path.path()) else {
        return;
    };

    assert_eq!(
        dump.variables.get("a"),
        Some(&OracleValue::Double {
            dims: vec![2, 3],
            data: vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        }),
        "dump={dump:#?}"
    );
}

#[test]
fn matio_oracle_reads_workspace_variables() {
    let _guard = lock_oracle_test();
    let mut object = Object::new();
    object.insert(Key::from("1 bad"), Value::from(7u8));
    object.insert(Key::from("beta"), Value::from("ok"));
    let options = MatV73Options {
        invalid_name_policy: InvalidNamePolicy::Sanitize,
        ..MatV73Options::default()
    };
    let path = write_mat(
        "workspace",
        &Value::Object(object),
        RootBinding::WorkspaceObject,
        &options,
    );
    let Some(dump) = run_matio_oracle(path.path()) else {
        return;
    };

    assert_eq!(
        dump.variables.get("x1_bad"),
        Some(&OracleValue::Uint8 {
            dims: vec![1, 1],
            data: vec![7],
        }),
        "dump={dump:#?}"
    );
    assert_eq!(
        dump.variables.get("beta"),
        Some(&OracleValue::Char {
            dims: vec![1, 2],
            value: "ok".to_owned(),
        }),
        "dump={dump:#?}"
    );
}

#[derive(Debug, Serialize)]
struct ScalarCoverage {
    nothing: Option<()>,
    truth: bool,
    greeting: &'static str,
    i8v: i8,
    i16v: i16,
    i32v: i32,
    i64v: i64,
    u8v: u8,
    u16v: u16,
    u32v: u32,
    u64v: u64,
    f32v: f32,
    f64v: f64,
    c32: Complex<f32>,
    c64: Complex<f64>,
}

#[test]
fn matio_oracle_roundtrips_supported_scalars() {
    let _guard = lock_oracle_test();
    let payload = ScalarCoverage {
        nothing: None,
        truth: true,
        greeting: "hello",
        i8v: -5,
        i16v: -1234,
        i32v: -123_456,
        i64v: -9_876_543_210,
        u8v: 7,
        u16v: 65_000,
        u32v: 1_000_000,
        u64v: 9_000_000_000,
        f32v: 1.5,
        f64v: -2.25,
        c32: Complex { re: 1.25, im: -2.5 },
        c64: Complex { re: 3.5, im: -4.75 },
    };
    let path = write_mat(
        "scalar-coverage",
        &payload,
        RootBinding::WorkspaceObject,
        &MatV73Options::default(),
    );
    let Some(dump) = run_matio_oracle(path.path()) else {
        return;
    };

    assert_eq!(
        expect_var(&dump, "nothing"),
        &OracleValue::Struct {
            dims: vec![0, 0],
            fields: BTreeMap::new(),
        }
    );
    assert_eq!(
        expect_var(&dump, "truth"),
        &OracleValue::Logical {
            dims: vec![1, 1],
            data: vec![true],
        }
    );
    assert_eq!(
        expect_var(&dump, "greeting"),
        &OracleValue::Char {
            dims: vec![1, 5],
            value: "hello".to_owned(),
        }
    );
    assert_eq!(
        expect_var(&dump, "i8v"),
        &OracleValue::Int8 {
            dims: vec![1, 1],
            data: vec![-5],
        }
    );
    assert_eq!(
        expect_var(&dump, "i16v"),
        &OracleValue::Int16 {
            dims: vec![1, 1],
            data: vec![-1234],
        }
    );
    assert_eq!(
        expect_var(&dump, "i32v"),
        &OracleValue::Int32 {
            dims: vec![1, 1],
            data: vec![-123_456],
        }
    );
    assert_eq!(
        expect_var(&dump, "i64v"),
        &OracleValue::Int64 {
            dims: vec![1, 1],
            data: vec![-9_876_543_210],
        }
    );
    assert_eq!(
        expect_var(&dump, "u8v"),
        &OracleValue::Uint8 {
            dims: vec![1, 1],
            data: vec![7],
        }
    );
    assert_eq!(
        expect_var(&dump, "u16v"),
        &OracleValue::Uint16 {
            dims: vec![1, 1],
            data: vec![65_000],
        }
    );
    assert_eq!(
        expect_var(&dump, "u32v"),
        &OracleValue::Uint32 {
            dims: vec![1, 1],
            data: vec![1_000_000],
        }
    );
    assert_eq!(
        expect_var(&dump, "u64v"),
        &OracleValue::Uint64 {
            dims: vec![1, 1],
            data: vec![9_000_000_000],
        }
    );
    assert_eq!(
        expect_var(&dump, "f32v"),
        &OracleValue::Single {
            dims: vec![1, 1],
            data: vec![1.5],
        }
    );
    assert_eq!(
        expect_var(&dump, "f64v"),
        &OracleValue::Double {
            dims: vec![1, 1],
            data: vec![-2.25],
        }
    );
    assert_eq!(
        expect_var(&dump, "c32"),
        &OracleValue::ComplexSingle {
            dims: vec![1, 1],
            real: vec![1.25],
            imag: vec![-2.5],
        }
    );
    assert_eq!(
        expect_var(&dump, "c64"),
        &OracleValue::ComplexDouble {
            dims: vec![1, 1],
            real: vec![3.5],
            imag: vec![-4.75],
        }
    );
}

#[derive(Debug, Serialize)]
struct ArrayCoverage {
    i8s: Vec<i8>,
    i16s: Vec<i16>,
    i32s: Vec<i32>,
    i64s: Vec<i64>,
    u8s: Vec<u8>,
    u16s: Vec<u16>,
    u32s: Vec<u32>,
    u64s: Vec<u64>,
    f32s: Vec<f32>,
    f64s: Vec<f64>,
    bools: Vec<bool>,
    labels: Vec<String>,
    c32s: Vec<Complex<f32>>,
    c64s: Vec<Complex<f64>>,
}

#[test]
fn matio_oracle_roundtrips_supported_arrays() {
    let _guard = lock_oracle_test();
    let payload = ArrayCoverage {
        i8s: vec![-1, 0, 1],
        i16s: vec![-10, 20, 30],
        i32s: vec![-100, 200, 300],
        i64s: vec![-1000, 2000, 3000],
        u8s: vec![1, 2, 3],
        u16s: vec![10, 20, 30],
        u32s: vec![100, 200, 300],
        u64s: vec![1000, 2000, 3000],
        f32s: vec![1.5, -2.5, 3.25],
        f64s: vec![10.0, -20.5, 30.25],
        bools: vec![true, false, true],
        labels: vec!["left".to_owned(), "right".to_owned()],
        c32s: vec![Complex { re: 1.0, im: 2.0 }, Complex { re: -3.0, im: 4.5 }],
        c64s: vec![Complex { re: 5.0, im: -6.0 }, Complex { re: 7.5, im: 8.25 }],
    };
    let path = write_mat(
        "array-coverage",
        &payload,
        RootBinding::WorkspaceObject,
        &MatV73Options::default(),
    );
    let Some(dump) = run_matio_oracle(path.path()) else {
        return;
    };

    assert_eq!(
        expect_var(&dump, "i8s"),
        &OracleValue::Int8 {
            dims: vec![3, 1],
            data: vec![-1, 0, 1],
        }
    );
    assert_eq!(
        expect_var(&dump, "i16s"),
        &OracleValue::Int16 {
            dims: vec![3, 1],
            data: vec![-10, 20, 30],
        }
    );
    assert_eq!(
        expect_var(&dump, "i32s"),
        &OracleValue::Int32 {
            dims: vec![3, 1],
            data: vec![-100, 200, 300],
        }
    );
    assert_eq!(
        expect_var(&dump, "i64s"),
        &OracleValue::Int64 {
            dims: vec![3, 1],
            data: vec![-1000, 2000, 3000],
        }
    );
    assert_eq!(
        expect_var(&dump, "u8s"),
        &OracleValue::Uint8 {
            dims: vec![3, 1],
            data: vec![1, 2, 3],
        }
    );
    assert_eq!(
        expect_var(&dump, "u16s"),
        &OracleValue::Uint16 {
            dims: vec![3, 1],
            data: vec![10, 20, 30],
        }
    );
    assert_eq!(
        expect_var(&dump, "u32s"),
        &OracleValue::Uint32 {
            dims: vec![3, 1],
            data: vec![100, 200, 300],
        }
    );
    assert_eq!(
        expect_var(&dump, "u64s"),
        &OracleValue::Uint64 {
            dims: vec![3, 1],
            data: vec![1000, 2000, 3000],
        }
    );
    assert_eq!(
        expect_var(&dump, "f32s"),
        &OracleValue::Single {
            dims: vec![3, 1],
            data: vec![1.5, -2.5, 3.25],
        }
    );
    assert_eq!(
        expect_var(&dump, "f64s"),
        &OracleValue::Double {
            dims: vec![3, 1],
            data: vec![10.0, -20.5, 30.25],
        }
    );
    assert_eq!(
        expect_var(&dump, "bools"),
        &OracleValue::Logical {
            dims: vec![3, 1],
            data: vec![true, false, true],
        }
    );
    assert_eq!(
        expect_var(&dump, "labels"),
        &OracleValue::Cell {
            dims: vec![2, 1],
            elements: vec![
                OracleValue::Char {
                    dims: vec![1, 4],
                    value: "left".to_owned(),
                },
                OracleValue::Char {
                    dims: vec![1, 5],
                    value: "right".to_owned(),
                },
            ],
        }
    );
    assert_eq!(
        expect_var(&dump, "c32s"),
        &OracleValue::ComplexSingle {
            dims: vec![2, 1],
            real: vec![1.0, -3.0],
            imag: vec![2.0, 4.5],
        }
    );
    assert_eq!(
        expect_var(&dump, "c64s"),
        &OracleValue::ComplexDouble {
            dims: vec![2, 1],
            real: vec![5.0, 7.5],
            imag: vec![-6.0, 8.25],
        }
    );
}

#[test]
fn matio_oracle_roundtrips_supported_empty_values() {
    let _guard = lock_oracle_test();
    let options = MatV73Options::default();

    let empty_i32: &[i32] = &[];
    let path = write_mat_bytes(
        "empty-ints",
        &beve::to_vec_typed_slice(empty_i32),
        RootBinding::NamedVariable("ints"),
        &options,
    );
    let dump = run_matio_oracle(path.path()).unwrap();
    assert_eq!(
        expect_var(&dump, "ints"),
        &OracleValue::Int32 {
            dims: vec![0, 1],
            data: Vec::new(),
        }
    );

    let empty_bools: &[bool] = &[];
    let path = write_mat_bytes(
        "empty-bools",
        &beve::to_vec_bool_slice(empty_bools),
        RootBinding::NamedVariable("flags"),
        &options,
    );
    let dump = run_matio_oracle(path.path()).unwrap();
    assert_eq!(
        expect_var(&dump, "flags"),
        &OracleValue::Logical {
            dims: vec![0, 1],
            data: Vec::new(),
        }
    );

    let empty_strings: &[String] = &[];
    let path = write_mat_bytes(
        "empty-strings",
        &beve::to_vec_string_slice(empty_strings),
        RootBinding::NamedVariable("strings"),
        &options,
    );
    let dump = run_matio_oracle(path.path()).unwrap();
    assert_eq!(
        expect_var(&dump, "strings"),
        &OracleValue::Cell {
            dims: vec![0, 1],
            elements: Vec::new(),
        }
    );

    let path = write_mat(
        "empty-cells",
        &Value::Array(Vec::new()),
        RootBinding::NamedVariable("cells"),
        &options,
    );
    let dump = run_matio_oracle(path.path()).unwrap();
    assert_eq!(
        expect_var(&dump, "cells"),
        &OracleValue::Cell {
            dims: vec![0, 1],
            elements: Vec::new(),
        }
    );

    let empty_complex: &[Complex<f64>] = &[];
    let path = write_mat_bytes(
        "empty-c64s",
        &beve::to_vec_complex64_slice(empty_complex),
        RootBinding::NamedVariable("c64s"),
        &options,
    );
    let dump = run_matio_oracle(path.path()).unwrap();
    match expect_var(&dump, "c64s") {
        OracleValue::ComplexDouble { dims, real, imag } => {
            assert_eq!(dims, &vec![0, 1]);
            assert!(real.is_empty());
            assert!(imag.is_empty());
        }
        // MATIO 1.5.30 currently collapses zero-length complex arrays to real
        // doubles on readback even though the writer emits a complex dataset.
        OracleValue::Double { dims, data } => {
            assert_eq!(dims, &vec![0, 1]);
            assert!(data.is_empty());
        }
        value => panic!("expected empty complex<double> roundtrip, got {value:#?}"),
    }
}

#[derive(Debug, Serialize)]
struct ExtensionCoverage {
    cell: Value,
    nested: OracleStruct<'static>,
    int_matrix: MatrixOwned<i16>,
    bool_matrix: MatrixOwned<bool>,
    complex_matrix: MatrixOwned<Complex<f32>>,
    row_major: MatrixOwned<f64>,
}

#[test]
fn matio_oracle_roundtrips_cells_structs_and_matrices() {
    let _guard = lock_oracle_test();
    let mut nested_obj = Object::new();
    nested_obj.insert(Key::from("answer"), Value::from(42u8));
    nested_obj.insert(Key::from("ok"), Value::from(true));
    let payload = ExtensionCoverage {
        cell: Value::Array(vec![
            Value::from(1u8),
            Value::from("hi"),
            Value::Object(nested_obj),
            Value::Null,
        ]),
        nested: OracleStruct {
            answer: 7,
            label: "ready",
        },
        int_matrix: MatrixOwned {
            layout: MatrixLayout::Left,
            extents: vec![2, 2],
            data: vec![-1i16, 2, 3, -4],
        },
        bool_matrix: MatrixOwned {
            layout: MatrixLayout::Left,
            extents: vec![2, 2],
            data: vec![true, false, false, true],
        },
        complex_matrix: MatrixOwned {
            layout: MatrixLayout::Left,
            extents: vec![2, 2],
            data: vec![
                Complex { re: 1.0, im: 2.0 },
                Complex { re: 3.0, im: 4.0 },
                Complex { re: 5.0, im: 6.0 },
                Complex { re: 7.0, im: 8.0 },
            ],
        },
        row_major: MatrixOwned {
            layout: MatrixLayout::Right,
            extents: vec![2, 3],
            data: vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0],
        },
    };
    let path = write_mat(
        "extension-coverage",
        &payload,
        RootBinding::WorkspaceObject,
        &MatV73Options::default(),
    );
    let Some(dump) = run_matio_oracle(path.path()) else {
        return;
    };

    assert_eq!(
        expect_var(&dump, "cell"),
        &OracleValue::Cell {
            dims: vec![4, 1],
            elements: vec![
                OracleValue::Uint8 {
                    dims: vec![1, 1],
                    data: vec![1],
                },
                OracleValue::Char {
                    dims: vec![1, 2],
                    value: "hi".to_owned(),
                },
                OracleValue::Struct {
                    dims: vec![1, 1],
                    fields: BTreeMap::from([
                        (
                            "answer".to_owned(),
                            vec![OracleValue::Uint8 {
                                dims: vec![1, 1],
                                data: vec![42],
                            }],
                        ),
                        (
                            "ok".to_owned(),
                            vec![OracleValue::Logical {
                                dims: vec![1, 1],
                                data: vec![true],
                            }],
                        ),
                    ]),
                },
                OracleValue::Struct {
                    dims: vec![0, 0],
                    fields: BTreeMap::new(),
                },
            ],
        }
    );

    let nested = expect_var(&dump, "nested");
    assert_eq!(
        expect_struct_field(nested, "answer"),
        &OracleValue::Uint8 {
            dims: vec![1, 1],
            data: vec![7],
        }
    );
    assert_eq!(
        expect_struct_field(nested, "label"),
        &OracleValue::Char {
            dims: vec![1, 5],
            value: "ready".to_owned(),
        }
    );

    assert_eq!(
        expect_var(&dump, "int_matrix"),
        &OracleValue::Int16 {
            dims: vec![2, 2],
            data: vec![-1, 2, 3, -4],
        }
    );
    assert_eq!(
        expect_var(&dump, "bool_matrix"),
        &OracleValue::Logical {
            dims: vec![2, 2],
            data: vec![true, false, false, true],
        }
    );
    assert_eq!(
        expect_var(&dump, "complex_matrix"),
        &OracleValue::ComplexSingle {
            dims: vec![2, 2],
            real: vec![1.0, 3.0, 5.0, 7.0],
            imag: vec![2.0, 4.0, 6.0, 8.0],
        }
    );
    assert_eq!(
        expect_var(&dump, "row_major"),
        &OracleValue::Double {
            dims: vec![2, 3],
            data: vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        }
    );
}

#[derive(Debug, Serialize)]
struct MatrixWidthCoverage {
    i8m: MatrixOwned<i8>,
    i32m: MatrixOwned<i32>,
    i64m: MatrixOwned<i64>,
    u8m: MatrixOwned<u8>,
    u16m: MatrixOwned<u16>,
    u32m: MatrixOwned<u32>,
    u64m: MatrixOwned<u64>,
    f32m: MatrixOwned<f32>,
    c64m: MatrixOwned<Complex<f64>>,
}

#[test]
fn matio_oracle_roundtrips_matrix_extension_widths() {
    let _guard = lock_oracle_test();
    let payload = MatrixWidthCoverage {
        i8m: MatrixOwned {
            layout: MatrixLayout::Left,
            extents: vec![2],
            data: vec![-1, 2],
        },
        i32m: MatrixOwned {
            layout: MatrixLayout::Left,
            extents: vec![2],
            data: vec![-100, 200],
        },
        i64m: MatrixOwned {
            layout: MatrixLayout::Left,
            extents: vec![2],
            data: vec![-1000, 2000],
        },
        u8m: MatrixOwned {
            layout: MatrixLayout::Left,
            extents: vec![2],
            data: vec![1, 2],
        },
        u16m: MatrixOwned {
            layout: MatrixLayout::Left,
            extents: vec![2],
            data: vec![10, 20],
        },
        u32m: MatrixOwned {
            layout: MatrixLayout::Left,
            extents: vec![2],
            data: vec![100, 200],
        },
        u64m: MatrixOwned {
            layout: MatrixLayout::Left,
            extents: vec![2],
            data: vec![1000, 2000],
        },
        f32m: MatrixOwned {
            layout: MatrixLayout::Left,
            extents: vec![2],
            data: vec![1.5, -2.5],
        },
        c64m: MatrixOwned {
            layout: MatrixLayout::Left,
            extents: vec![2],
            data: vec![Complex { re: 3.0, im: -4.0 }, Complex { re: 5.5, im: 6.25 }],
        },
    };
    let path = write_mat(
        "matrix-width-coverage",
        &payload,
        RootBinding::WorkspaceObject,
        &MatV73Options::default(),
    );
    let Some(dump) = run_matio_oracle(path.path()) else {
        return;
    };

    assert_eq!(
        expect_var(&dump, "i8m"),
        &OracleValue::Int8 {
            dims: vec![2, 1],
            data: vec![-1, 2],
        }
    );
    assert_eq!(
        expect_var(&dump, "i32m"),
        &OracleValue::Int32 {
            dims: vec![2, 1],
            data: vec![-100, 200],
        }
    );
    assert_eq!(
        expect_var(&dump, "i64m"),
        &OracleValue::Int64 {
            dims: vec![2, 1],
            data: vec![-1000, 2000],
        }
    );
    assert_eq!(
        expect_var(&dump, "u8m"),
        &OracleValue::Uint8 {
            dims: vec![2, 1],
            data: vec![1, 2],
        }
    );
    assert_eq!(
        expect_var(&dump, "u16m"),
        &OracleValue::Uint16 {
            dims: vec![2, 1],
            data: vec![10, 20],
        }
    );
    assert_eq!(
        expect_var(&dump, "u32m"),
        &OracleValue::Uint32 {
            dims: vec![2, 1],
            data: vec![100, 200],
        }
    );
    assert_eq!(
        expect_var(&dump, "u64m"),
        &OracleValue::Uint64 {
            dims: vec![2, 1],
            data: vec![1000, 2000],
        }
    );
    assert_eq!(
        expect_var(&dump, "f32m"),
        &OracleValue::Single {
            dims: vec![2, 1],
            data: vec![1.5, -2.5],
        }
    );
    assert_eq!(
        expect_var(&dump, "c64m"),
        &OracleValue::ComplexDouble {
            dims: vec![2, 1],
            real: vec![3.0, 5.5],
            imag: vec![-4.0, 6.25],
        }
    );
}

#[test]
fn matio_oracle_roundtrips_supported_fallback_policies() {
    let _guard = lock_oracle_test();

    let fallback = MatV73Options {
        unsupported_policy: UnsupportedPolicy::StringFallback,
        ..MatV73Options::default()
    };
    let path = write_mat(
        "i128-fallback",
        &i128::MIN,
        RootBinding::NamedVariable("big"),
        &fallback,
    );
    let dump = run_matio_oracle(path.path()).unwrap();
    assert_eq!(
        expect_var(&dump, "big"),
        &OracleValue::Char {
            dims: vec![1, i128::MIN.to_string().len()],
            value: i128::MIN.to_string(),
        }
    );

    let path = write_mat(
        "u128-fallback",
        &u128::MAX,
        RootBinding::NamedVariable("big_unsigned"),
        &fallback,
    );
    let dump = run_matio_oracle(path.path()).unwrap();
    assert_eq!(
        expect_var(&dump, "big_unsigned"),
        &OracleValue::Char {
            dims: vec![1, u128::MAX.to_string().len()],
            value: u128::MAX.to_string(),
        }
    );

    let lossy = MatV73Options {
        unsupported_policy: UnsupportedPolicy::LossyNumericWidening,
        ..MatV73Options::default()
    };
    let path = write_mat(
        "f16-lossy",
        &f16::from_f32(1.5),
        RootBinding::NamedVariable("half_value"),
        &lossy,
    );
    let dump = run_matio_oracle(path.path()).unwrap();
    assert_eq!(
        expect_var(&dump, "half_value"),
        &OracleValue::Single {
            dims: vec![1, 1],
            data: vec![1.5],
        }
    );

    let values = vec![bf16::from_f32(-1.0), bf16::from_f32(2.5)];
    let path = write_mat(
        "bf16-array-lossy",
        &values,
        RootBinding::NamedVariable("half_array"),
        &lossy,
    );
    let dump = run_matio_oracle(path.path()).unwrap();
    assert_eq!(
        expect_var(&dump, "half_array"),
        &OracleValue::Single {
            dims: vec![2, 1],
            data: vec![-1.0, 2.5],
        }
    );

    let matrix = MatrixOwned {
        layout: MatrixLayout::Left,
        extents: vec![2, 1],
        data: vec![f16::from_f32(3.0), f16::from_f32(-4.0)],
    };
    let path = write_mat(
        "f16-matrix-lossy",
        &matrix,
        RootBinding::NamedVariable("half_matrix"),
        &lossy,
    );
    let dump = run_matio_oracle(path.path()).unwrap();
    assert_eq!(
        expect_var(&dump, "half_matrix"),
        &OracleValue::Single {
            dims: vec![2, 1],
            data: vec![3.0, -4.0],
        }
    );

    let matrix = MatrixOwned {
        layout: MatrixLayout::Left,
        extents: vec![2, 1],
        data: vec![bf16::from_f32(5.0), bf16::from_f32(-6.5)],
    };
    let path = write_mat(
        "bf16-matrix-lossy",
        &matrix,
        RootBinding::NamedVariable("bf16_matrix"),
        &lossy,
    );
    let dump = run_matio_oracle(path.path()).unwrap();
    assert_eq!(
        expect_var(&dump, "bf16_matrix"),
        &OracleValue::Single {
            dims: vec![2, 1],
            data: vec![5.0, -6.5],
        }
    );
}

#[test]
fn matio_oracle_reports_expected_errors_for_unsupported_beve_types() {
    let _guard = lock_oracle_test();

    let default = MatV73Options::default();
    let err = write_mat_err(
        "i128-default-error",
        &i128::MIN,
        RootBinding::NamedVariable("big"),
        &default,
    );
    assert!(err.contains("unsupported 128-bit integer scalar"));

    let err = write_mat_err(
        "u128-array-error",
        &vec![1u128, 2u128],
        RootBinding::NamedVariable("big_array"),
        &default,
    );
    assert!(err.contains("unsupported u128 array"));

    let err = write_mat_err(
        "bf16-default-error",
        &bf16::from_f32(1.0),
        RootBinding::NamedVariable("half_value"),
        &default,
    );
    assert!(err.contains("unsupported bf16 scalar"));

    let err = write_mat_err(
        "f16-default-error",
        &f16::from_f32(1.0),
        RootBinding::NamedVariable("half_value"),
        &default,
    );
    assert!(err.contains("unsupported f16 scalar"));

    let err = write_mat_err(
        "f16-array-default-error",
        &vec![f16::from_f32(1.0), f16::from_f32(2.0)],
        RootBinding::NamedVariable("half_array"),
        &default,
    );
    assert!(err.contains("unsupported f16 array"));

    let matrix = MatrixOwned {
        layout: MatrixLayout::Left,
        extents: vec![2],
        data: vec![1u128, 2u128],
    };
    let err = write_mat_err(
        "u128-matrix-error",
        &matrix,
        RootBinding::NamedVariable("big_matrix"),
        &default,
    );
    assert!(err.contains("unsupported u128 matrix"));

    let null_error = MatV73Options {
        null_policy: NullPolicy::Error,
        ..MatV73Options::default()
    };
    let err = write_mat_err(
        "null-policy-error",
        &Option::<()>::None,
        RootBinding::NamedVariable("nothing"),
        &null_error,
    );
    assert!(err.contains("unsupported null value"));
}

#[test]
fn matio_oracle_reports_expected_errors_for_unmappable_beve_encodings() {
    let _guard = lock_oracle_test();
    let default = MatV73Options::default();

    let signed_keys = BTreeMap::from([(-2i16, 7u8)]);
    let err = write_mat_err(
        "signed-key-object-error",
        &signed_keys,
        RootBinding::NamedVariable("payload"),
        &default,
    );
    assert!(err.contains("unsupported non-string object keys"));

    let unsigned_keys = BTreeMap::from([(5u16, 7u8)]);
    let err = write_mat_err(
        "unsigned-key-object-error",
        &unsigned_keys,
        RootBinding::NamedVariable("payload"),
        &default,
    );
    assert!(err.contains("unsupported non-string object keys"));

    let enum_bytes = beve::to_vec_with_options(
        &TaggedEnum::Scalar(7),
        SerializerOptions {
            enum_encoding: EnumEncoding::String,
        },
    )
    .unwrap();
    let err = write_mat_bytes_err(
        "type-tag-error",
        &enum_bytes,
        RootBinding::NamedVariable("payload"),
        &default,
    );
    assert!(err.contains("unsupported BEVE extension 0x1"));

    let err = write_mat_bytes_err(
        "delimiter-error",
        &[0x06],
        RootBinding::NamedVariable("payload"),
        &default,
    );
    assert!(err.contains("unsupported BEVE extension 0x0"));

    let options = MatV73Options {
        row_major_policy: RowMajorPolicy::Error,
        ..MatV73Options::default()
    };
    let matrix = MatrixOwned {
        layout: MatrixLayout::Right,
        extents: vec![2, 2],
        data: vec![1.0f64, 2.0, 3.0, 4.0],
    };
    let err = write_mat_err(
        "row-major-policy-error",
        &matrix,
        RootBinding::NamedVariable("payload"),
        &options,
    );
    assert!(err.contains("requires reordering to MATLAB column-major layout"));

    let options = MatV73Options {
        invalid_name_policy: InvalidNamePolicy::Error,
        ..MatV73Options::default()
    };
    let mut object = Object::new();
    object.insert(Key::from("1 bad"), Value::from(7u8));
    let err = write_mat_err(
        "invalid-name-error",
        &Value::Object(object),
        RootBinding::WorkspaceObject,
        &options,
    );
    assert!(err.contains("invalid MATLAB name"));
}

#[test]
fn matio_oracle_respects_row_vector_mode() {
    let _guard = lock_oracle_test();
    let options = MatV73Options {
        one_dimensional_mode: OneDimensionalMode::RowVector,
        ..MatV73Options::default()
    };
    let path = write_mat(
        "row-vector-mode",
        &vec![1u16, 2, 3],
        RootBinding::NamedVariable("values"),
        &options,
    );
    let Some(dump) = run_matio_oracle(path.path()) else {
        return;
    };

    assert_eq!(
        expect_var(&dump, "values"),
        &OracleValue::Uint16 {
            dims: vec![1, 3],
            data: vec![1, 2, 3],
        }
    );
}
