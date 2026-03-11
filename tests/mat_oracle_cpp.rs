#![cfg(feature = "mat")]
#![deny(warnings)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use beve::{InvalidNamePolicy, Key, MatV73Options, Object, RootBinding, Value};
use serde::{Deserialize, Serialize};

fn matio_oracle_bin() -> Option<String> {
    std::env::var("MATIO_ORACLE_BIN").ok()
}

fn oracle_test_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
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
    Logical {
        dims: Vec<usize>,
        data: Vec<bool>,
    },
    Double {
        dims: Vec<usize>,
        data: Vec<f64>,
    },
    Uint8 {
        dims: Vec<usize>,
        data: Vec<u8>,
    },
    Cell {
        dims: Vec<usize>,
        elements: Vec<OracleValue>,
    },
    Struct {
        dims: Vec<usize>,
        fields: BTreeMap<String, Vec<OracleValue>>,
    },
}

#[derive(Debug, Serialize)]
struct OracleStruct<'a> {
    answer: u8,
    label: &'a str,
}

#[test]
fn matio_oracle_reads_string_scalar() {
    let _guard = oracle_test_lock().lock().unwrap();
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
    let _guard = oracle_test_lock().lock().unwrap();
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
    let _guard = oracle_test_lock().lock().unwrap();
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
    let _guard = oracle_test_lock().lock().unwrap();
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
    let _guard = oracle_test_lock().lock().unwrap();
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
    let _guard = oracle_test_lock().lock().unwrap();
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
    let _guard = oracle_test_lock().lock().unwrap();
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
