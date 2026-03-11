#![cfg(all(feature = "mat", feature = "mat-oracle"))]
#![deny(warnings)]

// `matio-rs-sys` 0.2.5 builds MATIO with `MATIO_MAT73=OFF`, so it cannot open the
// v7.3 files emitted by this crate. Keep these tests compiled but ignored until the
// oracle dependency is patched or replaced with a reader that actually supports v7.3.

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use beve::{Key, MatV73Options, Object, RootBinding, Value};
use matio_rs::{MatFile, MayBeInto};
use serde::Serialize;

fn temp_path(test_name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("beve-mat-oracle-{test_name}-{nanos}.mat"))
}

fn write_mat<T: Serialize>(test_name: &str, value: &T, root: RootBinding<'_>) -> PathBuf {
    let path = temp_path(test_name);
    let bytes = beve::to_vec(value).unwrap();
    beve::beve_slice_to_mat_v73_file(&bytes, &path, root, &MatV73Options::default()).unwrap();
    path
}

#[derive(Debug, Serialize)]
struct OracleStruct {
    answer: u8,
    samples: Vec<i32>,
}

#[test]
#[ignore = "matio-rs-sys 0.2.5 disables MAT73 support in build.rs"]
fn mat_oracle_reads_string_scalar() {
    let path = write_mat("string", &"hello", RootBinding::NamedVariable("greeting"));
    let mat = MatFile::load(&path).unwrap();

    let greeting: String = mat.var("greeting").unwrap();

    assert_eq!(greeting, "hello");
    std::fs::remove_file(path).unwrap();
}

#[test]
#[ignore = "matio-rs-sys 0.2.5 disables MAT73 support in build.rs"]
fn mat_oracle_reads_struct_into_rust_type() {
    let path = write_mat(
        "struct",
        &OracleStruct {
            answer: 7,
            samples: vec![4, 5, 6],
        },
        RootBinding::NamedVariable("payload"),
    );
    let mat = MatFile::load(&path).unwrap();

    let payload = mat.read("payload").unwrap();
    let answer: u8 = payload.field("answer").unwrap()[0].maybe_into().unwrap();
    let samples: Vec<i32> = payload.field("samples").unwrap()[0].maybe_into().unwrap();

    assert_eq!(payload.dims(), vec![1, 1]);
    assert_eq!(answer, 7);
    assert_eq!(samples, vec![4, 5, 6]);
    std::fs::remove_file(path).unwrap();
}

#[test]
#[ignore = "matio-rs-sys 0.2.5 disables MAT73 support in build.rs"]
fn mat_oracle_reads_workspace_object_variables() {
    let mut object = Object::new();
    object.insert(Key::from("alpha"), Value::from(3u32));
    object.insert(Key::from("beta"), Value::from("ok"));
    let path = write_mat(
        "workspace",
        &Value::Object(object),
        RootBinding::WorkspaceObject,
    );
    let mat = MatFile::load(&path).unwrap();

    let alpha: u8 = mat.var("alpha").unwrap();
    let beta: String = mat.var("beta").unwrap();

    assert_eq!(alpha, 3);
    assert_eq!(beta, "ok");
    std::fs::remove_file(path).unwrap();
}

#[test]
#[ignore = "matio-rs-sys 0.2.5 disables MAT73 support in build.rs"]
fn mat_oracle_reads_generic_array_as_cell_array_of_strings() {
    let value = Value::Array(vec![Value::from("left"), Value::from("right")]);
    let path = write_mat("cell", &value, RootBinding::NamedVariable("cells"));
    let mat = MatFile::load(&path).unwrap();

    let cells: Vec<String> = mat.var("cells").unwrap();

    assert_eq!(cells, vec!["left".to_owned(), "right".to_owned()]);
    std::fs::remove_file(path).unwrap();
}

#[test]
#[ignore = "matio-rs-sys 0.2.5 disables MAT73 support in build.rs"]
fn mat_oracle_reads_reordered_matrix_values() {
    let matrix = beve::MatrixOwned {
        layout: beve::MatrixLayout::Right,
        extents: vec![2, 3],
        data: vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    };
    let path = write_mat("matrix", &matrix, RootBinding::NamedVariable("a"));
    let mat = MatFile::load(&path).unwrap();
    let dims = mat.read("a").unwrap().dims();
    let values: Vec<f64> = mat.var("a").unwrap();

    assert_eq!(dims, vec![3, 2]);
    assert_eq!(values, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    std::fs::remove_file(path).unwrap();
}
