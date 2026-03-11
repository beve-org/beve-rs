#![cfg(feature = "mat")]
#![deny(warnings)]

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use beve::{
    Complex, InvalidNamePolicy, Key, MatV73Options, NullPolicy, Object, RootBinding,
    UnsupportedPolicy, Value,
};
use hdf5::types::{FixedAscii, TypeDescriptor, VarLenArray};
use hdf5::{ObjectReference1, ReferencedObject};
use hdf5_metno as hdf5;

fn temp_path(test_name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("beve-mat-{test_name}-{nanos}.mat"))
}

fn read_matlab_class(loc: &hdf5::Location, name: &str) -> String {
    let attr = loc.attr(name).unwrap();
    match attr.dtype().unwrap().size() {
        1 => attr
            .read_scalar::<FixedAscii<1>>()
            .unwrap()
            .as_str()
            .to_owned(),
        2 => attr
            .read_scalar::<FixedAscii<2>>()
            .unwrap()
            .as_str()
            .to_owned(),
        3 => attr
            .read_scalar::<FixedAscii<3>>()
            .unwrap()
            .as_str()
            .to_owned(),
        4 => attr
            .read_scalar::<FixedAscii<4>>()
            .unwrap()
            .as_str()
            .to_owned(),
        5 => attr
            .read_scalar::<FixedAscii<5>>()
            .unwrap()
            .as_str()
            .to_owned(),
        6 => attr
            .read_scalar::<FixedAscii<6>>()
            .unwrap()
            .as_str()
            .to_owned(),
        7 => attr
            .read_scalar::<FixedAscii<7>>()
            .unwrap()
            .as_str()
            .to_owned(),
        8 => attr
            .read_scalar::<FixedAscii<8>>()
            .unwrap()
            .as_str()
            .to_owned(),
        len => panic!("unsupported MATLAB_class width: {len}"),
    }
}

#[test]
fn mat_v73_scalar_string_and_userblock() {
    let path = temp_path("string");
    let bytes = beve::to_vec(&"hello").unwrap();
    beve::beve_slice_to_mat_v73_file(
        &bytes,
        &path,
        RootBinding::NamedVariable("greeting"),
        &MatV73Options::default(),
    )
    .unwrap();

    let raw = std::fs::read(&path).unwrap();
    assert!(raw.starts_with(b"MATLAB 7.3 MAT-file"));
    assert_eq!(&raw[126..128], b"IM");

    let file = hdf5::File::open(&path).unwrap();
    assert_eq!(file.userblock(), 512);

    let ds = file.dataset("greeting").unwrap();
    assert_eq!(ds.shape(), vec![5, 1]);
    assert_eq!(read_matlab_class(&ds, "MATLAB_class"), "char");
    assert_eq!(
        ds.attr("MATLAB_int_decode")
            .unwrap()
            .read_scalar::<i32>()
            .unwrap(),
        2
    );
    assert_eq!(
        ds.read_raw::<u16>().unwrap(),
        "hello".encode_utf16().collect::<Vec<_>>()
    );

    std::fs::remove_file(path).unwrap();
}

#[test]
fn mat_v73_logical_array() {
    let path = temp_path("logical");
    let bytes = beve::to_vec(&vec![true, false, true]).unwrap();
    beve::beve_slice_to_mat_v73_file(
        &bytes,
        &path,
        RootBinding::NamedVariable("flags"),
        &MatV73Options::default(),
    )
    .unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("flags").unwrap();
    assert_eq!(ds.shape(), vec![1, 3]);
    assert_eq!(read_matlab_class(&ds, "MATLAB_class"), "logical");
    assert_eq!(
        ds.attr("MATLAB_int_decode")
            .unwrap()
            .read_scalar::<i32>()
            .unwrap(),
        1
    );
    assert_eq!(ds.read_raw::<u8>().unwrap(), vec![1, 0, 1]);

    std::fs::remove_file(path).unwrap();
}

#[test]
fn mat_v73_cell_array_uses_references() {
    let path = temp_path("cell");
    let value = Value::Array(vec![Value::from(1u32), Value::from("hi")]);
    let bytes = beve::to_vec(&value).unwrap();
    beve::beve_slice_to_mat_v73_file(
        &bytes,
        &path,
        RootBinding::NamedVariable("cells"),
        &MatV73Options::default(),
    )
    .unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("cells").unwrap();
    assert_eq!(ds.shape(), vec![1, 2]);
    assert_eq!(read_matlab_class(&ds, "MATLAB_class"), "cell");

    let refs = ds.read_raw::<ObjectReference1>().unwrap();
    assert_eq!(refs.len(), 2);

    match file.dereference(&refs[0]).unwrap() {
        ReferencedObject::Dataset(item) => {
            assert_eq!(read_matlab_class(&item, "MATLAB_class"), "uint8");
            assert_eq!(item.read_raw::<u8>().unwrap(), vec![1]);
        }
        _ => panic!("expected dataset reference"),
    }

    match file.dereference(&refs[1]).unwrap() {
        ReferencedObject::Dataset(item) => {
            assert_eq!(read_matlab_class(&item, "MATLAB_class"), "char");
            assert_eq!(
                item.read_raw::<u16>().unwrap(),
                "hi".encode_utf16().collect::<Vec<_>>()
            );
        }
        _ => panic!("expected dataset reference"),
    }

    std::fs::remove_file(path).unwrap();
}

#[test]
fn mat_v73_null_defaults_to_empty_struct_array() {
    let path = temp_path("null");
    let bytes = beve::to_vec(&Value::Null).unwrap();
    beve::beve_slice_to_mat_v73_file(
        &bytes,
        &path,
        RootBinding::NamedVariable("nothing"),
        &MatV73Options::default(),
    )
    .unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("nothing").unwrap();
    assert_eq!(read_matlab_class(&ds, "MATLAB_class"), "struct");
    assert_eq!(
        ds.attr("MATLAB_empty")
            .unwrap()
            .read_scalar::<u32>()
            .unwrap(),
        1
    );
    assert_eq!(ds.read_raw::<u64>().unwrap(), vec![0, 0]);

    std::fs::remove_file(path).unwrap();
}

#[test]
fn mat_v73_row_major_matrix_reorders_to_column_major() {
    let path = temp_path("matrix");
    let matrix = beve::MatrixOwned {
        layout: beve::MatrixLayout::Right,
        extents: vec![2, 3],
        data: vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    };
    let bytes = beve::to_vec(&matrix).unwrap();
    beve::beve_slice_to_mat_v73_file(
        &bytes,
        &path,
        RootBinding::NamedVariable("a"),
        &MatV73Options::default(),
    )
    .unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("a").unwrap();
    assert_eq!(ds.shape(), vec![3, 2]);
    assert_eq!(read_matlab_class(&ds, "MATLAB_class"), "double");
    assert_eq!(
        ds.read_raw::<f64>().unwrap(),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );

    std::fs::remove_file(path).unwrap();
}

#[test]
fn mat_v73_empty_complex_array_uses_complex_dataset_type() {
    let path = temp_path("empty-complex");
    let empty: &[Complex<f64>] = &[];
    let bytes = beve::to_vec_complex64_slice(empty);
    beve::beve_slice_to_mat_v73_file(
        &bytes,
        &path,
        RootBinding::NamedVariable("z"),
        &MatV73Options::default(),
    )
    .unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("z").unwrap();
    assert_eq!(ds.shape(), vec![1, 0]);
    assert_eq!(read_matlab_class(&ds, "MATLAB_class"), "double");
    assert!(ds.attr("MATLAB_empty").is_err());

    match ds.dtype().unwrap().to_descriptor().unwrap() {
        TypeDescriptor::Compound(compound) => {
            assert_eq!(compound.size, 16);
            assert_eq!(compound.fields.len(), 2);
            assert_eq!(compound.fields[0].name, "real");
            assert_eq!(compound.fields[1].name, "imag");
        }
        other => panic!("expected compound complex dtype, got {other:?}"),
    }

    std::fs::remove_file(path).unwrap();
}

#[test]
fn mat_v73_workspace_object_sanitizes_names() {
    let path = temp_path("workspace");
    let mut object = Object::new();
    object.insert(Key::from("1 bad"), Value::from(7u32));
    let bytes = beve::to_vec(&Value::Object(object)).unwrap();

    let options = MatV73Options {
        invalid_name_policy: InvalidNamePolicy::Sanitize,
        unsupported_policy: UnsupportedPolicy::Error,
        ..MatV73Options::default()
    };
    beve::beve_slice_to_mat_v73_file(&bytes, &path, RootBinding::WorkspaceObject, &options)
        .unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("x1_bad").unwrap();
    assert_eq!(read_matlab_class(&ds, "MATLAB_class"), "uint8");
    assert_eq!(ds.read_raw::<u8>().unwrap(), vec![7]);

    std::fs::remove_file(path).unwrap();
}

#[test]
fn mat_v73_struct_groups_include_fields_metadata() {
    let path = temp_path("struct");
    let value = Object::from_iter([
        (Key::from("answer"), Value::from(7u32)),
        (Key::from("label"), Value::from("ready")),
    ]);
    let bytes = beve::to_vec(&Value::Object(value)).unwrap();
    beve::beve_slice_to_mat_v73_file(
        &bytes,
        &path,
        RootBinding::NamedVariable("payload"),
        &MatV73Options::default(),
    )
    .unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let group = file.group("payload").unwrap();
    assert_eq!(read_matlab_class(&group, "MATLAB_class"), "struct");
    let fields = group
        .attr("MATLAB_fields")
        .unwrap()
        .read_raw::<VarLenArray<FixedAscii<1>>>()
        .unwrap();
    assert_eq!(
        fields
            .iter()
            .map(|field| field.iter().map(|ch| ch.as_str()).collect::<String>())
            .collect::<Vec<_>>(),
        vec!["answer", "label"]
    );

    std::fs::remove_file(path).unwrap();
}

#[test]
fn mat_v73_null_policy_error_rejects_null() {
    let path = temp_path("null-error");
    let bytes = beve::to_vec(&Value::Null).unwrap();
    let options = MatV73Options {
        null_policy: NullPolicy::Error,
        ..MatV73Options::default()
    };
    let err = beve::beve_slice_to_mat_v73_file(
        &bytes,
        &path,
        RootBinding::NamedVariable("nothing"),
        &options,
    )
    .unwrap_err();
    assert!(err.to_string().contains("unsupported null value"));
}

#[test]
fn mat_v73_failed_overwrite_preserves_existing_file() {
    let path = temp_path("preserve-existing");

    let good = beve::to_vec(&"hello").unwrap();
    beve::beve_slice_to_mat_v73_file(
        &good,
        &path,
        RootBinding::NamedVariable("greeting"),
        &MatV73Options::default(),
    )
    .unwrap();
    let original = std::fs::read(&path).unwrap();

    let bad = beve::to_vec(&Value::Null).unwrap();
    let options = MatV73Options {
        null_policy: NullPolicy::Error,
        ..MatV73Options::default()
    };
    let err = beve::beve_slice_to_mat_v73_file(
        &bad,
        &path,
        RootBinding::NamedVariable("nothing"),
        &options,
    )
    .unwrap_err();
    assert!(err.to_string().contains("unsupported null value"));

    let after = std::fs::read(&path).unwrap();
    assert_eq!(after, original);

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("greeting").unwrap();
    assert_eq!(read_matlab_class(&ds, "MATLAB_class"), "char");
    assert_eq!(
        ds.read_raw::<u16>().unwrap(),
        "hello".encode_utf16().collect::<Vec<_>>()
    );

    std::fs::remove_file(path).unwrap();
}
