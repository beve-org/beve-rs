#![cfg(feature = "mat")]
#![deny(warnings)]

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use beve::fast::{to_vec_matrix_f64, MatrixLayoutFast};
use beve::{
    Complex, InvalidNamePolicy, Key, MatV73Options, NullPolicy, Object, RootBinding,
    UnsupportedPolicy, Value,
};
use hdf5::types::{FixedAscii, TypeDescriptor, VarLenArray};
use hdf5::{ObjectReference1, ReferencedObject};
use hdf5_metno as hdf5;

const MCOS_MAGIC_NUMBER: u32 = 0xDD00_0000;

fn temp_path(test_name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("beve-mat-{test_name}-{nanos}.mat"))
}

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join("matlab_string_fixture")
        .join(name)
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
        9 => attr
            .read_scalar::<FixedAscii<9>>()
            .unwrap()
            .as_str()
            .to_owned(),
        10 => attr
            .read_scalar::<FixedAscii<10>>()
            .unwrap()
            .as_str()
            .to_owned(),
        11 => attr
            .read_scalar::<FixedAscii<11>>()
            .unwrap()
            .as_str()
            .to_owned(),
        12 => attr
            .read_scalar::<FixedAscii<12>>()
            .unwrap()
            .as_str()
            .to_owned(),
        13 => attr
            .read_scalar::<FixedAscii<13>>()
            .unwrap()
            .as_str()
            .to_owned(),
        14 => attr
            .read_scalar::<FixedAscii<14>>()
            .unwrap()
            .as_str()
            .to_owned(),
        15 => attr
            .read_scalar::<FixedAscii<15>>()
            .unwrap()
            .as_str()
            .to_owned(),
        16 => attr
            .read_scalar::<FixedAscii<16>>()
            .unwrap()
            .as_str()
            .to_owned(),
        len => panic!("unsupported MATLAB_class width: {len}"),
    }
}

fn decode_matlab_string_saveobj(raw: &[u64]) -> Vec<String> {
    assert!(raw.len() >= 2, "string saveobj payload is too short");

    let ndims = usize::try_from(raw[1]).unwrap();
    assert!(raw.len() >= 2 + ndims, "string saveobj dims are truncated");

    let dims = &raw[2..2 + ndims];
    let count = dims
        .iter()
        .copied()
        .try_fold(1usize, |acc, dim| {
            acc.checked_mul(usize::try_from(dim).unwrap())
        })
        .unwrap();

    let lens_start = 2 + ndims;
    let lens_end = lens_start + count;
    assert!(
        raw.len() >= lens_end,
        "string saveobj lengths are truncated"
    );

    let lengths: Vec<usize> = raw[lens_start..lens_end]
        .iter()
        .copied()
        .map(|len| usize::try_from(len).unwrap())
        .collect();

    let utf16_units = lengths
        .iter()
        .copied()
        .try_fold(0usize, |acc, len| acc.checked_add(len))
        .unwrap();
    let payload_words = &raw[lens_end..];
    let payload_bytes: Vec<u8> = payload_words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .take(utf16_units * 2)
        .collect();
    let utf16: Vec<u16> = payload_bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect();

    let mut offset = 0usize;
    lengths
        .into_iter()
        .map(|len| {
            let end = offset + len;
            let value = String::from_utf16(&utf16[offset..end]).unwrap();
            offset = end;
            value
        })
        .collect()
}

fn read_emitted_string_saveobj(file: &hdf5::File, object: &hdf5::Dataset) -> Vec<u64> {
    assert_eq!(read_matlab_class(object, "MATLAB_class"), "string");
    assert_eq!(
        object
            .attr("MATLAB_object_decode")
            .unwrap()
            .read_scalar::<i32>()
            .unwrap(),
        3
    );

    let metadata = object.read_raw::<u32>().unwrap();
    assert_eq!(metadata, vec![MCOS_MAGIC_NUMBER, 2, 1, 1, metadata[4], 1]);

    let subsystem = file.dataset("#subsystem#/MCOS").unwrap();
    assert_eq!(
        read_matlab_class(&subsystem, "MATLAB_class"),
        "FileWrapper__"
    );
    assert_eq!(
        subsystem
            .attr("MATLAB_object_decode")
            .unwrap()
            .read_scalar::<i32>()
            .unwrap(),
        3
    );
    let refs = subsystem.read_raw::<ObjectReference1>().unwrap();
    let payload_ref = refs[2 + metadata[4] as usize - 1];
    match file.dereference(&payload_ref).unwrap() {
        ReferencedObject::Dataset(payload) => payload.read_raw::<u64>().unwrap(),
        _ => panic!("expected string saveobj payload dataset"),
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
    assert_eq!(ds.shape(), vec![1, 6]);
    let payload = read_emitted_string_saveobj(&file, &ds);
    assert_eq!(payload[..5], [1, 2, 1, 1, 5]);
    assert_eq!(
        decode_matlab_string_saveobj(&payload),
        vec!["hello".to_owned()]
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
            let payload = read_emitted_string_saveobj(&file, &item);
            assert_eq!(
                decode_matlab_string_saveobj(&payload),
                vec!["hi".to_owned()]
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
fn matlab_string_fixture_v73_schema() {
    let file = hdf5::File::open(fixture_path("test_string_v73.mat")).unwrap();

    for (name, payload) in [
        ("string_scalar", vec![3707764736, 2, 1, 1, 1, 1]),
        ("string_array", vec![3707764736, 2, 1, 1, 2, 1]),
        ("string_empty", vec![3707764736, 2, 1, 1, 3, 1]),
    ] {
        let ds = file.dataset(name).unwrap();
        assert_eq!(ds.shape(), vec![1, 6]);
        assert_eq!(read_matlab_class(&ds, "MATLAB_class"), "string");
        assert_eq!(
            ds.attr("MATLAB_object_decode")
                .unwrap()
                .read_scalar::<i32>()
                .unwrap(),
            3
        );
        assert_eq!(ds.read_raw::<u32>().unwrap(), payload);
    }

    let subsystem = file.dataset("#subsystem#/MCOS").unwrap();
    assert_eq!(subsystem.shape(), vec![1, 8]);
    assert_eq!(
        read_matlab_class(&subsystem, "MATLAB_class"),
        "FileWrapper__"
    );
    assert_eq!(
        subsystem
            .attr("MATLAB_object_decode")
            .unwrap()
            .read_scalar::<i32>()
            .unwrap(),
        3
    );
}

#[test]
fn matlab_string_fixture_v73_saveobj_payloads() {
    let file = hdf5::File::open(fixture_path("test_string_v73.mat")).unwrap();

    let scalar = file.dataset("#refs#/c").unwrap();
    assert_eq!(read_matlab_class(&scalar, "MATLAB_class"), "uint64");
    assert_eq!(
        decode_matlab_string_saveobj(&scalar.read_raw::<u64>().unwrap()),
        vec!["Hello".to_owned()]
    );

    let array = file.dataset("#refs#/d").unwrap();
    assert_eq!(read_matlab_class(&array, "MATLAB_class"), "uint64");
    assert_eq!(
        decode_matlab_string_saveobj(&array.read_raw::<u64>().unwrap()),
        vec![
            "Apple".to_owned(),
            "Date".to_owned(),
            "Banana".to_owned(),
            "Fig".to_owned(),
            "Cherry".to_owned(),
            "Grapes".to_owned(),
        ]
    );

    let empty = file.dataset("#refs#/e").unwrap();
    assert_eq!(read_matlab_class(&empty, "MATLAB_class"), "uint64");
    assert_eq!(
        decode_matlab_string_saveobj(&empty.read_raw::<u64>().unwrap()),
        vec![String::new()]
    );
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
    let label = group.dataset("label").unwrap();
    let payload = read_emitted_string_saveobj(&file, &label);
    assert_eq!(
        decode_matlab_string_saveobj(&payload),
        vec!["ready".to_owned()]
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
    let payload = read_emitted_string_saveobj(&file, &ds);
    assert_eq!(
        decode_matlab_string_saveobj(&payload),
        vec!["hello".to_owned()]
    );

    std::fs::remove_file(path).unwrap();
}

#[test]
fn mat_v73_typed_string_array_uses_matlab_string() {
    let path = temp_path("typed-strings");
    let values = vec!["left".to_owned(), "right".to_owned()];
    let bytes = beve::to_vec_string_slice(&values);
    beve::beve_slice_to_mat_v73_file(
        &bytes,
        &path,
        RootBinding::NamedVariable("labels"),
        &MatV73Options::default(),
    )
    .unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("labels").unwrap();
    let payload = read_emitted_string_saveobj(&file, &ds);
    assert_eq!(payload[..6], [1, 2, 2, 1, 4, 5]);
    assert_eq!(
        decode_matlab_string_saveobj(&payload),
        vec!["left".to_owned(), "right".to_owned()]
    );

    std::fs::remove_file(path).unwrap();
}

#[test]
fn mat_v73_rejects_matrix_extension_with_empty_extents() {
    let path = temp_path("matrix-empty-extents");
    let bytes = to_vec_matrix_f64(MatrixLayoutFast::Left, &[], &[1.0]);
    let err = beve::beve_slice_to_mat_v73_file(
        &bytes,
        &path,
        RootBinding::NamedVariable("bad_matrix"),
        &MatV73Options::default(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("matrix extents cannot be empty"));
    assert!(!path.exists());
}

#[test]
fn mat_v73_rejects_matrix_extension_with_zero_extent() {
    let path = temp_path("matrix-zero-extent");
    let bytes = to_vec_matrix_f64(MatrixLayoutFast::Left, &[0, 2], &[]);
    let err = beve::beve_slice_to_mat_v73_file(
        &bytes,
        &path,
        RootBinding::NamedVariable("bad_matrix"),
        &MatV73Options::default(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("matrix dimensions cannot be zero"));
    assert!(!path.exists());
}
