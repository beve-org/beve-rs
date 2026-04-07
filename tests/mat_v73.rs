#![cfg(feature = "mat")]
#![deny(warnings)]

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use beve::fast::{MatrixLayoutFast, to_vec_matrix_f64};
use beve::{
    Complex, InvalidNamePolicy, Key, MatV73Options, NullPolicy, Object, RootBinding,
    UnsupportedPolicy, Value,
};
use hdf5_pure::{AttrValue, DType, File};

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

fn read_attr_string(attrs: &HashMap<String, AttrValue>, name: &str) -> String {
    match &attrs[name] {
        AttrValue::String(s) => s.clone(),
        AttrValue::AsciiString(s) => s.clone(),
        other => panic!("expected String for {name}, got {other:?}"),
    }
}

fn read_attr_i64(attrs: &HashMap<String, AttrValue>, name: &str) -> i64 {
    match &attrs[name] {
        AttrValue::I64(v) => *v,
        AttrValue::I32(v) => *v as i64,
        other => panic!("expected I64 for {name}, got {other:?}"),
    }
}

fn read_attr_u64(attrs: &HashMap<String, AttrValue>, name: &str) -> u64 {
    match &attrs[name] {
        AttrValue::U64(v) => *v,
        AttrValue::U32(v) => *v as u64,
        other => panic!("expected U64 for {name}, got {other:?}"),
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

/// Read the saveobj payload for a MATLAB string object.
///
/// Verifies the dataset has MATLAB_class="string" and MATLAB_object_decode=3,
/// then finds the corresponding uint64 payload dataset in `#refs#`.
fn read_string_saveobj_payload(file: &File, ds_path: &str) -> Vec<u64> {
    let ds = file.dataset(ds_path).unwrap();
    let attrs = ds.attrs().unwrap();
    assert_eq!(read_attr_string(&attrs, "MATLAB_class"), "string");
    assert_eq!(read_attr_i64(&attrs, "MATLAB_object_decode"), 3);

    let metadata = ds.read_u32().unwrap();
    assert_eq!(metadata[..4], [MCOS_MAGIC_NUMBER, 2, 1, 1]);

    // Verify subsystem exists
    let subsystem = file.dataset("#subsystem#/MCOS").unwrap();
    let sub_attrs = subsystem.attrs().unwrap();
    assert_eq!(
        read_attr_string(&sub_attrs, "MATLAB_class"),
        "FileWrapper__"
    );
    assert_eq!(read_attr_i64(&sub_attrs, "MATLAB_object_decode"), 3);

    // Find the saveobj payload: scan #refs# for uint64 datasets.
    // Each string object allocates one ref for the saveobj payload (a uint64 dataset).
    // The refs are named ref_{id:016x} in allocation order.
    // For object_id N (1-based from metadata[4]), the payload is at index N-1 among
    // the uint64 datasets sorted by name.
    let refs_group = file.group("#refs#").unwrap();
    let mut uint64_refs: Vec<String> = Vec::new();
    for name in refs_group.datasets().unwrap() {
        let Ok(ref_ds) = refs_group.dataset(&name) else {
            continue;
        };
        let Ok(ref_attrs) = ref_ds.attrs() else {
            continue;
        };
        let class_str = match ref_attrs.get("MATLAB_class") {
            Some(AttrValue::String(s) | AttrValue::AsciiString(s)) => s.as_str(),
            _ => "",
        };
        if class_str == "uint64" {
            uint64_refs.push(name);
        }
    }
    uint64_refs.sort();

    let payload_idx = (metadata[4] - 1) as usize;
    refs_group
        .dataset(&uint64_refs[payload_idx])
        .unwrap()
        .read_u64()
        .unwrap()
}

/// Read the saveobj payload for a string object nested under a group.
fn read_string_saveobj_payload_in_group(file: &File, group_path: &str, ds_name: &str) -> Vec<u64> {
    let group = file.group(group_path).unwrap();
    let ds = group.dataset(ds_name).unwrap();
    let attrs = ds.attrs().unwrap();
    assert_eq!(read_attr_string(&attrs, "MATLAB_class"), "string");
    assert_eq!(read_attr_i64(&attrs, "MATLAB_object_decode"), 3);

    let metadata = ds.read_u32().unwrap();
    assert_eq!(metadata[..4], [MCOS_MAGIC_NUMBER, 2, 1, 1]);

    let subsystem = file.dataset("#subsystem#/MCOS").unwrap();
    let sub_attrs = subsystem.attrs().unwrap();
    assert_eq!(
        read_attr_string(&sub_attrs, "MATLAB_class"),
        "FileWrapper__"
    );
    assert_eq!(read_attr_i64(&sub_attrs, "MATLAB_object_decode"), 3);

    let refs_group = file.group("#refs#").unwrap();
    let mut uint64_refs: Vec<String> = Vec::new();
    for name in refs_group.datasets().unwrap() {
        let Ok(ref_ds) = refs_group.dataset(&name) else {
            continue;
        };
        let Ok(ref_attrs) = ref_ds.attrs() else {
            continue;
        };
        let class_str = match ref_attrs.get("MATLAB_class") {
            Some(AttrValue::String(s) | AttrValue::AsciiString(s)) => s.as_str(),
            _ => "",
        };
        if class_str == "uint64" {
            uint64_refs.push(name);
        }
    }
    uint64_refs.sort();

    let payload_idx = (metadata[4] - 1) as usize;
    refs_group
        .dataset(&uint64_refs[payload_idx])
        .unwrap()
        .read_u64()
        .unwrap()
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

    let file = File::open(&path).unwrap();
    // The userblock size equals base_address (the superblock starts after the userblock).
    assert_eq!(file.superblock().base_address, 512);

    let ds = file.dataset("greeting").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![1, 6]);
    let payload = read_string_saveobj_payload(&file, "greeting");
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

    let file = File::open(&path).unwrap();
    let ds = file.dataset("flags").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![1, 3]);
    let attrs = ds.attrs().unwrap();
    assert_eq!(read_attr_string(&attrs, "MATLAB_class"), "logical");
    assert_eq!(read_attr_i64(&attrs, "MATLAB_int_decode"), 1);
    assert_eq!(ds.read_u8().unwrap(), vec![1, 0, 1]);

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

    let file = File::open(&path).unwrap();
    let ds = file.dataset("cells").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![1, 2]);
    let attrs = ds.attrs().unwrap();
    assert_eq!(read_attr_string(&attrs, "MATLAB_class"), "cell");

    // The dtype should be ObjectReference
    assert_eq!(ds.dtype().unwrap(), DType::ObjectReference);

    // Verify the referenced datasets in #refs#.
    // The cell array writes element references in order:
    //   ref_0000000000000000 = first element (uint8 scalar 1)
    //   ref_0000000000000001 = second element (string "hi" saveobj payload)
    //   ref_0000000000000002 = second element metadata (string object u32 dataset)
    // But the second element is a string, which allocates a saveobj payload ref first,
    // then the string metadata dataset goes into #refs# too.
    // Actually: cell array element refs go first. For "hi", write_value_reference
    // is called, which calls write_string_object, which:
    //   1. Calls write_string_saveobj_payload → allocates next ref (saveobj)
    //   2. Creates the metadata dataset in-place (returned as the cell element ref)
    // Wait, let me re-check. For cell arrays:
    //   write_cell_array → for each element, calls write_value_reference
    //   write_value_reference allocates a ref name and writes the value there
    //   For uint8: ref_0000000000000000 = uint8 dataset with value [1]
    //   For string: write_value_reference allocates ref_0000000000000001 for the string metadata
    //     But write_string_object first calls write_string_saveobj_payload which allocates
    //     ref_0000000000000002 for the saveobj payload
    //     Then the metadata dataset is placed at ref_0000000000000001
    // So: ref_0000000000000000 = uint8, ref_0000000000000001 = string metadata,
    //     ref_0000000000000002 = saveobj payload (uint64)

    let refs_group = file.group("#refs#").unwrap();

    // First ref: uint8 scalar with value 1
    let item0 = refs_group.dataset("ref_0000000000000000").unwrap();
    let item0_attrs = item0.attrs().unwrap();
    assert_eq!(read_attr_string(&item0_attrs, "MATLAB_class"), "uint8");
    assert_eq!(item0.read_u8().unwrap(), vec![1]);

    // Second ref: string metadata dataset
    let item1 = refs_group.dataset("ref_0000000000000001").unwrap();
    let item1_attrs = item1.attrs().unwrap();
    assert_eq!(read_attr_string(&item1_attrs, "MATLAB_class"), "string");
    assert_eq!(read_attr_i64(&item1_attrs, "MATLAB_object_decode"), 3);

    // Third ref: saveobj payload (uint64) for the string "hi"
    let item2 = refs_group.dataset("ref_0000000000000002").unwrap();
    let item2_attrs = item2.attrs().unwrap();
    assert_eq!(read_attr_string(&item2_attrs, "MATLAB_class"), "uint64");
    let payload = item2.read_u64().unwrap();
    assert_eq!(
        decode_matlab_string_saveobj(&payload),
        vec!["hi".to_owned()]
    );

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

    let file = File::open(&path).unwrap();
    let ds = file.dataset("nothing").unwrap();
    let attrs = ds.attrs().unwrap();
    assert_eq!(read_attr_string(&attrs, "MATLAB_class"), "struct");
    assert_eq!(read_attr_u64(&attrs, "MATLAB_empty"), 1);
    assert_eq!(ds.read_u64().unwrap(), vec![0, 0]);

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

    let file = File::open(&path).unwrap();
    let ds = file.dataset("a").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![3, 2]);
    let attrs = ds.attrs().unwrap();
    assert_eq!(read_attr_string(&attrs, "MATLAB_class"), "double");
    assert_eq!(ds.read_f64().unwrap(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    std::fs::remove_file(path).unwrap();
}

#[test]
fn mat_v73_empty_complex_array_uses_complex_dataset_type() {
    let path = temp_path("empty-complex");
    let empty: &[Complex<f64>] = &[];
    let bytes = beve::to_vec_complex_slice(empty);
    beve::beve_slice_to_mat_v73_file(
        &bytes,
        &path,
        RootBinding::NamedVariable("z"),
        &MatV73Options::default(),
    )
    .unwrap();

    let file = File::open(&path).unwrap();
    let ds = file.dataset("z").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![1, 0]);
    let attrs = ds.attrs().unwrap();
    assert_eq!(read_attr_string(&attrs, "MATLAB_class"), "double");
    assert!(!attrs.contains_key("MATLAB_empty"));

    match ds.dtype().unwrap() {
        DType::Compound(fields) => {
            assert_eq!(fields.len(), 2);
            assert_eq!(fields[0].0, "real");
            assert_eq!(fields[1].0, "imag");
        }
        other => panic!("expected compound complex dtype, got {other:?}"),
    }

    std::fs::remove_file(path).unwrap();
}

#[test]
fn matlab_string_fixture_v73_schema() {
    let file = File::open(fixture_path("test_string_v73.mat")).unwrap();

    for (name, payload) in [
        ("string_scalar", vec![3707764736u32, 2, 1, 1, 1, 1]),
        ("string_array", vec![3707764736, 2, 1, 1, 2, 1]),
        ("string_empty", vec![3707764736, 2, 1, 1, 3, 1]),
    ] {
        let ds = file.dataset(name).unwrap();
        assert_eq!(ds.shape().unwrap(), vec![1, 6]);
        let attrs = ds.attrs().unwrap();
        assert_eq!(read_attr_string(&attrs, "MATLAB_class"), "string");
        assert_eq!(read_attr_i64(&attrs, "MATLAB_object_decode"), 3);
        assert_eq!(ds.read_u32().unwrap(), payload);
    }

    let subsystem = file.dataset("#subsystem#/MCOS").unwrap();
    assert_eq!(subsystem.shape().unwrap(), vec![1, 8]);
    let sub_attrs = subsystem.attrs().unwrap();
    assert_eq!(
        read_attr_string(&sub_attrs, "MATLAB_class"),
        "FileWrapper__"
    );
    assert_eq!(read_attr_i64(&sub_attrs, "MATLAB_object_decode"), 3);
}

#[test]
fn matlab_string_fixture_v73_saveobj_payloads() {
    let file = File::open(fixture_path("test_string_v73.mat")).unwrap();

    let scalar = file.dataset("#refs#/c").unwrap();
    let scalar_attrs = scalar.attrs().unwrap();
    assert_eq!(read_attr_string(&scalar_attrs, "MATLAB_class"), "uint64");
    assert_eq!(
        decode_matlab_string_saveobj(&scalar.read_u64().unwrap()),
        vec!["Hello".to_owned()]
    );

    let array = file.dataset("#refs#/d").unwrap();
    let array_attrs = array.attrs().unwrap();
    assert_eq!(read_attr_string(&array_attrs, "MATLAB_class"), "uint64");
    assert_eq!(
        decode_matlab_string_saveobj(&array.read_u64().unwrap()),
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
    let empty_attrs = empty.attrs().unwrap();
    assert_eq!(read_attr_string(&empty_attrs, "MATLAB_class"), "uint64");
    assert_eq!(
        decode_matlab_string_saveobj(&empty.read_u64().unwrap()),
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

    let file = File::open(&path).unwrap();
    let ds = file.dataset("x1_bad").unwrap();
    let attrs = ds.attrs().unwrap();
    assert_eq!(read_attr_string(&attrs, "MATLAB_class"), "uint8");
    assert_eq!(ds.read_u8().unwrap(), vec![7]);

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

    let file = File::open(&path).unwrap();
    let group = file.group("payload").unwrap();
    let group_attrs = group.attrs().unwrap();
    assert_eq!(read_attr_string(&group_attrs, "MATLAB_class"), "struct");

    // Fields are written as fixed-width ASCII strings (AsciiStringArray).
    // The reader returns them as StringArray.
    let fields: Vec<String> = match &group_attrs["MATLAB_fields"] {
        AttrValue::StringArray(arr) => arr.clone(),
        AttrValue::AsciiStringArray(arr) => arr.clone(),
        other => panic!("expected StringArray for MATLAB_fields, got {other:?}"),
    };
    assert_eq!(fields, vec!["answer", "label"]);

    let payload = read_string_saveobj_payload_in_group(&file, "payload", "label");
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

    let file = File::open(&path).unwrap();
    let payload = read_string_saveobj_payload(&file, "greeting");
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

    let file = File::open(&path).unwrap();
    let ds = file.dataset("labels").unwrap();
    let payload = read_string_saveobj_payload(&file, "labels");
    assert_eq!(payload[..6], [1, 2, 2, 1, 4, 5]);
    assert_eq!(
        decode_matlab_string_saveobj(&payload),
        vec!["left".to_owned(), "right".to_owned()]
    );

    drop(ds);
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
