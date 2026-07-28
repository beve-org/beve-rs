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
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| u16::from_le_bytes(*chunk))
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

// ---------------------------------------------------------------------------
// Integer complex arrays
//
// MATLAB has first-class complex integer array types, so a complex `int16`
// payload converts to a complex `int16` MATLAB array rather than being widened
// to `single`. These pin the on-disk contract (compound field types, class
// string, and bytes per element), not just the round trip: a file this crate
// reads back happily can still be one MATLAB refuses.
// ---------------------------------------------------------------------------

/// Assert a dataset is a MATLAB complex array of `class` whose compound fields
/// are `real`/`imag` of `field_dtype`, and return its raw payload bytes.
fn assert_complex_dataset(file: &File, name: &str, class: &str, field_dtype: DType) -> Vec<u8> {
    let ds = file.dataset(name).unwrap();
    let attrs = ds.attrs().unwrap();
    assert_eq!(read_attr_string(&attrs, "MATLAB_class"), class);
    match ds.dtype().unwrap() {
        DType::Compound(fields) => {
            assert_eq!(fields.len(), 2, "{name}: expected two compound fields");
            assert_eq!(fields[0].0, "real");
            assert_eq!(fields[0].1, field_dtype, "{name}: real field type");
            assert_eq!(fields[1].0, "imag");
            assert_eq!(fields[1].1, field_dtype, "{name}: imag field type");
        }
        other => panic!("{name}: expected compound complex dtype, got {other:?}"),
    }
    ds.read_raw().unwrap()
}

/// Decode a complex `int16` dataset payload into `(re, im)` pairs.
fn decode_complex_i16(raw: &[u8]) -> Vec<(i16, i16)> {
    raw.as_chunks::<4>()
        .0
        .iter()
        .map(|c| {
            let (re, im) = c.split_at(2);
            (
                i16::from_le_bytes(re.try_into().unwrap()),
                i16::from_le_bytes(im.try_into().unwrap()),
            )
        })
        .collect()
}

fn convert_named(bytes: &[u8], path: &PathBuf, name: &str, options: &MatV73Options) -> File {
    beve::beve_slice_to_mat_v73_file(bytes, path, RootBinding::NamedVariable(name), options)
        .unwrap();
    File::open(path).unwrap()
}

#[test]
fn mat_v73_complex_i16_array_is_int16_not_widened() {
    let path = temp_path("complex-i16-array");
    let samples = [
        Complex { re: 1i16, im: -2 },
        Complex { re: 300, im: -400 },
        Complex { re: 0, im: 7 },
    ];
    let bytes = beve::to_vec_complex_slice(&samples);
    let file = convert_named(&bytes, &path, "iq", &MatV73Options::default());

    let raw = assert_complex_dataset(&file, "iq", "int16", DType::I16);
    // The whole point of the integer path: 4 bytes per complex element, half
    // what the same data costs once widened to `single`.
    assert_eq!(raw.len(), samples.len() * 4);
    let decoded = decode_complex_i16(&raw);
    assert_eq!(decoded, vec![(1, -2), (300, -400), (0, 7)]);

    let ds = file.dataset("iq").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![1, 3]);

    std::fs::remove_file(path).unwrap();
}

#[test]
fn mat_v73_complex_i16_scalar_is_int16() {
    let path = temp_path("complex-i16-scalar");
    let bytes = beve::to_vec(&Complex { re: -5i16, im: 9 }).unwrap();
    let file = convert_named(&bytes, &path, "s", &MatV73Options::default());

    let raw = assert_complex_dataset(&file, "s", "int16", DType::I16);
    assert_eq!(raw, vec![0xFB, 0xFF, 0x09, 0x00]);
    assert_eq!(file.dataset("s").unwrap().shape().unwrap(), vec![1, 1]);

    std::fs::remove_file(path).unwrap();
}

#[test]
fn mat_v73_complex_i16_preserves_extreme_values() {
    let path = temp_path("complex-i16-edges");
    let samples = [
        Complex {
            re: i16::MIN,
            im: i16::MAX,
        },
        Complex {
            re: i16::MAX,
            im: i16::MIN,
        },
        Complex { re: 0i16, im: 0 },
    ];
    let bytes = beve::to_vec_complex_slice(&samples);
    let file = convert_named(&bytes, &path, "edges", &MatV73Options::default());

    let raw = assert_complex_dataset(&file, "edges", "int16", DType::I16);
    let decoded = decode_complex_i16(&raw);
    assert_eq!(
        decoded,
        vec![(i16::MIN, i16::MAX), (i16::MAX, i16::MIN), (0, 0),]
    );

    std::fs::remove_file(path).unwrap();
}

#[test]
fn mat_v73_complex_integer_widths_map_to_matching_matlab_classes() {
    // Values, not just widths: the extremes are what catch a sign-extension
    // slip on the signed types and a lost high bit on the unsigned ones, and
    // the asymmetric middle pair catches a real/imag swap.
    macro_rules! check_width {
        ($ty:ty, $class:literal, $dtype:expr, $label:literal) => {{
            let path = temp_path(concat!("complex-", $label));
            let samples = [
                Complex {
                    re: <$ty>::MIN,
                    im: <$ty>::MAX,
                },
                Complex {
                    re: <$ty>::MAX,
                    im: <$ty>::MIN,
                },
                Complex {
                    re: 1 as $ty,
                    im: 2 as $ty,
                },
            ];
            let bytes = beve::to_vec_complex_slice(&samples);
            let file = convert_named(&bytes, &path, "v", &MatV73Options::default());
            let raw = assert_complex_dataset(&file, "v", $class, $dtype);

            const ELEM: usize = core::mem::size_of::<$ty>();
            assert_eq!(raw.len(), samples.len() * ELEM * 2);
            let decoded: Vec<$ty> = raw
                .as_chunks::<ELEM>()
                .0
                .iter()
                .map(|c| <$ty>::from_le_bytes(*c))
                .collect();
            assert_eq!(
                decoded,
                vec![
                    <$ty>::MIN,
                    <$ty>::MAX,
                    <$ty>::MAX,
                    <$ty>::MIN,
                    1 as $ty,
                    2 as $ty,
                ],
                concat!($label, ": component values or order are wrong")
            );
            std::fs::remove_file(path).unwrap();
        }};
    }

    check_width!(i8, "int8", DType::I8, "i8");
    check_width!(i16, "int16", DType::I16, "i16");
    check_width!(i32, "int32", DType::I32, "i32");
    check_width!(i64, "int64", DType::I64, "i64");
    check_width!(u8, "uint8", DType::U8, "u8");
    check_width!(u16, "uint16", DType::U16, "u16");
    check_width!(u32, "uint32", DType::U32, "u32");
    check_width!(u64, "uint64", DType::U64, "u64");
}

#[test]
fn mat_v73_float_complex_classes_are_unchanged() {
    let path = temp_path("complex-f32-class");
    let bytes = beve::to_vec_complex_slice(&[Complex {
        re: 1.5f32,
        im: -2.5,
    }]);
    let file = convert_named(&bytes, &path, "c", &MatV73Options::default());
    assert_complex_dataset(&file, "c", "single", DType::F32);
    std::fs::remove_file(path).unwrap();

    let path = temp_path("complex-f64-class");
    let bytes = beve::to_vec_complex_slice(&[Complex {
        re: 1.5f64,
        im: -2.5,
    }]);
    let file = convert_named(&bytes, &path, "c", &MatV73Options::default());
    assert_complex_dataset(&file, "c", "double", DType::F64);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn mat_v73_empty_complex_i16_array_keeps_int16_class() {
    let path = temp_path("empty-complex-i16");
    let empty: &[Complex<i16>] = &[];
    let bytes = beve::to_vec_complex_slice(empty);
    let file = convert_named(&bytes, &path, "z", &MatV73Options::default());

    let raw = assert_complex_dataset(&file, "z", "int16", DType::I16);
    assert!(raw.is_empty());
    let ds = file.dataset("z").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![1, 0]);
    assert!(!ds.attrs().unwrap().contains_key("MATLAB_empty"));

    std::fs::remove_file(path).unwrap();
}

#[test]
fn mat_v73_row_major_complex_i16_matrix_reorders_to_column_major() {
    let path = temp_path("complex-i16-matrix");
    let matrix = beve::MatrixOwned {
        layout: beve::MatrixLayout::Right,
        extents: vec![2, 3],
        data: vec![
            Complex { re: 1i16, im: -1 },
            Complex { re: 2, im: -2 },
            Complex { re: 3, im: -3 },
            Complex { re: 4, im: -4 },
            Complex { re: 5, im: -5 },
            Complex { re: 6, im: -6 },
        ],
    };
    let bytes = beve::to_vec(&matrix).unwrap();
    let file = convert_named(&bytes, &path, "m", &MatV73Options::default());

    let raw = assert_complex_dataset(&file, "m", "int16", DType::I16);
    let decoded: Vec<i16> = raw
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| i16::from_le_bytes(*c))
        .collect();
    // Row-major [1..6] over a 2x3 becomes column-major [1, 4, 2, 5, 3, 6],
    // each element still carrying its own imaginary part.
    assert_eq!(decoded, vec![1, -1, 4, -4, 2, -2, 5, -5, 3, -3, 6, -6]);
    assert_eq!(file.dataset("m").unwrap().shape().unwrap(), vec![3, 2]);

    std::fs::remove_file(path).unwrap();
}

#[test]
fn mat_v73_rejects_128_bit_complex() {
    let path = temp_path("complex-i128");
    let bytes = beve::to_vec_complex_slice(&[Complex { re: 1i128, im: 2 }]);
    let err = beve::beve_slice_to_mat_v73_file(
        &bytes,
        &path,
        RootBinding::NamedVariable("wide"),
        &MatV73Options::default(),
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("128-bit integer complex"),
        "unexpected error: {err}"
    );
    assert!(!path.exists());
}

#[test]
fn mat_v73_half_precision_complex_needs_lossy_widening() {
    let samples = [Complex {
        re: half::f16::from_f32(1.5),
        im: half::f16::from_f32(-2.5),
    }];
    let bytes = beve::to_vec_complex_slice(&samples);

    let path = temp_path("complex-f16-strict");
    let err = beve::beve_slice_to_mat_v73_file(
        &bytes,
        &path,
        RootBinding::NamedVariable("h"),
        &MatV73Options::default(),
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("f16 complex"),
        "error should name the encoding, got: {err}"
    );
    assert!(!path.exists());

    let path = temp_path("complex-f16-lossy");
    let options = MatV73Options {
        unsupported_policy: UnsupportedPolicy::LossyNumericWidening,
        ..MatV73Options::default()
    };
    let file = convert_named(&bytes, &path, "h", &options);
    let raw = assert_complex_dataset(&file, "h", "single", DType::F32);
    let decoded: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    assert_eq!(decoded, vec![1.5, -2.5]);

    std::fs::remove_file(path).unwrap();
}

#[test]
fn mat_v73_bf16_complex_widens_under_lossy_policy() {
    // The f16 sibling is covered above; bf16 takes the other branch of the
    // half-precision reader, and a swapped branch would decode 1.5 as 1.9375.
    let samples = [Complex {
        re: half::bf16::from_f32(1.5),
        im: half::bf16::from_f32(-2.5),
    }];
    let bytes = beve::to_vec_complex_slice(&samples);

    let path = temp_path("complex-bf16-strict");
    let err = beve::beve_slice_to_mat_v73_file(
        &bytes,
        &path,
        RootBinding::NamedVariable("h"),
        &MatV73Options::default(),
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("bf16 complex"),
        "error should name the encoding, got: {err}"
    );
    assert!(!path.exists());

    let path = temp_path("complex-bf16-lossy");
    let options = MatV73Options {
        unsupported_policy: UnsupportedPolicy::LossyNumericWidening,
        ..MatV73Options::default()
    };
    let file = convert_named(&bytes, &path, "h", &options);
    let raw = assert_complex_dataset(&file, "h", "single", DType::F32);
    let decoded: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    assert_eq!(decoded, vec![1.5, -2.5]);

    std::fs::remove_file(path).unwrap();
}

#[test]
fn mat_v73_complex_i16_nested_in_struct_and_cell() {
    // The realistic shape: complex arrays arrive as struct fields and cell
    // elements, not as the whole document. Both route through a different
    // MatBuilder scope than a bare root value.
    let path = temp_path("complex-i16-nested");

    #[derive(serde::Serialize)]
    struct Inner {
        samples: Vec<Complex<i16>>,
    }
    #[derive(serde::Serialize)]
    struct Outer {
        inner: Inner,
    }

    let doc = Outer {
        inner: Inner {
            samples: vec![Complex { re: -7i16, im: 11 }, Complex { re: 12, im: -13 }],
        },
    };
    let bytes = beve::to_vec(&doc).unwrap();
    let file = convert_named(&bytes, &path, "root", &MatV73Options::default());

    let ds = file
        .group("root")
        .unwrap()
        .group("inner")
        .unwrap()
        .dataset("samples")
        .unwrap();
    let attrs = ds.attrs().unwrap();
    assert_eq!(read_attr_string(&attrs, "MATLAB_class"), "int16");
    match ds.dtype().unwrap() {
        DType::Compound(fields) => {
            assert_eq!(fields[0].1, DType::I16);
            assert_eq!(fields[1].1, DType::I16);
        }
        other => panic!("expected compound complex dtype, got {other:?}"),
    }
    let decoded: Vec<i16> = ds
        .read_raw()
        .unwrap()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| i16::from_le_bytes(*c))
        .collect();
    assert_eq!(decoded, vec![-7, 11, 12, -13]);

    std::fs::remove_file(path).unwrap();
}

#[test]
fn mat_v73_complex_i16_3d_matrix_uses_general_reorder_path() {
    // A 2-D matrix hits the reorder fast path; only rank 3 and above exercise
    // the general stride walk, which had no complex coverage.
    let path = temp_path("complex-i16-3d");
    let data: Vec<Complex<i16>> = (0..12)
        .map(|i| Complex {
            re: i as i16,
            im: -(i as i16),
        })
        .collect();
    let matrix = beve::MatrixOwned {
        layout: beve::MatrixLayout::Right,
        extents: vec![2, 3, 2],
        data: data.clone(),
    };
    let bytes = beve::to_vec(&matrix).unwrap();
    let file = convert_named(&bytes, &path, "m", &MatV73Options::default());

    let raw = assert_complex_dataset(&file, "m", "int16", DType::I16);
    let decoded = decode_complex_i16(&raw);

    // Independently recompute the row-major -> column-major permutation for
    // extents [2, 3, 2] rather than trusting the converter's own ordering.
    let extents = [2usize, 3, 2];
    let mut expected = Vec::with_capacity(12);
    for col_pos in 0..12usize {
        let mut remaining = col_pos;
        let mut row_index = 0usize;
        let mut row_stride = [1usize; 3];
        for axis in (0..2).rev() {
            row_stride[axis] = row_stride[axis + 1] * extents[axis + 1];
        }
        for (axis, &extent) in extents.iter().enumerate() {
            let coord = remaining % extent;
            remaining /= extent;
            row_index += coord * row_stride[axis];
        }
        expected.push((data[row_index].re, data[row_index].im));
    }
    assert_eq!(decoded, expected);
    // Every element still carries its own imaginary part after the walk.
    assert!(decoded.iter().all(|&(re, im)| im == -re));

    std::fs::remove_file(path).unwrap();
}

#[test]
fn mat_v73_rejects_unknown_complex_class_without_panicking() {
    // class 3 (bool/string) is unreachable from this crate's encoder but
    // reachable from hostile bytes, so it must be a clean Err.
    let path = temp_path("complex-bad-class");
    // 0x1E = complex extension; second byte: byte_code 0, class 3, is_array.
    let bytes = vec![0x1E, 0b0001_1001, 0x00];
    let err = beve::beve_slice_to_mat_v73_file(
        &bytes,
        &path,
        RootBinding::NamedVariable("bad"),
        &MatV73Options::default(),
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("unsupported complex element type"),
        "unexpected error: {err}"
    );
    assert!(!path.exists());
}

#[test]
fn mat_v73_oversized_complex_length_errors_instead_of_allocating() {
    // A tiny payload declaring a huge element count must be rejected by the
    // bounds check, not turned into an allocation request. Covers the
    // fixed-width readers and the half-precision reader, which reach the
    // length through different code.
    for (label, byte_code, policy) in [
        ("i16", 0b0010_1001u8, UnsupportedPolicy::Error),
        ("f32", 0b0100_0001u8, UnsupportedPolicy::Error),
        (
            "f16",
            0b0010_0001u8,
            UnsupportedPolicy::LossyNumericWidening,
        ),
    ] {
        // SIZE 8-byte form (low 2 bits = 3, value in the remaining 62 bits)
        // encoding 2^61 elements, followed by no payload at all.
        let mut bytes = vec![0x1E, byte_code, 0x03];
        bytes.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0x80]);

        let path = temp_path(&format!("complex-oversize-{label}"));
        let options = MatV73Options {
            unsupported_policy: policy,
            ..MatV73Options::default()
        };
        let err = beve::beve_slice_to_mat_v73_file(
            &bytes,
            &path,
            RootBinding::NamedVariable("huge"),
            &options,
        )
        .unwrap_err();
        // The exact message differs by path; what matters is Err, not panic.
        assert!(
            !err.to_string().is_empty(),
            "{label}: expected a non-empty error"
        );
        assert!(!path.exists(), "{label}: no file should be produced");
    }
}
