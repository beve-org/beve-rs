use std::process::{Command, Stdio};
use std::io::Write;
use serde::{Serialize, Deserialize};
use beve::{Complex, MatrixLayout, Matrix, to_vec_complex64, to_vec_complex64_slice};

fn glaze_bin() -> Option<String> {
    std::env::var("GLAZE_INTEROP_BIN").ok()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
enum Color { Red, Green, Blue }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Basic {
    i: i32,
    d: f64,
    name: String,
    vs: Vec<f64>,
    vb: Vec<bool>,
    m: std::collections::BTreeMap<String, i32>,
    mi: std::collections::BTreeMap<i32, f64>,
    color: Color,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Inner { x: i32, y: String }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Outer {
    title: String,
    inner: Inner,
    list: Vec<Inner>,
    maybe: Option<Inner>,
}

fn sample_vec_f64() -> Vec<f64> { vec![1.1, -2.25, 3.5] }
fn sample_vec_u32() -> Vec<u32> { vec![1, 2, 1000] }
fn sample_vec_bool() -> Vec<bool> { vec![true,false,true,true,false] }
fn sample_vec_string() -> Vec<String> { vec!["a".into(), "bb".into(), "ccc".into()] }
fn sample_color() -> Color { Color::Green }
fn sample_basic() -> Basic {
    let mut m = std::collections::BTreeMap::new(); m.insert("a".into(),1); m.insert("bb".into(),2);
    let mut mi = std::collections::BTreeMap::new(); mi.insert(5, 3.14); mi.insert(7, 7.42);
    Basic { i: 42, d: -3.125, name: "hello".into(), vs: sample_vec_f64(), vb: sample_vec_bool(), m, mi, color: Color::Green }
}

fn sample_nested() -> Outer {
    Outer {
        title: "outer".into(),
        inner: Inner { x: 7, y: "inner".into() },
        list: vec![Inner { x: 1, y: "a".into() }, Inner { x: 2, y: "b".into() }],
        maybe: Some(Inner { x: -5, y: "maybe".into() }),
    }
}

fn run_glaze(args: &[&str], stdin: Option<&[u8]>) -> Option<(i32, Vec<u8>, Vec<u8>)> {
    let bin = glaze_bin()?;
    let mut cmd = Command::new(bin);
    cmd.args(args).stdin(if stdin.is_some() { Stdio::piped() } else { Stdio::null() }).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn().ok()?;
    if let Some(input) = stdin { child.stdin.as_mut().unwrap().write_all(input).ok()?; }
    let out = child.wait_with_output().ok()?;
    Some((out.status.code().unwrap_or(-1), out.stdout, out.stderr))
}

macro_rules! glaze_to_rust {
    ($case:literal, $ty:ty, $val:expr) => {{
        if glaze_bin().is_none() { return; }
        let (code, out, _err) = run_glaze(&["write", $case], None).unwrap();
        assert_eq!(code, 0, "glaze write {} failed", $case);
        let got: $ty = beve::from_slice(&out).unwrap();
        assert_eq!(got, $val, "glaze->rust mismatch for {}", $case);
    }};
}

macro_rules! rust_to_glaze_ok {
    ($case:literal, $val:expr) => {{
        if glaze_bin().is_none() { return; }
        let bytes = beve::to_vec(&$val).unwrap();
        let (code, out, err) = run_glaze(&["read", $case], Some(&bytes)).unwrap();
        if code != 0 {
            panic!("glaze read {} failed: {}", $case, String::from_utf8_lossy(&err));
        }
        assert_eq!(String::from_utf8_lossy(&out), "OK\n");
    }};
}

#[test]
fn glaze_to_rust_roundtrip_cases() {
    glaze_to_rust!("vec_f64", Vec<f64>, sample_vec_f64());
    glaze_to_rust!("vec_u32", Vec<u32>, sample_vec_u32());
    glaze_to_rust!("vec_bool", Vec<bool>, sample_vec_bool());
    glaze_to_rust!("vec_string", Vec<String>, sample_vec_string());
    glaze_to_rust!("color", Color, sample_color());
    glaze_to_rust!("basic", Basic, sample_basic());
    glaze_to_rust!("nested", Outer, sample_nested());
}

#[test]
fn rust_to_glaze_roundtrip_cases() {
    rust_to_glaze_ok!("vec_f64", sample_vec_f64());
    rust_to_glaze_ok!("vec_u32", sample_vec_u32());
    rust_to_glaze_ok!("vec_bool", sample_vec_bool());
    rust_to_glaze_ok!("vec_string", sample_vec_string());
    rust_to_glaze_ok!("color", sample_color());
    rust_to_glaze_ok!("basic", sample_basic());
    rust_to_glaze_ok!("nested", sample_nested());
}

#[test]
fn interop_complex_numbers() {
    // Glaze -> Rust
    if glaze_bin().is_some() {
        // Single complex<double>
        let (code, out, _err) = run_glaze(&["write", "cplx64"], None).unwrap();
        assert_eq!(code, 0);
        let got: Complex<f64> = beve::from_slice(&out).unwrap();
        assert_eq!(got, Complex { re: 1.5, im: -2.25 });

        // Vector of complex<double>
        let (code, out, _err) = run_glaze(&["write", "vcplx64"], None).unwrap();
        assert_eq!(code, 0);
        let got: Vec<Complex<f64>> = beve::from_slice(&out).unwrap();
        assert_eq!(got, vec![Complex { re: 1.0, im: 2.0 }, Complex { re: -3.0, im: 4.5 }]);
    }

    // Rust -> Glaze
    if glaze_bin().is_some() {
        let bytes = to_vec_complex64(1.5, -2.25);
        let (code, out, err) = run_glaze(&["read", "cplx64"], Some(&bytes)).unwrap();
        assert_eq!(code, 0, "{}", String::from_utf8_lossy(&err));
        assert_eq!(String::from_utf8_lossy(&out), "OK\n");

        let vec = vec![Complex { re: 1.0, im: 2.0 }, Complex { re: -3.0, im: 4.5 }];
        let bytes = to_vec_complex64_slice(&vec);
        let (code, out, err) = run_glaze(&["read", "vcplx64"], Some(&bytes)).unwrap();
        assert_eq!(code, 0, "{}", String::from_utf8_lossy(&err));
        assert_eq!(String::from_utf8_lossy(&out), "OK\n");
    }
}

#[test]
fn interop_matrices_to_json() {
    // Rust -> Glaze JSON: ensure Glaze parses matrices extension to JSON
    if glaze_bin().is_none() { return; }
    // A 2x3 row-major matrix of f64
    let rows = 2u64; let cols = 3u64;
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bytes = beve::fast::to_vec_matrix_f64(beve::fast::MatrixLayoutFast::Right, &[rows, cols], &data);
    let (code, out, err) = run_glaze(&["tojson", "matrix"], Some(&bytes)).unwrap();
    assert_eq!(code, 0, "{}", String::from_utf8_lossy(&err));
    let json = String::from_utf8_lossy(&out);
    // Extents and values must be present and correct
    assert!(json.contains("\"extents\":[2,3]"), "json missing extents: {}", json);
    assert!(json.contains("\"value\":[1,2,3,4,5,6]"), "json missing values: {}", json);
}

#[test]
fn interop_integer_keyed_objects() {
    if glaze_bin().is_none() { return; }
    use std::collections::BTreeMap;

    // Glaze -> Rust (i8)
    let (code, out, _err) = run_glaze(&["write", "mapi8"], None).unwrap();
    assert_eq!(code, 0);
    let got: BTreeMap<i8, i32> = beve::from_slice(&out).unwrap();
    let mut expect = BTreeMap::new(); expect.insert(-1, 10); expect.insert(2, 20);
    assert_eq!(got, expect);

    // u16
    let (code, out, _err) = run_glaze(&["write", "mapu16"], None).unwrap();
    assert_eq!(code, 0);
    let got: BTreeMap<u16, i32> = beve::from_slice(&out).unwrap();
    let mut expect = BTreeMap::new(); expect.insert(1, 10); expect.insert(65535, 20);
    assert_eq!(got, expect);

    // u64
    let (code, out, _err) = run_glaze(&["write", "mapu64"], None).unwrap();
    assert_eq!(code, 0);
    let got: BTreeMap<u64, i32> = beve::from_slice(&out).unwrap();
    let mut expect = BTreeMap::new(); expect.insert(1, 10); expect.insert(1u64<<40, 20);
    assert_eq!(got, expect);

    // Rust -> Glaze validation
    fn expect_i8() -> BTreeMap<i8, i32> { let mut m=BTreeMap::new(); m.insert(-1,10); m.insert(2,20); m }
    let bytes = beve::to_vec(&expect_i8()).unwrap(); let (code, out, err) = run_glaze(&["read","mapi8"], Some(&bytes)).unwrap();
    assert_eq!(code, 0, "{}", String::from_utf8_lossy(&err)); assert_eq!(String::from_utf8_lossy(&out), "OK\n");

    let mut m16=BTreeMap::new(); m16.insert(1u16,10); m16.insert(65535u16,20);
    let (code, out, err) = run_glaze(&["read","mapu16"], Some(&beve::to_vec(&m16).unwrap())).unwrap();
    assert_eq!(code, 0, "{}", String::from_utf8_lossy(&err)); assert_eq!(String::from_utf8_lossy(&out), "OK\n");

    let mut m64=BTreeMap::new(); m64.insert(1u64,10); m64.insert(1u64<<40,20);
    let (code, out, err) = run_glaze(&["read","mapu64"], Some(&beve::to_vec(&m64).unwrap())).unwrap();
    assert_eq!(code, 0, "{}", String::from_utf8_lossy(&err)); assert_eq!(String::from_utf8_lossy(&out), "OK\n");
}
