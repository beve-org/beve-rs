#![deny(warnings)]

use beve::Complex;
use serde::{Deserialize, Serialize};

// Minimal SIZE encoder for tests (matches src/size.rs behavior)
fn write_size_for_test(mut n: u64, out: &mut Vec<u8>) {
    if n < (1 << 6) {
        out.push((n as u8) << 2);
        return;
    }
    if n < (1 << 14) {
        out.push((((n & 0x3f) as u8) << 2) | 0b01);
        n >>= 6;
        out.push(n as u8);
        return;
    }
    if n < (1 << 30) {
        out.push((((n & 0x3f) as u8) << 2) | 0b10);
        n >>= 6;
        out.push(n as u8);
        out.push((n >> 8) as u8);
        out.push((n >> 16) as u8);
        return;
    }
    out.push((((n & 0x3f) as u8) << 2) | 0b11);
    n >>= 6;
    for i in 0..7 {
        out.push((n >> (i * 8)) as u8);
    }
}

#[test]
fn fast_numeric_vs_serde() {
    macro_rules! check {
        ($t:ty, $val:expr) => {
            let v: Vec<$t> = $val;
            let fast = beve::to_vec_typed_slice(&v);
            let via_serde = beve::to_vec(&v).unwrap();
            assert_eq!(
                fast,
                via_serde,
                "mismatch for {}",
                std::any::type_name::<$t>()
            );
        };
    }
    check!(u8, vec![1, 2, 3, 4, 5]);
    check!(u16, vec![1, 512, 1024]);
    check!(u32, vec![1, 2, 3, 4]);
    check!(u64, vec![9, 10, 11]);
    check!(i8, vec![-1, 0, 1]);
    check!(i16, vec![-1, 0, 1, 1024]);
    check!(i32, vec![-1, 0, 1, 1_000_000]);
    check!(i64, vec![-1, 0, 1]);
    check!(f32, vec![1.0, -2.5, 3.25]);
    check!(f64, vec![1.0, -2.5, 3.25]);
}

#[test]
fn fast_bool_vs_serde() {
    let v = vec![
        true, false, true, true, false, true, false, true, false, true,
    ];
    let fast = beve::to_vec_bool_slice(&v);
    let via_serde = beve::to_vec(&v).unwrap();
    assert_eq!(fast, via_serde);

    // Ensure deserialization from fast-path bytes matches original
    let back: Vec<bool> = beve::from_slice(&fast).unwrap();
    assert_eq!(v, back);
}

#[test]
fn bool_packing_uses_lsb_order() {
    let bytes = beve::to_vec_bool_slice(&[true, false, true]);
    assert_eq!(bytes, vec![0x1c, 0x0c, 0x05]);
}

#[test]
fn fast_string_vs_serde() {
    let v = vec!["a".to_string(), "bb".to_string(), "ccc".to_string()];
    let fast = beve::to_vec_string_slice(&v);
    let via_serde = beve::to_vec(&v).unwrap();
    assert_eq!(fast, via_serde);

    let v2: Vec<&str> = vec!["x", "yy", "zzz"];
    let fast2 = beve::to_vec_str_slice(&v2);
    let via_serde2 = beve::to_vec(&v2).unwrap();
    assert_eq!(fast2, via_serde2);
}

#[test]
fn fast_bool_in_struct_vs_serde() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct S {
        flags: Vec<bool>,
    }

    let v = vec![true, false, true, true, false, false, true, false, true];

    // Build expected struct bytes manually using fast-path for the field value
    let mut expected = Vec::new();
    expected.push(0x03); // string-keyed object header
    write_size_for_test(1, &mut expected); // one field
    let key = "flags";
    write_size_for_test(key.len() as u64, &mut expected);
    expected.extend_from_slice(key.as_bytes());
    expected.extend_from_slice(&beve::to_vec_bool_slice(&v));

    // Compare to serde-produced bytes
    let s = S { flags: v.clone() };
    let via_serde = beve::to_vec(&s).unwrap();
    assert_eq!(expected, via_serde);

    // And ensure from_slice can read the same payload back
    let back: S = beve::from_slice(&expected).unwrap();
    assert_eq!(back, s);
}

#[test]
fn fast_numeric_in_struct_vs_serde() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct S<T> {
        data: Vec<T>,
    }

    macro_rules! check {
        ($t:ty, $val:expr) => {
            let v: Vec<$t> = $val;
            // Build expected object with typed array as the field value
            let mut expected = Vec::new();
            expected.push(0x03); // string-keyed object header
            write_size_for_test(1, &mut expected); // one field
            let key = "data";
            write_size_for_test(key.len() as u64, &mut expected);
            expected.extend_from_slice(key.as_bytes());
            expected.extend_from_slice(&beve::to_vec_typed_slice(&v));

            let s = S { data: v.clone() };
            let via_serde = beve::to_vec(&s).unwrap();
            assert_eq!(
                expected,
                via_serde,
                "mismatch for {}",
                std::any::type_name::<$t>()
            );

            // Deserialization roundtrip
            let back: S<$t> = beve::from_slice(&expected).unwrap();
            assert_eq!(back, s);
        };
    }

    check!(u8, vec![1, 2, 3, 4, 5]);
    check!(u16, vec![1, 512, 1024]);
    check!(u32, vec![1, 2, 3, 4]);
    check!(u64, vec![9, 10, 11]);
    check!(i8, vec![-1, 0, 1]);
    check!(i16, vec![-1, 0, 1, 1024]);
    check!(i32, vec![-1, 0, 1, 1_000_000]);
    check!(i64, vec![-1, 0, 1]);
    check!(f32, vec![1.0, -2.5, 3.25]);
    check!(f64, vec![1.0, -2.5, 3.25]);
}

#[test]
fn read_complex_slice_roundtrips_writer() {
    macro_rules! check {
        ($t:ty, $val:expr) => {
            let v: Vec<Complex<$t>> = $val;
            let bytes = beve::to_vec_complex_slice(&v);

            // Bulk reader round-trips the bulk writer.
            let bulk = beve::read_complex_slice::<$t>(&bytes).unwrap();
            assert_eq!(
                bulk,
                v,
                "bulk read mismatch for {}",
                std::any::type_name::<$t>()
            );

            // ...and agrees with the generic serde decode of the same bytes.
            let via_serde: Vec<Complex<$t>> = beve::from_slice(&bytes).unwrap();
            assert_eq!(
                bulk,
                via_serde,
                "bulk vs serde mismatch for {}",
                std::any::type_name::<$t>()
            );
        };
    }

    check!(
        f32,
        vec![
            Complex { re: 1.0, im: -2.0 },
            Complex { re: 3.5, im: 4.25 },
            Complex { re: -0.0, im: 1e9 },
        ]
    );
    check!(
        f64,
        vec![Complex { re: 1.0, im: -2.0 }, Complex { re: 3.5, im: 4.25 },]
    );
    check!(
        i32,
        vec![
            Complex { re: -1, im: 0 },
            Complex {
                re: 1,
                im: 1_000_000
            },
        ]
    );
    check!(
        u16,
        vec![Complex { re: 0, im: 65535 }, Complex { re: 7, im: 9 }]
    );
    // Empty array: header only, no payload.
    check!(f64, vec![]);
}

#[test]
fn read_complex_slice_rejects_mismatched_and_truncated() {
    let v = vec![
        Complex {
            re: 1.0f64,
            im: 2.0,
        },
        Complex { re: 3.0, im: 4.0 },
    ];
    let bytes = beve::to_vec_complex_slice(&v);

    // Wrong element width (decoding an f64 array as f32) is a type mismatch, not
    // a silent reinterpretation.
    assert!(beve::read_complex_slice::<f32>(&bytes).is_err());

    // A truncated payload is an EOF, not an out-of-bounds read.
    assert!(beve::read_complex_slice::<f64>(&bytes[..bytes.len() - 1]).is_err());

    // A non-complex value (a plain typed numeric array) is rejected.
    let typed = beve::to_vec_typed_slice(&[1.0f64, 2.0, 3.0]);
    assert!(beve::read_complex_slice::<f64>(&typed).is_err());

    // Same width, different class: an i64 complex array decoded as u64 trips the
    // class half of the type check independently of byte width (both byte_code 3).
    let signed = beve::to_vec_complex_slice(&[Complex { re: -1i64, im: 2 }]);
    assert!(beve::read_complex_slice::<u64>(&signed).is_err());

    // A scalar (non-array) complex is rejected: serde encodes a bare `Complex`
    // as a complex extension with the array flag clear.
    let scalar = beve::to_vec(&Complex {
        re: 1.0f64,
        im: 2.0,
    })
    .unwrap();
    assert!(beve::read_complex_slice::<f64>(&scalar).is_err());

    // Empty input: there is not even a header byte to read.
    assert!(beve::read_complex_slice::<f64>(&[]).is_err());
}

#[test]
fn fast_string_in_struct_vs_serde() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct S {
        labels: Vec<String>,
    }

    let v = vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()];

    let mut expected = Vec::new();
    expected.push(0x03); // string-keyed object header
    write_size_for_test(1, &mut expected); // one field
    let key = "labels";
    write_size_for_test(key.len() as u64, &mut expected);
    expected.extend_from_slice(key.as_bytes());
    expected.extend_from_slice(&beve::to_vec_string_slice(&v));

    let s = S { labels: v.clone() };
    let via_serde = beve::to_vec(&s).unwrap();
    assert_eq!(expected, via_serde);

    let back: S = beve::from_slice(&expected).unwrap();
    assert_eq!(back, s);
}
