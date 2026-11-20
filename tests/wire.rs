//! `beve::wire`: the public framing pieces.
//!
//! The module promises one thing: a document assembled from its pieces is
//! byte-identical to the same document encoded whole. Each test pins a piece
//! against what beve's own encoders write for it. The module doctest covers the
//! motivating case, a complex payload appended after its header.

use beve::{BeveTypedSlice, Complex, wire};
use half::{bf16, f16};
use serde::Serialize;

/// Values on both sides of every SIZE width boundary, and the extremes.
const SIZES: &[u64] = &[
    0,
    1,
    63,
    64,
    16_383,
    16_384,
    (1 << 30) - 1,
    1 << 30,
    u32::MAX as u64,
    wire::MAX_SIZE,
];

/// Element counts spanning the 1/2/4-byte SIZE widths, for headers whose
/// reference encoding needs a real payload behind it.
const LENGTHS: &[usize] = &[0, 1, 63, 64, 16_383, 16_384];

#[test]
fn size_round_trips_at_every_width() {
    for &n in SIZES {
        let mut out = Vec::new();
        wire::write_size(&mut out, n);
        assert_eq!(out.len(), wire::size_encoded_len(n), "width of {n}");

        let mut pos = 0;
        assert_eq!(wire::read_size(&out, &mut pos).unwrap(), n);
        assert_eq!(pos, out.len(), "read consumed all of {n}");

        let mut pos = 0;
        assert!(
            matches!(
                wire::read_size(&out[..out.len() - 1], &mut pos),
                Err(beve::Error::Eof)
            ),
            "truncated {n}"
        );
    }
}

#[test]
#[should_panic(expected = "exceeds the 62-bit maximum")]
fn a_size_past_62_bits_panics() {
    wire::write_size(&mut Vec::new(), wire::MAX_SIZE + 1);
}

#[test]
fn generic_array_header_matches_the_encoder() {
    for &n in LENGTHS {
        // Alternating types keep the encoder off the typed-array path.
        let values: Vec<beve::Value> = (0..n)
            .map(|i| {
                if i % 2 == 0 {
                    beve::Value::Null
                } else {
                    beve::Value::Bool(true)
                }
            })
            .collect();
        let mut ours = Vec::new();
        wire::write_generic_array_header(&mut ours, n as u64);
        for v in &values {
            beve::to_writer(&mut ours, v).unwrap();
        }
        assert_eq!(ours, beve::to_vec(&values).unwrap(), "len={n}");
    }
}

#[test]
fn object_header_and_keys_match_the_encoder() {
    #[derive(Serialize)]
    struct Doc {
        a: u8,
        long_key_name_past_the_one_byte_size_width_of_sixty_three_bytes_x: (),
    }
    let mut ours = Vec::new();
    wire::write_object_header(&mut ours, 2);
    wire::write_object_key(&mut ours, "a");
    beve::to_writer(&mut ours, &7u8).unwrap();
    wire::write_object_key(
        &mut ours,
        "long_key_name_past_the_one_byte_size_width_of_sixty_three_bytes_x",
    );
    beve::to_writer(&mut ours, &()).unwrap();
    let doc = Doc {
        a: 7,
        long_key_name_past_the_one_byte_size_width_of_sixty_three_bytes_x: (),
    };
    assert_eq!(ours, beve::to_vec(&doc).unwrap());
}

/// Header then payload, against `to_vec_typed_slice` and the complex
/// counterpart, for one scalar type at every length.
fn check_array_headers<T: BeveTypedSlice + Copy + Default>() {
    for &n in LENGTHS {
        let values = vec![T::default(); n];
        let mut ours = Vec::new();
        wire::write_typed_array_header::<T>(&mut ours, n as u64);
        for v in &values {
            T::write_one_le(v, &mut ours);
        }
        assert_eq!(
            ours,
            beve::to_vec_typed_slice(&values),
            "typed {} len={n}",
            std::any::type_name::<T>()
        );

        let complex = vec![
            Complex {
                re: T::default(),
                im: T::default()
            };
            n
        ];
        let mut ours = Vec::new();
        wire::write_complex_array_header::<T>(&mut ours, n as u64);
        for c in &complex {
            T::write_one_le(&c.re, &mut ours);
            T::write_one_le(&c.im, &mut ours);
        }
        assert_eq!(
            ours,
            beve::to_vec_complex_slice(&complex),
            "complex {} len={n}",
            std::any::type_name::<T>()
        );
    }
}

#[test]
fn array_headers_match_the_encoders_for_every_scalar() {
    check_array_headers::<i8>();
    check_array_headers::<i16>();
    check_array_headers::<i32>();
    check_array_headers::<i64>();
    check_array_headers::<i128>();
    check_array_headers::<u8>();
    check_array_headers::<u16>();
    check_array_headers::<u32>();
    check_array_headers::<u64>();
    check_array_headers::<u128>();
    check_array_headers::<f32>();
    check_array_headers::<f64>();
    // The two half widths, where the complex width is not `1 << byte_code`.
    check_array_headers::<f16>();
    check_array_headers::<bf16>();
}
