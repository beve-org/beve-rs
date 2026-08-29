//! Decode throughput of a struct array field: bulk `#[serde(with = ...)]` vs the
//! unannotated element-wise `Vec<T>`. Both decode the *same* wire bytes (the
//! encodings are identical), so this isolates the decode-path difference.

use benchit::{Bench, Throughput};
use beve::Complex;
use serde::{Deserialize, Serialize};
use std::hint::black_box;

const N: usize = 1_000_000;

#[derive(Serialize, Deserialize)]
struct ComplexBulk {
    #[serde(with = "beve::complex_array::f32")]
    iq: Vec<Complex<f32>>,
}
#[derive(Serialize, Deserialize)]
struct ComplexPlain {
    iq: Vec<Complex<f32>>,
}

#[derive(Serialize, Deserialize)]
struct TypedBulk {
    #[serde(with = "beve::typed::f64")]
    x: Vec<f64>,
}
#[derive(Serialize, Deserialize)]
struct TypedPlain {
    x: Vec<f64>,
}

fn main() {
    let mut bench = Bench::from_args();

    let iq: Vec<Complex<f32>> = (0..N)
        .map(|i| Complex {
            re: i as f32,
            im: -(i as f32),
        })
        .collect();
    let cbytes = beve::to_vec(&ComplexPlain { iq }).unwrap();

    let mut g = bench.group("serde_with/complex_f32_1M");
    g.throughput(Throughput::Bytes((N * 8) as u64)); // 2 * f32
    g.bench("with (bulk)", |b| {
        b.iter(|| {
            let d: ComplexBulk = beve::from_slice(black_box(&cbytes)).unwrap();
            black_box(d);
        })
    });
    g.bench("plain (element-wise)", |b| {
        b.iter(|| {
            let d: ComplexPlain = beve::from_slice(black_box(&cbytes)).unwrap();
            black_box(d);
        })
    });
    g.finish();

    let x: Vec<f64> = (0..N).map(|i| i as f64).collect();
    let tbytes = beve::to_vec(&TypedPlain { x }).unwrap();

    let mut g = bench.group("serde_with/typed_f64_1M");
    g.throughput(Throughput::Bytes((N * 8) as u64));
    g.bench("with (bulk)", |b| {
        b.iter(|| {
            let d: TypedBulk = beve::from_slice(black_box(&tbytes)).unwrap();
            black_box(d);
        })
    });
    g.bench("plain (element-wise)", |b| {
        b.iter(|| {
            let d: TypedPlain = beve::from_slice(black_box(&tbytes)).unwrap();
            black_box(d);
        })
    });
    g.finish();
}
