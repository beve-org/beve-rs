//! Throughput of the streaming bulk slice readers vs. the serde streaming path.
//!
//! `read_typed_slice_from_reader` / `read_complex_slice_from_reader` drain a typed
//! or complex array's payload straight into the result `Vec` in bulk, where
//! `from_reader_streaming::<Vec<T>>` visits every element through serde. The
//! in-memory `read_typed_slice` / `from_slice` rows bound the range (one bulk
//! copy vs. per-element serde).

use std::io::Cursor;

use beve::Complex;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

fn bench_typed_f64(c: &mut Criterion) {
    let v: Vec<f64> = (0..1_000_000).map(|i| i as f64 * 0.5).collect();
    let bytes = beve::to_vec_typed_slice(&v);

    let mut g = c.benchmark_group("bulk_decode/vec_f64_1M");
    g.throughput(Throughput::Bytes((v.len() * 8) as u64));

    g.bench_function("read_typed_slice (in-mem bulk)", |b| {
        b.iter(|| black_box(beve::read_typed_slice::<f64>(black_box(&bytes)).unwrap()))
    });
    g.bench_function("read_typed_slice_from_reader (stream bulk)", |b| {
        b.iter(|| {
            black_box(
                beve::read_typed_slice_from_reader::<f64, _>(Cursor::new(black_box(&bytes)))
                    .unwrap(),
            )
        })
    });
    g.bench_function("from_reader_streaming (serde stream)", |b| {
        b.iter(|| {
            let d: Vec<f64> =
                beve::from_reader_streaming(Cursor::new(black_box(&bytes))).unwrap();
            black_box(d);
        })
    });
    g.bench_function("from_slice (serde in-mem)", |b| {
        b.iter(|| {
            let d: Vec<f64> = beve::from_slice(black_box(&bytes)).unwrap();
            black_box(d);
        })
    });
    g.finish();
}

fn bench_complex_f32(c: &mut Criterion) {
    // IQ-style buffer: a large run of Complex<f32>.
    let v: Vec<Complex<f32>> = (0..1_000_000)
        .map(|i| Complex {
            re: (i as f32) * 1e-3,
            im: -(i as f32) * 1e-3,
        })
        .collect();
    let bytes = beve::to_vec_complex_slice(&v);

    let mut g = c.benchmark_group("bulk_decode/vec_complex_f32_1M");
    g.throughput(Throughput::Bytes((v.len() * 8) as u64)); // 2 * f32 per value

    g.bench_function("read_complex_slice (in-mem bulk)", |b| {
        b.iter(|| black_box(beve::read_complex_slice::<f32>(black_box(&bytes)).unwrap()))
    });
    g.bench_function("read_complex_slice_from_reader (stream bulk)", |b| {
        b.iter(|| {
            black_box(
                beve::read_complex_slice_from_reader::<f32, _>(Cursor::new(black_box(&bytes)))
                    .unwrap(),
            )
        })
    });
    g.finish();
}

criterion_group!(benches, bench_typed_f64, bench_complex_f32);
criterion_main!(benches);
