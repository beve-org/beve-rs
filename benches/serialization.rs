#![deny(warnings)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use serde::{Deserialize, Serialize};

const WORDS: [&str; 6] = ["alpha", "beta", "gamma", "delta", "eta", "theta"];

#[derive(Serialize, Deserialize, Clone)]
struct Record {
    id: u64,
    flag: bool,
    intensity: f64,
    readings: [f32; 4],
    label: String,
    tags: Vec<String>,
}

fn sample_records(count: usize) -> Vec<Record> {
    (0..count)
        .map(|i| Record {
            id: i as u64,
            flag: i % 3 == 0,
            intensity: (i as f64 * 0.125).sin() * (1.0 + (i % 7) as f64),
            readings: [
                i as f32,
                (i as f32 * 0.25).sin(),
                (i as f32 * 0.5).cos(),
                ((i * 31) % 13) as f32,
            ],
            label: format!("sensor-{i:04}"),
            tags: vec![
                format!("batch-{}", i / 32),
                if i % 2 == 0 {
                    "even".into()
                } else {
                    "odd".into()
                },
            ],
        })
        .collect()
}

fn bench_struct_roundtrip(c: &mut Criterion) {
    let records = sample_records(512);
    let encoded = beve::to_vec(&records).expect("serialize records");

    let mut group = c.benchmark_group("struct_roundtrip");
    group.bench_with_input(
        BenchmarkId::new("to_vec", records.len()),
        &records,
        |b, data| {
            b.iter(|| {
                let bytes = beve::to_vec(black_box(data)).expect("serialize");
                black_box(bytes);
            });
        },
    );
    group.bench_function(BenchmarkId::new("from_slice", records.len()), |b| {
        b.iter(|| {
            let decoded: Vec<Record> =
                beve::from_slice(black_box(&encoded)).expect("decode records");
            black_box(decoded);
        });
    });
    group.finish();
}

fn bench_numeric_arrays(c: &mut Criterion) {
    let values: Vec<f64> = (0..4096)
        .map(|i| ((i as f64 * 0.25).sin() + 1.0) * (i % 1024) as f64)
        .collect();
    let slice = values.as_slice();
    let typed_bytes = beve::to_vec_typed_slice(slice);

    let mut group = c.benchmark_group("numeric_arrays_f64");
    group.bench_function("serde_to_vec", |b| {
        b.iter(|| {
            let bytes = beve::to_vec(black_box(&values)).expect("serialize f64 slice");
            black_box(bytes);
        });
    });
    group.bench_function("typed_to_vec", |b| {
        b.iter(|| {
            let bytes = beve::to_vec_typed_slice(black_box(slice));
            black_box(bytes);
        });
    });
    group.bench_function("from_slice", |b| {
        b.iter(|| {
            let decoded: Vec<f64> =
                beve::from_slice(black_box(&typed_bytes)).expect("decode f64 slice");
            black_box(decoded);
        });
    });
    group.finish();
}

fn bench_bool_arrays(c: &mut Criterion) {
    let flags: Vec<bool> = (0..8192).map(|i| (i % 3 == 0) ^ (i % 5 == 0)).collect();
    let slice = flags.as_slice();
    let typed_bytes = beve::to_vec_bool_slice(slice);

    let mut group = c.benchmark_group("bool_arrays");
    group.bench_function("serde_to_vec", |b| {
        b.iter(|| {
            let bytes = beve::to_vec(black_box(&flags)).expect("serialize bool vec");
            black_box(bytes);
        });
    });
    group.bench_function("typed_to_vec", |b| {
        b.iter(|| {
            let bytes = beve::to_vec_bool_slice(black_box(slice));
            black_box(bytes);
        });
    });
    group.bench_function("from_slice", |b| {
        b.iter(|| {
            let decoded: Vec<bool> =
                beve::from_slice(black_box(&typed_bytes)).expect("decode bool vec");
            black_box(decoded);
        });
    });
    group.finish();
}

fn bench_string_arrays(c: &mut Criterion) {
    let owned: Vec<String> = (0..2048)
        .map(|i| format!("sensor-{i:04}-{}", WORDS[i % WORDS.len()]))
        .collect();
    let owned_slice = owned.as_slice();
    let borrowed: Vec<&str> = owned.iter().map(|s| s.as_str()).collect();
    let borrowed_slice = borrowed.as_slice();
    let typed_owned = beve::to_vec_string_slice(owned_slice);

    let mut group = c.benchmark_group("string_arrays");
    group.bench_function("serde_to_vec", |b| {
        b.iter(|| {
            let bytes = beve::to_vec(black_box(&owned)).expect("serialize strings");
            black_box(bytes);
        });
    });
    group.bench_function("typed_to_vec_owned", |b| {
        b.iter(|| {
            let bytes = beve::to_vec_string_slice(black_box(owned_slice));
            black_box(bytes);
        });
    });
    group.bench_function("typed_to_vec_borrowed", |b| {
        b.iter(|| {
            let bytes = beve::to_vec_str_slice(black_box(borrowed_slice));
            black_box(bytes);
        });
    });
    group.bench_function("from_slice", |b| {
        b.iter(|| {
            let decoded: Vec<String> =
                beve::from_slice(black_box(&typed_owned)).expect("decode strings");
            black_box(decoded);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_struct_roundtrip,
    bench_numeric_arrays,
    bench_bool_arrays,
    bench_string_arrays
);
criterion_main!(benches);
