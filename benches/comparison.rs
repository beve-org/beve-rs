#![deny(warnings)]

use beve::{Complex, Matrix, MatrixLayout};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use serde::{Deserialize, Serialize};
use std::time::Duration;

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

#[derive(Deserialize)]
struct OwnedMatrix {
    layout: MatrixLayout,
    extents: Vec<usize>,
    value: Vec<f64>,
}

fn sample_matrix(rows: usize, cols: usize) -> (Vec<usize>, Vec<f64>) {
    let extents = vec![rows, cols];
    let data = (0..rows * cols)
        .map(|i| ((i as f64 * 0.03125).sin() + (i as f64 * 0.015625).cos()) * 0.5)
        .collect();
    (extents, data)
}

#[derive(Serialize, Deserialize, Clone)]
struct MixedFrame {
    id: u64,
    scalars: Vec<f64>,
    counts: Vec<u32>,
    complex: Vec<Complex<f64>>,
    notes: String,
    status: Status,
}

#[derive(Serialize, Deserialize, Clone)]
enum Status {
    Idle,
    Running {
        stage: u16,
        gains: Vec<f32>,
    },
    Error {
        code: i32,
        tags: Vec<String>,
        retry: bool,
    },
}

fn mixed_frames(count: usize) -> Vec<MixedFrame> {
    (0..count)
        .map(|i| {
            let scalars = (0..32)
                .map(|j| ((i as f64 * 0.5 + j as f64).sin() * 0.25) + j as f64)
                .collect();
            let counts = (0..32).map(|j| (j as u32) ^ (i as u32)).collect();
            let complex = (0..16)
                .map(|j| Complex {
                    re: (i as f64 * 0.25 + j as f64).cos(),
                    im: (i as f64 * 0.125 + j as f64).sin(),
                })
                .collect();
            let notes = format!("frame-{i:05}");
            let status = match i % 3 {
                0 => Status::Idle,
                1 => Status::Running {
                    stage: (i % 8) as u16,
                    gains: (0..12)
                        .map(|j| (j as f32 * 0.03125).sin() + (i as f32 * 0.0625))
                        .collect(),
                },
                _ => Status::Error {
                    code: -((i % 11) as i32),
                    tags: vec![
                        format!("sensor-{}", WORDS[i % WORDS.len()]),
                        format!("batch-{}", i / 64),
                    ],
                    retry: i % 2 == 0,
                },
            };
            MixedFrame {
                id: i as u64,
                scalars,
                counts,
                complex,
                notes,
                status,
            }
        })
        .collect()
}

fn bench_struct_roundtrip(c: &mut Criterion) {
    let records = sample_records(128);
    let beve_encoded = beve::to_vec(&records).expect("beve serialize records");
    let serde_encoded = serde_beve::to_bytes(&records).expect("serde-beve serialize records");

    let mut group = c.benchmark_group("struct_roundtrip_comparison");
    group.bench_with_input(
        BenchmarkId::new("beve_to_vec", records.len()),
        &records,
        |b, data| {
            b.iter(|| {
                let bytes = beve::to_vec(black_box(data)).expect("beve serialize");
                black_box(bytes);
            });
        },
    );
    group.bench_with_input(
        BenchmarkId::new("serde_beve_to_bytes", records.len()),
        &records,
        |b, data| {
            b.iter(|| {
                let bytes = serde_beve::to_bytes(black_box(data)).expect("serde-beve serialize");
                black_box(bytes);
            });
        },
    );
    group.bench_function(BenchmarkId::new("beve_from_slice", records.len()), |b| {
        let encoded = beve_encoded.clone();
        b.iter(|| {
            let decoded: Vec<Record> =
                beve::from_slice(black_box(encoded.as_slice())).expect("beve decode");
            black_box(decoded);
        });
    });
    group.bench_function(
        BenchmarkId::new("serde_beve_from_bytes", records.len()),
        |b| {
            let encoded = serde_encoded.clone();
            b.iter(|| {
                let decoded: Vec<Record> = serde_beve::from_bytes(black_box(encoded.as_slice()))
                    .expect("serde-beve decode");
                black_box(decoded);
            });
        },
    );
    group.finish();
}

fn bench_numeric_arrays(c: &mut Criterion) {
    let values: Vec<f64> = (0..1024)
        .map(|i| ((i as f64 * 0.25).sin() + 1.0) * (i % 1024) as f64)
        .collect();
    let beve_encoded = beve::to_vec(&values).expect("beve serialize f64 slice");
    let serde_encoded = serde_beve::to_bytes(&values).expect("serde-beve serialize f64 slice");

    let mut group = c.benchmark_group("numeric_arrays_f64_comparison");
    group.bench_function("beve_to_vec", |b| {
        b.iter(|| {
            let bytes = beve::to_vec(black_box(&values)).expect("beve serialize f64");
            black_box(bytes);
        });
    });
    group.bench_function("serde_beve_to_bytes", |b| {
        b.iter(|| {
            let bytes = serde_beve::to_bytes(black_box(&values)).expect("serde-beve serialize f64");
            black_box(bytes);
        });
    });
    group.bench_function("beve_from_slice", |b| {
        let encoded = beve_encoded.clone();
        b.iter(|| {
            let decoded: Vec<f64> =
                beve::from_slice(black_box(encoded.as_slice())).expect("beve decode f64");
            black_box(decoded);
        });
    });
    group.bench_function("serde_beve_from_bytes", |b| {
        let encoded = serde_encoded.clone();
        b.iter(|| {
            let decoded: Vec<f64> = serde_beve::from_bytes(black_box(encoded.as_slice()))
                .expect("serde-beve decode f64");
            black_box(decoded);
        });
    });
    group.finish();
}

fn bench_bool_arrays(c: &mut Criterion) {
    let flags: Vec<bool> = (0..2048).map(|i| (i % 3 == 0) ^ (i % 5 == 0)).collect();
    let beve_encoded = beve::to_vec(&flags).expect("beve serialize bool vec");
    let serde_encoded = serde_beve::to_bytes(&flags).expect("serde-beve serialize bool vec");

    let mut group = c.benchmark_group("bool_arrays_comparison");
    group.bench_function("beve_to_vec", |b| {
        b.iter(|| {
            let bytes = beve::to_vec(black_box(&flags)).expect("beve serialize bool");
            black_box(bytes);
        });
    });
    group.bench_function("serde_beve_to_bytes", |b| {
        b.iter(|| {
            let bytes = serde_beve::to_bytes(black_box(&flags)).expect("serde-beve serialize bool");
            black_box(bytes);
        });
    });
    group.bench_function("beve_from_slice", |b| {
        let encoded = beve_encoded.clone();
        b.iter(|| {
            let decoded: Vec<bool> =
                beve::from_slice(black_box(encoded.as_slice())).expect("beve decode bool");
            black_box(decoded);
        });
    });
    group.bench_function("serde_beve_from_bytes", |b| {
        let encoded = serde_encoded.clone();
        b.iter(|| {
            let decoded: Vec<bool> = serde_beve::from_bytes(black_box(encoded.as_slice()))
                .expect("serde-beve decode bool");
            black_box(decoded);
        });
    });
    group.finish();
}

fn bench_string_arrays(c: &mut Criterion) {
    let owned: Vec<String> = (0..512)
        .map(|i| format!("sensor-{i:04}-{}", WORDS[i % WORDS.len()]))
        .collect();
    let beve_encoded = beve::to_vec(&owned).expect("beve serialize strings");
    let serde_encoded = serde_beve::to_bytes(&owned).expect("serde-beve serialize strings");

    let mut group = c.benchmark_group("string_arrays_comparison");
    group.bench_function("beve_to_vec", |b| {
        b.iter(|| {
            let bytes = beve::to_vec(black_box(&owned)).expect("beve serialize strings");
            black_box(bytes);
        });
    });
    group.bench_function("serde_beve_to_bytes", |b| {
        b.iter(|| {
            let bytes =
                serde_beve::to_bytes(black_box(&owned)).expect("serde-beve serialize strings");
            black_box(bytes);
        });
    });
    group.bench_function("beve_from_slice", |b| {
        let encoded = beve_encoded.clone();
        b.iter(|| {
            let decoded: Vec<String> =
                beve::from_slice(black_box(encoded.as_slice())).expect("beve decode strings");
            black_box(decoded);
        });
    });
    group.bench_function("serde_beve_from_bytes", |b| {
        let encoded = serde_encoded.clone();
        b.iter(|| {
            let decoded: Vec<String> = serde_beve::from_bytes(black_box(encoded.as_slice()))
                .expect("serde-beve decode strings");
            black_box(decoded);
        });
    });
    group.finish();
}

fn bench_matrix_payloads(c: &mut Criterion) {
    let (extents, data) = sample_matrix(32, 32);
    let matrix = Matrix {
        layout: MatrixLayout::Right,
        extents: &extents,
        data: &data,
    };
    let beve_encoded = beve::to_vec(&matrix).expect("beve serialize matrix");
    let serde_encoded = serde_beve::to_bytes(&matrix).expect("serde-beve serialize matrix");

    let mut group = c.benchmark_group("matrix_payloads_comparison");
    group.bench_function("beve_to_vec", |b| {
        let extents = extents.clone();
        let data = data.clone();
        b.iter(move || {
            let matrix = Matrix {
                layout: MatrixLayout::Right,
                extents: &extents,
                data: &data,
            };
            let bytes = beve::to_vec(black_box(&matrix)).expect("beve serialize matrix");
            black_box(bytes);
        });
    });
    group.bench_function("serde_beve_to_bytes", |b| {
        let extents = extents.clone();
        let data = data.clone();
        b.iter(move || {
            let matrix = Matrix {
                layout: MatrixLayout::Right,
                extents: &extents,
                data: &data,
            };
            let bytes =
                serde_beve::to_bytes(black_box(&matrix)).expect("serde-beve serialize matrix");
            black_box(bytes);
        });
    });
    group.bench_function("beve_from_slice", |b| {
        let encoded = beve_encoded.clone();
        b.iter(|| {
            let OwnedMatrix {
                layout,
                extents,
                value,
            } = beve::from_slice(black_box(encoded.as_slice())).expect("beve decode matrix");
            black_box(layout);
            black_box(extents);
            black_box(value);
        });
    });
    group.bench_function("serde_beve_from_bytes", |b| {
        let encoded = serde_encoded.clone();
        b.iter(|| {
            let OwnedMatrix {
                layout,
                extents,
                value,
            } = serde_beve::from_bytes(black_box(encoded.as_slice()))
                .expect("serde-beve decode matrix");
            black_box(layout);
            black_box(extents);
            black_box(value);
        });
    });
    group.finish();
}

fn bench_mixed_structs(c: &mut Criterion) {
    let frames = mixed_frames(96);
    let beve_encoded = beve::to_vec(&frames).expect("beve serialize mixed frames");
    let serde_encoded = serde_beve::to_bytes(&frames).expect("serde-beve serialize mixed frames");

    let mut group = c.benchmark_group("mixed_structs_comparison");
    group.bench_with_input(
        BenchmarkId::new("beve_to_vec", frames.len()),
        &frames,
        |b, data| {
            b.iter(|| {
                let bytes = beve::to_vec(black_box(data)).expect("beve serialize mixed");
                black_box(bytes);
            });
        },
    );
    group.bench_with_input(
        BenchmarkId::new("serde_beve_to_bytes", frames.len()),
        &frames,
        |b, data| {
            b.iter(|| {
                let bytes =
                    serde_beve::to_bytes(black_box(data)).expect("serde-beve serialize mixed");
                black_box(bytes);
            });
        },
    );
    group.bench_function(BenchmarkId::new("beve_from_slice", frames.len()), |b| {
        let encoded = beve_encoded.clone();
        b.iter(|| {
            let decoded: Vec<MixedFrame> =
                beve::from_slice(black_box(encoded.as_slice())).expect("beve decode mixed");
            black_box(decoded);
        });
    });
    group.bench_function(
        BenchmarkId::new("serde_beve_from_bytes", frames.len()),
        |b| {
            let encoded = serde_encoded.clone();
            b.iter(|| {
                let decoded: Vec<MixedFrame> =
                    serde_beve::from_bytes(black_box(encoded.as_slice()))
                        .expect("serde-beve decode mixed");
                black_box(decoded);
            });
        },
    );
    group.finish();
}

fn configure_criterion() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_millis(200))
}

criterion_group!(
    name = benches;
    config = configure_criterion();
    targets =
        bench_struct_roundtrip,
        bench_numeric_arrays,
        bench_bool_arrays,
        bench_string_arrays,
        bench_matrix_payloads,
        bench_mixed_structs
);
criterion_main!(benches);
