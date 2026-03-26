#![deny(warnings)]

use beve::{Complex, Matrix, MatrixLayout};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
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

fn bench_matrix_payloads(c: &mut Criterion) {
    let (extents, data) = sample_matrix(64, 64);
    let matrix = Matrix {
        layout: MatrixLayout::Right,
        extents: &extents,
        data: &data,
    };
    let encoded = beve::to_vec(&matrix).expect("serialize matrix");

    let mut group = c.benchmark_group("matrix_payloads");
    group.bench_function("to_vec", move |b| {
        let extents = extents.clone();
        let data = data.clone();
        b.iter(move || {
            let matrix = Matrix {
                layout: MatrixLayout::Right,
                extents: &extents,
                data: &data,
            };
            let bytes = beve::to_vec(black_box(&matrix)).expect("matrix encode");
            black_box(bytes);
        });
    });
    group.bench_function("from_slice", move |b| {
        let encoded = encoded.clone();
        b.iter(move || {
            let OwnedMatrix {
                layout,
                extents,
                value,
            } = beve::from_slice(black_box(&encoded)).expect("matrix decode");
            black_box(layout);
            black_box(extents);
            black_box(value);
        });
    });
    group.finish();
}

fn bench_mixed_structs(c: &mut Criterion) {
    let frames = mixed_frames(256);
    let encoded = beve::to_vec(&frames).expect("serialize mixed frames");

    let mut group = c.benchmark_group("mixed_structs");
    group.bench_with_input(
        BenchmarkId::new("to_vec", frames.len()),
        &frames,
        |b, data| {
            b.iter(|| {
                let bytes = beve::to_vec(black_box(data)).expect("serialize mixed");
                black_box(bytes);
            });
        },
    );
    group.bench_function(BenchmarkId::new("from_slice", frames.len()), |b| {
        let encoded = encoded.clone();
        b.iter(move || {
            let decoded: Vec<MixedFrame> =
                beve::from_slice(black_box(&encoded)).expect("decode mixed");
            black_box(decoded);
        });
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Glaze-comparable benchmark: struct with large vectors
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone)]
struct LargeVecs {
    name: String,
    values: Vec<f64>,
    ids: Vec<u32>,
    flags: Vec<bool>,
}

fn make_large_vecs(n: usize) -> LargeVecs {
    LargeVecs {
        name: "benchmark".into(),
        values: (0..n).map(|i| (i as f64 * 0.001).sin() * 100.0).collect(),
        ids: (0..n).map(|i| i as u32).collect(),
        flags: (0..n).map(|i| i % 3 == 0).collect(),
    }
}

fn bench_large_vecs(c: &mut Criterion) {
    for &n in &[1_000, 10_000, 100_000] {
        let data = make_large_vecs(n);
        let encoded = beve::to_vec(&data).expect("serialize");

        let mut group = c.benchmark_group(format!("large_vecs_{n}"));

        group.bench_function("serde_to_vec", |b| {
            b.iter(|| {
                let bytes = beve::to_vec(black_box(&data)).expect("serialize");
                black_box(bytes);
            });
        });

        group.bench_function("serde_from_slice", |b| {
            b.iter(|| {
                let decoded: LargeVecs =
                    beve::from_slice(black_box(&encoded)).expect("deserialize");
                black_box(decoded);
            });
        });

        group.bench_function("streaming_to_vec", |b| {
            b.iter(|| {
                let mut buf = Vec::new();
                beve::to_writer_streaming(black_box(&mut buf), black_box(&data))
                    .expect("streaming");
                black_box(buf);
            });
        });

        let prealloc_size = encoded.len();
        group.bench_function("serde_to_vec_prealloc", |b| {
            b.iter(|| {
                let mut ser = beve::Serializer::with_capacity(prealloc_size);
                black_box(&data).serialize(&mut ser).expect("serialize");
                black_box(ser.into_vec());
            });
        });

        group.bench_function("streaming_to_vec_prealloc", |b| {
            b.iter(|| {
                let mut buf = Vec::with_capacity(prealloc_size);
                beve::to_writer_streaming(black_box(&mut buf), black_box(&data))
                    .expect("streaming");
                black_box(buf);
            });
        });

        group.finish();
    }
}

criterion_group!(
    benches,
    bench_struct_roundtrip,
    bench_numeric_arrays,
    bench_bool_arrays,
    bench_string_arrays,
    bench_matrix_payloads,
    bench_mixed_structs,
    bench_large_vecs
);
criterion_main!(benches);
