//! Per-element serde decode throughput for integer typed arrays.
//!
//! Integer typed arrays decode element-by-element through `NumDe`, which carries
//! every value as a 128-bit integer. This benchmark isolates that path (the
//! buffered `from_slice` reader, so no per-element reader cost confounds it) so a
//! change to the per-element integer conversion can be measured directly. Floats
//! are included as a control: they do not take the 128-bit carrier, so they
//! should be unaffected by integer-path changes.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

const N: usize = 1_000_000;

fn bench(c: &mut Criterion) {
    macro_rules! case {
        ($name:literal, $t:ty, $make:expr) => {{
            let v: Vec<$t> = (0..N).map($make).collect();
            let bytes = beve::to_vec_typed_slice(&v);
            let mut g = c.benchmark_group(concat!("int_decode/", $name));
            g.throughput(Throughput::Bytes((N * std::mem::size_of::<$t>()) as u64));
            g.bench_function("from_slice", |b| {
                b.iter(|| {
                    let d: Vec<$t> = beve::from_slice(black_box(&bytes)).unwrap();
                    black_box(d);
                })
            });
            g.bench_function("from_reader_streaming", |b| {
                b.iter(|| {
                    let d: Vec<$t> =
                        beve::from_reader_streaming(std::io::Cursor::new(black_box(&bytes)))
                            .unwrap();
                    black_box(d);
                })
            });
            g.finish();
        }};
    }

    case!("vec_u8", u8, |i| (i % 251) as u8);
    case!("vec_u16", u16, |i| (i as u32 % 65521) as u16);
    case!("vec_u32", u32, |i| (i as u32).wrapping_mul(2654435761));
    case!("vec_u64", u64, |i| (i as u64).wrapping_mul(0x9E3779B97F4A7C15));
    case!("vec_i32", i32, |i| (i as i32).wrapping_mul(-1640531527));
    case!("vec_i64", i64, |i| (i as i64).wrapping_mul(-0x61C8864680B583EB));
    // Control: float path, should be unchanged by integer-path edits.
    case!("vec_f64", f64, |i| i as f64 * 0.5);
}

criterion_group!(benches, bench);
criterion_main!(benches);
