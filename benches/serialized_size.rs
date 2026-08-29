#![deny(warnings)]

//! Benchmarks for `serialized_size`, used to validate the optimization analysis
//! in PR #22.
//!
//! Two questions:
//!
//! 1. **Asymptotics.** A bare numeric `Vec<f64>` is measured element-by-element
//!    (O(N)), while a `TypedSlice<f64>` body is one bulk write (O(1)) and
//!    `typed_slice_size` is closed-form (no traversal). How far apart are they?
//!
//! 2. **Is byte-materialization worth eliminating?** The counting pass still calls
//!    `to_le_bytes()` per element and then discards the bytes. A hypothetical
//!    "length-only sink" would skip that materialization. To gauge the ceiling on
//!    that win, compare `serialized_size(&Vec<f64>)` (materialize-then-discard)
//!    against `to_writer_streaming` into a preallocated `Vec` (materialize-then-
//!    copy). If the two are close, the per-element serde traversal dominates and
//!    skipping materialization would save almost nothing.

use benchit::Bench;
use beve::TypedSlice;
use std::hint::black_box;

fn make_f64s(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64 * 0.125).sin()).collect()
}

fn main() {
    let mut bench = Bench::from_args();

    for &n in &[1_000usize, 100_000, 1_000_000] {
        let data = make_f64s(n);
        let cap = beve::serialized_size(&data).expect("measure") as usize;

        let mut group = bench.group(format!("serialized_size_f64_{n}"));

        // O(N): serde drives the Vec element-by-element; each f64 does a
        // to_le_bytes() that the counter discards.
        group.bench("vec_count", |b| {
            b.iter(|| black_box(beve::serialized_size(black_box(&data)).expect("measure")));
        });

        // Same O(N) traversal, but the bytes are actually moved into a
        // preallocated buffer instead of discarded. The gap vs `vec_count` is the
        // cost of the byte movement alone — i.e. the ceiling on what a length-only
        // sink could save.
        group.bench("vec_stream_prealloc", |b| {
            b.iter(|| {
                let mut buf = Vec::with_capacity(cap);
                beve::to_writer_streaming(black_box(&mut buf), black_box(&data)).expect("stream");
                black_box(buf);
            });
        });

        // O(1): the TypedSlice body is a single bulk write -> one `len +=`.
        group.bench("typedslice_count", |b| {
            b.iter(|| {
                black_box(beve::serialized_size(black_box(&TypedSlice(&data))).expect("measure"))
            });
        });

        // O(1) closed form: no encoder traversal at all.
        group.bench("typed_slice_size", |b| {
            b.iter(|| black_box(beve::typed_slice_size(black_box(&data))));
        });

        group.finish();
    }
}
