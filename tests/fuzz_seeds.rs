//! Replays the committed fuzz seed corpus on stable, so the fuzz targets are
//! regression tests and not just an activity.
//!
//! `cargo fuzz` needs nightly and runs for as long as someone gives it, which
//! makes it a poor guard against a regression: a bug it found last month is
//! only re-found if someone spends the wall-clock to re-find it. The corpus
//! fixes that. Every input under `tests/fuzz/seeds/` is replayed here, through the
//! same target bodies the fuzzer calls, on every toolchain, in milliseconds.
//!
//! The workflow that keeps the two halves in step:
//!
//! 1. `cargo +nightly fuzz run <target> fuzz/corpus/<target> tests/fuzz/seeds/<target>`
//! 2. A crash leaves a file in `fuzz/artifacts/<target>/`.
//! 3. Move it (minimised first, with `cargo fuzz tmin`) into
//!    `tests/fuzz/seeds/<target>/` under a name that says what it is.
//! 4. It is now a permanent stable-CI test, and it stays one after the fix.
//!
//! `fuzz/corpus/` and `fuzz/artifacts/` are the fuzzer's scratch space and are
//! gitignored; `tests/fuzz/seeds/` is the curated set, and only files a human put
//! there are committed.

// One `#[test]`, walking every seed, rather than one per file. The target
// bodies measure allocation through a process-wide counter, and libtest runs
// tests on parallel threads: a second test in this binary would be allocating
// inside another's measurement window.
#[path = "fuzz/shared.rs"]
mod shared;

use std::fs;
use std::path::{Path, PathBuf};

/// Seed files for one target, sorted so a failure reports the same first file
/// on every machine.
fn seeds(target: &str) -> Vec<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fuzz/seeds")
        .join(target);
    let mut paths: Vec<_> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("reading seed directory {}: {e}", dir.display()))
        .map(|entry| entry.expect("seed directory entry").path())
        .filter(|p| p.is_file())
        .collect();
    paths.sort();
    assert!(
        !paths.is_empty(),
        "no seeds in {} -- a target with an empty corpus guards nothing",
        dir.display()
    );
    paths
}

/// Run `body` over every seed, naming the file in the panic message.
///
/// The bodies assert with `panic!`, which is what libFuzzer wants, so a failure
/// here would otherwise point at `shared.rs` and leave the reader to work out
/// which of thirty files produced it.
fn replay(target: &str, body: fn(&[u8])) -> usize {
    let paths = seeds(target);
    for path in &paths {
        let data = fs::read(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        let outcome = std::panic::catch_unwind(|| body(&data));
        assert!(
            outcome.is_ok(),
            "fuzz seed {target}/{name} failed ({} bytes); reproduce with \
             `cargo +nightly fuzz run {target} tests/fuzz/seeds/{target}/{name}`",
            data.len()
        );
    }
    paths.len()
}

#[test]
fn every_seed_still_passes_its_target() {
    let decode = replay("decode_arbitrary", shared::decode_arbitrary);
    let roundtrip = replay("roundtrip", shared::roundtrip);
    println!("replayed {decode} decode_arbitrary and {roundtrip} roundtrip seeds");
}
