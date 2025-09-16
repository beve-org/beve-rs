# Repository Guidelines

## Project Structure & Module Organization
The crate lives in `src/` with `lib.rs` exporting the public API and delegating to focused modules: `ser.rs` for serde-driven encoding, `de.rs` for decoding, `fast.rs` for typed-array fast paths, `header.rs` and `size.rs` for framing helpers, and `ext.rs` for serde extensions. Errors reside in `error.rs`. Integration tests sit in `tests/`, split across `basic.rs`, `fast.rs`, and interop suites under `tests/interop.rs` and `tests/cpp/` that validate cross-language compatibility. Example binaries demonstrating emission lives in `examples/` (see `emit_bool.rs` and `emit_color.rs`). Specification notes and external references live under `reference/` for design context.

## Build, Test, and Development Commands
Use `cargo build` for a debug build and `cargo build --release` when benchmarking encoding throughput. Run `cargo test` to execute unit, doc, and integration suites; add `-- --nocapture` when you need stdout. `cargo fmt` formats the codebase and doubles as a CI guard with `cargo fmt -- --check`. Lint with `cargo clippy --all-targets --all-features` before submitting changes. Example binaries can be exercised with `cargo run --example emit_bool`.

## Coding Style & Naming Conventions
Follow idiomatic Rust style with rustfmt defaults (4-space indentation, trailing commas, snake_case modules). Public types and traits use UpperCamelCase, while functions, modules, and files stay snake_case. Keep error variants descriptive and suffix serde helper types with `Visitor` or `Adapter` when appropriate. Prefer `&[T]` and iterators over owning collections to keep encoding zero-copy. Let clippy warnings guide refactors; do not ignore new lints without justification.

## Testing Guidelines
Add integration coverage in `tests/` for external behavior and unit tests inline with modules via `#[cfg(test)]` blocks. Mirror the naming of the code under test (`fast_typed_arrays`, `object_roundtrip`, etc.) and assert byte-level expectations when feasible. Validate interop by extending fixtures in `tests/cpp/` when touching the format. Aim to keep new features accompanied by round-trip tests and property checks where practical.

## Commit & Pull Request Guidelines
Write concise, present-tense commit subjects similar to `git log` (<72 chars, imperative verbs such as “Add fast header encode”). Squash noisy commits before review. Each pull request should describe the change, link related issues or spec sections, list validation commands run, and include benchmarks when performance-sensitive behavior shifts. Confirm `cargo fmt`, `cargo clippy`, and `cargo test` succeed before requesting review.
