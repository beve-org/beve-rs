//! Bytes interpreted as a structured value, then put through every
//! encoder/decoder pairing. See `shared::roundtrip`.

#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../../tests/fuzz/shared.rs"]
mod shared;

fuzz_target!(|data: &[u8]| {
    shared::roundtrip(data);
});
