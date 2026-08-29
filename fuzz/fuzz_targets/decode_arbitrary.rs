//! Unconstrained bytes into the slice decoders. See `shared::decode_arbitrary`.

#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../shared.rs"]
mod shared;

fuzz_target!(|data: &[u8]| {
    shared::decode_arbitrary(data);
});
