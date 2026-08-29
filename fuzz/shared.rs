//! The bodies of the fuzz targets, kept out of the target files themselves so
//! that a stable toolchain can run them too.
//!
//! `cargo fuzz` needs nightly for its sanitizer flags, so a target that lives
//! only inside `fuzz_targets/` is exercised exactly when someone remembers to
//! spend wall-clock on it. Everything that decides whether an input passes lives
//! here instead, and two callers include this file as a module:
//!
//! * `fuzz/fuzz_targets/*.rs`, which hand it bytes libFuzzer generated;
//! * `tests/fuzz_seeds.rs`, which hands it the bytes committed under
//!   `fuzz/seeds/` on every `cargo test`, on every toolchain.
//!
//! So the fuzzer explores and the seed corpus defends: a crash found on nightly
//! becomes a file in `fuzz/seeds/`, and from then on it is a stable-CI
//! regression test that no one has to remember to run.

// Each target binary includes this whole file but calls one of its bodies, and
// the seed replay uses the generator only through `roundtrip`.
#![allow(dead_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use beve::Value;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Allocation accounting
// ---------------------------------------------------------------------------

/// Live bytes handed out by the allocator, and the high-water mark since the
/// last [`peak_growth`] reset.
///
/// `Relaxed` is deliberate: these are counters, not a synchronisation
/// mechanism, and nothing reads them to establish an ordering. Both fuzz
/// targets are single-threaded, and the seed replay is a single `#[test]` in
/// its own binary, so the only writer during a measured window is the thread
/// doing the decode.
static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

/// A pass-through allocator that records how much is live.
///
/// A decoder that sizes a buffer from an untrusted length does not return an
/// error, it asks the allocator for an impossible number of bytes and the
/// process aborts. An abort is invisible to an `assert!`, so the bound below
/// has to be checked against something that sees the request. Counting live
/// bytes catches the whole family: the outright impossible request that aborts,
/// and the merely enormous one that succeeds on a big machine and would have
/// been a denial of service in production.
struct TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            grew(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            grew(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        shrank(layout.size());
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let out = unsafe { System.realloc(ptr, layout, new_size) };
        if !out.is_null() {
            if new_size >= layout.size() {
                grew(new_size - layout.size());
            } else {
                shrank(layout.size() - new_size);
            }
        }
        out
    }
}

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

fn grew(bytes: usize) {
    let live = LIVE_BYTES.fetch_add(bytes, Ordering::Relaxed) + bytes;
    PEAK_BYTES.fetch_max(live, Ordering::Relaxed);
}

fn shrank(bytes: usize) {
    LIVE_BYTES.fetch_sub(bytes, Ordering::Relaxed);
}

/// Runs `f` and returns how far live bytes rose above where they started.
fn peak_growth<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let base = LIVE_BYTES.load(Ordering::Relaxed);
    PEAK_BYTES.store(base, Ordering::Relaxed);
    let out = f();
    let peak = PEAK_BYTES.load(Ordering::Relaxed);
    (out, peak.saturating_sub(base))
}

/// Allocation a decode of `input_len` bytes is allowed to reach.
///
/// This is beve's own stated policy, restated as an assertion. The decoder caps
/// any single buffer it sizes from an untrusted length at `MAX_PREALLOC_BYTES`
/// (8 MiB, in `src/de.rs`), and caps nesting at [`beve::MAX_RECURSION_DEPTH`],
/// so at most that many capped reserves can be live at once. Their product is
/// the most a conforming decode can ask for, plus room for the values actually
/// built out of the bytes that really are there.
///
/// A failure here therefore means a reserve escaped the ceiling entirely, which
/// is the failure that matters: the first run of this target found a
/// nine-byte document driving a 35 TB request, because `Value`'s `visit_seq`
/// passed the wire's element count to `Vec::reserve` unclamped.
fn allocation_bound(input_len: usize) -> usize {
    // Kept in step with `de::MAX_PREALLOC_BYTES`, which is `pub(crate)` and so
    // not nameable from the fuzz crate. It is a policy ceiling that moves
    // rarely; if it moves, this fails loudly rather than silently.
    const RESERVE_CEILING: usize = 8 * 1024 * 1024;
    const PER_INPUT_BYTE: usize = 1024;

    RESERVE_CEILING
        .saturating_mul(beve::MAX_RECURSION_DEPTH)
        .saturating_add(input_len.saturating_mul(PER_INPUT_BYTE))
}

// ---------------------------------------------------------------------------
// Target: decode_arbitrary
// ---------------------------------------------------------------------------

/// Feed unconstrained bytes to the two decoder entry points that take a slice.
///
/// Neither call is expected to succeed; almost every input is not a BEVE
/// document, and `Err` is the correct answer. What is being asserted is that
/// the wrong answer is never a panic, an abort, or an allocation sized from a
/// number the input never backed with bytes.
pub fn decode_arbitrary(data: &[u8]) {
    let (_, peak) = peak_growth(|| {
        // `Value` accepts any document the format can express, so it reaches
        // more of the decoder than any concrete type would.
        let _ = beve::from_slice::<Value>(data);
        // The validation-only path walks the same structure without building
        // it, and skips through values with its own length handling.
        let _ = beve::validate_slice(data);
    });

    let bound = allocation_bound(data.len());
    assert!(
        peak <= bound,
        "decoding {} bytes reached {peak} live bytes, over the {bound}-byte bound",
        data.len()
    );
}

// ---------------------------------------------------------------------------
// Target: roundtrip
// ---------------------------------------------------------------------------

/// A shape chosen to cover the serde enum representations, which is where the
/// encoder has the most cases to get wrong: a newtype variant holding a
/// sequence, a newtype variant holding another enum, and a struct variant. The
/// nesting matters as much as the variants; an empty collection reached the
/// encoder in a different state depending on what enclosed it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Message {
    id: u32,
    payload: Payload,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
enum Payload {
    Telemetry(Vec<f64>),
    MetaOnly(Meta),
    Snapshot { label: String, counters: Vec<u32> },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
enum Meta {
    Unit,
    Codes(Vec<u32>),
    Window { start: u64, metrics: Vec<f32> },
}

/// A cursor that turns a byte string into values.
///
/// This is what a property-testing crate's generator would do, minus the
/// dependency: libFuzzer already produces and mutates byte strings, and it does
/// so under coverage guidance, so all that is missing is a deterministic
/// mapping from those bytes onto a `Message`. Determinism is the whole
/// requirement — a seed file has to rebuild the same value years later, on any
/// toolchain, or it is not a regression test.
///
/// Running off the end yields zeroes rather than stopping. That makes short
/// inputs mean something (an empty file is the all-defaults `Message`) and
/// keeps libFuzzer's length mutations cheap.
struct Generator<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Generator<'a> {
    fn new(data: &'a [u8]) -> Self {
        Generator { data, pos: 0 }
    }

    fn byte(&mut self) -> u8 {
        let b = self.data.get(self.pos).copied().unwrap_or(0);
        self.pos = self.pos.saturating_add(1);
        b
    }

    fn array<const N: usize>(&mut self) -> [u8; N] {
        let mut out = [0u8; N];
        for slot in out.iter_mut() {
            *slot = self.byte();
        }
        out
    }

    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.array::<4>())
    }

    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.array::<8>())
    }

    /// A finite `f32`/`f64`: NaN is not equal to itself, so a generated NaN
    /// would fail the round-trip comparison no matter how correct the codec is.
    /// Infinities survive a round trip, but they are excluded alongside NaN so
    /// that the generated set is exactly the set the comparison can judge.
    fn f32(&mut self) -> f32 {
        let v = f32::from_bits(self.u32());
        if v.is_finite() { v } else { 0.0 }
    }

    fn f64(&mut self) -> f64 {
        let v = f64::from_bits(self.u64());
        if v.is_finite() { v } else { 0.0 }
    }

    /// A length in `0..=max`, one byte wide so that a mutation of that byte is
    /// a small change to the length rather than a jump to an unrelated size.
    fn len(&mut self, max: usize) -> usize {
        (self.byte() as usize) % (max + 1)
    }

    fn variant(&mut self, count: u8) -> u8 {
        self.byte() % count
    }

    fn message(&mut self) -> Message {
        Message {
            id: self.u32(),
            payload: self.payload(),
        }
    }

    fn payload(&mut self) -> Payload {
        match self.variant(3) {
            0 => Payload::Telemetry((0..self.len(47)).map(|_| self.f64()).collect()),
            1 => Payload::MetaOnly(self.meta()),
            _ => Payload::Snapshot {
                label: self.label(),
                counters: (0..self.len(31)).map(|_| self.u32()).collect(),
            },
        }
    }

    fn meta(&mut self) -> Meta {
        match self.variant(3) {
            0 => Meta::Unit,
            1 => Meta::Codes((0..self.len(23)).map(|_| self.u32()).collect()),
            _ => Meta::Window {
                start: self.u64(),
                metrics: (0..self.len(23)).map(|_| self.f32()).collect(),
            },
        }
    }

    /// Alphanumerics rather than arbitrary `char`s: string *content* is the
    /// subject of the UTF-8 tests, and letting it vary freely here would spend
    /// the fuzzer's budget rediscovering those instead of exploring structure.
    fn label(&mut self) -> String {
        const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-";
        let n = 1 + self.len(23);
        (0..n)
            .map(|_| ALPHABET[self.byte() as usize % ALPHABET.len()] as char)
            .collect()
    }
}

/// Build a `Message` from `data`, then require every encoder to agree with
/// every decoder about it.
///
/// The crate ships two encoders and two decoders, and `serialized_size` as a
/// third way to ask the streaming encoder a question. They are separate
/// implementations, so this is a differential test as much as a round trip:
/// all four encode/decode pairings must land back on the value we started with.
///
/// Note what is *not* asserted: that the two encoders emit identical bytes.
/// They are allowed to differ, because the buffered encoder can backpatch a
/// size that the streaming one has to commit to before it knows the answer.
/// `serialized_size` is the dual of the streaming encoder specifically, which
/// is why it is compared against that length and not against `to_vec`.
pub fn roundtrip(data: &[u8]) {
    let message = Generator::new(data).message();

    let buffered = beve::to_vec(&message).expect("buffered encode of a generated message");

    let mut streamed = Vec::new();
    beve::to_writer_streaming(&mut streamed, &message)
        .expect("streaming encode of a generated message");

    let measured = beve::serialized_size(&message).expect("serialized_size of a generated message");
    assert_eq!(
        measured,
        streamed.len() as u64,
        "serialized_size disagreed with the streamed length for {message:?}"
    );

    for (name, bytes) in [("buffered", &buffered), ("streamed", &streamed)] {
        beve::validate_slice(bytes)
            .unwrap_or_else(|e| panic!("{name} encoding failed validation: {e:?} for {message:?}"));

        let decoded: Message = beve::from_slice(bytes)
            .unwrap_or_else(|e| panic!("buffered decode of {name} encoding: {e:?}"));
        assert_eq!(decoded, message, "buffered decode of {name} encoding");

        let streamed_back: Message = beve::from_reader_streaming(std::io::Cursor::new(bytes))
            .unwrap_or_else(|e| panic!("streaming decode of {name} encoding: {e:?}"));
        assert_eq!(
            streamed_back, message,
            "streaming decode of {name} encoding"
        );
    }
}
