//! The decoders bound how deeply input may nest.
//!
//! Nesting is declared by the input, not by the destination type, so a decoder
//! with no ceiling recurses as far as an attacker asks. That is not an ordinary
//! parse failure: a Rust stack overflow **aborts the process** rather than
//! unwinding, so no `Result` carries it, no `catch_unwind` contains it, and a
//! server decoding an untrusted body loses every other connection it was
//! serving along with the request that caused it.
//!
//! These tests exist because that was reachable: a few kilobytes of nested
//! array headers killed a REPE gateway (and the server co-hosted with it)
//! through one unauthenticated request.

use beve::{Error, MAX_RECURSION_DEPTH, Value};

/// A document of exactly `values` nested generic arrays, innermost empty.
///
/// `05 04` opens a generic array of one element; `05 00` is an empty one. Sized
/// in total nested values rather than wrappers, because that is the unit the
/// limit is stated in. This is the shape of the payload that aborted the
/// gateway.
fn nested_arrays(values: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values * 2);
    for _ in 0..values - 1 {
        bytes.extend_from_slice(&[0x05, 0x04]);
    }
    bytes.extend_from_slice(&[0x05, 0x00]);
    bytes
}

/// The same, as nested objects: `03 04` opens a one-entry object, `04 61` is
/// the one-byte key `"a"`, `03 00` is an empty object.
fn nested_objects(values: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values * 4);
    for _ in 0..values - 1 {
        bytes.extend_from_slice(&[0x03, 0x04, 0x04, b'a']);
    }
    bytes.extend_from_slice(&[0x03, 0x00]);
    bytes
}

/// The same shape as [`nested_arrays`], as JSON text: `values` nested arrays,
/// innermost empty.
fn nested_json_arrays(values: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values * 2);
    bytes.extend(std::iter::repeat_n(b'[', values));
    bytes.extend(std::iter::repeat_n(b']', values));
    bytes
}

fn is_too_deep(err: &Error) -> bool {
    matches!(err, Error::RecursionLimitExceeded)
}

// -- the regression ---------------------------------------------------------

#[test]
fn a_depth_bomb_is_refused_rather_than_aborting_the_process() {
    // 40 KB, and before the limit existed this did not fail the test — it
    // killed the test binary. Reaching the assertion at all is the point.
    let err = beve::from_slice::<Value>(&nested_arrays(20_000))
        .expect_err("a 20,000-deep array must be refused");
    assert!(is_too_deep(&err), "{err}");
}

#[test]
fn the_refusal_is_an_ordinary_error_a_caller_can_act_on() {
    let err = beve::from_slice::<Value>(&nested_arrays(20_000)).unwrap_err();
    // The property that matters to a server: it can answer the request with a
    // status instead of dying, so the message has to be its own rather than a
    // generic parse failure.
    assert!(err.to_string().contains("nests deeper"), "{err}");
}

// -- the boundary -----------------------------------------------------------

#[test]
fn nesting_at_the_limit_still_decodes() {
    // A ceiling that rejected real data would just move the outage. Anything
    // inside the documented bound must still work.
    let decoded = beve::from_slice::<Value>(&nested_arrays(MAX_RECURSION_DEPTH))
        .expect("nesting at the limit is legal input");
    assert!(decoded.is_array());
}

#[test]
fn nesting_one_past_the_limit_is_refused() {
    let err = beve::from_slice::<Value>(&nested_arrays(MAX_RECURSION_DEPTH + 1))
        .expect_err("one level past the limit must be refused");
    assert!(is_too_deep(&err), "{err}");
}

#[test]
fn objects_are_bounded_too() {
    assert!(beve::from_slice::<Value>(&nested_objects(MAX_RECURSION_DEPTH)).is_ok());
    let err = beve::from_slice::<Value>(&nested_objects(MAX_RECURSION_DEPTH + 1))
        .expect_err("nested objects are bounded on the same counter");
    assert!(is_too_deep(&err), "{err}");
}

// -- the counter is depth, not volume ---------------------------------------

#[test]
fn siblings_do_not_accumulate_depth() {
    // The failure mode of a counter that is incremented but never restored: a
    // wide, shallow document would be refused for being long. Ten thousand
    // siblings two levels deep is ordinary data.
    let wide: Vec<Vec<u8>> = (0..10_000u32).map(|_| vec![1u8, 2, 3]).collect();
    let bytes = beve::to_vec(&wide).unwrap();
    let decoded: Vec<Vec<u8>> = beve::from_slice(&bytes).expect("width is not depth");
    assert_eq!(decoded.len(), 10_000);
}

#[test]
fn a_refused_document_does_not_poison_a_later_one() {
    // The depth is restored on the error path too, so a caller that recovers
    // and decodes again does not inherit a phantom level.
    assert!(beve::from_slice::<Value>(&nested_arrays(MAX_RECURSION_DEPTH + 1)).is_err());
    assert!(beve::from_slice::<Value>(&nested_arrays(MAX_RECURSION_DEPTH)).is_ok());
}

// -- every entry point ------------------------------------------------------

#[test]
fn the_streaming_decoder_is_bounded() {
    // It needs the bound more than the slice decoder: a reader need not hold
    // the whole document, so the input length that incidentally caps nesting
    // for a slice does not exist here at all.
    let bomb = nested_arrays(20_000);
    let err = beve::from_reader_streaming::<_, Value>(bomb.as_slice())
        .expect_err("the streaming decoder must be bounded too");
    assert!(is_too_deep(&err), "{err}");

    assert!(
        beve::from_reader_streaming::<_, Value>(nested_arrays(MAX_RECURSION_DEPTH).as_slice())
            .is_ok()
    );
}

#[test]
fn skipping_a_value_is_bounded() {
    // `skip_value` does not build anything, but it still *walks* the value, so
    // it recurses exactly as decoding does and ends the same way. It is public
    // API and the navigation behind `from_field` runs on it.
    let bomb = nested_arrays(20_000);
    let mut pos = 0;
    let err = beve::skip_value(&bomb, &mut pos).expect_err("skipping must be bounded too");
    assert!(is_too_deep(&err), "{err}");

    let shallow = nested_arrays(MAX_RECURSION_DEPTH);
    let mut pos = 0;
    beve::skip_value(&shallow, &mut pos).expect("nesting at the limit still skips");
    assert_eq!(pos, shallow.len());
}

#[test]
fn every_entry_point_draws_the_line_in_the_same_place() {
    // A depth that `validate_slice` accepted but `from_slice` refused -- or that
    // `skip_value` refused one level earlier than either -- would be a trap, so
    // the limit is stated as a property of the document and they all agree.
    let ok = nested_arrays(MAX_RECURSION_DEPTH);
    let too_deep = nested_arrays(MAX_RECURSION_DEPTH + 1);

    assert!(beve::from_slice::<Value>(&ok).is_ok());
    assert!(beve::validate_slice(&ok).is_ok());
    assert!(beve::from_reader_streaming::<_, Value>(ok.as_slice()).is_ok());
    assert!(beve::skip_value(&ok, &mut 0).is_ok());

    assert!(beve::from_slice::<Value>(&too_deep).is_err());
    assert!(beve::validate_slice(&too_deep).is_err());
    assert!(beve::from_reader_streaming::<_, Value>(too_deep.as_slice()).is_err());
    assert!(beve::skip_value(&too_deep, &mut 0).is_err());
}

#[test]
fn json_conversion_is_bounded_on_the_same_constant() {
    // The BEVE->JSON walker had its own private ceiling with a different value,
    // which meant one crate gave two answers to "how deep may input nest".
    let err =
        beve::beve_slice_to_json(&nested_arrays(20_000)).expect_err("conversion must be bounded");
    assert!(is_too_deep(&err), "{err}");
    assert!(beve::beve_slice_to_json(&nested_arrays(MAX_RECURSION_DEPTH)).is_ok());
}

#[test]
fn json_input_is_bounded_on_the_same_constant() {
    // The other direction, which is the one that could hand `from_slice` a
    // document it would then have aborted on: the JSON parser accepted 256
    // levels while the decoders now stop at 128.
    let err = beve::json_slice_to_beve(&nested_json_arrays(20_000))
        .expect_err("conversion must be bounded");
    assert!(is_too_deep(&err), "{err}");
    assert!(beve::json_slice_to_beve(&nested_json_arrays(MAX_RECURSION_DEPTH)).is_ok());
}

#[test]
fn validation_is_bounded() {
    // `validate_slice` walks the input with `IgnoredAny` rather than building a
    // value, but it walks it by the same recursion — a validator that aborted
    // on the input it exists to screen would be worse than useless.
    let err = beve::validate_slice(&nested_arrays(20_000))
        .expect_err("validation must be bounded on the same counter");
    assert!(is_too_deep(&err), "{err}");
}
