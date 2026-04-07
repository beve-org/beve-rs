#![deny(warnings)]

use beve::{Complex, from_slice, to_vec, to_vec_complex_slice, to_vec_typed_slice};
use proptest::collection::vec;
use proptest::prelude::*;
use proptest::sample::select;
use serde::{Deserialize, Serialize};

fn finite_f64() -> BoxedStrategy<f64> {
    any::<f64>()
        .prop_filter("finite f64", |v| v.is_finite())
        .boxed()
}

fn complex64_strategy() -> BoxedStrategy<Complex<f64>> {
    (finite_f64(), finite_f64())
        .prop_map(|(re, im)| Complex { re, im })
        .boxed()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Message {
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

fn meta_strategy() -> BoxedStrategy<Meta> {
    let window = (any::<u64>(), vec(any::<f32>(), 0..24))
        .prop_map(|(start, metrics)| Meta::Window { start, metrics })
        .boxed();
    prop_oneof![
        Just(Meta::Unit).boxed(),
        vec(any::<u32>(), 0..24).prop_map(Meta::Codes).boxed(),
        window,
    ]
    .boxed()
}

fn label_strategy() -> BoxedStrategy<String> {
    let alphabet: Vec<char> = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        .chars()
        .collect();
    prop::collection::vec(select(alphabet), 1..25)
        .prop_map(|chars| chars.into_iter().collect())
        .boxed()
}

fn payload_strategy() -> BoxedStrategy<Payload> {
    prop_oneof![
        vec(finite_f64(), 0..48)
            .prop_map(Payload::Telemetry)
            .boxed(),
        meta_strategy().prop_map(Payload::MetaOnly).boxed(),
        (label_strategy(), vec(any::<u32>(), 0..32))
            .prop_map(|(label, counters)| Payload::Snapshot { label, counters })
            .boxed(),
    ]
    .boxed()
}

fn message_strategy() -> BoxedStrategy<Message> {
    (any::<u32>(), payload_strategy())
        .prop_map(|(id, payload)| Message { id, payload })
        .boxed()
}

proptest! {
    #[test]
    fn prop_numeric_roundtrip_u64(values in vec(any::<u64>(), 0..96)) {
        let bytes = to_vec_typed_slice(&values);
        let decoded: Vec<u64> = from_slice(&bytes).expect("decode u64 slice");
        prop_assert_eq!(decoded, values);
    }
}

proptest! {
    #[test]
    fn prop_numeric_roundtrip_f64(values in vec(finite_f64(), 0..96)) {
        let bytes = to_vec_typed_slice(&values);
        let decoded: Vec<f64> = from_slice(&bytes).expect("decode f64 slice");
        prop_assert_eq!(decoded, values);
    }
}

proptest! {
    #[test]
    fn prop_complex_roundtrip(values in vec(complex64_strategy(), 0..48)) {
        let bytes = to_vec_complex_slice(&values);
        let decoded: Vec<Complex<f64>> = from_slice(&bytes).expect("decode complex64 slice");
        prop_assert_eq!(decoded, values);
    }
}

proptest! {
    #[test]
    fn prop_nested_enum_roundtrip(message in message_strategy()) {
        let bytes = to_vec(&message).expect("serialize message");
        let decoded: Message = from_slice(&bytes).expect("decode message");
        prop_assert_eq!(decoded, message);
    }
}
