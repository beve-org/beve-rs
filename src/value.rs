//! Dynamic Value type for schemaless BEVE deserialization.
//!
//! The [`Value`] enum can represent any BEVE value, allowing deserialization
//! when the schema is not known at compile time.
//!
//! # Example
//!
//! ```rust
//! use beve::Value;
//!
//! // Deserialize unknown BEVE data
//! let bytes = beve::to_vec(&42i32).unwrap();
//! let value: Value = beve::from_slice(&bytes).unwrap();
//!
//! // Access the value dynamically
//! assert_eq!(value.as_i64(), Some(42));
//! ```

use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::ser::{Serialize, SerializeMap, SerializeSeq, Serializer};
use std::collections::BTreeMap;
use std::fmt;

/// A BEVE number, which can be signed, unsigned, or floating-point.
#[derive(Clone, Debug, PartialEq)]
pub enum Number {
    /// Signed integer (up to 128-bit)
    Signed(i128),
    /// Unsigned integer (up to 128-bit)
    Unsigned(u128),
    /// Floating-point number (stored as f64)
    Float(f64),
}

impl Number {
    /// Returns `true` if the number is signed.
    pub fn is_signed(&self) -> bool {
        matches!(self, Number::Signed(_))
    }

    /// Returns `true` if the number is unsigned.
    pub fn is_unsigned(&self) -> bool {
        matches!(self, Number::Unsigned(_))
    }

    /// Returns `true` if the number is a float.
    pub fn is_float(&self) -> bool {
        matches!(self, Number::Float(_))
    }

    /// Returns the number as an i64, if it fits.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Number::Signed(n) => i64::try_from(*n).ok(),
            Number::Unsigned(n) => i64::try_from(*n).ok(),
            Number::Float(f) => {
                if f.fract() == 0.0 && *f >= i64::MIN as f64 && *f <= i64::MAX as f64 {
                    Some(*f as i64)
                } else {
                    None
                }
            }
        }
    }

    /// Returns the number as a u64, if it fits.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Number::Signed(n) => u64::try_from(*n).ok(),
            Number::Unsigned(n) => u64::try_from(*n).ok(),
            Number::Float(f) => {
                if f.fract() == 0.0 && *f >= 0.0 && *f <= u64::MAX as f64 {
                    Some(*f as u64)
                } else {
                    None
                }
            }
        }
    }

    /// Returns the number as an i128.
    pub fn as_i128(&self) -> Option<i128> {
        match self {
            Number::Signed(n) => Some(*n),
            Number::Unsigned(n) => i128::try_from(*n).ok(),
            Number::Float(_) => None,
        }
    }

    /// Returns the number as a u128.
    pub fn as_u128(&self) -> Option<u128> {
        match self {
            Number::Signed(n) => u128::try_from(*n).ok(),
            Number::Unsigned(n) => Some(*n),
            Number::Float(_) => None,
        }
    }

    /// Returns the number as an f64.
    pub fn as_f64(&self) -> f64 {
        match self {
            Number::Signed(n) => *n as f64,
            Number::Unsigned(n) => *n as f64,
            Number::Float(f) => *f,
        }
    }
}

impl fmt::Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Number::Signed(n) => write!(f, "{}", n),
            Number::Unsigned(n) => write!(f, "{}", n),
            Number::Float(n) => write!(f, "{}", n),
        }
    }
}

/// A map key that can be a string or integer.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Key {
    /// String key
    String(String),
    /// Signed integer key
    Signed(i128),
    /// Unsigned integer key
    Unsigned(u128),
}

impl Key {
    /// Returns the key as a string reference, if it is a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Key::String(s) => Some(s),
            _ => None,
        }
    }

    /// Returns the key as an i128, if it is a signed integer.
    pub fn as_i128(&self) -> Option<i128> {
        match self {
            Key::Signed(n) => Some(*n),
            _ => None,
        }
    }

    /// Returns the key as a u128, if it is an unsigned integer.
    pub fn as_u128(&self) -> Option<u128> {
        match self {
            Key::Unsigned(n) => Some(*n),
            _ => None,
        }
    }
}

impl fmt::Display for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Key::String(s) => write!(f, "{}", s),
            Key::Signed(n) => write!(f, "{}", n),
            Key::Unsigned(n) => write!(f, "{}", n),
        }
    }
}

impl From<String> for Key {
    fn from(s: String) -> Self {
        Key::String(s)
    }
}

impl From<&str> for Key {
    fn from(s: &str) -> Self {
        Key::String(s.to_owned())
    }
}

impl From<i128> for Key {
    fn from(n: i128) -> Self {
        Key::Signed(n)
    }
}

impl From<u128> for Key {
    fn from(n: u128) -> Self {
        Key::Unsigned(n)
    }
}

/// A BEVE object with ordered key-value pairs.
///
/// Keys can be strings or integers, matching BEVE's object key types.
/// Uses `BTreeMap` internally for deterministic ordering.
pub type Object = BTreeMap<Key, Value>;

/// Represents any valid BEVE value.
///
/// This enum can represent all BEVE types and enables dynamic/schemaless
/// deserialization of BEVE data.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// Represents a null value.
    Null,
    /// Represents a boolean.
    Bool(bool),
    /// Represents a number (signed, unsigned, or float).
    Number(Number),
    /// Represents a string.
    String(String),
    /// Represents an array of values.
    Array(Vec<Value>),
    /// Represents an object with string or integer keys.
    Object(Object),
}

impl Value {
    /// Returns `true` if the value is null.
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// Returns `true` if the value is a boolean.
    pub fn is_bool(&self) -> bool {
        matches!(self, Value::Bool(_))
    }

    /// Returns `true` if the value is a number.
    pub fn is_number(&self) -> bool {
        matches!(self, Value::Number(_))
    }

    /// Returns `true` if the value is a string.
    pub fn is_string(&self) -> bool {
        matches!(self, Value::String(_))
    }

    /// Returns `true` if the value is an array.
    pub fn is_array(&self) -> bool {
        matches!(self, Value::Array(_))
    }

    /// Returns `true` if the value is an object.
    pub fn is_object(&self) -> bool {
        matches!(self, Value::Object(_))
    }

    /// Returns the boolean value, if this is a `Bool`.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Returns a reference to the number, if this is a `Number`.
    pub fn as_number(&self) -> Option<&Number> {
        match self {
            Value::Number(n) => Some(n),
            _ => None,
        }
    }

    /// Returns the value as an i64, if applicable.
    pub fn as_i64(&self) -> Option<i64> {
        self.as_number().and_then(|n| n.as_i64())
    }

    /// Returns the value as a u64, if applicable.
    pub fn as_u64(&self) -> Option<u64> {
        self.as_number().and_then(|n| n.as_u64())
    }

    /// Returns the value as an f64, if applicable.
    pub fn as_f64(&self) -> Option<f64> {
        self.as_number().map(|n| n.as_f64())
    }

    /// Returns a reference to the string, if this is a `String`.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    /// Returns a reference to the array, if this is an `Array`.
    pub fn as_array(&self) -> Option<&Vec<Value>> {
        match self {
            Value::Array(arr) => Some(arr),
            _ => None,
        }
    }

    /// Returns a mutable reference to the array, if this is an `Array`.
    pub fn as_array_mut(&mut self) -> Option<&mut Vec<Value>> {
        match self {
            Value::Array(arr) => Some(arr),
            _ => None,
        }
    }

    /// Returns a reference to the object, if this is an `Object`.
    pub fn as_object(&self) -> Option<&Object> {
        match self {
            Value::Object(obj) => Some(obj),
            _ => None,
        }
    }

    /// Returns a mutable reference to the object, if this is an `Object`.
    pub fn as_object_mut(&mut self) -> Option<&mut Object> {
        match self {
            Value::Object(obj) => Some(obj),
            _ => None,
        }
    }

    /// Index into an array. Returns `None` if not an array or index out of bounds.
    pub fn get(&self, index: usize) -> Option<&Value> {
        self.as_array().and_then(|arr| arr.get(index))
    }

    /// Index into an object by string key. Returns `None` if not an object or key not found.
    pub fn get_key(&self, key: &str) -> Option<&Value> {
        self.as_object()
            .and_then(|obj| obj.get(&Key::String(key.to_owned())))
    }

    /// Index into an object by integer key. Returns `None` if not an object or key not found.
    pub fn get_int_key(&self, key: i128) -> Option<&Value> {
        self.as_object().and_then(|obj| obj.get(&Key::Signed(key)))
    }

    /// Index into an object by unsigned integer key. Returns `None` if not an object or key not found.
    pub fn get_uint_key(&self, key: u128) -> Option<&Value> {
        self.as_object()
            .and_then(|obj| obj.get(&Key::Unsigned(key)))
    }
}

impl Default for Value {
    fn default() -> Self {
        Value::Null
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Null => write!(f, "null"),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Number(n) => write!(f, "{}", n),
            Value::String(s) => write!(f, "\"{}\"", s),
            Value::Array(arr) => {
                write!(f, "[")?;
                for (i, v) in arr.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            }
            Value::Object(obj) => {
                write!(f, "{{")?;
                for (i, (k, v)) in obj.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", k, v)?;
                }
                write!(f, "}}")
            }
        }
    }
}

// ============ From implementations ============

impl From<()> for Value {
    fn from(_: ()) -> Self {
        Value::Null
    }
}

impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Value::Bool(b)
    }
}

impl From<i8> for Value {
    fn from(n: i8) -> Self {
        Value::Number(Number::Signed(n as i128))
    }
}

impl From<i16> for Value {
    fn from(n: i16) -> Self {
        Value::Number(Number::Signed(n as i128))
    }
}

impl From<i32> for Value {
    fn from(n: i32) -> Self {
        Value::Number(Number::Signed(n as i128))
    }
}

impl From<i64> for Value {
    fn from(n: i64) -> Self {
        Value::Number(Number::Signed(n as i128))
    }
}

impl From<i128> for Value {
    fn from(n: i128) -> Self {
        Value::Number(Number::Signed(n))
    }
}

impl From<u8> for Value {
    fn from(n: u8) -> Self {
        Value::Number(Number::Unsigned(n as u128))
    }
}

impl From<u16> for Value {
    fn from(n: u16) -> Self {
        Value::Number(Number::Unsigned(n as u128))
    }
}

impl From<u32> for Value {
    fn from(n: u32) -> Self {
        Value::Number(Number::Unsigned(n as u128))
    }
}

impl From<u64> for Value {
    fn from(n: u64) -> Self {
        Value::Number(Number::Unsigned(n as u128))
    }
}

impl From<u128> for Value {
    fn from(n: u128) -> Self {
        Value::Number(Number::Unsigned(n))
    }
}

impl From<f32> for Value {
    fn from(n: f32) -> Self {
        Value::Number(Number::Float(n as f64))
    }
}

impl From<f64> for Value {
    fn from(n: f64) -> Self {
        Value::Number(Number::Float(n))
    }
}

impl From<String> for Value {
    fn from(s: String) -> Self {
        Value::String(s)
    }
}

impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Value::String(s.to_owned())
    }
}

impl<T: Into<Value>> From<Vec<T>> for Value {
    fn from(v: Vec<T>) -> Self {
        Value::Array(v.into_iter().map(Into::into).collect())
    }
}

impl<T: Into<Value>> From<Option<T>> for Value {
    fn from(opt: Option<T>) -> Self {
        match opt {
            Some(v) => v.into(),
            None => Value::Null,
        }
    }
}

// ============ Deserialize implementation ============

impl<'de> Deserialize<'de> for Value {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(ValueVisitor)
    }
}

struct ValueVisitor;

impl<'de> Visitor<'de> for ValueVisitor {
    type Value = Value;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("any valid BEVE value")
    }

    fn visit_unit<E: de::Error>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_none<E: de::Error>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_some<D: Deserializer<'de>>(self, deserializer: D) -> Result<Value, D::Error> {
        Deserialize::deserialize(deserializer)
    }

    fn visit_bool<E: de::Error>(self, v: bool) -> Result<Value, E> {
        Ok(Value::Bool(v))
    }

    fn visit_i8<E: de::Error>(self, v: i8) -> Result<Value, E> {
        Ok(Value::Number(Number::Signed(v as i128)))
    }

    fn visit_i16<E: de::Error>(self, v: i16) -> Result<Value, E> {
        Ok(Value::Number(Number::Signed(v as i128)))
    }

    fn visit_i32<E: de::Error>(self, v: i32) -> Result<Value, E> {
        Ok(Value::Number(Number::Signed(v as i128)))
    }

    fn visit_i64<E: de::Error>(self, v: i64) -> Result<Value, E> {
        Ok(Value::Number(Number::Signed(v as i128)))
    }

    fn visit_i128<E: de::Error>(self, v: i128) -> Result<Value, E> {
        Ok(Value::Number(Number::Signed(v)))
    }

    fn visit_u8<E: de::Error>(self, v: u8) -> Result<Value, E> {
        Ok(Value::Number(Number::Unsigned(v as u128)))
    }

    fn visit_u16<E: de::Error>(self, v: u16) -> Result<Value, E> {
        Ok(Value::Number(Number::Unsigned(v as u128)))
    }

    fn visit_u32<E: de::Error>(self, v: u32) -> Result<Value, E> {
        Ok(Value::Number(Number::Unsigned(v as u128)))
    }

    fn visit_u64<E: de::Error>(self, v: u64) -> Result<Value, E> {
        Ok(Value::Number(Number::Unsigned(v as u128)))
    }

    fn visit_u128<E: de::Error>(self, v: u128) -> Result<Value, E> {
        Ok(Value::Number(Number::Unsigned(v)))
    }

    fn visit_f32<E: de::Error>(self, v: f32) -> Result<Value, E> {
        Ok(Value::Number(Number::Float(v as f64)))
    }

    fn visit_f64<E: de::Error>(self, v: f64) -> Result<Value, E> {
        Ok(Value::Number(Number::Float(v)))
    }

    fn visit_str<E: de::Error>(self, v: &str) -> Result<Value, E> {
        Ok(Value::String(v.to_owned()))
    }

    fn visit_borrowed_str<E: de::Error>(self, v: &'de str) -> Result<Value, E> {
        Ok(Value::String(v.to_owned()))
    }

    fn visit_string<E: de::Error>(self, v: String) -> Result<Value, E> {
        Ok(Value::String(v))
    }

    fn visit_bytes<E: de::Error>(self, v: &[u8]) -> Result<Value, E> {
        // Represent bytes as an array of unsigned integers
        Ok(Value::Array(
            v.iter().map(|&b| Value::Number(Number::Unsigned(b as u128))).collect(),
        ))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Value, A::Error> {
        let mut values = Vec::new();
        if let Some(hint) = seq.size_hint() {
            values.reserve(hint);
        }
        while let Some(elem) = seq.next_element()? {
            values.push(elem);
        }
        Ok(Value::Array(values))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
        let mut values = Object::new();
        while let Some((key, value)) = map.next_entry::<KeyDeserialize, Value>()? {
            values.insert(key.0, value);
        }
        Ok(Value::Object(values))
    }

    fn visit_enum<A: de::EnumAccess<'de>>(self, data: A) -> Result<Value, A::Error> {
        // For enums, we'll represent them as an object with variant info
        let (variant, _): (String, _) = data.variant()?;
        // Simple representation: just the variant name as a string
        // A more complete representation would include the variant data
        Ok(Value::String(variant))
    }
}

// Helper for deserializing map keys
struct KeyDeserialize(Key);

impl<'de> Deserialize<'de> for KeyDeserialize {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(KeyVisitor)
    }
}

struct KeyVisitor;

impl<'de> Visitor<'de> for KeyVisitor {
    type Value = KeyDeserialize;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a string or integer key")
    }

    fn visit_str<E: de::Error>(self, v: &str) -> Result<KeyDeserialize, E> {
        Ok(KeyDeserialize(Key::String(v.to_owned())))
    }

    fn visit_borrowed_str<E: de::Error>(self, v: &'de str) -> Result<KeyDeserialize, E> {
        Ok(KeyDeserialize(Key::String(v.to_owned())))
    }

    fn visit_string<E: de::Error>(self, v: String) -> Result<KeyDeserialize, E> {
        Ok(KeyDeserialize(Key::String(v)))
    }

    fn visit_i8<E: de::Error>(self, v: i8) -> Result<KeyDeserialize, E> {
        Ok(KeyDeserialize(Key::Signed(v as i128)))
    }

    fn visit_i16<E: de::Error>(self, v: i16) -> Result<KeyDeserialize, E> {
        Ok(KeyDeserialize(Key::Signed(v as i128)))
    }

    fn visit_i32<E: de::Error>(self, v: i32) -> Result<KeyDeserialize, E> {
        Ok(KeyDeserialize(Key::Signed(v as i128)))
    }

    fn visit_i64<E: de::Error>(self, v: i64) -> Result<KeyDeserialize, E> {
        Ok(KeyDeserialize(Key::Signed(v as i128)))
    }

    fn visit_i128<E: de::Error>(self, v: i128) -> Result<KeyDeserialize, E> {
        Ok(KeyDeserialize(Key::Signed(v)))
    }

    fn visit_u8<E: de::Error>(self, v: u8) -> Result<KeyDeserialize, E> {
        Ok(KeyDeserialize(Key::Unsigned(v as u128)))
    }

    fn visit_u16<E: de::Error>(self, v: u16) -> Result<KeyDeserialize, E> {
        Ok(KeyDeserialize(Key::Unsigned(v as u128)))
    }

    fn visit_u32<E: de::Error>(self, v: u32) -> Result<KeyDeserialize, E> {
        Ok(KeyDeserialize(Key::Unsigned(v as u128)))
    }

    fn visit_u64<E: de::Error>(self, v: u64) -> Result<KeyDeserialize, E> {
        Ok(KeyDeserialize(Key::Unsigned(v as u128)))
    }

    fn visit_u128<E: de::Error>(self, v: u128) -> Result<KeyDeserialize, E> {
        Ok(KeyDeserialize(Key::Unsigned(v)))
    }
}

// ============ Serialize implementation ============

impl Serialize for Value {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Value::Null => serializer.serialize_unit(),
            Value::Bool(b) => serializer.serialize_bool(*b),
            Value::Number(n) => n.serialize(serializer),
            Value::String(s) => serializer.serialize_str(s),
            Value::Array(arr) => {
                let mut seq = serializer.serialize_seq(Some(arr.len()))?;
                for elem in arr {
                    seq.serialize_element(elem)?;
                }
                seq.end()
            }
            Value::Object(obj) => {
                let mut map = serializer.serialize_map(Some(obj.len()))?;
                for (k, v) in obj {
                    map.serialize_entry(k, v)?;
                }
                map.end()
            }
        }
    }
}

impl Serialize for Number {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            // Choose smallest fitting type for better BEVE encoding
            Number::Signed(n) => {
                if *n >= i8::MIN as i128 && *n <= i8::MAX as i128 {
                    serializer.serialize_i8(*n as i8)
                } else if *n >= i16::MIN as i128 && *n <= i16::MAX as i128 {
                    serializer.serialize_i16(*n as i16)
                } else if *n >= i32::MIN as i128 && *n <= i32::MAX as i128 {
                    serializer.serialize_i32(*n as i32)
                } else if *n >= i64::MIN as i128 && *n <= i64::MAX as i128 {
                    serializer.serialize_i64(*n as i64)
                } else {
                    serializer.serialize_i128(*n)
                }
            }
            Number::Unsigned(n) => {
                if *n <= u8::MAX as u128 {
                    serializer.serialize_u8(*n as u8)
                } else if *n <= u16::MAX as u128 {
                    serializer.serialize_u16(*n as u16)
                } else if *n <= u32::MAX as u128 {
                    serializer.serialize_u32(*n as u32)
                } else if *n <= u64::MAX as u128 {
                    serializer.serialize_u64(*n as u64)
                } else {
                    serializer.serialize_u128(*n)
                }
            }
            Number::Float(f) => serializer.serialize_f64(*f),
        }
    }
}

impl Serialize for Key {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Key::String(s) => serializer.serialize_str(s),
            Key::Signed(n) => {
                if *n >= i8::MIN as i128 && *n <= i8::MAX as i128 {
                    serializer.serialize_i8(*n as i8)
                } else if *n >= i16::MIN as i128 && *n <= i16::MAX as i128 {
                    serializer.serialize_i16(*n as i16)
                } else if *n >= i32::MIN as i128 && *n <= i32::MAX as i128 {
                    serializer.serialize_i32(*n as i32)
                } else if *n >= i64::MIN as i128 && *n <= i64::MAX as i128 {
                    serializer.serialize_i64(*n as i64)
                } else {
                    serializer.serialize_i128(*n)
                }
            }
            Key::Unsigned(n) => {
                if *n <= u8::MAX as u128 {
                    serializer.serialize_u8(*n as u8)
                } else if *n <= u16::MAX as u128 {
                    serializer.serialize_u16(*n as u16)
                } else if *n <= u32::MAX as u128 {
                    serializer.serialize_u32(*n as u32)
                } else if *n <= u64::MAX as u128 {
                    serializer.serialize_u64(*n as u64)
                } else {
                    serializer.serialize_u128(*n)
                }
            }
        }
    }
}

// ============ Index implementations ============

impl std::ops::Index<usize> for Value {
    type Output = Value;

    fn index(&self, index: usize) -> &Self::Output {
        static NULL: Value = Value::Null;
        self.get(index).unwrap_or(&NULL)
    }
}

impl std::ops::Index<&str> for Value {
    type Output = Value;

    fn index(&self, key: &str) -> &Self::Output {
        static NULL: Value = Value::Null;
        self.get_key(key).unwrap_or(&NULL)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_creation() {
        assert!(Value::Null.is_null());
        assert!(Value::from(true).is_bool());
        assert!(Value::from(42i32).is_number());
        assert!(Value::from("hello").is_string());
        assert!(Value::from(vec![1, 2, 3]).is_array());
    }

    #[test]
    fn test_number_conversions() {
        let num = Number::Signed(42);
        assert_eq!(num.as_i64(), Some(42));
        assert_eq!(num.as_u64(), Some(42));
        assert_eq!(num.as_f64(), 42.0);

        let num = Number::Unsigned(100);
        assert_eq!(num.as_i64(), Some(100));
        assert_eq!(num.as_u64(), Some(100));

        let num = Number::Float(3.14);
        assert_eq!(num.as_i64(), None); // Has fractional part
        assert!((num.as_f64() - 3.14).abs() < f64::EPSILON);
    }

    #[test]
    fn test_value_accessors() {
        let v = Value::from(42i32);
        assert_eq!(v.as_i64(), Some(42));

        let v = Value::from("test");
        assert_eq!(v.as_str(), Some("test"));

        let v = Value::Array(vec![Value::from(1), Value::from(2)]);
        assert_eq!(v.get(0), Some(&Value::from(1)));
        assert_eq!(v.get(5), None);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Value::Null), "null");
        assert_eq!(format!("{}", Value::Bool(true)), "true");
        assert_eq!(format!("{}", Value::Number(Number::Signed(42))), "42");
        assert_eq!(format!("{}", Value::String("hi".into())), "\"hi\"");
    }
}
