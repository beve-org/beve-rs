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
#[derive(Clone, Debug, PartialEq, Default)]
pub enum Value {
    /// Represents a null value.
    #[default]
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
            v.iter()
                .map(|&b| Value::Number(Number::Unsigned(b as u128)))
                .collect(),
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

// ============ Value as Deserializer (for from_value) ============

/// Error type for Value deserialization
#[derive(Debug)]
pub struct ValueError(String);

impl std::fmt::Display for ValueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ValueError {}

impl de::Error for ValueError {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        ValueError(msg.to_string())
    }
}

// Helper methods for integer conversion in Deserializer
impl Value {
    fn to_i128(&self) -> Result<i128, ValueError> {
        match self {
            Value::Number(Number::Signed(i)) => Ok(*i),
            Value::Number(Number::Unsigned(u)) => {
                if *u <= i128::MAX as u128 {
                    Ok(*u as i128)
                } else {
                    Err(ValueError("integer overflow".to_string()))
                }
            }
            _ => Err(ValueError("expected integer".to_string())),
        }
    }

    fn to_u128(&self) -> Result<u128, ValueError> {
        match self {
            Value::Number(Number::Unsigned(u)) => Ok(*u),
            Value::Number(Number::Signed(i)) => {
                if *i >= 0 {
                    Ok(*i as u128)
                } else {
                    Err(ValueError("expected unsigned integer".to_string()))
                }
            }
            _ => Err(ValueError("expected integer".to_string())),
        }
    }
}

/// Deserialize a concrete type from a `Value`.
///
/// This allows efficient conversion from `Value` to typed data without
/// going through BEVE bytes.
///
/// # Example
///
/// ```rust
/// use beve::{Value, Number, Object, Key, from_value};
/// use serde::Deserialize;
///
/// #[derive(Deserialize, Debug, PartialEq)]
/// struct Point { x: f64, y: f64 }
///
/// let mut obj = Object::new();
/// obj.insert(Key::String("x".to_string()), Value::Number(Number::Float(1.0)));
/// obj.insert(Key::String("y".to_string()), Value::Number(Number::Float(2.0)));
/// let value = Value::Object(obj);
///
/// let point: Point = from_value(value).unwrap();
/// assert_eq!(point, Point { x: 1.0, y: 2.0 });
/// ```
pub fn from_value<T: de::DeserializeOwned>(value: Value) -> Result<T, ValueError> {
    T::deserialize(value)
}

/// Deserialize a concrete type from a `Value` reference.
///
/// This is useful when you want to keep the `Value` around after deserialization.
pub fn from_value_ref<'de, T: Deserialize<'de>>(value: &'de Value) -> Result<T, ValueError> {
    T::deserialize(value)
}

impl<'de> Deserializer<'de> for Value {
    type Error = ValueError;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        match self {
            Value::Null => visitor.visit_unit(),
            Value::Bool(b) => visitor.visit_bool(b),
            Value::Number(n) => match n {
                Number::Signed(i) => {
                    // Use smallest fitting type for better compatibility
                    if i >= i8::MIN as i128 && i <= i8::MAX as i128 {
                        visitor.visit_i8(i as i8)
                    } else if i >= i16::MIN as i128 && i <= i16::MAX as i128 {
                        visitor.visit_i16(i as i16)
                    } else if i >= i32::MIN as i128 && i <= i32::MAX as i128 {
                        visitor.visit_i32(i as i32)
                    } else if i >= i64::MIN as i128 && i <= i64::MAX as i128 {
                        visitor.visit_i64(i as i64)
                    } else {
                        visitor.visit_i128(i)
                    }
                }
                Number::Unsigned(u) => {
                    if u <= u8::MAX as u128 {
                        visitor.visit_u8(u as u8)
                    } else if u <= u16::MAX as u128 {
                        visitor.visit_u16(u as u16)
                    } else if u <= u32::MAX as u128 {
                        visitor.visit_u32(u as u32)
                    } else if u <= u64::MAX as u128 {
                        visitor.visit_u64(u as u64)
                    } else {
                        visitor.visit_u128(u)
                    }
                }
                Number::Float(f) => visitor.visit_f64(f),
            },
            Value::String(s) => visitor.visit_string(s),
            Value::Array(arr) => {
                let len = arr.len();
                let mut seq = ValueSeqAccess {
                    iter: arr.into_iter(),
                };
                let result = visitor.visit_seq(&mut seq)?;
                if seq.iter.len() != 0 {
                    return Err(de::Error::custom(format!(
                        "expected {} elements, got more",
                        len
                    )));
                }
                Ok(result)
            }
            Value::Object(obj) => {
                let mut map = ValueMapAccess {
                    iter: obj.into_iter(),
                    pending_value: None,
                };
                let result = visitor.visit_map(&mut map)?;
                Ok(result)
            }
        }
    }

    fn deserialize_bool<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        match self {
            Value::Bool(b) => visitor.visit_bool(b),
            _ => Err(de::Error::custom("expected bool")),
        }
    }

    fn deserialize_i8<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        let i = self.to_i128()?;
        if i >= i8::MIN as i128 && i <= i8::MAX as i128 {
            visitor.visit_i8(i as i8)
        } else {
            Err(de::Error::custom("integer out of range for i8"))
        }
    }

    fn deserialize_i16<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        let i = self.to_i128()?;
        if i >= i16::MIN as i128 && i <= i16::MAX as i128 {
            visitor.visit_i16(i as i16)
        } else {
            Err(de::Error::custom("integer out of range for i16"))
        }
    }

    fn deserialize_i32<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        let i = self.to_i128()?;
        if i >= i32::MIN as i128 && i <= i32::MAX as i128 {
            visitor.visit_i32(i as i32)
        } else {
            Err(de::Error::custom("integer out of range for i32"))
        }
    }

    fn deserialize_i64<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        let i = self.to_i128()?;
        if i >= i64::MIN as i128 && i <= i64::MAX as i128 {
            visitor.visit_i64(i as i64)
        } else {
            Err(de::Error::custom("integer out of range for i64"))
        }
    }

    fn deserialize_i128<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        visitor.visit_i128(self.to_i128()?)
    }

    fn deserialize_u8<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        let u = self.to_u128()?;
        if u <= u8::MAX as u128 {
            visitor.visit_u8(u as u8)
        } else {
            Err(de::Error::custom("integer out of range for u8"))
        }
    }

    fn deserialize_u16<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        let u = self.to_u128()?;
        if u <= u16::MAX as u128 {
            visitor.visit_u16(u as u16)
        } else {
            Err(de::Error::custom("integer out of range for u16"))
        }
    }

    fn deserialize_u32<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        let u = self.to_u128()?;
        if u <= u32::MAX as u128 {
            visitor.visit_u32(u as u32)
        } else {
            Err(de::Error::custom("integer out of range for u32"))
        }
    }

    fn deserialize_u64<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        let u = self.to_u128()?;
        if u <= u64::MAX as u128 {
            visitor.visit_u64(u as u64)
        } else {
            Err(de::Error::custom("integer out of range for u64"))
        }
    }

    fn deserialize_u128<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        visitor.visit_u128(self.to_u128()?)
    }

    fn deserialize_f32<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        self.deserialize_f64(visitor)
    }

    fn deserialize_f64<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        match self {
            Value::Number(Number::Float(f)) => visitor.visit_f64(f),
            Value::Number(Number::Signed(i)) => visitor.visit_f64(i as f64),
            Value::Number(Number::Unsigned(u)) => visitor.visit_f64(u as f64),
            _ => Err(de::Error::custom("expected number")),
        }
    }

    fn deserialize_char<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        match self {
            Value::String(s) => {
                let mut chars = s.chars();
                if let Some(c) = chars.next() {
                    if chars.next().is_none() {
                        return visitor.visit_char(c);
                    }
                }
                Err(de::Error::custom("expected single character"))
            }
            _ => Err(de::Error::custom("expected string")),
        }
    }

    fn deserialize_str<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        match self {
            Value::String(s) => visitor.visit_string(s),
            _ => Err(de::Error::custom("expected string")),
        }
    }

    fn deserialize_string<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        self.deserialize_str(visitor)
    }

    fn deserialize_bytes<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        match self {
            Value::Array(arr) => {
                let bytes: Result<Vec<u8>, _> = arr
                    .into_iter()
                    .map(|v| match v {
                        Value::Number(Number::Unsigned(u)) if u <= 255 => Ok(u as u8),
                        Value::Number(Number::Signed(i)) if (0..=255).contains(&i) => Ok(i as u8),
                        _ => Err(de::Error::custom("expected byte array")),
                    })
                    .collect();
                visitor.visit_byte_buf(bytes?)
            }
            _ => Err(de::Error::custom("expected array")),
        }
    }

    fn deserialize_byte_buf<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        self.deserialize_bytes(visitor)
    }

    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        match self {
            Value::Null => visitor.visit_none(),
            _ => visitor.visit_some(self),
        }
    }

    fn deserialize_unit<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        match self {
            Value::Null => visitor.visit_unit(),
            _ => Err(de::Error::custom("expected null")),
        }
    }

    fn deserialize_unit_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        self.deserialize_unit(visitor)
    }

    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        visitor.visit_newtype_struct(self)
    }

    fn deserialize_seq<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        match self {
            Value::Array(arr) => {
                let mut seq = ValueSeqAccess {
                    iter: arr.into_iter(),
                };
                visitor.visit_seq(&mut seq)
            }
            _ => Err(de::Error::custom("expected array")),
        }
    }

    fn deserialize_tuple<V: Visitor<'de>>(
        self,
        _len: usize,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        self.deserialize_seq(visitor)
    }

    fn deserialize_tuple_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _len: usize,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        self.deserialize_seq(visitor)
    }

    fn deserialize_map<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        match self {
            Value::Object(obj) => {
                let mut map = ValueMapAccess {
                    iter: obj.into_iter(),
                    pending_value: None,
                };
                visitor.visit_map(&mut map)
            }
            _ => Err(de::Error::custom("expected object")),
        }
    }

    fn deserialize_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        self.deserialize_map(visitor)
    }

    fn deserialize_enum<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        match self {
            Value::String(s) => visitor.visit_enum(ValueEnumAccess::Unit(s)),
            Value::Object(obj) => {
                if obj.len() != 1 {
                    return Err(de::Error::custom("expected enum with single key"));
                }
                let (key, value) = obj.into_iter().next().unwrap();
                match key {
                    Key::String(s) => visitor.visit_enum(ValueEnumAccess::WithValue(s, value)),
                    _ => Err(de::Error::custom("expected string key for enum")),
                }
            }
            _ => Err(de::Error::custom("expected string or object for enum")),
        }
    }

    fn deserialize_identifier<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        self.deserialize_str(visitor)
    }

    fn deserialize_ignored_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        visitor.visit_unit()
    }
}

// Deserializer for &Value (borrowed version)
impl<'de> Deserializer<'de> for &'de Value {
    type Error = ValueError;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        match self {
            Value::Null => visitor.visit_unit(),
            Value::Bool(b) => visitor.visit_bool(*b),
            Value::Number(n) => match n {
                Number::Signed(i) => {
                    let i = *i;
                    if i >= i8::MIN as i128 && i <= i8::MAX as i128 {
                        visitor.visit_i8(i as i8)
                    } else if i >= i16::MIN as i128 && i <= i16::MAX as i128 {
                        visitor.visit_i16(i as i16)
                    } else if i >= i32::MIN as i128 && i <= i32::MAX as i128 {
                        visitor.visit_i32(i as i32)
                    } else if i >= i64::MIN as i128 && i <= i64::MAX as i128 {
                        visitor.visit_i64(i as i64)
                    } else {
                        visitor.visit_i128(i)
                    }
                }
                Number::Unsigned(u) => {
                    let u = *u;
                    if u <= u8::MAX as u128 {
                        visitor.visit_u8(u as u8)
                    } else if u <= u16::MAX as u128 {
                        visitor.visit_u16(u as u16)
                    } else if u <= u32::MAX as u128 {
                        visitor.visit_u32(u as u32)
                    } else if u <= u64::MAX as u128 {
                        visitor.visit_u64(u as u64)
                    } else {
                        visitor.visit_u128(u)
                    }
                }
                Number::Float(f) => visitor.visit_f64(*f),
            },
            Value::String(s) => visitor.visit_borrowed_str(s),
            Value::Array(arr) => {
                let mut seq = ValueSeqAccessRef { iter: arr.iter() };
                visitor.visit_seq(&mut seq)
            }
            Value::Object(obj) => {
                let mut map = ValueMapAccessRef {
                    iter: obj.iter(),
                    pending_value: None,
                };
                visitor.visit_map(&mut map)
            }
        }
    }

    fn deserialize_bool<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        match self {
            Value::Bool(b) => visitor.visit_bool(*b),
            _ => Err(de::Error::custom("expected bool")),
        }
    }

    fn deserialize_i8<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        let i = self.to_i128()?;
        if i >= i8::MIN as i128 && i <= i8::MAX as i128 {
            visitor.visit_i8(i as i8)
        } else {
            Err(de::Error::custom("integer out of range for i8"))
        }
    }

    fn deserialize_i16<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        let i = self.to_i128()?;
        if i >= i16::MIN as i128 && i <= i16::MAX as i128 {
            visitor.visit_i16(i as i16)
        } else {
            Err(de::Error::custom("integer out of range for i16"))
        }
    }

    fn deserialize_i32<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        let i = self.to_i128()?;
        if i >= i32::MIN as i128 && i <= i32::MAX as i128 {
            visitor.visit_i32(i as i32)
        } else {
            Err(de::Error::custom("integer out of range for i32"))
        }
    }

    fn deserialize_i64<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        let i = self.to_i128()?;
        if i >= i64::MIN as i128 && i <= i64::MAX as i128 {
            visitor.visit_i64(i as i64)
        } else {
            Err(de::Error::custom("integer out of range for i64"))
        }
    }

    fn deserialize_i128<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        visitor.visit_i128(self.to_i128()?)
    }

    fn deserialize_u8<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        let u = self.to_u128()?;
        if u <= u8::MAX as u128 {
            visitor.visit_u8(u as u8)
        } else {
            Err(de::Error::custom("integer out of range for u8"))
        }
    }

    fn deserialize_u16<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        let u = self.to_u128()?;
        if u <= u16::MAX as u128 {
            visitor.visit_u16(u as u16)
        } else {
            Err(de::Error::custom("integer out of range for u16"))
        }
    }

    fn deserialize_u32<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        let u = self.to_u128()?;
        if u <= u32::MAX as u128 {
            visitor.visit_u32(u as u32)
        } else {
            Err(de::Error::custom("integer out of range for u32"))
        }
    }

    fn deserialize_u64<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        let u = self.to_u128()?;
        if u <= u64::MAX as u128 {
            visitor.visit_u64(u as u64)
        } else {
            Err(de::Error::custom("integer out of range for u64"))
        }
    }

    fn deserialize_u128<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        visitor.visit_u128(self.to_u128()?)
    }

    fn deserialize_f32<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        self.deserialize_f64(visitor)
    }

    fn deserialize_f64<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        match self {
            Value::Number(Number::Float(f)) => visitor.visit_f64(*f),
            Value::Number(Number::Signed(i)) => visitor.visit_f64(*i as f64),
            Value::Number(Number::Unsigned(u)) => visitor.visit_f64(*u as f64),
            _ => Err(de::Error::custom("expected number")),
        }
    }

    fn deserialize_char<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        match self {
            Value::String(s) => {
                let mut chars = s.chars();
                if let Some(c) = chars.next() {
                    if chars.next().is_none() {
                        return visitor.visit_char(c);
                    }
                }
                Err(de::Error::custom("expected single character"))
            }
            _ => Err(de::Error::custom("expected string")),
        }
    }

    fn deserialize_str<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        match self {
            Value::String(s) => visitor.visit_borrowed_str(s),
            _ => Err(de::Error::custom("expected string")),
        }
    }

    fn deserialize_string<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        self.deserialize_str(visitor)
    }

    fn deserialize_bytes<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        match self {
            Value::Array(arr) => {
                let bytes: Result<Vec<u8>, _> = arr
                    .iter()
                    .map(|v| match v {
                        Value::Number(Number::Unsigned(u)) if *u <= 255 => Ok(*u as u8),
                        Value::Number(Number::Signed(i)) if (0..=255).contains(i) => Ok(*i as u8),
                        _ => Err(de::Error::custom("expected byte array")),
                    })
                    .collect();
                visitor.visit_byte_buf(bytes?)
            }
            _ => Err(de::Error::custom("expected array")),
        }
    }

    fn deserialize_byte_buf<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        self.deserialize_bytes(visitor)
    }

    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        match self {
            Value::Null => visitor.visit_none(),
            _ => visitor.visit_some(self),
        }
    }

    fn deserialize_unit<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        match self {
            Value::Null => visitor.visit_unit(),
            _ => Err(de::Error::custom("expected null")),
        }
    }

    fn deserialize_unit_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        self.deserialize_unit(visitor)
    }

    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        visitor.visit_newtype_struct(self)
    }

    fn deserialize_seq<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        match self {
            Value::Array(arr) => {
                let mut seq = ValueSeqAccessRef { iter: arr.iter() };
                visitor.visit_seq(&mut seq)
            }
            _ => Err(de::Error::custom("expected array")),
        }
    }

    fn deserialize_tuple<V: Visitor<'de>>(
        self,
        _len: usize,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        self.deserialize_seq(visitor)
    }

    fn deserialize_tuple_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _len: usize,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        self.deserialize_seq(visitor)
    }

    fn deserialize_map<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        match self {
            Value::Object(obj) => {
                let mut map = ValueMapAccessRef {
                    iter: obj.iter(),
                    pending_value: None,
                };
                visitor.visit_map(&mut map)
            }
            _ => Err(de::Error::custom("expected object")),
        }
    }

    fn deserialize_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        self.deserialize_map(visitor)
    }

    fn deserialize_enum<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        match self {
            Value::String(s) => visitor.visit_enum(ValueEnumAccessRef::Unit(s)),
            Value::Object(obj) => {
                if obj.len() != 1 {
                    return Err(de::Error::custom("expected enum with single key"));
                }
                let (key, value) = obj.iter().next().unwrap();
                match key {
                    Key::String(s) => visitor.visit_enum(ValueEnumAccessRef::WithValue(s, value)),
                    _ => Err(de::Error::custom("expected string key for enum")),
                }
            }
            _ => Err(de::Error::custom("expected string or object for enum")),
        }
    }

    fn deserialize_identifier<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        self.deserialize_str(visitor)
    }

    fn deserialize_ignored_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        visitor.visit_unit()
    }
}

// SeqAccess for owned Value
struct ValueSeqAccess {
    iter: std::vec::IntoIter<Value>,
}

impl<'de> SeqAccess<'de> for ValueSeqAccess {
    type Error = ValueError;

    fn next_element_seed<T: de::DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>, Self::Error> {
        match self.iter.next() {
            Some(value) => seed.deserialize(value).map(Some),
            None => Ok(None),
        }
    }

    fn size_hint(&self) -> Option<usize> {
        Some(self.iter.len())
    }
}

// SeqAccess for borrowed Value
struct ValueSeqAccessRef<'de> {
    iter: std::slice::Iter<'de, Value>,
}

impl<'de> SeqAccess<'de> for ValueSeqAccessRef<'de> {
    type Error = ValueError;

    fn next_element_seed<T: de::DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>, Self::Error> {
        match self.iter.next() {
            Some(value) => seed.deserialize(value).map(Some),
            None => Ok(None),
        }
    }

    fn size_hint(&self) -> Option<usize> {
        Some(self.iter.len())
    }
}

// MapAccess for owned Value
struct ValueMapAccess {
    iter: std::collections::btree_map::IntoIter<Key, Value>,
    pending_value: Option<Value>,
}

impl<'de> MapAccess<'de> for ValueMapAccess {
    type Error = ValueError;

    fn next_key_seed<K: de::DeserializeSeed<'de>>(
        &mut self,
        seed: K,
    ) -> Result<Option<K::Value>, Self::Error> {
        match self.iter.next() {
            Some((key, value)) => {
                self.pending_value = Some(value);
                let key_value = match key {
                    Key::String(s) => Value::String(s),
                    Key::Signed(i) => Value::Number(Number::Signed(i)),
                    Key::Unsigned(u) => Value::Number(Number::Unsigned(u)),
                };
                seed.deserialize(key_value).map(Some)
            }
            None => Ok(None),
        }
    }

    fn next_value_seed<V: de::DeserializeSeed<'de>>(
        &mut self,
        seed: V,
    ) -> Result<V::Value, Self::Error> {
        match self.pending_value.take() {
            Some(value) => seed.deserialize(value),
            None => Err(de::Error::custom("missing value")),
        }
    }

    fn size_hint(&self) -> Option<usize> {
        Some(self.iter.len())
    }
}

// MapAccess for borrowed Value
struct ValueMapAccessRef<'de> {
    iter: std::collections::btree_map::Iter<'de, Key, Value>,
    pending_value: Option<&'de Value>,
}

impl<'de> MapAccess<'de> for ValueMapAccessRef<'de> {
    type Error = ValueError;

    fn next_key_seed<K: de::DeserializeSeed<'de>>(
        &mut self,
        seed: K,
    ) -> Result<Option<K::Value>, Self::Error> {
        match self.iter.next() {
            Some((key, value)) => {
                self.pending_value = Some(value);
                seed.deserialize(KeyAsDeserializer(key)).map(Some)
            }
            None => Ok(None),
        }
    }

    fn next_value_seed<V: de::DeserializeSeed<'de>>(
        &mut self,
        seed: V,
    ) -> Result<V::Value, Self::Error> {
        match self.pending_value.take() {
            Some(value) => seed.deserialize(value),
            None => Err(de::Error::custom("missing value")),
        }
    }

    fn size_hint(&self) -> Option<usize> {
        Some(self.iter.len())
    }
}

// Key as deserializer for borrowed access
struct KeyAsDeserializer<'de>(&'de Key);

impl<'de> Deserializer<'de> for KeyAsDeserializer<'de> {
    type Error = ValueError;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        match self.0 {
            Key::String(s) => visitor.visit_borrowed_str(s),
            Key::Signed(i) => visitor.visit_i128(*i),
            Key::Unsigned(u) => visitor.visit_u128(*u),
        }
    }

    fn deserialize_str<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        match self.0 {
            Key::String(s) => visitor.visit_borrowed_str(s),
            _ => Err(de::Error::custom("expected string key")),
        }
    }

    fn deserialize_string<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        self.deserialize_str(visitor)
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char bytes byte_buf
        option unit unit_struct newtype_struct seq tuple tuple_struct map struct
        enum identifier ignored_any
    }
}

// EnumAccess for owned Value
enum ValueEnumAccess {
    Unit(String),
    WithValue(String, Value),
}

impl<'de> de::EnumAccess<'de> for ValueEnumAccess {
    type Error = ValueError;
    type Variant = ValueVariantAccess;

    fn variant_seed<V: de::DeserializeSeed<'de>>(
        self,
        seed: V,
    ) -> Result<(V::Value, Self::Variant), Self::Error> {
        match self {
            ValueEnumAccess::Unit(s) => {
                let variant = seed.deserialize(Value::String(s))?;
                Ok((variant, ValueVariantAccess::Unit))
            }
            ValueEnumAccess::WithValue(s, v) => {
                let variant = seed.deserialize(Value::String(s))?;
                Ok((variant, ValueVariantAccess::WithValue(v)))
            }
        }
    }
}

enum ValueVariantAccess {
    Unit,
    WithValue(Value),
}

impl<'de> de::VariantAccess<'de> for ValueVariantAccess {
    type Error = ValueError;

    fn unit_variant(self) -> Result<(), Self::Error> {
        match self {
            ValueVariantAccess::Unit => Ok(()),
            ValueVariantAccess::WithValue(Value::Null) => Ok(()),
            _ => Err(de::Error::custom("expected unit variant")),
        }
    }

    fn newtype_variant_seed<T: de::DeserializeSeed<'de>>(
        self,
        seed: T,
    ) -> Result<T::Value, Self::Error> {
        match self {
            ValueVariantAccess::WithValue(v) => seed.deserialize(v),
            ValueVariantAccess::Unit => Err(de::Error::custom("expected newtype variant")),
        }
    }

    fn tuple_variant<V: Visitor<'de>>(
        self,
        _len: usize,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        match self {
            ValueVariantAccess::WithValue(v) => v.deserialize_seq(visitor),
            ValueVariantAccess::Unit => Err(de::Error::custom("expected tuple variant")),
        }
    }

    fn struct_variant<V: Visitor<'de>>(
        self,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        match self {
            ValueVariantAccess::WithValue(v) => v.deserialize_map(visitor),
            ValueVariantAccess::Unit => Err(de::Error::custom("expected struct variant")),
        }
    }
}

// EnumAccess for borrowed Value
enum ValueEnumAccessRef<'de> {
    Unit(&'de str),
    WithValue(&'de str, &'de Value),
}

impl<'de> de::EnumAccess<'de> for ValueEnumAccessRef<'de> {
    type Error = ValueError;
    type Variant = ValueVariantAccessRef<'de>;

    fn variant_seed<V: de::DeserializeSeed<'de>>(
        self,
        seed: V,
    ) -> Result<(V::Value, Self::Variant), Self::Error> {
        match self {
            ValueEnumAccessRef::Unit(s) => {
                let variant = seed.deserialize(de::value::BorrowedStrDeserializer::new(s))?;
                Ok((variant, ValueVariantAccessRef::Unit))
            }
            ValueEnumAccessRef::WithValue(s, v) => {
                let variant = seed.deserialize(de::value::BorrowedStrDeserializer::new(s))?;
                Ok((variant, ValueVariantAccessRef::WithValue(v)))
            }
        }
    }
}

enum ValueVariantAccessRef<'de> {
    Unit,
    WithValue(&'de Value),
}

impl<'de> de::VariantAccess<'de> for ValueVariantAccessRef<'de> {
    type Error = ValueError;

    fn unit_variant(self) -> Result<(), Self::Error> {
        match self {
            ValueVariantAccessRef::Unit => Ok(()),
            ValueVariantAccessRef::WithValue(Value::Null) => Ok(()),
            _ => Err(de::Error::custom("expected unit variant")),
        }
    }

    fn newtype_variant_seed<T: de::DeserializeSeed<'de>>(
        self,
        seed: T,
    ) -> Result<T::Value, Self::Error> {
        match self {
            ValueVariantAccessRef::WithValue(v) => seed.deserialize(v),
            ValueVariantAccessRef::Unit => Err(de::Error::custom("expected newtype variant")),
        }
    }

    fn tuple_variant<V: Visitor<'de>>(
        self,
        _len: usize,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        match self {
            ValueVariantAccessRef::WithValue(v) => v.deserialize_seq(visitor),
            ValueVariantAccessRef::Unit => Err(de::Error::custom("expected tuple variant")),
        }
    }

    fn struct_variant<V: Visitor<'de>>(
        self,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        match self {
            ValueVariantAccessRef::WithValue(v) => v.deserialize_map(visitor),
            ValueVariantAccessRef::Unit => Err(de::Error::custom("expected struct variant")),
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

        let num = Number::Float(2.5);
        assert_eq!(num.as_i64(), None); // Has fractional part
        assert!((num.as_f64() - 2.5).abs() < f64::EPSILON);
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
