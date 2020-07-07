use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::hash::Hash;
use std::str::FromStr;

use bytes::Bytes;
use regex::Regex;
use serde::de;
use serde::ser::{Serialize, SerializeMap, SerializeSeq, Serializer};
use uuid::Uuid;

use crate::error;

use super::link::{Link, TCPath};
use super::op::Op;
use super::{TCRef, TCResult, TCType};

const RESERVED_CHARS: [&str; 21] = [
    "/", "..", "~", "$", "`", "^", "&", "|", "=", "^", "{", "}", "<", ">", "'", "\"", "?", ":",
    "//", "@", "#",
];

fn validate_id(id: &str) -> TCResult<()> {
    if id.is_empty() {
        return Err(error::bad_request("ValueId cannot be empty", id));
    }

    let filtered: &str = &id.chars().filter(|c| *c as u8 > 32).collect::<String>();
    if filtered != id {
        return Err(error::bad_request(
            "This value ID contains an ASCII control character",
            filtered,
        ));
    }

    for pattern in &RESERVED_CHARS {
        if id.contains(pattern) {
            return Err(error::bad_request(
                "A value ID may not contain this pattern",
                pattern,
            ));
        }
    }

    if let Some(w) = Regex::new(r"\s").unwrap().find(id) {
        return Err(error::bad_request(
            "A value ID may not contain whitespace",
            format!("{:?}", w),
        ));
    }

    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct ValueId {
    id: String,
}

impl ValueId {
    pub fn as_str(&self) -> &str {
        self.id.as_str()
    }

    pub fn starts_with(&self, prefix: &str) -> bool {
        self.id.starts_with(prefix)
    }
}

impl From<Uuid> for ValueId {
    fn from(id: Uuid) -> ValueId {
        id.to_hyphenated().to_string().parse().unwrap()
    }
}

impl From<u64> for ValueId {
    fn from(i: u64) -> ValueId {
        i.to_string().parse().unwrap()
    }
}

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.id)
    }
}

impl<'de> serde::Deserialize<'de> for ValueId {
    fn deserialize<D>(deserializer: D) -> Result<ValueId, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        let s: &str = de::Deserialize::deserialize(deserializer)?;
        s.parse().map_err(de::Error::custom)
    }
}

impl Serialize for ValueId {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        s.serialize_str(&self.id)
    }
}

impl PartialEq<&str> for ValueId {
    fn eq(&self, other: &&str) -> bool {
        &self.id == other
    }
}

impl FromStr for ValueId {
    type Err = error::TCError;

    fn from_str(id: &str) -> TCResult<ValueId> {
        validate_id(id)?;
        Ok(ValueId { id: id.to_string() })
    }
}

impl TryFrom<TCPath> for ValueId {
    type Error = error::TCError;

    fn try_from(path: TCPath) -> TCResult<ValueId> {
        ValueId::try_from(&path)
    }
}

impl TryFrom<&TCPath> for ValueId {
    type Error = error::TCError;

    fn try_from(path: &TCPath) -> TCResult<ValueId> {
        if path.len() == 1 {
            Ok(path[0].clone())
        } else {
            Err(error::bad_request("Expected a ValueId, found", path))
        }
    }
}

impl From<&ValueId> for String {
    fn from(value_id: &ValueId) -> String {
        value_id.id.to_string()
    }
}

#[derive(Clone, PartialEq)]
pub enum Value {
    None,
    Bool(bool),
    Bytes(Bytes),
    Complex32(num::Complex<f32>),
    Complex64(num::Complex<f64>),
    Id(ValueId),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    UInt8(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),
    Link(Link),
    Op(Box<Op>),
    Ref(TCRef),
    r#String(String),
    Vector(Vec<Value>),
}

impl Value {
    pub fn is_a(&self, dtype: &TCType) -> bool {
        dtype == &self.dtype()
    }

    pub fn dtype(&self) -> TCType {
        use Value::*;
        match self {
            None => TCType::None,
            Bool(_) => TCType::Bool,
            Bytes(_) => TCType::Bytes,
            Complex32(_) => TCType::Complex32,
            Complex64(_) => TCType::Complex64,
            Id(_) => TCType::Id,
            Int16(_) => TCType::Int16,
            Int32(_) => TCType::Int32,
            Int64(_) => TCType::Int64,
            UInt8(_) => TCType::UInt8,
            UInt16(_) => TCType::UInt16,
            UInt32(_) => TCType::UInt32,
            UInt64(_) => TCType::UInt64,
            Link(_) => TCType::Link,
            Op(_) => TCType::Op,
            Ref(_) => TCType::Ref,
            r#String(_) => TCType::String,
            Vector(_) => TCType::Vector,
        }
    }
}

impl From<()> for Value {
    fn from(_: ()) -> Value {
        Value::None
    }
}

impl From<&'static [u8]> for Value {
    fn from(b: &'static [u8]) -> Value {
        Value::Bytes(Bytes::from(b))
    }
}

impl From<Bytes> for Value {
    fn from(b: Bytes) -> Value {
        Value::Bytes(b)
    }
}

impl From<Link> for Value {
    fn from(l: Link) -> Value {
        Value::Link(l)
    }
}

impl From<Op> for Value {
    fn from(op: Op) -> Value {
        Value::Op(Box::new(op))
    }
}

impl From<TCPath> for Value {
    fn from(path: TCPath) -> Value {
        Value::Link(path.into())
    }
}

impl From<ValueId> for Value {
    fn from(id: ValueId) -> Value {
        Value::Id(id)
    }
}

impl<T: Into<Value>> From<Option<T>> for Value {
    fn from(opt: Option<T>) -> Value {
        match opt {
            Some(val) => val.into(),
            None => Value::None,
        }
    }
}

impl From<TCRef> for Value {
    fn from(r: TCRef) -> Value {
        Value::Ref(r)
    }
}

impl<T: Into<Value>> From<Vec<T>> for Value {
    fn from(mut v: Vec<T>) -> Value {
        Value::Vector(v.drain(..).map(|i| i.into()).collect())
    }
}

impl From<String> for Value {
    fn from(s: String) -> Value {
        Value::r#String(s)
    }
}

impl From<Vec<u8>> for Value {
    fn from(b: Vec<u8>) -> Value {
        Value::Bytes(Bytes::copy_from_slice(&b[..]))
    }
}

impl TryFrom<Value> for Bytes {
    type Error = error::TCError;

    fn try_from(v: Value) -> TCResult<Bytes> {
        match v {
            Value::Bytes(b) => Ok(b),
            other => Err(error::bad_request("Expected Bytes but found", other)),
        }
    }
}

impl TryFrom<Value> for Link {
    type Error = error::TCError;

    fn try_from(v: Value) -> TCResult<Link> {
        match v {
            Value::Link(l) => Ok(l),
            other => Err(error::bad_request("Expected Link but found", other)),
        }
    }
}

impl TryFrom<Value> for TCPath {
    type Error = error::TCError;

    fn try_from(v: Value) -> TCResult<TCPath> {
        match v {
            Value::Link(l) => {
                if l.host().is_none() {
                    Ok(l.path().clone())
                } else {
                    Err(error::bad_request("Expected Path but found Link", l))
                }
            }
            other => Err(error::bad_request("Expected Path but found", other)),
        }
    }
}

impl TryFrom<Value> for String {
    type Error = error::TCError;

    fn try_from(v: Value) -> TCResult<String> {
        match v {
            Value::r#String(s) => Ok(s),
            other => Err(error::bad_request("Expected a String but found", other)),
        }
    }
}

impl TryFrom<Value> for i32 {
    type Error = error::TCError;

    fn try_from(v: Value) -> TCResult<i32> {
        match v {
            Value::Int32(i) => Ok(i),
            other => Err(error::bad_request("Expected an Int32 but found", other)),
        }
    }
}

impl TryFrom<&Value> for i32 {
    type Error = error::TCError;

    fn try_from(v: &Value) -> TCResult<i32> {
        match v {
            Value::Int32(i) => Ok(*i),
            other => Err(error::bad_request("Expected an Int32 but found", other)),
        }
    }
}

impl TryFrom<Value> for u64 {
    type Error = error::TCError;

    fn try_from(v: Value) -> TCResult<u64> {
        match v {
            Value::UInt64(i) => Ok(i),
            other => Err(error::bad_request("Expected a UInt64 but found", other)),
        }
    }
}

impl TryFrom<&Value> for u64 {
    type Error = error::TCError;

    fn try_from(v: &Value) -> TCResult<u64> {
        match v {
            Value::UInt64(i) => Ok(*i),
            other => Err(error::bad_request("Expected an Int32 but found", other)),
        }
    }
}

impl TryFrom<Value> for ValueId {
    type Error = error::TCError;

    fn try_from(v: Value) -> TCResult<ValueId> {
        let s: String = v.try_into()?;
        s.parse()
    }
}

impl<E: Into<error::TCError>, T: TryFrom<Value, Error = E>> TryFrom<Value> for Vec<T> {
    type Error = error::TCError;

    fn try_from(v: Value) -> TCResult<Vec<T>> {
        match v {
            Value::Vector(mut v) => v
                .drain(..)
                .map(|i| i.try_into().map_err(|e: E| e.into()))
                .collect(),
            other => Err(error::bad_request("Expected a Vector but found", other)),
        }
    }
}

struct ValueVisitor;

impl<'de> de::Visitor<'de> for ValueVisitor {
    type Value = Value;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a Tinychain Value, e.g. \"foo\" or 123 or {\"$object_ref: [\"slice_id\", \"$state\"]\"}")
    }

    fn visit_i32<E>(self, value: i32) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::Int32(value))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::r#String(value.to_string()))
    }

    fn visit_map<M>(self, mut access: M) -> Result<Self::Value, M::Error>
    where
        M: de::MapAccess<'de>,
    {
        if let Some(key) = access.next_key::<&str>()? {
            let mut value: Vec<Value> = access.next_value()?;

            if key.starts_with('$') {
                let subject = key.parse::<TCRef>().map_err(de::Error::custom)?;
                match value.len() {
                    0 => Ok(subject.into()),
                    1 => Ok(Op::Get(subject.into(), value.remove(0)).into()),
                    2 => Ok(Op::Put(subject.into(), value.remove(0), value.remove(0)).into()),
                    _ => Err(de::Error::custom(format!(
                        "Expected a Get or Put op, found {}",
                        Value::Vector(value)
                    ))),
                }
            } else if let Ok(link) = key.parse::<Link>() {
                Ok(link.into())
            } else {
                panic!("NOT IMPLEMENTED")
            }
        } else {
            Err(de::Error::custom("Unable to parse map entry"))
        }
    }
}

impl<'de> de::Deserialize<'de> for Value {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        d.deserialize_any(ValueVisitor)
    }
}

impl Serialize for Value {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Value::None => s.serialize_none(),
            Value::Bool(b) => s.serialize_bool(*b),
            Value::Bytes(b) => {
                let mut map = s.serialize_map(Some(1))?;
                map.serialize_entry("/sbin/value/bytes", &[base64::encode(b)])?;
                map.end()
            }
            Value::Complex32(c) => {
                let mut map = s.serialize_map(Some(1))?;
                map.serialize_entry("/sbin/value/complex/32", &[[c.re, c.im]])?;
                map.end()
            }
            Value::Complex64(c) => {
                let mut map = s.serialize_map(Some(1))?;
                map.serialize_entry("/sbin/value/complex/64", &[[c.re, c.im]])?;
                map.end()
            }
            Value::Id(i) => {
                let mut map = s.serialize_map(Some(1))?;
                map.serialize_entry("/sbin/value/id", &[i.as_str()])?;
                map.end()
            }
            Value::Int16(i) => s.serialize_i16(*i),
            Value::Int32(i) => s.serialize_i32(*i),
            Value::Int64(i) => s.serialize_i64(*i),
            Value::UInt8(i) => s.serialize_u8(*i),
            Value::UInt16(i) => s.serialize_u16(*i),
            Value::UInt32(i) => s.serialize_u32(*i),
            Value::UInt64(i) => s.serialize_u64(*i),
            Value::Link(l) => l.serialize(s),
            Value::Op(op) => {
                let mut map = s.serialize_map(Some(1))?;
                match &**op {
                    Op::Get(subject, selector) => {
                        map.serialize_entry(&subject.to_string(), &[selector])?
                    }
                    Op::Put(subject, selector, value) => {
                        map.serialize_entry(&subject.to_string(), &[selector, value])?
                    }
                }
                map.end()
            }
            Value::Ref(r) => r.serialize(s),
            Value::r#String(v) => s.serialize_str(v),
            Value::Vector(v) => {
                let mut seq = s.serialize_seq(Some(v.len()))?;
                for item in v {
                    seq.serialize_element(item)?;
                }
                seq.end()
            }
        }
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Value::None => write!(f, "None"),
            Value::Bytes(b) => write!(f, "Bytes({})", b.len()),
            Value::Bool(b) => write!(f, "Bool({})", b),
            Value::Complex32(c) => write!(f, "Complex32({})", c),
            Value::Complex64(c) => write!(f, "Complex64({})", c),
            Value::Id(id) => write!(f, "ValueId: {}", id),
            Value::Int16(i) => write!(f, "Int16: {}", i),
            Value::Int32(i) => write!(f, "Int32: {}", i),
            Value::Int64(i) => write!(f, "Int64: {}", i),
            Value::UInt8(i) => write!(f, "UInt8: {}", i),
            Value::UInt16(i) => write!(f, "UInt16: {}", i),
            Value::UInt32(i) => write!(f, "UInt32: {}", i),
            Value::UInt64(i) => write!(f, "UInt64: {}", i),
            Value::Link(l) => write!(f, "Link: {}", l),
            Value::Op(op) => write!(f, "Op: {}", op),
            Value::Ref(r) => write!(f, "Ref: {}", r),
            Value::r#String(s) => write!(f, "String: {}", s),
            Value::Vector(v) => write!(
                f,
                "[{}]",
                v.iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
        }
    }
}
