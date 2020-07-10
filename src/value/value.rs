use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::hash::Hash;
use std::str::FromStr;

use bytes::Bytes;
use num::Complex;
use regex::Regex;
use serde::de;
use serde::ser::{Serialize, SerializeMap, SerializeSeq, Serializer};
use uuid::Uuid;

use crate::error;

use super::link::{Link, TCPath};
use super::op::Op;
use super::*;

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
pub enum Number {
    Bool(bool),
    Complex32(Complex<f32>),
    Complex64(Complex<f64>),
    Float32(f32),
    Float64(f64),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    UInt8(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),
}

impl TypeImpl for Number {
    type DType = NumberType;

    fn dtype(&self) -> NumberType {
        match self {
            Number::Bool(_) => NumberType::Bool,
            Number::Complex32(_) => NumberType::Complex32,
            Number::Complex64(_) => NumberType::Complex64,
            Number::Float32(_) => NumberType::Float32,
            Number::Float64(_) => NumberType::Float64,
            Number::Int16(_) => NumberType::Int16,
            Number::Int32(_) => NumberType::Int32,
            Number::Int64(_) => NumberType::Int64,
            Number::UInt8(_) => NumberType::UInt8,
            Number::UInt16(_) => NumberType::UInt16,
            Number::UInt32(_) => NumberType::UInt32,
            Number::UInt64(_) => NumberType::UInt64,
        }
    }
}

impl From<bool> for Number {
    fn from(b: bool) -> Number {
        Number::Bool(b)
    }
}

impl From<Complex<f32>> for Number {
    fn from(c: Complex<f32>) -> Number {
        Number::Complex32(c)
    }
}

impl From<Complex<f64>> for Number {
    fn from(c: Complex<f64>) -> Number {
        Number::Complex64(c)
    }
}

impl From<f32> for Number {
    fn from(f: f32) -> Number {
        Number::Float32(f)
    }
}

impl From<f64> for Number {
    fn from(f: f64) -> Number {
        Number::Float64(f)
    }
}

impl From<i16> for Number {
    fn from(i: i16) -> Number {
        Number::Int16(i)
    }
}

impl From<i32> for Number {
    fn from(i: i32) -> Number {
        Number::Int32(i)
    }
}

impl From<i64> for Number {
    fn from(i: i64) -> Number {
        Number::Int64(i)
    }
}

impl From<u8> for Number {
    fn from(i: u8) -> Number {
        Number::UInt8(i)
    }
}

impl From<u16> for Number {
    fn from(i: u16) -> Number {
        Number::UInt16(i)
    }
}

impl From<u32> for Number {
    fn from(i: u32) -> Number {
        Number::UInt32(i)
    }
}

impl From<u64> for Number {
    fn from(i: u64) -> Number {
        Number::UInt64(i)
    }
}

impl TryFrom<Number> for bool {
    type Error = error::TCError;

    fn try_from(n: Number) -> TCResult<bool> {
        match n {
            Number::Bool(b) => Ok(b),
            other => Err(error::bad_request("Expected Bool but found", other)),
        }
    }
}

impl TryFrom<Number> for Complex<f32> {
    type Error = error::TCError;

    fn try_from(v: Number) -> TCResult<Complex<f32>> {
        match v {
            Number::Complex32(c) => Ok(c),
            other => Err(error::bad_request("Expected Complex32 but found", other)),
        }
    }
}

impl TryFrom<Number> for Complex<f64> {
    type Error = error::TCError;

    fn try_from(v: Number) -> TCResult<Complex<f64>> {
        match v {
            Number::Complex64(c) => Ok(c),
            other => Err(error::bad_request("Expected Complex64 but found", other)),
        }
    }
}

impl TryFrom<Number> for f32 {
    type Error = error::TCError;

    fn try_from(n: Number) -> TCResult<f32> {
        match n {
            Number::Float32(f) => Ok(f),
            other => Err(error::bad_request("Expected Float32 but found", other)),
        }
    }
}

impl TryFrom<Number> for f64 {
    type Error = error::TCError;

    fn try_from(n: Number) -> TCResult<f64> {
        match n {
            Number::Float64(f) => Ok(f),
            other => Err(error::bad_request("Expected Float64 but found", other)),
        }
    }
}

impl TryFrom<Number> for i16 {
    type Error = error::TCError;

    fn try_from(n: Number) -> TCResult<i16> {
        match n {
            Number::Int16(i) => Ok(i),
            other => Err(error::bad_request("Expected Int16 but found", other)),
        }
    }
}

impl TryFrom<Number> for i32 {
    type Error = error::TCError;

    fn try_from(n: Number) -> TCResult<i32> {
        match n {
            Number::Int32(i) => Ok(i),
            other => Err(error::bad_request("Expected an Int32 but found", other)),
        }
    }
}

impl TryFrom<Number> for i64 {
    type Error = error::TCError;

    fn try_from(n: Number) -> TCResult<i64> {
        match n {
            Number::Int64(i) => Ok(i),
            other => Err(error::bad_request("Expected an Int64 but found", other)),
        }
    }
}

impl TryFrom<Number> for u8 {
    type Error = error::TCError;

    fn try_from(n: Number) -> TCResult<u8> {
        match n {
            Number::UInt8(i) => Ok(i),
            other => Err(error::bad_request("Expected a UInt8 but found", other)),
        }
    }
}

impl TryFrom<Number> for u16 {
    type Error = error::TCError;

    fn try_from(n: Number) -> TCResult<u16> {
        match n {
            Number::UInt16(i) => Ok(i),
            other => Err(error::bad_request("Expected a UInt16 but found", other)),
        }
    }
}

impl TryFrom<Number> for u32 {
    type Error = error::TCError;

    fn try_from(n: Number) -> TCResult<u32> {
        match n {
            Number::UInt32(i) => Ok(i),
            other => Err(error::bad_request("Expected a UInt32 but found", other)),
        }
    }
}

impl TryFrom<Number> for u64 {
    type Error = error::TCError;

    fn try_from(n: Number) -> TCResult<u64> {
        match n {
            Number::UInt64(i) => Ok(i),
            other => Err(error::bad_request("Expected a UInt64 but found", other)),
        }
    }
}

impl Serialize for Number {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Number::Bool(b) => s.serialize_bool(*b),
            Number::Complex32(c) => {
                let mut map = s.serialize_map(Some(1))?;
                map.serialize_entry("/sbin/value/complex/32", &[[c.re, c.im]])?;
                map.end()
            }
            Number::Complex64(c) => {
                let mut map = s.serialize_map(Some(1))?;
                map.serialize_entry("/sbin/value/complex/64", &[[c.re, c.im]])?;
                map.end()
            }
            Number::Float32(f) => s.serialize_f32(*f),
            Number::Float64(f) => s.serialize_f64(*f),
            Number::Int16(i) => s.serialize_i16(*i),
            Number::Int32(i) => s.serialize_i32(*i),
            Number::Int64(i) => s.serialize_i64(*i),
            Number::UInt8(i) => s.serialize_u8(*i),
            Number::UInt16(i) => s.serialize_u16(*i),
            Number::UInt32(i) => s.serialize_u32(*i),
            Number::UInt64(i) => s.serialize_u64(*i),
        }
    }
}

impl fmt::Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Number::Bool(b) => write!(f, "Bool({})", b),
            Number::Complex32(c) => write!(f, "Complex32({})", c),
            Number::Complex64(c) => write!(f, "Complex64({})", c),
            Number::Float32(n) => write!(f, "Float32({})", n),
            Number::Float64(n) => write!(f, "Float64({})", n),
            Number::Int16(i) => write!(f, "Int16: {}", i),
            Number::Int32(i) => write!(f, "Int32: {}", i),
            Number::Int64(i) => write!(f, "Int64: {}", i),
            Number::UInt8(i) => write!(f, "UInt8: {}", i),
            Number::UInt16(i) => write!(f, "UInt16: {}", i),
            Number::UInt32(i) => write!(f, "UInt32: {}", i),
            Number::UInt64(i) => write!(f, "UInt64: {}", i),
        }
    }
}

#[derive(Clone, PartialEq)]
pub enum TCString {
    Id(ValueId),
    Link(Link),
    Ref(TCRef),
    r#String(String),
}

impl TypeImpl for TCString {
    type DType = StringType;

    fn dtype(&self) -> StringType {
        match self {
            TCString::Id(_) => StringType::Id,
            TCString::Link(_) => StringType::Link,
            TCString::Ref(_) => StringType::Ref,
            TCString::r#String(_) => StringType::r#String,
        }
    }
}

impl From<Link> for TCString {
    fn from(l: Link) -> TCString {
        TCString::Link(l)
    }
}

impl From<TCPath> for TCString {
    fn from(path: TCPath) -> TCString {
        TCString::Link(path.into())
    }
}

impl From<ValueId> for TCString {
    fn from(id: ValueId) -> TCString {
        TCString::Id(id)
    }
}

impl From<TCRef> for TCString {
    fn from(r: TCRef) -> TCString {
        TCString::Ref(r)
    }
}

impl From<String> for TCString {
    fn from(s: String) -> TCString {
        TCString::r#String(s)
    }
}

impl TryFrom<TCString> for Link {
    type Error = error::TCError;

    fn try_from(s: TCString) -> TCResult<Link> {
        match s {
            TCString::Link(l) => Ok(l),
            other => Err(error::bad_request("Expected Link but found", other)),
        }
    }
}

impl TryFrom<TCString> for String {
    type Error = error::TCError;

    fn try_from(s: TCString) -> TCResult<String> {
        match s {
            TCString::r#String(s) => Ok(s),
            other => Err(error::bad_request("Expected a String but found", other)),
        }
    }
}

impl TryFrom<TCString> for TCPath {
    type Error = error::TCError;

    fn try_from(s: TCString) -> TCResult<TCPath> {
        match s {
            TCString::Link(l) => {
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

impl TryFrom<TCString> for ValueId {
    type Error = error::TCError;

    fn try_from(s: TCString) -> TCResult<ValueId> {
        let s: String = s.try_into()?;
        s.parse()
    }
}

impl Serialize for TCString {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use TCString::*;
        match self {
            Id(i) => {
                let mut map = s.serialize_map(Some(1))?;
                map.serialize_entry("/sbin/value/id", &[i.as_str()])?;
                map.end()
            }
            Link(l) => l.serialize(s),
            Ref(r) => r.serialize(s),
            r#String(v) => s.serialize_str(v),
        }
    }
}

impl fmt::Display for TCString {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TCString::Id(id) => write!(f, "ValueId: {}", id),
            TCString::Link(l) => write!(f, "Link: {}", l),
            TCString::Ref(r) => write!(f, "Ref: {}", r),
            TCString::r#String(s) => write!(f, "String: {}", s),
        }
    }
}

#[derive(Clone, PartialEq)]
pub enum Value {
    None,
    Bytes(Bytes),
    Number(Number),
    TCString(TCString),
    Op(Box<Op>),
    Vector(Vec<Value>),
}

impl TypeImpl for Value {
    type DType = TCType;

    fn dtype(&self) -> TCType {
        match self {
            Value::None => TCType::None,
            Value::Bytes(_) => TCType::Bytes,
            Value::Number(n) => TCType::Number(n.dtype()),
            Value::TCString(s) => TCType::TCString(s.dtype()),
            Value::Op(_) => TCType::Op,
            Value::Vector(_) => TCType::Vector,
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

impl From<Number> for Value {
    fn from(n: Number) -> Value {
        Value::Number(n)
    }
}

impl From<TCString> for Value {
    fn from(s: TCString) -> Value {
        Value::TCString(s)
    }
}

impl From<Op> for Value {
    fn from(op: Op) -> Value {
        Value::Op(Box::new(op))
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

impl<T: Into<Value>> From<Vec<T>> for Value {
    fn from(mut v: Vec<T>) -> Value {
        Value::Vector(v.drain(..).map(|i| i.into()).collect())
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

impl TryFrom<Value> for Number {
    type Error = error::TCError;

    fn try_from(v: Value) -> TCResult<Number> {
        match v {
            Value::Number(n) => Ok(n),
            other => Err(error::bad_request("Expected Number but found", other)),
        }
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
        Ok(Value::Number(Number::Int32(value)))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::TCString(TCString::r#String(value.to_string())))
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
                    0 => Ok(Value::TCString(TCString::Ref(subject))),
                    1 => Ok(Op::Get(subject.into(), value.remove(0)).into()),
                    2 => Ok(Op::Put(subject.into(), value.remove(0), value.remove(0)).into()),
                    _ => Err(de::Error::custom(format!(
                        "Expected a Get or Put op, found {}",
                        Value::Vector(value)
                    ))),
                }
            } else if let Ok(link) = key.parse::<Link>() {
                Ok(Value::TCString(TCString::Link(link)))
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
            Value::Bytes(b) => {
                let mut map = s.serialize_map(Some(1))?;
                map.serialize_entry("/sbin/value/bytes", &[base64::encode(b)])?;
                map.end()
            }
            Value::Number(n) => n.serialize(s),
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
            Value::TCString(tc_string) => tc_string.serialize(s),
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
            Value::Number(n) => write!(f, "Number({})", n),
            Value::TCString(s) => write!(f, "String({})", s),
            Value::Op(op) => write!(f, "Op: {}", op),
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
