use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::hash::Hash;
use std::str::FromStr;

use bytes::Bytes;
use num::PrimInt;
use regex::Regex;
use serde::de;
use serde::ser::{Serialize, SerializeMap, SerializeSeq, Serializer};

use crate::error;
use crate::state::State;
use crate::value::link::{Link, TCPath};
use crate::value::op::Request;
use crate::value::{TCRef, TCResult};

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

impl<T: fmt::Display + PrimInt> From<T> for ValueId {
    fn from(i: T) -> ValueId {
        ValueId {
            id: format!("{}", i),
        }
    }
}

impl From<&ValueId> for String {
    fn from(value_id: &ValueId) -> String {
        value_id.id.to_string()
    }
}

#[derive(Clone, Hash, Eq, PartialEq)]
pub enum Value {
    None,
    Bytes(Bytes),
    Id(ValueId),
    Int32(i32),
    UInt64(u64),
    Link(Link),
    Ref(TCRef),
    Request(Box<Request>),
    r#String(String),
    Vector(Vec<Value>),
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

impl From<TCPath> for Value {
    fn from(path: TCPath) -> Value {
        Value::Link(path.into())
    }
}

impl From<Request> for Value {
    fn from(op: Request) -> Value {
        Value::Request(Box::new(op))
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

impl<T: Into<Value>> From<Vec<T>> for Value {
    fn from(v: Vec<T>) -> Value {
        Value::Vector(v.into_iter().map(|i| i.into()).collect())
    }
}

impl<T1: Into<Value>, T2: Into<Value>> From<(T1, T2)> for Value {
    fn from(tuple: (T1, T2)) -> Value {
        Value::Vector(vec![tuple.0.into(), tuple.1.into()])
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

impl TryFrom<Value> for Request {
    type Error = error::TCError;

    fn try_from(v: Value) -> TCResult<Request> {
        match v {
            Value::Request(request) => Ok(*request),
            other => Err(error::bad_request("Expected Request but found", other)),
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

impl TryFrom<Value> for u64 {
    type Error = error::TCError;

    fn try_from(v: Value) -> TCResult<u64> {
        match v {
            Value::UInt64(i) => Ok(i),
            other => Err(error::bad_request("Expected a UInt64 but found", other)),
        }
    }
}

impl<
        E1: Into<error::TCError>,
        E2: Into<error::TCError>,
        T1: TryFrom<Value, Error = E1>,
        T2: TryFrom<Value, Error = E2>,
    > TryFrom<Value> for (T1, T2)
{
    type Error = error::TCError;

    fn try_from(v: Value) -> TCResult<(T1, T2)> {
        let mut v: Vec<Value> = v.try_into()?;
        if v.len() == 2 {
            Ok((
                v.remove(0).try_into().map_err(|e: E1| e.into())?,
                v.remove(0).try_into().map_err(|e: E2| e.into())?,
            ))
        } else {
            Err(error::bad_request(
                "Expected a 2-tuple, found",
                Value::Vector(v),
            ))
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
            Value::Vector(v) => v
                .into_iter()
                .map(|i| i.try_into().map_err(|e: E| e.into()))
                .collect(),
            other => Err(error::bad_request("Expected a Vector, found", other)),
        }
    }
}

impl<
        E1: Into<error::TCError>,
        E2: Into<error::TCError>,
        T1: Eq + Hash + TryFrom<Value, Error = E1>,
        T2: TryFrom<Value, Error = E2>,
    > TryFrom<Value> for HashMap<T1, T2>
{
    type Error = error::TCError;

    fn try_from(v: Value) -> TCResult<HashMap<T1, T2>> {
        let items: Vec<Value> = v.try_into()?;
        items.into_iter().map(|i| i.try_into()).collect()
    }
}

impl<T: Into<Value>> std::iter::FromIterator<T> for Value {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut v: Vec<Value> = vec![];
        for item in iter {
            v.push(item.into());
        }

        v.into()
    }
}

impl TryFrom<State> for Value {
    type Error = error::TCError;

    fn try_from(state: State) -> TCResult<Value> {
        match state {
            State::Value(value) => Ok(value),
            other => Err(error::bad_request("Expected a Value but found", other)),
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
            let value: Vec<Value> = access.next_value()?;
            if value.is_empty() {
                if key.starts_with('$') {
                    let r: TCRef = key.parse().map_err(de::Error::custom)?;
                    Ok(r.into())
                } else {
                    let link: Link = key.parse().map_err(de::Error::custom)?;
                    Ok(link.into())
                }
            } else {
                let request: Request = (key, value).try_into().map_err(de::Error::custom)?;
                Ok(request.into())
            }
        } else {
            Err(de::Error::custom("Unable to parse map entry"))
        }
    }

    fn visit_seq<S>(self, mut access: S) -> Result<Self::Value, S::Error>
    where
        S: de::SeqAccess<'de>,
    {
        let mut seq: Vec<Value> = vec![];

        while let Some(e) = access.next_element()? {
            seq.push(e);
        }

        Ok(Value::Vector(seq))
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
            Value::Id(i) => {
                let mut map = s.serialize_map(Some(1))?;
                map.serialize_entry("/sbin/value/id", &[i.as_str()])?;
                map.end()
            }
            Value::Int32(i) => s.serialize_i32(*i),
            Value::UInt64(i) => s.serialize_u64(*i),
            Value::Link(l) => l.serialize(s),
            Value::Ref(r) => r.serialize(s),
            Value::Request(r) => r.serialize(s),
            Value::r#String(v) => s.serialize_str(v),
            Value::Vector(v) => {
                let mut seq = s.serialize_seq(Some(v.len()))?;
                for element in v {
                    seq.serialize_element(element)?;
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
            Value::Id(id) => write!(f, "ValueId: {}", id),
            Value::Int32(i) => write!(f, "Int32: {}", i),
            Value::UInt64(i) => write!(f, "UInt64: {}", i),
            Value::Link(l) => write!(f, "Link: {}", l),
            Value::Ref(r) => write!(f, "Ref: {}", r),
            Value::Request(r) => write!(f, "Request: {}", r),
            Value::r#String(s) => write!(f, "String: {}", s),
            Value::Vector(v) => write!(f, "Vector: {:?}", v),
        }
    }
}
