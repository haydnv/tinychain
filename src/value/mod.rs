use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::str::FromStr;

use bytes::Bytes;
use regex::Regex;
use serde::de;
use serde::ser::{Serialize, SerializeMap, SerializeSeq, Serializer};

use crate::error;
use crate::state::State;

mod link;
mod op;
mod reference;
mod version;

pub type Args = op::Args;
pub type Link = link::Link;
pub type PathSegment = link::PathSegment;
pub type Op = op::Op;
pub type TCPath = link::TCPath;
pub type TCRef = reference::TCRef;
pub type TCResult<T> = Result<T, error::TCError>;
pub type Subject = op::Subject;
pub type Version = version::Version;

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

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ValueId {
    id: String,
}

impl ValueId {
    pub fn as_str(&self) -> &str {
        self.id.as_str()
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
        if path.len() == 1 {
            Ok(path[0].clone())
        } else {
            Err(error::bad_request("Expected a ValueId, found", path))
        }
    }
}

impl From<u64> for ValueId {
    fn from(u: u64) -> ValueId {
        ValueId {
            id: format!("{}", u),
        }
    }
}

impl From<&ValueId> for String {
    fn from(value_id: &ValueId) -> String {
        value_id.id.to_string()
    }
}

#[derive(Clone, Hash, Eq, PartialEq)]
pub enum TCValue {
    None,
    Bytes(Bytes),
    Int32(i32),
    Link(Link),
    Op(Op),
    Path(TCPath),
    Ref(TCRef),
    r#String(String),
    Vector(Vec<TCValue>),
}

impl From<()> for TCValue {
    fn from(_: ()) -> TCValue {
        TCValue::None
    }
}

impl From<&'static [u8]> for TCValue {
    fn from(b: &'static [u8]) -> TCValue {
        TCValue::Bytes(Bytes::from(b))
    }
}

impl From<Bytes> for TCValue {
    fn from(b: Bytes) -> TCValue {
        TCValue::Bytes(b)
    }
}

impl From<TCPath> for TCValue {
    fn from(path: TCPath) -> TCValue {
        TCValue::Path(path)
    }
}

impl From<Op> for TCValue {
    fn from(op: Op) -> TCValue {
        TCValue::Op(op)
    }
}

impl<T: Into<TCValue>> From<Option<T>> for TCValue {
    fn from(opt: Option<T>) -> TCValue {
        match opt {
            Some(val) => val.into(),
            None => TCValue::None,
        }
    }
}

impl From<TCRef> for TCValue {
    fn from(r: TCRef) -> TCValue {
        TCValue::Ref(r)
    }
}

impl From<String> for TCValue {
    fn from(s: String) -> TCValue {
        TCValue::r#String(s)
    }
}

impl From<Vec<u8>> for TCValue {
    fn from(v: Vec<u8>) -> TCValue {
        TCValue::Bytes(Bytes::copy_from_slice(&v[..]))
    }
}

impl<T: Into<TCValue>> From<Vec<T>> for TCValue {
    fn from(v: Vec<T>) -> TCValue {
        TCValue::Vector(v.into_iter().map(|i| i.into()).collect())
    }
}

impl<T1: Into<TCValue>, T2: Into<TCValue>> From<(T1, T2)> for TCValue {
    fn from(tuple: (T1, T2)) -> TCValue {
        TCValue::Vector(vec![tuple.0.into(), tuple.1.into()])
    }
}

impl TryFrom<TCValue> for Bytes {
    type Error = error::TCError;

    fn try_from(v: TCValue) -> TCResult<Bytes> {
        match v {
            TCValue::Bytes(b) => Ok(b),
            other => Err(error::bad_request("Expected Bytes but found", other)),
        }
    }
}

impl TryFrom<TCValue> for Link {
    type Error = error::TCError;

    fn try_from(v: TCValue) -> TCResult<Link> {
        match v {
            TCValue::Link(l) => Ok(l),
            other => Err(error::bad_request("Expected Link but found", other)),
        }
    }
}

impl TryFrom<TCValue> for TCPath {
    type Error = error::TCError;

    fn try_from(v: TCValue) -> TCResult<TCPath> {
        match v {
            TCValue::Path(p) => Ok(p),
            other => Err(error::bad_request("Expected Path but found", other)),
        }
    }
}

impl TryFrom<TCValue> for String {
    type Error = error::TCError;

    fn try_from(v: TCValue) -> TCResult<String> {
        match v {
            TCValue::r#String(s) => Ok(s),
            other => Err(error::bad_request("Expected a String but found", other)),
        }
    }
}

impl<
        E1: fmt::Display,
        E2: fmt::Display,
        T1: TryFrom<TCValue, Error = E1>,
        T2: TryFrom<TCValue, Error = E2>,
    > TryFrom<TCValue> for (T1, T2)
{
    type Error = error::TCError;

    fn try_from(v: TCValue) -> TCResult<(T1, T2)> {
        let mut v: Vec<TCValue> = v.try_into()?;
        if v.len() == 2 {
            Ok((
                v.remove(0)
                    .try_into()
                    .map_err(|e| error::bad_request("Unable to convert from Value", e))?,
                v.remove(0)
                    .try_into()
                    .map_err(|e| error::bad_request("Unable to convert from Value", e))?,
            ))
        } else {
            Err(error::bad_request(
                "Expected a 2-tuple, found",
                TCValue::Vector(v),
            ))
        }
    }
}

impl TryFrom<TCValue> for Vec<TCValue> {
    type Error = error::TCError;

    fn try_from(v: TCValue) -> TCResult<Vec<TCValue>> {
        match v {
            TCValue::Vector(v) => Ok(v.to_vec()),
            other => Err(error::bad_request("Expected Vector but found", other)),
        }
    }
}

impl TryFrom<TCValue> for ValueId {
    type Error = error::TCError;

    fn try_from(v: TCValue) -> TCResult<ValueId> {
        let s: String = v.try_into()?;
        s.parse()
    }
}

impl<T: TryFrom<TCValue, Error = error::TCError>> TryFrom<TCValue> for Vec<T> {
    type Error = error::TCError;

    fn try_from(v: TCValue) -> TCResult<Vec<T>> {
        let items: Vec<TCValue> = v.try_into()?;
        items.into_iter().map(|i| i.try_into()).collect()
    }
}

impl<T: Into<TCValue>> std::iter::FromIterator<T> for TCValue {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut v: Vec<TCValue> = vec![];
        for item in iter {
            v.push(item.into());
        }

        v.into()
    }
}

impl TryFrom<State> for TCValue {
    type Error = error::TCError;

    fn try_from(state: State) -> TCResult<TCValue> {
        match state {
            State::Value(value) => Ok(value),
            other => Err(error::bad_request("Expected a Value but found", other)),
        }
    }
}

struct TCValueVisitor;

impl<'de> de::Visitor<'de> for TCValueVisitor {
    type Value = TCValue;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("A Tinychain Value, e.g. \"foo\" or 123 or {\"$object_ref: [\"slice_id\", \"$state\"]\"}")
    }

    fn visit_i32<E>(self, value: i32) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(TCValue::Int32(value))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(TCValue::r#String(value.to_string()))
    }

    fn visit_map<M>(self, mut access: M) -> Result<Self::Value, M::Error>
    where
        M: de::MapAccess<'de>,
    {
        if let Some(key) = access.next_key::<&str>()? {
            if key.starts_with('/') {
                let mut value = access.next_value::<Vec<TCValue>>()?;

                let path: TCPath = key.parse().map_err(de::Error::custom)?;

                if value.is_empty() {
                    Ok(path.into())
                } else if value.len() == 1 {
                    Ok(Op::get(path.into(), value.remove(0)).into())
                } else if value.len() == 2 {
                    Ok(Op::put(path.into(), value.remove(0), value.remove(0)).into())
                } else {
                    Err(de::Error::custom(format!(
                        "Expected a list of 0, 1, or 2 values for {}",
                        key
                    )))
                }
            } else if key.starts_with('$') {
                if key.contains('/') {
                    let key: Vec<&str> = key.split('/').collect();
                    let subject: TCRef = key[0][1..].parse().map_err(de::Error::custom)?;
                    let method: TCPath = key[1..]
                        .iter()
                        .map(|s| s.parse())
                        .collect::<TCResult<Vec<PathSegment>>>()
                        .map_err(de::Error::custom)?
                        .into();
                    let requires = access.next_value::<Vec<(ValueId, TCValue)>>()?;

                    Ok(Op::post(subject.into(), method, requires).into())
                } else {
                    let subject: TCRef = key[1..].parse().map_err(de::Error::custom)?;
                    let mut value = access.next_value::<Vec<TCValue>>()?;

                    if value.is_empty() {
                        Ok(subject.into())
                    } else if value.len() == 1 {
                        Ok(Op::get(subject.into(), value.remove(0)).into())
                    } else if value.len() == 2 {
                        Ok(Op::put(subject.into(), value.remove(0), value.remove(0)).into())
                    } else {
                        Err(de::Error::custom(format!(
                            "Expected a list of 0, 1, or 2 Values for {}",
                            key
                        )))
                    }
                }
            } else if key == "Bytes" {
                let value: &str = access.next_value()?;
                let value: Vec<u8> = base64::decode(value)
                    .map_err(|e| de::Error::custom(format!("Invalid base64 string: {}", e)))?;
                Ok(value.into())
            } else {
                Err(de::Error::custom(format!(
                    "Unable to determine data type of {}",
                    key
                )))
            }
        } else {
            Err(de::Error::custom("Unable to parse map entry"))
        }
    }

    fn visit_seq<S>(self, mut access: S) -> Result<Self::Value, S::Error>
    where
        S: de::SeqAccess<'de>,
    {
        let mut seq: Vec<TCValue> = vec![];

        while let Some(e) = access.next_element()? {
            seq.push(e);
        }

        Ok(TCValue::Vector(seq))
    }
}

impl<'de> de::Deserialize<'de> for TCValue {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        d.deserialize_any(TCValueVisitor)
    }
}

impl Serialize for TCValue {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            TCValue::None => s.serialize_none(),
            TCValue::Bytes(b) => {
                let mut map = s.serialize_map(Some(1))?;
                map.serialize_entry("Bytes", &base64::encode(b))?;
                map.end()
            }
            TCValue::Int32(i) => s.serialize_i32(*i),
            TCValue::Link(l) => l.serialize(s),
            TCValue::Op(o) => o.serialize(s),
            TCValue::Path(p) => p.serialize(s),
            TCValue::Ref(r) => r.serialize(s),
            TCValue::r#String(v) => s.serialize_str(v),
            TCValue::Vector(v) => {
                let mut seq = s.serialize_seq(Some(v.len()))?;
                for element in v {
                    seq.serialize_element(element)?;
                }
                seq.end()
            }
        }
    }
}

impl fmt::Debug for TCValue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl fmt::Display for TCValue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TCValue::None => write!(f, "None"),
            TCValue::Bytes(b) => write!(f, "binary data: {} bytes", b.len()),
            TCValue::Int32(i) => write!(f, "Int32: {}", i),
            TCValue::Link(l) => write!(f, "Link: {}", l),
            TCValue::Op(o) => write!(f, "Op: {}", o),
            TCValue::Path(p) => write!(f, "Path: {}", p),
            TCValue::Ref(r) => write!(f, "Ref: {}", r),
            TCValue::r#String(s) => write!(f, "string: {}", s),
            TCValue::Vector(v) => write!(f, "vector: {:?}", v),
        }
    }
}
