use std::convert::{TryFrom, TryInto};
use std::fmt;

use regex::Regex;
use serde::de;
use serde::ser::{SerializeSeq, SerializeStructVariant, Serializer};
use serde::{Deserialize, Serialize};

use crate::context::TCResult;
use crate::error;
use crate::state::TCState;

pub type ValueId = String;

pub trait TCValueExt: TryFrom<TCValue, Error = error::TCError> {}

const RESERVED_CHARS: [&str; 17] = [
    "..", "~", "$", "&", "?", "|", "{", "}", "//", ":", "=", "^", ">", "<", "'", "`", "\"",
];

fn valid_id(id: &str) -> bool {
    let mut eot_char = [0];
    let eot_char = (4 as char).encode_utf8(&mut eot_char);

    let reserved = [&RESERVED_CHARS[..], &[eot_char]].concat();

    for pattern in reserved.iter() {
        if id.contains(pattern) {
            return false;
        }
    }

    if Regex::new(r"\s").unwrap().find(id).is_some() {
        return false;
    }

    true
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct Link {
    segments: Vec<String>,
}

impl Link {
    fn _validate(to: &str) -> TCResult<()> {
        if !valid_id(to) {
            Err(error::bad_request(
                "A link may not contain whitespace or any of these patterns",
                RESERVED_CHARS.join(", "),
            ))
        } else if !to.starts_with('/') {
            Err(error::bad_request(
                "Expected an absolute path starting with '/' but found",
                to,
            ))
        } else if to != "/" && to.ends_with('/') {
            Err(error::bad_request("Trailing slash is not allowed", to))
        } else {
            Ok(())
        }
    }

    pub fn to(destination: &str) -> TCResult<Link> {
        Link::_validate(destination)?;

        let segments: Vec<String> = if destination == "/" {
            vec![]
        } else {
            destination[1..].split('/').map(|s| s.to_string()).collect()
        };

        Ok(Link { segments })
    }

    pub fn append(&self, suffix: &Link) -> Link {
        Link::to(&format!("{}{}", self, suffix)).unwrap()
    }

    pub fn as_str(&self, index: usize) -> &str {
        self.segments[index].as_str()
    }

    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    pub fn len(&self) -> usize {
        self.segments.len()
    }

    pub fn nth(&self, i: usize) -> Link {
        Link {
            segments: vec![self.segments[i].clone()],
        }
    }

    pub fn slice_from(&self, start: usize) -> Link {
        Link {
            segments: self.segments[start..].to_vec(),
        }
    }

    pub fn slice_to(&self, end: usize) -> Link {
        Link {
            segments: self.segments[..end].to_vec(),
        }
    }
}

impl TCValueExt for Link {}

impl From<u64> for Link {
    fn from(i: u64) -> Link {
        Link::to(&format!("/{}", i)).unwrap()
    }
}

impl IntoIterator for Link {
    type Item = Link;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        let mut segments = Vec::with_capacity(self.len());
        for i in 0..self.len() {
            segments.push(self.nth(i));
        }
        segments.into_iter()
    }
}

impl PartialEq<str> for Link {
    fn eq(&self, other: &str) -> bool {
        self.to_string().as_str() == other
    }
}

impl<'de> serde::Deserialize<'de> for Link {
    fn deserialize<D>(deserializer: D) -> Result<Link, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        let s: &str = Deserialize::deserialize(deserializer)?;
        Link::to(s).map_err(de::Error::custom)
    }
}

impl serde::Serialize for Link {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        s.serialize_str(&format!("{}", self))
    }
}

impl fmt::Display for Link {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", format!("/{}", self.segments.join("/")))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Ref(ValueId);

impl Ref {
    pub fn to(id: &str) -> TCResult<Ref> {
        if valid_id(id) {
            Ok(Ref(id.to_string()))
        } else {
            Err(error::bad_request(
                "A reference id may not contain whitespace or any of these patterns",
                RESERVED_CHARS.join(", "),
            ))
        }
    }

    pub fn value_id(&self) -> ValueId {
        self.0.to_string()
    }
}

impl fmt::Display for Ref {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "${}", self.0)
    }
}

struct RefVisitor;

impl<'de> de::Visitor<'de> for RefVisitor {
    type Value = Ref;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("A reference to a local variable (e.g. '$foo')")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ref::to(value).map_err(de::Error::custom)
    }
}

impl<'de> de::Deserialize<'de> for Ref {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        d.deserialize_str(RefVisitor)
    }
}

impl Serialize for Ref {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        s.serialize_str(&format!("${}", self))
    }
}

#[derive(Clone, Debug, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub enum Subject {
    Link(Link),
    Ref(Ref),
}

impl fmt::Display for Subject {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Subject::Link(l) => write!(f, "{}", l),
            Subject::Ref(r) => write!(f, "{}", r),
        }
    }
}

impl From<Link> for Subject {
    fn from(l: Link) -> Subject {
        Subject::Link(l)
    }
}

impl From<Ref> for Subject {
    fn from(r: Ref) -> Subject {
        Subject::Ref(r)
    }
}

#[derive(Clone, Hash, Eq, PartialEq)]
pub enum Op {
    Get {
        subject: Subject,
        key: Box<TCValue>,
    },
    Put {
        subject: Subject,
        key: Box<TCValue>,
        value: Box<TCValue>,
    },
    Post {
        subject: Option<Ref>,
        action: Link,
        requires: Vec<(String, TCValue)>,
    },
}

impl Op {
    pub fn get(subject: Subject, key: TCValue) -> Op {
        Op::Get {
            subject,
            key: Box::new(key),
        }
    }

    pub fn put(subject: Subject, key: TCValue, value: TCValue) -> Op {
        Op::Put {
            subject,
            key: Box::new(key),
            value: Box::new(value),
        }
    }

    pub fn post(subject: Option<Ref>, action: Link, requires: Vec<(String, TCValue)>) -> Op {
        Op::Post {
            subject,
            action,
            requires,
        }
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Op::Get { subject, key } => write!(f, "subject: {}, key: {}", subject, key),
            Op::Put {
                subject,
                key,
                value,
            } => write!(f, "subject: {}, key: {}, value: {}", subject, key, value),
            Op::Post {
                subject,
                action,
                requires,
            } => write!(
                f,
                "subject: {}, action: {}, requires: {:?}",
                subject
                    .clone()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| String::from("None")),
                action,
                requires
            ),
        }
    }
}

#[derive(Clone, Hash, Eq, PartialEq)]
pub enum TCValue {
    None,
    Int32(i32),
    Link(Link),
    Op(Op),
    r#String(String),
    Vector(Vec<TCValue>),
    Ref(Ref),
}

impl TCValueExt for String {}

impl From<Link> for TCValue {
    fn from(link: Link) -> TCValue {
        TCValue::Link(link)
    }
}

impl From<Op> for TCValue {
    fn from(op: Op) -> TCValue {
        TCValue::Op(op)
    }
}

impl From<Ref> for TCValue {
    fn from(r: Ref) -> TCValue {
        TCValue::Ref(r)
    }
}

impl From<String> for TCValue {
    fn from(s: String) -> TCValue {
        TCValue::r#String(s)
    }
}

impl From<Vec<TCValue>> for TCValue {
    fn from(v: Vec<TCValue>) -> TCValue {
        TCValue::Vector(v)
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

impl TryFrom<TCValue> for String {
    type Error = error::TCError;

    fn try_from(v: TCValue) -> TCResult<String> {
        match v {
            TCValue::r#String(s) => Ok(s),
            other => Err(error::bad_request("Expected a String but found", other)),
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

impl TryFrom<TCValue> for Vec<(ValueId, TCValue)> {
    type Error = error::TCError;

    fn try_from(value: TCValue) -> TCResult<Vec<(ValueId, TCValue)>> {
        let v: Vec<(String, TCValue)> = value.try_into()?;
        Ok(v.into_iter().collect())
    }
}

impl From<Vec<(ValueId, TCValue)>> for TCValue {
    fn from(values: Vec<(ValueId, TCValue)>) -> TCValue {
        let mut result: Vec<TCValue> = vec![];
        for (id, val) in values {
            let mut r_item: Vec<TCValue> = vec![];
            r_item.push(id.into());
            r_item.push(val);
            result.push(r_item.into());
        }
        result.into()
    }
}

impl<T1: TCValueExt, T2: TCValueExt> std::iter::FromIterator<TCValue> for TCResult<Vec<(T1, T2)>> {
    fn from_iter<I: IntoIterator<Item = TCValue>>(iter: I) -> Self {
        let mut v: Vec<(T1, T2)> = vec![];
        for item in iter {
            v.push(item.try_into()?);
        }
        Ok(v)
    }
}

impl<T1: TCValueExt, T2: TCValueExt> TryFrom<TCValue> for Vec<(T1, T2)> {
    type Error = error::TCError;

    fn try_from(value: TCValue) -> TCResult<Vec<(T1, T2)>> {
        let v: Vec<TCValue> = value.try_into()?;
        v.into_iter().collect()
    }
}

impl<T1: TCValueExt, T2: TCValueExt> TryFrom<TCValue> for (T1, T2) {
    type Error = error::TCError;

    fn try_from(value: TCValue) -> TCResult<(T1, T2)> {
        let v: Vec<TCValue> = value.clone().try_into()?;
        if v.len() == 2 {
            Ok((v[0].clone().try_into()?, v[1].clone().try_into()?))
        } else {
            Err(error::bad_request("Expected 2-tuple but found", value))
        }
    }
}

impl TryFrom<TCState> for TCValue {
    type Error = error::TCError;

    fn try_from(state: TCState) -> TCResult<TCValue> {
        match state {
            TCState::Value(value) => Ok(value),
            other => Err(error::bad_request("Expected a Value but found", other)),
        }
    }
}

struct TCValueVisitor;

impl<'de> de::Visitor<'de> for TCValueVisitor {
    type Value = TCValue;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("A JSON value")
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
        if let Some(key) = access.next_key::<String>()? {
            if key.starts_with('/') {
                let value = access.next_value::<Vec<(String, TCValue)>>()?;

                let link = Link::to(&key).map_err(de::Error::custom)?;

                if value.is_empty() {
                    Ok(link.into())
                } else {
                    Ok(Op::post(None, link, value).into())
                }
            } else if key.starts_with('$') {
                if key.contains('/') {
                    let key: Vec<&str> = key.split('/').collect();
                    let subject = Ref::to(&key[0][1..]).map_err(de::Error::custom)?;
                    let method =
                        Link::to(&format!("/{}", key[1..].join("/"))).map_err(de::Error::custom)?;
                    let requires = access.next_value::<Vec<(String, TCValue)>>()?;

                    Ok(Op::post(subject.into(), method, requires).into())
                } else {
                    let subject = Ref::to(&key[1..].to_string()).map_err(de::Error::custom)?;
                    let value = access.next_value::<Vec<TCValue>>()?;

                    if value.is_empty() {
                        Ok(subject.into())
                    } else if value.len() == 1 {
                        Ok(Op::get(subject.into(), value[0].clone()).into())
                    } else if value.len() == 2 {
                        Ok(Op::put(subject.into(), value[0].clone(), value[1].clone()).into())
                    } else {
                        Err(de::Error::custom(format!(
                            "Expected a list of 0, 1, or 2 Values for {}",
                            key
                        )))
                    }
                }
            } else {
                Err(de::Error::custom(format!(
                    "Expected a Link starting with '/' or a Ref starting with '$', found {}",
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
            TCValue::Int32(i) => s.serialize_i32(*i),
            TCValue::Link(l) => l.serialize(s),
            TCValue::Op(o) => match o {
                Op::Get { subject, key } => {
                    let mut op = s.serialize_struct_variant("Op", 0, "Get", 2)?;
                    op.serialize_field("subject", subject)?;
                    op.serialize_field("key", key)?;
                    op.end()
                }
                Op::Put {
                    subject,
                    key,
                    value,
                } => {
                    let mut op = s.serialize_struct_variant("Op", 1, "Put", 3)?;
                    op.serialize_field("subject", subject)?;
                    op.serialize_field("key", key)?;
                    op.serialize_field("value", value)?;
                    op.end()
                }
                Op::Post {
                    subject,
                    action,
                    requires,
                } => {
                    let mut op = s.serialize_struct_variant("Op", 2, "Post", 3)?;
                    op.serialize_field("subject", subject)?;
                    op.serialize_field("action", action)?;
                    op.serialize_field("requires", requires)?;
                    op.end()
                }
            },
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
            TCValue::Int32(i) => write!(f, "Int32: {}", i),
            TCValue::Link(l) => write!(f, "Link: {}", l),
            TCValue::Op(o) => write!(f, "Op: {}", o),
            TCValue::Ref(r) => write!(f, "Ref: {}", r),
            TCValue::r#String(s) => write!(f, "string: {}", s),
            TCValue::Vector(v) => write!(f, "vector: {:?}", v),
        }
    }
}
