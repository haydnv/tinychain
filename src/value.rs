use std::fmt;

use regex::Regex;
use serde::de;
use serde::ser::{SerializeSeq, SerializeStruct, Serializer};
use serde::{Deserialize, Serialize};

use crate::context::TCResult;
use crate::error;

const RESERVED_CHARS: [&str; 17] = [
    "..", "~", "$", "&", "?", "|", "{", "}", "//", ":", "=", "^", ">", "<", "'", "`", "\"",
];

pub type ValueId = String;

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

#[derive(Clone, Hash, Eq, PartialEq)]
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

        Ok(Link {
            segments: destination[1..].split('/').map(|s| s.to_string()).collect(),
        })
    }

    pub fn append(&self, suffix: &Link) -> Link {
        Link::to(&format!("{}{}", self, suffix)).unwrap()
    }

    pub fn as_str(&self, index: usize) -> &str {
        self.segments[index].as_str()
    }

    pub fn slice_from(&self, start: usize) -> Link {
        Link {
            segments: self.segments[start..].to_vec(),
        }
    }

    pub fn len(&self) -> usize {
        self.segments.len()
    }

    pub fn nth(&self, i: usize) -> Link {
        Link {
            segments: vec![self.segments[i].clone()],
        }
    }

    pub fn slice_to(&self, end: usize) -> Link {
        Link {
            segments: self.segments[..end].to_vec(),
        }
    }
}

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

#[derive(Clone, Eq, PartialEq, Hash)]
pub struct Ref {
    id: String,
}

impl Ref {
    pub fn to(id: &str) -> TCResult<Ref> {
        if valid_id(id) {
            Ok(Ref { id: id.to_string() })
        } else {
            Err(error::bad_request(
                "A reference id may not contain whitespace or any of these patterns",
                RESERVED_CHARS.join(", "),
            ))
        }
    }
}

impl fmt::Display for Ref {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "${}", self.id)
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
        s.serialize_str(&format!("${}", self.id))
    }
}

#[derive(Clone, Deserialize, Serialize, Hash, Eq, PartialEq)]
pub enum OpMethod {
    Get,
    Put,
    Post,
}

#[derive(Clone, Deserialize, Serialize, Hash, Eq, PartialEq)]
pub struct Op {
    method: OpMethod,
    subject: Option<Ref>,
    action: Link,
    requires: Vec<(String, TCValue)>,
}

impl Op {
    pub fn new(action: Link, requires: Vec<(String, TCValue)>) -> Op {
        Op {
            method: OpMethod::Post,
            subject: None,
            action,
            requires,
        }
    }

    pub fn action(&self) -> Link {
        self.action.clone()
    }

    pub fn requires(&self) -> Vec<(String, TCValue)> {
        self.requires.clone()
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}({:?})", self.action, self.requires)
    }
}

#[derive(Clone, Hash, Eq, PartialEq)]
pub enum TCValue {
    Int32(i32),
    Link(Link),
    Op(Op),
    r#String(String),
    Vector(Vec<TCValue>),
    Ref(Ref),
}

impl TCValue {
    pub fn as_link(&self) -> TCResult<Link> {
        match self {
            TCValue::Link(l) => Ok(l.clone()),
            other => Err(error::bad_request("Expected a Link but found", other)),
        }
    }
}

impl From<Link> for TCValue {
    fn from(link: Link) -> TCValue {
        TCValue::Link(link)
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
        if let Some((key, value)) = access.next_entry::<String, Vec<(String, TCValue)>>()? {
            if key.starts_with('/') {
                let link = match Link::to(&key) {
                    Ok(l) => l,
                    Err(cause) => {
                        return Err(de::Error::custom(cause));
                    }
                };

                if value.is_empty() {
                    Ok(TCValue::Link(link))
                } else {
                    Ok(TCValue::Op(Op {
                        method: OpMethod::Post,
                        subject: None,
                        action: link,
                        requires: value,
                    }))
                }
            } else if key.starts_with('$') {
                if let Some(pos) = key.find('/') {
                    let method = if value.is_empty() {
                        OpMethod::Get
                    } else {
                        OpMethod::Post
                    };
                    let subject = Some(Ref::to(&key[1..pos]).map_err(de::Error::custom)?);
                    let action = Link::to(&key[pos..]).map_err(de::Error::custom)?;

                    Ok(TCValue::Op(Op {
                        method,
                        subject,
                        action,
                        requires: value,
                    }))
                } else {
                    Ok(TCValue::Ref(Ref::to(&key).map_err(de::Error::custom)?))
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
            TCValue::Int32(i) => s.serialize_i32(*i),
            TCValue::Link(l) => l.serialize(s),
            TCValue::Op(o) => {
                let mut op = s.serialize_struct("Op", 2)?;
                op.serialize_field("action", &o.action)?;
                op.serialize_field("requires", &o.requires)?;
                op.end()
            }
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
            TCValue::Int32(i) => write!(f, "Int32: {}", i),
            TCValue::Link(l) => write!(f, "Link: {}", l),
            TCValue::Op(o) => write!(f, "Op: {}", o),
            TCValue::Ref(r) => write!(f, "Ref: {}", r),
            TCValue::r#String(s) => write!(f, "string: {}", s),
            TCValue::Vector(v) => write!(f, "vector: {:?}", v),
        }
    }
}
