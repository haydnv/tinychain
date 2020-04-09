use std::fmt;

use regex::Regex;
use serde::de;
use serde::ser::{SerializeSeq, SerializeStruct, Serializer};
use serde::{Deserialize, Serialize};

use crate::context::TCResult;
use crate::error;

const LINK_BLACKLIST: [&str; 11] = ["..", "~", "$", "&", "?", "|", "{", "}", "//", ":", "="];

pub type ValueId = String;

#[derive(Clone, Hash, Eq, PartialEq)]
pub struct Link {
    segments: Vec<String>,
}

impl Link {
    fn _validate(to: &str) -> TCResult<()> {
        for pattern in LINK_BLACKLIST.iter() {
            if to.contains(pattern) {
                return Err(error::bad_request(
                    "Tinychain links do not allow this pattern",
                    pattern,
                ));
            }
        }

        if !to.starts_with('/') {
            Err(error::bad_request(
                "Expected an absolute path starting with '/' but found",
                to,
            ))
        } else if to != "/" && to.ends_with('/') {
            Err(error::bad_request("Trailing slash is not allowed", to))
        } else if Regex::new(r"\s").unwrap().find(to).is_some() {
            Err(error::bad_request(
                "Tinychain links do not allow whitespace",
                to,
            ))
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

#[derive(Clone, Deserialize, Serialize, Hash, Eq, PartialEq)]
pub struct Op {
    action: Link,
    requires: Vec<(String, TCValue)>,
}

impl Op {
    pub fn new(action: Link, requires: Vec<(String, TCValue)>) -> Op {
        Op { action, requires }
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
}

impl From<Link> for TCValue {
    fn from(link: Link) -> TCValue {
        TCValue::Link(link)
    }
}

impl PartialEq<str> for Link {
    fn eq(&self, other: &str) -> bool {
        self.to_string().as_str() == other
    }
}

impl TCValue {
    pub fn as_link(&self) -> TCResult<Link> {
        match self {
            TCValue::Link(l) => Ok(l.clone()),
            other => Err(error::bad_request("Expected a Link but found", other)),
        }
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
                        action: link,
                        requires: value,
                    }))
                }
            } else {
                Err(de::Error::custom("Link must start with a '/'"))
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
            TCValue::r#String(s) => write!(f, "string: {}", s),
            TCValue::Vector(v) => write!(f, "vector: {:?}", v),
        }
    }
}
