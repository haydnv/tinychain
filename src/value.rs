use std::collections::HashMap;
use std::fmt;

use regex::Regex;
use serde::de;
use serde::ser::{SerializeSeq, Serializer};
use serde::{Deserialize, Serialize};

use crate::context::TCResult;
use crate::error;

const LINK_BLACKLIST: [&str; 11] = ["..", "~", "$", "&", "?", "|", "{", "}", "//", ":", "="];

#[derive(Clone, Hash, Eq, PartialEq)]
pub struct Link {
    to: String,
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
            to: destination.to_string(),
            segments: destination[1..].split('/').map(|s| s.to_string()).collect(),
        })
    }

    pub fn as_str(&self) -> &str {
        &self.to
    }

    pub fn append(&self, suffix: Link) -> Link {
        Link::to(&format!("{}{}", self.to, suffix.to)).unwrap()
    }

    pub fn from(&self, prefix: &str) -> TCResult<Link> {
        if prefix.ends_with('/') {
            return Err(error::bad_request("Link prefix cannot end in a /", prefix));
        }
        if !self.to.starts_with(prefix) {
            return Err(error::bad_request(
                &format!("Cannot link {} from", self),
                prefix,
            ));
        }

        Link::to(&self.to[prefix.len()..])
    }

    pub fn len(&self) -> usize {
        self.segments.len()
    }

    pub fn segment(&self, i: usize) -> Link {
        let to = format!("/{}", self.segments[i]);
        Link {
            to,
            segments: vec![self.segments[i].clone()],
        }
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
        s.serialize_str(self.as_str())
    }
}

impl fmt::Display for Link {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to)
    }
}

impl IntoIterator for Link {
    type Item = Link;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        let mut segments = Vec::with_capacity(self.len());
        for i in 0..self.len() {
            segments.push(self.segment(i));
        }
        segments.into_iter()
    }
}

impl<Idx> std::ops::Index<Idx> for Link
where
    Idx: std::slice::SliceIndex<[String]>,
{
    type Output = Idx::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.segments[index]
    }
}

#[derive(Clone, Hash, Eq, PartialEq)]
pub enum TCValue {
    Bytes(Vec<u8>),
    Int32(i32),
    Link(Link),
    r#String(String),
    Vector(Vec<TCValue>),
}

impl TCValue {
    pub fn to_bytes(&self) -> TCResult<Vec<u8>> {
        match self {
            TCValue::Bytes(b) => Ok(b.clone()),
            other => Err(error::bad_request("Expected bytes but found", other)),
        }
    }

    pub fn to_link(&self) -> TCResult<Link> {
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
            TCValue::Bytes(b) => s.serialize_bytes(b),
            TCValue::Int32(i) => s.serialize_i32(*i),
            TCValue::Link(l) => l.serialize(s),
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
        if let Some((link, value)) = access.next_entry::<String, HashMap<String, TCValue>>()? {
            if value.is_empty() && link.starts_with('/') {
                match Link::to(&link) {
                    Ok(l) => Ok(TCValue::Link(l)),
                    Err(cause) => Err(de::Error::custom(cause)),
                }
            } else {
                Err(de::Error::custom(
                    "Link must start with a '/' and does not support any options",
                ))
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
            TCValue::Bytes(b) => write!(f, "binary of length {}", b.len()),
            TCValue::Int32(i) => write!(f, "Int32: {}", i),
            TCValue::Link(l) => write!(f, "Link: {}", l),
            TCValue::r#String(s) => write!(f, "string: {}", s),
            TCValue::Vector(v) => write!(f, "vector: {:?}", v),
        }
    }
}
