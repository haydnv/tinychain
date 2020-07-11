use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::hash::Hash;
use std::str::FromStr;

use regex::Regex;
use serde::de;
use serde::ser::{Serialize, SerializeMap, Serializer};
use uuid::Uuid;

use crate::error;

use super::class::{Impl, StringType};
use super::link::{Link, TCPath};
use super::reference::TCRef;
use super::TCResult;

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
pub enum TCString {
    Id(ValueId),
    Link(Link),
    Ref(TCRef),
    r#String(String),
}

impl Impl for TCString {
    type Class = StringType;

    fn class(&self) -> StringType {
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
        match self {
            Self::Id(i) => {
                let mut map = s.serialize_map(Some(1))?;
                map.serialize_entry("/sbin/value/id", &[i.as_str()])?;
                map.end()
            }
            Self::Link(l) => l.serialize(s),
            Self::Ref(r) => r.serialize(s),
            Self::r#String(v) => s.serialize_str(v),
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
