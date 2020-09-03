use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::hash::Hash;
use std::str::FromStr;

use regex::Regex;
use serde::de;
use serde::ser::{Serialize, SerializeMap, Serializer};
use uuid::Uuid;

use crate::class::{Class, Instance};
use crate::error;

use super::class::{ValueClass, ValueInstance};
use super::link::{Link, TCPath};
use super::reference::TCRef;
use super::{TCResult, ValueType};

const RESERVED_CHARS: [&str; 21] = [
    "/", "..", "~", "$", "`", "^", "&", "|", "=", "^", "{", "}", "<", ">", "'", "\"", "?", ":",
    "//", "@", "#",
];

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub enum StringType {
    Id,
    Link,
    Ref,
    UString,
}

impl Class for StringType {
    type Instance = TCString;

    fn from_path(path: &TCPath) -> TCResult<Self> {
        let path = path.from_path(&Self::prefix())?;

        if path.len() == 1 {
            match path[0].as_str() {
                "id" => Ok(StringType::Id),
                "link" => Ok(StringType::Link),
                "ref" => Ok(StringType::Ref),
                "ustring" => Ok(StringType::UString),
                other => Err(error::not_found(other)),
            }
        } else {
            Err(error::not_found(path))
        }
    }

    fn prefix() -> TCPath {
        ValueType::prefix().join(label("string").into())
    }
}

impl ValueClass for StringType {
    type Instance = TCString;

    fn get(path: &TCPath, value: TCString) -> TCResult<TCString> {
        let suffix = path.from_path(&Self::prefix())?;

        if suffix.is_empty() {
            return Ok(value);
        }

        match suffix[0].as_str() {
            "id" => value.try_into().map(TCString::Id),
            "link" => value.try_into().map(TCString::Link),
            "ref" => value.try_into().map(TCString::Ref),
            "ustring" => value.try_into().map(TCString::UString),
            other => Err(error::not_found(other)),
        }
    }

    fn size(self) -> Option<usize> {
        None
    }
}

impl From<StringType> for Link {
    fn from(st: StringType) -> Link {
        let prefix = StringType::prefix();

        use StringType::*;
        match st {
            Id => prefix.join(label("id").into()).into(),
            Link => prefix.join(label("link").into()).into(),
            Ref => prefix.join(label("ref").into()).into(),
            UString => prefix.join(label("ustring").into()).into(),
        }
    }
}

impl fmt::Display for StringType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Id => write!(f, "type Id"),
            Self::Link => write!(f, "type Link"),
            Self::Ref => write!(f, "type Ref"),
            Self::UString => write!(f, "type UString"),
        }
    }
}

pub struct Label {
    id: &'static str,
}

pub const fn label(id: &'static str) -> Label {
    Label { id }
}

impl From<Label> for ValueId {
    fn from(l: Label) -> ValueId {
        ValueId {
            id: l.id.to_string(),
        }
    }
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

impl From<usize> for ValueId {
    fn from(u: usize) -> ValueId {
        u.to_string().parse().unwrap()
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

impl PartialEq<Label> for ValueId {
    fn eq(&self, other: &Label) -> bool {
        self.id == other.id
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
    UString(String),
}

impl Instance for TCString {
    type Class = StringType;

    fn class(&self) -> StringType {
        match self {
            TCString::Id(_) => StringType::Id,
            TCString::Link(_) => StringType::Link,
            TCString::Ref(_) => StringType::Ref,
            TCString::UString(_) => StringType::UString,
        }
    }
}

impl ValueInstance for TCString {
    type Class = StringType;
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
            TCString::UString(s) => Ok(s),
            other => Err(error::bad_request("Expected String but found", other)),
        }
    }
}

impl<'a> TryFrom<&'a TCString> for &'a String {
    type Error = error::TCError;

    fn try_from(s: &'a TCString) -> TCResult<&'a String> {
        match s {
            TCString::UString(s) => Ok(s),
            other => Err(error::bad_request("Expected String but found", other)),
        }
    }
}

impl TryFrom<TCString> for ValueId {
    type Error = error::TCError;

    fn try_from(s: TCString) -> TCResult<ValueId> {
        match s {
            TCString::Id(id) => Ok(id),
            other => Err(error::bad_request("Expected ValueId but found", other)),
        }
    }
}

impl<'a> TryFrom<&'a TCString> for &'a ValueId {
    type Error = error::TCError;

    fn try_from(s: &'a TCString) -> TCResult<&'a ValueId> {
        match s {
            TCString::Id(id) => Ok(id),
            other => Err(error::bad_request("Expected ValueId but found", other)),
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

impl TryFrom<TCString> for TCRef {
    type Error = error::TCError;

    fn try_from(s: TCString) -> TCResult<TCRef> {
        match s {
            TCString::Ref(tc_ref) => Ok(tc_ref),
            other => Err(error::bad_request("Expected ValueId but found", other)),
        }
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
            Self::UString(u) => s.serialize_str(u.as_str()),
        }
    }
}

impl fmt::Display for TCString {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TCString::Id(id) => write!(f, "ValueId: {}", id),
            TCString::Link(l) => write!(f, "Link: {}", l),
            TCString::Ref(r) => write!(f, "Ref: {}", r),
            TCString::UString(u) => write!(f, "UString: \"{}\"", u),
        }
    }
}

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
