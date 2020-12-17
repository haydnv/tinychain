use std::convert::TryFrom;
use std::fmt;
use std::hash::Hash;
use std::str::FromStr;

use regex::Regex;
use serde::de;
use serde::ser::{Serialize, SerializeMap, Serializer};
use uuid::Uuid;

use crate::class::{Class, Instance, NativeClass, TCType};
use crate::error;
use crate::general::TCResult;
use crate::scalar::{Scalar, ScalarClass, ScalarInstance, TryCastFrom};

use super::class::{ValueClass, ValueInstance, ValueType};
use super::link::{Link, PathSegment, TCPath, TCPathBuf};
use super::Value;

const EMPTY: &[usize] = &[];

const RESERVED_CHARS: [&str; 21] = [
    "/", "..", "~", "$", "`", "^", "&", "|", "=", "^", "{", "}", "<", ">", "'", "\"", "?", ":",
    "//", "@", "#",
];

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub enum StringType {
    Id,
    Link,
    UString,
}

impl Class for StringType {
    type Instance = TCString;
}

impl NativeClass for StringType {
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        let suffix = Self::prefix().try_suffix(path)?;

        if suffix.len() == 1 {
            match suffix[0].as_str() {
                "id" => Ok(StringType::Id),
                "link" => Ok(StringType::Link),
                "ustring" => Ok(StringType::UString),
                other => Err(error::not_found(other)),
            }
        } else {
            Err(error::not_found(TCPath::from(suffix)))
        }
    }

    fn prefix() -> TCPathBuf {
        ValueType::prefix().append(label("string"))
    }
}

impl ScalarClass for StringType {
    type Instance = TCString;

    fn try_cast<S>(&self, scalar: S) -> TCResult<TCString>
    where
        Scalar: From<S>,
    {
        let value = Value::try_cast_from(Scalar::from(scalar), |s| {
            error::bad_request("Can't cast into Value from", s)
        })?;

        TCString::try_cast_from(value, |v| {
            error::bad_request("Can't cast into String from", v)
        })
    }
}

impl ValueClass for StringType {
    type Instance = TCString;

    fn size(self) -> Option<usize> {
        None
    }
}

impl From<StringType> for Link {
    fn from(st: StringType) -> Link {
        use StringType::*;
        let suffix = match st {
            Id => label("id"),
            Link => label("link"),
            UString => label("ustring"),
        };

        StringType::prefix().append(suffix).into()
    }
}

impl From<StringType> for TCType {
    fn from(st: StringType) -> TCType {
        ValueType::TCString(st).into()
    }
}

impl fmt::Display for StringType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Id => write!(f, "type Id"),
            Self::Link => write!(f, "type Link"),
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

impl From<Label> for Id {
    fn from(l: Label) -> Id {
        Id {
            id: l.id.to_string(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Id {
    id: String,
}

impl Id {
    pub fn as_str(&self) -> &str {
        self.id.as_str()
    }

    pub fn starts_with(&self, prefix: &str) -> bool {
        self.id.starts_with(prefix)
    }
}

impl From<Uuid> for Id {
    fn from(id: Uuid) -> Id {
        id.to_hyphenated().to_string().parse().unwrap()
    }
}

impl From<usize> for Id {
    fn from(u: usize) -> Id {
        u.to_string().parse().unwrap()
    }
}

impl From<u64> for Id {
    fn from(i: u64) -> Id {
        i.to_string().parse().unwrap()
    }
}

impl<'de> de::Deserialize<'de> for Id {
    fn deserialize<D: de::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_any(IdVisitor)
    }
}

struct IdVisitor;

impl<'de> de::Visitor<'de> for IdVisitor {
    type Value = Id;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a Tinychain Id like {\"foo\": []}")
    }

    fn visit_str<E: de::Error>(self, s: &str) -> Result<Self::Value, E> {
        // TODO: call parsing logic a different way
        Id::from_str(s).map_err(de::Error::custom)
    }

    fn visit_map<M: de::MapAccess<'de>>(self, mut access: M) -> Result<Self::Value, M::Error> {
        if let Some(key) = access.next_key::<&str>()? {
            let value: Vec<super::Value> = access.next_value()?;
            if value.is_empty() {
                // TODO: call parsing logic a different way
                Id::from_str(key).map_err(de::Error::custom)
            } else {
                Err(de::Error::custom("Expected Id but found OpRef"))
            }
        } else {
            Err(de::Error::custom("Unable to parse Id"))
        }
    }
}

impl Serialize for Id {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.id)
    }
}

impl PartialEq<Label> for Id {
    fn eq(&self, other: &Label) -> bool {
        self.id == other.id
    }
}

impl PartialEq<str> for Id {
    fn eq(&self, other: &str) -> bool {
        &self.id == other
    }
}

impl PartialEq<Id> for &str {
    fn eq(&self, other: &Id) -> bool {
        self == &other.id
    }
}

impl FromStr for Id {
    type Err = error::TCError;

    fn from_str(id: &str) -> TCResult<Id> {
        validate_id(id)?;
        Ok(Id { id: id.to_string() })
    }
}

impl TryCastFrom<String> for Id {
    fn can_cast_from(id: &String) -> bool {
        validate_id(id).is_ok()
    }

    // TODO: move parsing logic here and depend on opt_cast_from in Id::from_str
    fn opt_cast_from(id: String) -> Option<Id> {
        match id.parse() {
            Ok(id) => Some(id),
            Err(_) => None,
        }
    }
}

impl TryCastFrom<TCPathBuf> for Id {
    fn can_cast_from(path: &TCPathBuf) -> bool {
        path.as_slice().len() == 1
    }

    fn opt_cast_from(path: TCPathBuf) -> Option<Id> {
        let mut segments = path.into_vec();
        if segments.len() == 1 {
            segments.pop()
        } else {
            None
        }
    }
}

impl TryCastFrom<Id> for usize {
    fn can_cast_from(id: &Id) -> bool {
        id.as_str().parse::<usize>().is_ok()
    }

    fn opt_cast_from(id: Id) -> Option<usize> {
        id.as_str().parse::<usize>().ok()
    }
}

impl fmt::Display for Id {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.id)
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum TCString {
    Id(Id),
    Link(Link),
    UString(String),
}

impl Instance for TCString {
    type Class = StringType;

    fn class(&self) -> StringType {
        match self {
            TCString::Id(_) => StringType::Id,
            TCString::Link(_) => StringType::Link,
            TCString::UString(_) => StringType::UString,
        }
    }
}

impl ScalarInstance for TCString {
    type Class = StringType;
}

impl ValueInstance for TCString {
    type Class = StringType;
}

impl Default for TCString {
    fn default() -> TCString {
        TCString::UString(String::default())
    }
}

impl From<Link> for TCString {
    fn from(l: Link) -> TCString {
        TCString::Link(l)
    }
}

impl From<TCPathBuf> for TCString {
    fn from(path: TCPathBuf) -> TCString {
        TCString::Link(path.into())
    }
}

impl From<Id> for TCString {
    fn from(id: Id) -> TCString {
        TCString::Id(id)
    }
}

impl TryFrom<TCString> for Link {
    type Error = error::TCError;

    fn try_from(s: TCString) -> TCResult<Link> {
        match s {
            TCString::Link(l) => Ok(l),
            TCString::UString(s) => Link::from_str(&s),
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

impl TryFrom<TCString> for Id {
    type Error = error::TCError;

    fn try_from(s: TCString) -> TCResult<Id> {
        match s {
            TCString::Id(id) => Ok(id),
            TCString::UString(s) => Id::from_str(&s),
            other => Err(error::bad_request("Expected Id but found", other)),
        }
    }
}

impl<'a> TryFrom<&'a TCString> for &'a Id {
    type Error = error::TCError;

    fn try_from(s: &'a TCString) -> TCResult<&'a Id> {
        match s {
            TCString::Id(id) => Ok(id),
            other => Err(error::bad_request("Expected Id but found", other)),
        }
    }
}

impl TryFrom<TCString> for TCPathBuf {
    type Error = error::TCError;

    fn try_from(s: TCString) -> TCResult<TCPathBuf> {
        match s {
            TCString::Link(l) => {
                if l.host().is_none() {
                    Ok(l.into_path())
                } else {
                    Err(error::bad_request("Expected Path but found Link", l))
                }
            }
            other => Err(error::bad_request("Expected Path but found", other)),
        }
    }
}

impl TryCastFrom<Link> for Id {
    fn can_cast_from(link: &Link) -> bool {
        if link.host().is_none() {
            Id::can_cast_from(link.path())
        } else {
            false
        }
    }

    fn opt_cast_from(link: Link) -> Option<Id> {
        if link.host().is_none() {
            Id::opt_cast_from(link.into_path())
        } else {
            None
        }
    }
}

impl TryCastFrom<TCString> for Link {
    fn can_cast_from(s: &TCString) -> bool {
        match s {
            TCString::UString(s) => Link::from_str(s).is_ok(),
            _ => true,
        }
    }

    fn opt_cast_from(s: TCString) -> Option<Link> {
        match s {
            TCString::Id(id) => Some(TCPathBuf::from(id).into()),
            TCString::Link(link) => Some(link),
            // TODO: move Link::from_str logic here and rely on this function in Link::from_str
            TCString::UString(s) => Link::from_str(&s).ok(),
        }
    }
}

impl TryCastFrom<TCString> for Id {
    fn can_cast_from(s: &TCString) -> bool {
        match s {
            TCString::Id(_) => true,
            TCString::Link(link) => Id::can_cast_from(link),
            TCString::UString(s) => Id::can_cast_from(s),
        }
    }

    fn opt_cast_from(s: TCString) -> Option<Id> {
        match s {
            TCString::Id(id) => Some(id),
            TCString::Link(link) => Id::opt_cast_from(link),
            TCString::UString(s) => Id::opt_cast_from(s),
        }
    }
}

impl Serialize for TCString {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        if let Self::UString(ustring) = self {
            s.serialize_str(ustring.as_str())
        } else {
            let mut map = s.serialize_map(Some(1))?;
            map.serialize_entry(&self.to_string(), EMPTY)?;
            map.end()
        }
    }
}

impl fmt::Display for TCString {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TCString::Id(id) => write!(f, "{}", id),
            TCString::Link(l) => write!(f, "{}", l),
            TCString::UString(u) => write!(f, "{}", u),
        }
    }
}

fn validate_id(id: &str) -> TCResult<()> {
    if id.is_empty() {
        return Err(error::bad_request("Id cannot be empty", id));
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
