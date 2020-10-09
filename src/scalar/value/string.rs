use std::convert::TryFrom;
use std::fmt;
use std::hash::Hash;
use std::str::FromStr;

use regex::Regex;
use serde::de;
use serde::ser::{Serialize, SerializeMap, Serializer};
use uuid::Uuid;

use crate::class::{Class, Instance, TCResult};
use crate::error;
use crate::scalar::{Scalar, ScalarClass, ScalarInstance, TryCastFrom};

use super::class::{ValueClass, ValueInstance, ValueType};
use super::link::{Link, TCPath};
use super::reference::TCRef;
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

impl ScalarClass for StringType {
    type Instance = TCString;

    fn try_cast<S: Into<Scalar>>(&self, scalar: S) -> TCResult<TCString> {
        let scalar: Scalar = scalar.into();
        let value = Value::try_cast_from(scalar, |s| {
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

impl<'de> de::Deserialize<'de> for ValueId {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        d.deserialize_any(ValueIdVisitor)
    }
}

struct ValueIdVisitor;

impl<'de> de::Visitor<'de> for ValueIdVisitor {
    type Value = ValueId;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a Tinychain ValueId like {\"foo\": []}")
    }

    fn visit_map<M>(self, mut access: M) -> Result<Self::Value, M::Error>
    where
        M: de::MapAccess<'de>,
    {
        if let Some(key) = access.next_key::<&str>()? {
            let value: Vec<super::Value> = access.next_value()?;
            if value.is_empty() {
                ValueId::from_str(key).map_err(de::Error::custom)
            } else {
                Err(de::Error::custom("Expected ValueId but found OpRef"))
            }
        } else {
            Err(de::Error::custom("Unable to parse ValueId"))
        }
    }

    fn visit_str<E>(self, s: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        ValueId::from_str(s).map_err(de::Error::custom)
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

impl TryCastFrom<Link> for ValueId {
    fn can_cast_from(link: &Link) -> bool {
        if link.host().is_none() {
            ValueId::can_cast_from(link.path())
        } else {
            false
        }
    }

    fn opt_cast_from(link: Link) -> Option<ValueId> {
        if link.host().is_none() {
            ValueId::opt_cast_from(link.into_path())
        } else {
            None
        }
    }
}

impl TryCastFrom<String> for ValueId {
    fn can_cast_from(id: &String) -> bool {
        validate_id(id).is_ok()
    }

    fn opt_cast_from(id: String) -> Option<ValueId> {
        match id.parse() {
            Ok(value_id) => Some(value_id),
            Err(_) => None,
        }
    }
}

impl TryCastFrom<TCPath> for ValueId {
    fn can_cast_from(path: &TCPath) -> bool {
        path.len() == 1
    }

    fn opt_cast_from(path: TCPath) -> Option<ValueId> {
        if path.len() == 1 {
            let mut value_id = path.into_segments();
            value_id.pop()
        } else {
            None
        }
    }
}

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.id)
    }
}

#[derive(Clone, Eq, PartialEq)]
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

impl TryFrom<TCString> for ValueId {
    type Error = error::TCError;

    fn try_from(s: TCString) -> TCResult<ValueId> {
        match s {
            TCString::Id(id) => Ok(id),
            TCString::UString(s) => ValueId::from_str(&s),
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

impl TryCastFrom<TCString> for Link {
    fn can_cast_from(s: &TCString) -> bool {
        match s {
            TCString::UString(s) => Link::from_str(s).is_ok(),
            _ => true,
        }
    }

    fn opt_cast_from(s: TCString) -> Option<Link> {
        match s {
            TCString::Id(id) => Some(TCPath::from(id).into()),
            TCString::Link(link) => Some(link),
            TCString::Ref(tc_ref) => Some(TCPath::from(tc_ref.into_id()).into()),
            TCString::UString(s) => Link::from_str(&s).ok(),
        }
    }
}

impl TryCastFrom<TCString> for ValueId {
    fn can_cast_from(s: &TCString) -> bool {
        match s {
            TCString::Id(_) => true,
            TCString::Link(link) => ValueId::can_cast_from(link),
            TCString::Ref(_) => true,
            TCString::UString(s) => ValueId::can_cast_from(s),
        }
    }

    fn opt_cast_from(s: TCString) -> Option<ValueId> {
        match s {
            TCString::Id(id) => Some(id),
            TCString::Link(link) => ValueId::opt_cast_from(link),
            TCString::Ref(tc_ref) => Some(tc_ref.into_id()),
            TCString::UString(s) => ValueId::opt_cast_from(s),
        }
    }
}

impl TryCastFrom<TCString> for TCRef {
    fn can_cast_from(s: &TCString) -> bool {
        ValueId::can_cast_from(s)
    }

    fn opt_cast_from(s: TCString) -> Option<TCRef> {
        ValueId::opt_cast_from(s).map(TCRef::from)
    }
}

impl Serialize for TCString {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
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
            TCString::Ref(r) => write!(f, "{}", r),
            TCString::UString(u) => write!(f, "{}", u),
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
