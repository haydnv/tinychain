use std::fmt;
use std::str::FromStr;

use async_trait::async_trait;
use destream::{de, Decoder, Encoder, FromStream, ToStream};
use safecast::TryCastFrom;
use regex::Regex;

use error::*;

const RESERVED_CHARS: [&str; 21] = [
    "/", "..", "~", "$", "`", "^", "&", "|", "=", "^", "{", "}", "<", ">", "'", "\"", "?", ":",
    "//", "@", "#",
];

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

#[async_trait]
impl FromStream for Id {
    async fn from_stream<D: Decoder>(d: &mut D) -> Result<Self, D::Error> {
        d.decode_any(IdVisitor).await
    }
}

struct IdVisitor;

#[async_trait]
impl de::Visitor for IdVisitor {
    type Value = Id;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a Tinychain Id like {\"foo\": []}")
    }

    fn visit_string<E: de::Error>(self, s: String) -> Result<Self::Value, E> {
        Id::from_str(&s).map_err(de::Error::custom)
    }

    async fn visit_map<M: de::MapAccess>(self, mut access: M) -> Result<Self::Value, M::Error> {
        if let Some(key) = access.next_key::<String>().await? {
            let value: [u8; 0] = access.next_value().await?;
            if value.is_empty() {
                Id::from_str(&key).map_err(de::Error::custom)
            } else {
                Err(de::Error::custom("Expected Id but found OpRef"))
            }
        } else {
            Err(de::Error::custom("Unable to parse Id"))
        }
    }
}

impl<'en> ToStream<'en> for Id {
    fn to_stream<E: Encoder<'en>>(&'en self, e: E) -> Result<E::Ok, E::Error> {
        e.encode_str(&self.id)
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
    type Err = TCError;

    fn from_str(id: &str) -> TCResult<Id> {
        validate_id(id)?;
        Ok(Id { id: id.to_string() })
    }
}

impl TryCastFrom<String> for Id {
    fn can_cast_from(id: &String) -> bool {
        validate_id(id).is_ok()
    }

    fn opt_cast_from(id: String) -> Option<Id> {
        match id.parse() {
            Ok(id) => Some(id),
            Err(_) => None,
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

fn validate_id(id: &str) -> TCResult<()> {
    if id.is_empty() {
        return Err(TCError::bad_request("Id cannot be empty", id));
    }

    let filtered: &str = &id.chars().filter(|c| *c as u8 > 32).collect::<String>();
    if filtered != id {
        return Err(TCError::bad_request(
            "This value ID contains an ASCII control character",
            filtered,
        ));
    }

    for pattern in &RESERVED_CHARS {
        if id.contains(pattern) {
            return Err(TCError::bad_request(
                "A value ID may not contain this pattern",
                pattern,
            ));
        }
    }

    if let Some(w) = Regex::new(r"\s").unwrap().find(id) {
        return Err(TCError::bad_request(
            "A value ID may not contain whitespace",
            format!("{:?}", w),
        ));
    }

    Ok(())
}
