use std::fmt;
use std::str::FromStr;

use serde::de;
use serde::ser::{Serialize, SerializeMap, Serializer};

use crate::error::{self, TCResult};
use crate::scalar::Id;

const EMPTY_SLICE: &[usize] = &[];

#[derive(Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct TCRef {
    to: Id,
}

impl TCRef {
    pub fn into_id(self) -> Id {
        self.to
    }

    pub fn id(&'_ self) -> &'_ Id {
        &self.to
    }
}

impl From<Id> for TCRef {
    fn from(to: Id) -> TCRef {
        TCRef { to }
    }
}

impl FromStr for TCRef {
    type Err = error::TCError;

    fn from_str(to: &str) -> TCResult<TCRef> {
        if !to.starts_with('$') || to.len() < 2 {
            Err(error::bad_request("Invalid Ref", to))
        } else {
            Ok(TCRef {
                to: to[1..].parse()?,
            })
        }
    }
}

impl PartialEq<Id> for TCRef {
    fn eq(&self, other: &Id) -> bool {
        self.id() == other
    }
}

impl From<TCRef> for Id {
    fn from(r: TCRef) -> Id {
        r.to
    }
}

impl fmt::Display for TCRef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "${}", self.to)
    }
}

struct RefVisitor;

impl<'de> de::Visitor<'de> for RefVisitor {
    type Value = TCRef;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("A reference to a local variable (e.g. '$foo')")
    }

    fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
        if !value.starts_with('$') {
            Err(de::Error::custom(format!(
                "Expected Ref starting with $, found {}",
                value
            )))
        } else {
            // TODO: move deserialization logic here
            value[1..].parse().map_err(de::Error::custom)
        }
    }
}

impl<'de> de::Deserialize<'de> for TCRef {
    fn deserialize<D: de::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_str(RefVisitor)
    }
}

impl Serialize for TCRef {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut map = s.serialize_map(Some(1))?;
        map.serialize_entry(&self.to_string(), EMPTY_SLICE)?;
        map.end()
    }
}
