use std::fmt;
use std::str::FromStr;

use serde::de;
use serde::ser::Serializer;
use serde::Serialize;

use crate::error;
use crate::value::{TCResult, ValueId};

#[derive(Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct TCRef {
    to: ValueId,
}

impl TCRef {
    pub fn value_id(&'_ self) -> &'_ ValueId {
        &self.to
    }
}

impl From<ValueId> for TCRef {
    fn from(to: ValueId) -> TCRef {
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

impl From<TCRef> for ValueId {
    fn from(r: TCRef) -> ValueId {
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

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if !value.starts_with('$') {
            Err(de::Error::custom(format!(
                "Expected Ref starting with $, found {}",
                value
            )))
        } else {
            value[1..].parse().map_err(de::Error::custom)
        }
    }
}

impl<'de> de::Deserialize<'de> for TCRef {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        d.deserialize_str(RefVisitor)
    }
}

impl Serialize for TCRef {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        s.serialize_str(&format!("{}", self))
    }
}
