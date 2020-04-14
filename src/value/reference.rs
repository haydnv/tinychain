use std::fmt;

use serde::de;
use serde::ser::Serializer;
use serde::Serialize;

use crate::context::TCResult;
use crate::value::{validate_id, ValueId};

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct TCRef(ValueId);

impl TCRef {
    pub fn to(id: &str) -> TCResult<TCRef> {
        validate_id(id)?;
        Ok(TCRef(id.to_string()))
    }

    pub fn value_id(&self) -> ValueId {
        self.0.to_string()
    }
}

impl fmt::Display for TCRef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "${}", self.0)
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
        TCRef::to(value).map_err(de::Error::custom)
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
        s.serialize_str(&format!("${}", self))
    }
}
