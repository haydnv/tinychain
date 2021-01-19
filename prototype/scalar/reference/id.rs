use std::collections::{HashMap, HashSet};
use std::fmt;
use std::str::FromStr;

use async_trait::async_trait;
use serde::de;
use serde::ser::{Serialize, SerializeMap, Serializer};

use crate::class::State;
use crate::error;
use crate::request::Request;
use crate::scalar::{Id, Scalar, TCString, Value};
use crate::transaction::Txn;
use crate::{TCResult, TryCastFrom};

use super::{Refer, TCRef};

const EMPTY_SLICE: &[usize] = &[];

#[derive(Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct IdRef {
    to: Id,
}

impl IdRef {
    pub fn into_id(self) -> Id {
        self.to
    }

    pub fn id(&'_ self) -> &'_ Id {
        &self.to
    }
}

#[async_trait]
impl Refer for IdRef {
    fn requires(&self, deps: &mut HashSet<Id>) {
        deps.insert(self.to.clone());
    }

    async fn resolve(
        self,
        _request: &Request,
        _txn: &Txn,
        context: &HashMap<Id, State>,
    ) -> TCResult<State> {
        context
            .get(&self.to)
            .cloned()
            .ok_or_else(|| error::not_found(self))
    }
}

impl From<Id> for IdRef {
    fn from(to: Id) -> IdRef {
        IdRef { to }
    }
}

impl FromStr for IdRef {
    type Err = error::TCError;

    fn from_str(to: &str) -> TCResult<IdRef> {
        if !to.starts_with('$') || to.len() < 2 {
            Err(error::bad_request("Invalid Ref", to))
        } else {
            Ok(IdRef {
                to: to[1..].parse()?,
            })
        }
    }
}

impl PartialEq<Id> for IdRef {
    fn eq(&self, other: &Id) -> bool {
        self.id() == other
    }
}

impl TryCastFrom<Value> for IdRef {
    fn can_cast_from(value: &Value) -> bool {
        match value {
            Value::TCString(tc_string) => match tc_string {
                TCString::Id(_) => true,
                TCString::UString(ustring) => IdRef::from_str(ustring).is_ok(),
                _ => false,
            },
            _ => false,
        }
    }

    fn opt_cast_from(value: Value) -> Option<IdRef> {
        match value {
            Value::TCString(tc_string) => match tc_string {
                TCString::Id(id) => Some(id.into()),
                TCString::UString(ustring) => IdRef::from_str(&ustring).ok(),
                _ => None,
            },
            _ => None,
        }
    }
}

impl TryCastFrom<Scalar> for IdRef {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Ref(tc_ref) => match **tc_ref {
                TCRef::Id(_) => true,
                _ => false,
            },
            Scalar::Value(value) => IdRef::can_cast_from(value),
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<IdRef> {
        match scalar {
            Scalar::Ref(tc_ref) => match *tc_ref {
                TCRef::Id(id_ref) => Some(id_ref),
                _ => None,
            },
            Scalar::Value(value) => IdRef::opt_cast_from(value),
            _ => None,
        }
    }
}

impl From<IdRef> for Id {
    fn from(r: IdRef) -> Id {
        r.to
    }
}

impl fmt::Display for IdRef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "${}", self.to)
    }
}

struct RefVisitor;

impl<'de> de::Visitor<'de> for RefVisitor {
    type Value = IdRef;

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

impl<'de> de::Deserialize<'de> for IdRef {
    fn deserialize<D: de::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_str(RefVisitor)
    }
}

impl Serialize for IdRef {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut map = s.serialize_map(Some(1))?;
        map.serialize_entry(&self.to_string(), EMPTY_SLICE)?;
        map.end()
    }
}
