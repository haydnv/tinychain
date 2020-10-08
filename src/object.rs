use std::collections::HashMap;
use std::fmt;

use serde::de::{Deserialize, Deserializer};
use serde::ser::{Serialize, Serializer};

use crate::class::{Class, Instance, TCType};
use crate::error::{self, TCResult};
use crate::scalar::{label, Link, Scalar, TCPath, ValueId};

#[derive(Clone, Copy, Eq, PartialEq)]
struct ObjectType;

impl Class for ObjectType {
    type Instance = Object;

    fn from_path(path: &TCPath) -> TCResult<Self> {
        if path == &Self::prefix() {
            Ok(ObjectType)
        } else {
            Err(error::not_found(path))
        }
    }

    fn prefix() -> TCPath {
        TCType::prefix().join(label("object").into())
    }
}

impl From<ObjectType> for Link {
    fn from(_: ObjectType) -> Link {
        ObjectType::prefix().into()
    }
}

impl fmt::Display for ObjectType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "user-defined Class")
    }
}

struct Object {
    data: HashMap<ValueId, Scalar>,
}

impl<'de> Deserialize<'de> for Object {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Deserialize::deserialize(d).map(|data| Object { data })
    }
}

impl Instance for Object {
    type Class = ObjectType;

    fn class(&self) -> Self::Class {
        ObjectType
    }
}

impl Serialize for Object {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.data.serialize(s)
    }
}
