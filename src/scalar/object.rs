use std::collections::HashMap;
use std::fmt;

use crate::class::{Class, Instance, TCType};
use crate::error::{self, TCResult};
use crate::scalar::{label, Link, Scalar, TCPath, ValueId};

use super::{ScalarClass, ScalarInstance};

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct ObjectType;

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

impl ScalarClass for ObjectType {
    type Instance = Object;

    fn try_cast<S: Into<Scalar>>(&self, scalar: S) -> TCResult<Object> {
        let scalar: Scalar = scalar.into();

        match scalar {
            Scalar::Map(data) => Ok(Object { data }),
            other => Err(error::bad_request("Cannot cast into Object from", other))
        }
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

#[derive(Clone, Eq, PartialEq)]
pub struct Object {
    data: HashMap<ValueId, Scalar>,
}

impl Object {
    pub fn data(&'_ self) -> &'_ HashMap<ValueId, Scalar> {
        &self.data
    }
}

impl Instance for Object {
    type Class = ObjectType;

    fn class(&self) -> Self::Class {
        ObjectType
    }
}

impl ScalarInstance for Object {
    type Class = ObjectType;
}

impl fmt::Display for Object {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Object of type {}", self.class())
    }
}
