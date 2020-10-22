use std::collections::HashMap;
use std::fmt;

use crate::class::{Class, Instance, NativeClass, TCType};
use crate::scalar::{self, label, Link, Scalar, TCPath, ValueId};

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct ObjectClassType;

impl Class for ObjectClassType {
    type Instance = ObjectType;
}

impl From<ObjectClassType> for Link {
    fn from(_: ObjectClassType) -> Link {
        TCType::prefix().join(label("class").into()).into()
    }
}

impl fmt::Display for ObjectClassType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "user-defined Class")
    }
}

#[derive(Clone, Default, Eq, PartialEq)]
pub struct ObjectType {
    extends: Option<Link>,
    proto: HashMap<ValueId, Scalar>,
}

impl ObjectType {
    pub fn prefix() -> TCPath {
        TCType::prefix().join(label("object").into())
    }
}

impl Class for ObjectType {
    type Instance = Object;
}

impl Instance for ObjectType {
    type Class = ObjectClassType;

    fn class(&self) -> ObjectClassType {
        ObjectClassType
    }
}

impl From<ObjectType> for Link {
    fn from(ot: ObjectType) -> Link {
        if let Some(link) = ot.extends {
            link
        } else {
            ObjectType::prefix().into()
        }
    }
}

impl fmt::Display for ObjectType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(link) = &self.extends {
            write!(f, "class {}", link)
        } else {
            write!(f, "user-defined Class")
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct Object {
    class: ObjectType,
    data: scalar::object::Object,
}

impl Instance for Object {
    type Class = ObjectType;

    fn class(&self) -> Self::Class {
        self.class.clone()
    }
}

impl From<scalar::object::Object> for Object {
    fn from(generic: scalar::object::Object) -> Object {
        Object {
            class: ObjectType::default(),
            data: generic,
        }
    }
}

impl fmt::Display for Object {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Object of type {}", self.class())
    }
}
