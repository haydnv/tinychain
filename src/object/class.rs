use std::collections::HashMap;
use std::fmt;

use crate::class::{Class, Instance, NativeClass, TCType};
use crate::scalar::{label, Link, Scalar, TCPath, ValueId};

use super::ObjectInstance;

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct InstanceClassType;

impl Class for InstanceClassType {
    type Instance = InstanceClass;
}

impl From<InstanceClassType> for Link {
    fn from(_: InstanceClassType) -> Link {
        TCType::prefix().join(label("class").into()).into()
    }
}

impl fmt::Display for InstanceClassType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "user-defined Class")
    }
}

#[derive(Clone, Default, Eq, PartialEq)]
pub struct InstanceClass {
    extends: Option<Link>,
    proto: HashMap<ValueId, Scalar>,
}

impl InstanceClass {
    pub fn prefix() -> TCPath {
        TCType::prefix().join(label("object").into())
    }
}

impl Class for InstanceClass {
    type Instance = ObjectInstance;
}

impl Instance for InstanceClass {
    type Class = InstanceClassType;

    fn class(&self) -> InstanceClassType {
        InstanceClassType
    }
}

impl From<InstanceClass> for Link {
    fn from(ic: InstanceClass) -> Link {
        if let Some(link) = ic.extends {
            link
        } else {
            InstanceClass::prefix().into()
        }
    }
}

impl fmt::Display for InstanceClass {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(link) = &self.extends {
            write!(f, "class {}", link)
        } else {
            write!(f, "generic Object type")
        }
    }
}
