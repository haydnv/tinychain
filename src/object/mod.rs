use std::fmt;

use log::debug;

use crate::class::{Class, Instance, NativeClass, State, TCType};
use crate::error::{self, TCResult};
use crate::handler::*;
use crate::scalar::{self, label, Link, MethodType, PathSegment, TCPath, TCPathBuf};

mod class;
mod instance;

pub use class::{InstanceClass, InstanceClassType};
pub use instance::InstanceExt;

#[derive(Clone, Eq, PartialEq)]
pub enum ObjectType {
    Class(InstanceClassType),
    Instance(InstanceClass),
}

impl ObjectType {
    pub fn post(path: &[PathSegment], data: scalar::Map) -> TCResult<Object> {
        debug!("ObjectType::post {} <- {}", TCPath::from(path), data);

        if path.starts_with(&InstanceClassType::prefix()) {
            InstanceClassType::post(path, data).map(Object::Class)
        } else {
            Err(error::path_not_found(path))
        }
    }
}

impl Class for ObjectType {
    type Instance = Object;
}

impl NativeClass for ObjectType {
    fn from_path(path: &[PathSegment]) -> TCResult<ObjectType> {
        let suffix = Self::prefix().try_suffix(path)?;

        if suffix.is_empty() {
            Ok(ObjectType::Instance(InstanceClass::default()))
        } else if suffix.len() == 1 && &suffix[0] == "class" {
            Ok(ObjectType::Class(InstanceClassType))
        } else {
            Err(error::path_not_found(suffix))
        }
    }

    fn prefix() -> TCPathBuf {
        TCType::prefix().append(label("object"))
    }
}

impl From<ObjectType> for Link {
    fn from(ot: ObjectType) -> Link {
        match ot {
            ObjectType::Class(ict) => ict.into(),
            ObjectType::Instance(ic) => ic.into(),
        }
    }
}

impl fmt::Display for ObjectType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Class(ict) => write!(f, "{}", ict),
            Self::Instance(ic) => write!(f, "{}", ic),
        }
    }
}

#[derive(Clone)]
pub enum Object {
    Class(InstanceClass),
    Instance(InstanceExt<State>),
}

impl Route for Object {
    fn route(&'_ self, method: MethodType, path: &[PathSegment]) -> Option<Box<dyn Handler + '_>> {
        match self {
            Self::Class(ic) => ic.route(method, path),
            Self::Instance(instance) => instance.route(method, path),
        }
    }
}

impl Instance for Object {
    type Class = ObjectType;

    fn class(&self) -> ObjectType {
        match self {
            Self::Class(ic) => ObjectType::Class(ic.class()),
            Self::Instance(i) => ObjectType::Instance(i.class()),
        }
    }
}

impl From<InstanceClass> for Object {
    fn from(ic: InstanceClass) -> Object {
        Object::Class(ic)
    }
}

impl From<InstanceExt<State>> for Object {
    fn from(instance: InstanceExt<State>) -> Object {
        Object::Instance(instance)
    }
}

impl fmt::Display for Object {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Class(ict) => write!(f, "{}", ict),
            Self::Instance(ic) => write!(f, "{}", ic),
        }
    }
}
