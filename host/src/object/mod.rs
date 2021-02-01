use std::fmt;

use generic::{PathSegment, TCPathBuf};

use crate::state::State;

mod class;
mod instance;

pub use class::*;
pub use instance::*;

const PREFIX: generic::PathLabel = generic::path_label(&["state", "object"]);

#[derive(Clone, Eq, PartialEq)]
pub enum ObjectType {
    Class(InstanceClassType),
    Instance(InstanceClass),
}

impl generic::Class for ObjectType {
    type Instance = Object;
}

impl generic::NativeClass for ObjectType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if path.len() == 3 && &path[..2] == &PREFIX[..] {
            match path[2].as_str() {
                "class" => Some(Self::Class(InstanceClassType)),
                "instance" => Some(Self::Instance(InstanceClass::default())),
                _ => None,
            }
        } else {
            None
        }
    }

    fn path(&self) -> TCPathBuf {
        let suffix = match self {
            Self::Class(_) => "class",
            Self::Instance(_) => "instance",
        };

        TCPathBuf::from(PREFIX).append(generic::label(suffix))
    }
}

impl fmt::Display for ObjectType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Class(ict) => fmt::Display::fmt(ict, f),
            Self::Instance(ic) => fmt::Display::fmt(ic, f),
        }
    }
}

#[derive(Clone)]
pub enum Object {
    Class(InstanceClass),
    Instance(InstanceExt<State>),
}

impl generic::Instance for Object {
    type Class = ObjectType;

    fn class(&self) -> ObjectType {
        match self {
            Self::Class(ic) => ObjectType::Class(ic.class()),
            Self::Instance(i) => ObjectType::Instance(i.class()),
        }
    }
}

impl fmt::Display for Object {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Class(ict) => fmt::Display::fmt(ict, f),
            Self::Instance(ic) => fmt::Display::fmt(ic, f),
        }
    }
}
