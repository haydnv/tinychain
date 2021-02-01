use std::fmt;

use generic::{path_label, Map, PathLabel, PathSegment, TCPathBuf};

use crate::scalar::*;
use crate::state::State;

use super::InstanceExt;

const PATH: PathLabel = path_label(&["state", "class"]);

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct InstanceClassType;

impl generic::Class for InstanceClassType {
    type Instance = InstanceClass;
}

impl generic::NativeClass for InstanceClassType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if path == &PATH[..] {
            Some(Self)
        } else {
            None
        }
    }

    fn path(&self) -> TCPathBuf {
        PATH.into()
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
    proto: Map<Scalar>,
}

impl InstanceClass {
    pub fn extends(&self) -> Link {
        if let Some(link) = &self.extends {
            link.clone()
        } else {
            TCPathBuf::from(PATH).into()
        }
    }

    pub fn proto(&'_ self) -> &'_ Map<Scalar> {
        &self.proto
    }
}

impl generic::Class for InstanceClass {
    type Instance = InstanceExt<State>;
}

impl generic::Instance for InstanceClass {
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
            TCPathBuf::from(PATH).into()
        }
    }
}

impl fmt::Display for InstanceClass {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(link) = &self.extends {
            write!(f, "class {}", link)
        } else {
            f.write_str("generic Object type")
        }
    }
}
