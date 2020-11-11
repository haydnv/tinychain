use std::fmt;

use serde::ser::{Serialize, SerializeMap, Serializer};

use crate::class::{Class, Instance, NativeClass, TCResult};
use crate::error;
use crate::scalar::{
    label, Link, PathSegment, Scalar, ScalarClass, ScalarInstance, TCPathBuf, TryCastFrom,
    TryCastInto,
};

use super::{RefType, TCRef};

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub enum FlowControlType {
    If,
}

impl Class for FlowControlType {
    type Instance = FlowControl;
}

impl NativeClass for FlowControlType {
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        let suffix = Self::prefix().try_suffix(path)?;

        if suffix.len() == 1 {
            match suffix[0].as_str() {
                "if" => Ok(Self::If),
                other => Err(error::not_found(other)),
            }
        } else {
            Err(error::path_not_found(suffix))
        }
    }

    fn prefix() -> TCPathBuf {
        RefType::prefix().append(label("flow"))
    }
}

impl ScalarClass for FlowControlType {
    type Instance = FlowControl;

    fn try_cast<S: Into<Scalar>>(&self, scalar: S) -> TCResult<FlowControl> {
        FlowControl::try_cast_from(scalar.into(), |s| {
            error::bad_request("Cannot cast into FlowControl from", s)
        })
    }
}

impl From<FlowControlType> for Link {
    fn from(fct: FlowControlType) -> Link {
        use FlowControlType as FCT;
        let suffix = match fct {
            FCT::If => label("if"),
        };
        FCT::prefix().append(suffix).into()
    }
}

impl fmt::Display for FlowControlType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "type: control flow - {}",
            match self {
                Self::If => "if",
            }
        )
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum FlowControl {
    If(TCRef, Scalar, Scalar),
}

impl Instance for FlowControl {
    type Class = FlowControlType;

    fn class(&self) -> FlowControlType {
        match self {
            Self::If(_, _, _) => FlowControlType::If,
        }
    }
}

impl ScalarInstance for FlowControl {
    type Class = FlowControlType;
}

impl TryCastFrom<Scalar> for FlowControl {
    fn can_cast_from(s: &Scalar) -> bool {
        s.matches::<(TCRef, Scalar, Scalar)>()
    }

    fn opt_cast_from(s: Scalar) -> Option<FlowControl> {
        s.opt_cast_into()
            .map(|(cond, then, or_else)| FlowControl::If(cond, then, or_else))
    }
}

impl Serialize for FlowControl {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let class = Link::from(self.class()).to_string();

        match self {
            FlowControl::If(cond, then, or_else) => {
                let mut map = s.serialize_map(Some(1))?;
                map.serialize_entry(&class, &(cond, then, or_else))?;
                map.end()
            }
        }
    }
}

impl fmt::Display for FlowControl {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::If(cond, then, or_else) => {
                write!(f, "If ({}) then {} else {}", cond, then, or_else)
            }
        }
    }
}
