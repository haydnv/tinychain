use std::fmt;
use std::ops::{Bound, Range};

use crate::class::{Class, Instance, NativeClass, TCType};
use crate::error::{self, TCResult};

use super::{
    label, Link, PathSegment, Scalar, ScalarClass, ScalarInstance, TCPath, TCPathBuf, Value,
};

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub enum SliceType {
    Bound,
    Range,
}

impl Class for SliceType {
    type Instance = Slice;
}

impl NativeClass for SliceType {
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        let suffix = Self::prefix().try_suffix(path)?;

        if suffix.is_empty() {
            Err(error::method_not_allowed(TCPath::from(path)))
        } else if suffix.len() == 1 {
            match suffix[0].as_str() {
                "bound" => Ok(Self::Bound),
                "range" => Ok(Self::Range),
                other => Err(error::not_found(other)),
            }
        } else {
            Err(error::path_not_found(suffix))
        }
    }

    fn prefix() -> TCPathBuf {
        TCType::prefix().append(label("slice"))
    }
}

impl ScalarClass for SliceType {
    type Instance = Slice;

    fn try_cast<S: Into<Scalar>>(&self, _scalar: S) -> TCResult<Slice> {
        Err(error::not_implemented("SliceType::try_cast"))
    }
}

impl From<SliceType> for Link {
    fn from(st: SliceType) -> Link {
        let suffix = match st {
            SliceType::Bound => label("bound"),
            SliceType::Range => label("range"),
        };

        SliceType::prefix().append(suffix).into()
    }
}

impl fmt::Display for SliceType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Bound => write!(f, "type Bound"),
            Self::Range => write!(f, "type Range"),
        }
    }
}

pub enum Slice {
    Bound(Bound<Value>),
    Range(Range<Value>),
}

impl Instance for Slice {
    type Class = SliceType;

    fn class(&self) -> Self::Class {
        match self {
            Self::Bound(_) => SliceType::Bound,
            Self::Range(_) => SliceType::Range,
        }
    }
}

impl ScalarInstance for Slice {
    type Class = SliceType;
}
