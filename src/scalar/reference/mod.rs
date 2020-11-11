use std::fmt;

use crate::class::{Class, Instance, NativeClass, TCResult, TCType};
use crate::error;
use crate::scalar::{
    label, Link, PathSegment, Scalar, ScalarClass, ScalarInstance, ScalarType, TCPath, TCPathBuf,
    TryCastInto,
};

pub mod id;

pub use id::*;

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum RefType {
    Id,
}

impl Class for RefType {
    type Instance = Scalar;
}

impl NativeClass for RefType {
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        let suffix = Self::prefix().try_suffix(path)?;

        if suffix.is_empty() {
            Err(error::method_not_allowed(TCPath::from(path)))
        } else if suffix.len() == 1 {
            match suffix[0].as_str() {
                "id" if suffix.len() == 1 => Ok(RefType::Id),
                other => Err(error::not_found(other)),
            }
        } else {
            Err(error::path_not_found(suffix))
        }
    }

    fn prefix() -> TCPathBuf {
        ScalarType::prefix().append(label("ref"))
    }
}

impl ScalarClass for RefType {
    type Instance = TCRef;

    fn try_cast<S: Into<Scalar>>(&self, scalar: S) -> TCResult<TCRef> {
        let scalar: Scalar = scalar.into();

        match self {
            Self::Id => scalar
                .try_cast_into(|v| error::bad_request("Cannot cast into Ref from", v))
                .map(TCRef::Id),
        }
    }
}

impl From<RefType> for Link {
    fn from(rt: RefType) -> Link {
        match rt {
            RefType::Id => RefType::prefix().append(label("id")).into(),
        }
    }
}

impl From<RefType> for TCType {
    fn from(rt: RefType) -> TCType {
        TCType::Scalar(ScalarType::Ref(rt))
    }
}

impl fmt::Display for RefType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Id => write!(f, "type Ref"),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum TCRef {
    Id(IdRef),
}

impl Instance for TCRef {
    type Class = RefType;

    fn class(&self) -> RefType {
        match self {
            TCRef::Id(_) => RefType::Id,
        }
    }
}

impl ScalarInstance for TCRef {
    type Class = RefType;
}

impl From<IdRef> for TCRef {
    fn from(id_ref: IdRef) -> TCRef {
        TCRef::Id(id_ref)
    }
}

impl fmt::Display for TCRef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Id(id_ref) => write!(f, "{}", id_ref),
        }
    }
}
