use std::fmt;

use serde::ser::{Serialize, SerializeMap, Serializer};

use crate::class::{Class, Instance, NativeClass, TCType};
use crate::error::{self, TCResult};

use super::{
    label, Link, PathSegment, Scalar, ScalarClass, ScalarInstance, TCPath, TCPathBuf, TryCastFrom,
    TryCastInto, Value,
};

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub enum BoundType {
    In,
    Ex,
}

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub enum SliceType {
    Bound(BoundType),
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
                "bound" => Err(error::method_not_allowed(TCPath::from(path))),
                "range" => Ok(Self::Range),
                other => Err(error::not_found(other)),
            }
        } else if suffix.len() == 2 {
            match suffix[0].as_str() {
                "range" => Err(error::method_not_allowed(TCPath::from(path))),
                "bound" => match suffix[1].as_str() {
                    "in" => Ok(SliceType::Bound(BoundType::In)),
                    "ex" => Ok(SliceType::Bound(BoundType::Ex)),
                    other => Err(error::not_found(other)),
                },
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

    fn try_cast<S: Into<Scalar>>(&self, scalar: S) -> TCResult<Slice> {
        let scalar: Scalar = scalar.into();

        match self {
            Self::Bound(bt) => {
                let value: Value =
                    scalar.try_cast_into(|s| error::bad_request("Invalid Value for Slice", s))?;

                match bt {
                    BoundType::In => Ok(Slice::Bound(Bound::In(value))),
                    BoundType::Ex => Ok(Slice::Bound(Bound::Ex(value))),
                }
            }
            Self::Range => {
                let (start, end): (Bound, Bound) =
                    scalar.try_cast_into(|v| error::bad_request("Invalid Range", v))?;
                Ok(Slice::Range(start, end))
            }
        }
    }
}

impl From<SliceType> for Link {
    fn from(st: SliceType) -> Link {
        let prefix = SliceType::prefix();

        match st {
            SliceType::Bound(bt) => {
                let prefix = prefix.append(label("bound"));
                match bt {
                    BoundType::In => prefix.append(label("in")).into(),
                    BoundType::Ex => prefix.append(label("ex")).into(),
                }
            }
            SliceType::Range => prefix.append(label("range")).into(),
        }
    }
}

impl fmt::Display for SliceType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Bound(bt) => match bt {
                BoundType::In => write!(f, "type Bound (inclusive)"),
                BoundType::Ex => write!(f, "type Bound (exclusive)"),
            },
            Self::Range => write!(f, "type Range"),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum Bound {
    In(Value),
    Ex(Value),
}

impl fmt::Display for Bound {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::In(value) => write!(f, "include({})", value),
            Self::Ex(value) => write!(f, "exclude({})", value),
        }
    }
}

impl Bound {
    pub fn value(&'_ self) -> &'_ Value {
        match self {
            Self::In(value) => value,
            Self::Ex(value) => value,
        }
    }
}

impl TryCastFrom<Scalar> for Bound {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Slice(Slice::Bound(_)) => true,
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Bound> {
        match scalar {
            Scalar::Slice(Slice::Bound(bound)) => Some(bound),
            _ => None,
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum Slice {
    Bound(Bound),
    Range(Bound, Bound),
}

impl Instance for Slice {
    type Class = SliceType;

    fn class(&self) -> Self::Class {
        match self {
            Self::Bound(bound) => SliceType::Bound(match bound {
                Bound::In(_) => BoundType::In,
                Bound::Ex(_) => BoundType::Ex,
            }),
            Self::Range(_, _) => SliceType::Range,
        }
    }
}

impl ScalarInstance for Slice {
    type Class = SliceType;
}

impl Serialize for Slice {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let class = Link::from(self.class()).to_string();

        let mut map = s.serialize_map(Some(1))?;
        match self {
            Self::Bound(bound) => {
                map.serialize_entry(
                    &class,
                    match bound {
                        Bound::In(value) => value,
                        Bound::Ex(value) => value,
                    },
                )?;
            }
            Self::Range(start, end) => {
                map.serialize_entry(&class, &[start.value(), end.value()])?
            }
        };
        map.end()
    }
}

impl fmt::Display for Slice {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Bound(bound) => fmt::Display::fmt(bound, f),
            Self::Range(start, end) => write!(f, "Range({}, {})", start, end),
        }
    }
}
