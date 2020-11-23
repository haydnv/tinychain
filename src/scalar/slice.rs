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
    Un,
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
                    "un" => Ok(SliceType::Bound(BoundType::Un)),
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
                    BoundType::Un if value.is_none() => Ok(Slice::Bound(Bound::Unbounded)),
                    BoundType::Un => Err(error::bad_request(
                        "Unbounded requires None as a value, not",
                        value,
                    )),
                }
            }
            Self::Range => Range::try_cast_from(scalar, |s| {
                error::bad_request("Cannot cast into Range from", s)
            })
            .map(Slice::Range),
        }
    }
}

impl From<SliceType> for Link {
    fn from(st: SliceType) -> Link {
        let prefix = SliceType::prefix();

        let path = match st {
            SliceType::Bound(bt) => {
                let prefix = prefix.append(label("bound"));
                match bt {
                    BoundType::In => prefix.append(label("in")),
                    BoundType::Ex => prefix.append(label("ex")),
                    BoundType::Un => prefix.append(label("un")),
                }
            }
            SliceType::Range => prefix.append(label("range")),
        };

        path.into()
    }
}

impl fmt::Display for SliceType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Bound(bt) => match bt {
                BoundType::In => write!(f, "type Bound (inclusive)"),
                BoundType::Ex => write!(f, "type Bound (exclusive)"),
                BoundType::Un => write!(f, "type Bound (unbounded)"),
            },
            Self::Range => write!(f, "type Range"),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum Bound {
    In(Value),
    Ex(Value),
    Unbounded,
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

impl Serialize for Bound {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let (class, value) = match self {
            Self::In(value) => (BoundType::In, value),
            Self::Ex(value) => (BoundType::Ex, value),
            Self::Unbounded => (BoundType::Un, &Value::None),
        };

        let mut map = s.serialize_map(Some(1))?;
        map.serialize_entry(&Link::from(SliceType::Bound(class)).to_string(), value)?;
        map.end()
    }
}

impl fmt::Display for Bound {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::In(value) => write!(f, "include({})", value),
            Self::Ex(value) => write!(f, "exclude({})", value),
            Self::Unbounded => write!(f, "unbounded"),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct Range(pub Bound, pub Bound);

impl Range {
    pub fn into_inner(self) -> (Bound, Bound) {
        (self.0, self.1)
    }
}

impl TryCastFrom<Scalar> for Range {
    fn can_cast_from(scalar: &Scalar) -> bool {
        if let Scalar::Slice(Slice::Range(_)) = scalar {
            true
        } else {
            scalar.matches::<(Bound, Bound)>()
                || scalar.matches::<(Value, Value)>()
                || scalar.matches::<(Value, Bound)>()
                || scalar.matches::<(Bound, Value)>()
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Range> {
        let start_bound = |val: Value| {
            if val.is_none() {
                Bound::Unbounded
            } else {
                Bound::In(val)
            }
        };

        let end_bound = |val: Value| {
            if val.is_none() {
                Bound::Unbounded
            } else {
                Bound::Ex(val)
            }
        };

        if let Scalar::Slice(Slice::Range(range)) = scalar {
            Some(range)
        } else if scalar.matches::<(Bound, Bound)>() {
            let (start, end) = scalar.opt_cast_into().unwrap();
            Some(Range(start, end))
        } else if scalar.matches::<(Value, Value)>() {
            let (start, end): (Value, Value) = scalar.opt_cast_into().unwrap();
            Some(Range(start_bound(start), end_bound(end)))
        } else if scalar.matches::<(Value, Bound)>() {
            let (start, end): (Value, Bound) = scalar.opt_cast_into().unwrap();
            Some(Range(start_bound(start), end))
        } else if scalar.matches::<(Bound, Value)>() {
            let (start, end): (Bound, Value) = scalar.opt_cast_into().unwrap();
            Some(Range(start, end_bound(end)))
        } else {
            None
        }
    }
}

impl Serialize for Range {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut map = s.serialize_map(Some(1))?;
        map.serialize_entry(
            &Link::from(SliceType::Range).to_string(),
            &[&self.0, &self.1],
        )?;
        map.end()
    }
}

impl fmt::Display for Range {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Range({}, {})", self.0, self.1)
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum Slice {
    Bound(Bound),
    Range(Range),
}

impl Instance for Slice {
    type Class = SliceType;

    fn class(&self) -> Self::Class {
        match self {
            Self::Bound(bound) => SliceType::Bound(match bound {
                Bound::In(_) => BoundType::In,
                Bound::Ex(_) => BoundType::Ex,
                Bound::Unbounded => BoundType::Un,
            }),
            Self::Range(_) => SliceType::Range,
        }
    }
}

impl ScalarInstance for Slice {
    type Class = SliceType;
}

impl Serialize for Slice {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Bound(bound) => bound.serialize(s),
            Self::Range(range) => range.serialize(s),
        }
    }
}

impl fmt::Display for Slice {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Bound(bound) => fmt::Display::fmt(bound, f),
            Self::Range(range) => fmt::Display::fmt(range, f),
        }
    }
}
