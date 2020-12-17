use std::fmt;

use crate::class::{Class, NativeClass, TCType};
use crate::error;
use crate::general::TCResult;
use crate::scalar::{Scalar, ScalarClass, ScalarInstance, ScalarType, TryCastFrom, TryCastInto};

use super::link::{Link, PathSegment, TCPathBuf};
use super::number::NumberType;
use super::string::{label, StringType};
use super::Value;

pub trait ValueInstance: ScalarInstance {
    type Class: ValueClass;
}

pub trait ValueClass: ScalarClass {
    type Instance: ValueInstance;

    fn size(self) -> Option<usize>;
}

impl From<NumberType> for ValueType {
    fn from(nt: NumberType) -> ValueType {
        ValueType::Number(nt)
    }
}

impl From<StringType> for ValueType {
    fn from(st: StringType) -> ValueType {
        ValueType::TCString(st)
    }
}

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub enum ValueType {
    None,
    Bytes,
    Class,
    Number(NumberType),
    TCString(StringType),
    Tuple,
    Value, // self
}

impl ValueType {
    pub fn uint64() -> Self {
        ValueType::Number(NumberType::uint64())
    }
}

impl Class for ValueType {
    type Instance = Value;
}

impl NativeClass for ValueType {
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        let suffix = Self::prefix().try_suffix(path)?;

        if suffix.is_empty() {
            return Ok(ValueType::Value);
        }

        match suffix[0].as_str() {
            "none" if suffix.len() == 1 => Ok(ValueType::None),
            "bytes" if suffix.len() == 1 => Ok(ValueType::Bytes),
            "number" => NumberType::from_path(path).map(ValueType::Number),
            "string" => StringType::from_path(path).map(ValueType::TCString),
            "tuple" if suffix.len() == 1 => Ok(ValueType::Tuple),
            other => Err(error::not_found(other)),
        }
    }

    fn prefix() -> TCPathBuf {
        TCType::prefix().append(label("value"))
    }
}

impl ScalarClass for ValueType {
    type Instance = Value;

    fn try_cast<S>(&self, scalar: S) -> TCResult<Value>
    where
        Scalar: From<S>,
    {
        match self {
            Self::None => Ok(Value::None),
            Self::Number(nt) => nt.try_cast(scalar).map(Value::Number),
            Self::TCString(st) => st.try_cast(scalar).map(Value::TCString),
            Self::Tuple => Scalar::from(scalar)
                .try_cast_into(|v| error::bad_request("Not a Value Tuple", v))
                .map(Value::Tuple),
            other => Scalar::from(scalar).try_cast_into(|v| {
                error::not_implemented(format!("Cast into {} from {}", other, v))
            }),
        }
    }
}

impl ValueClass for ValueType {
    type Instance = Value;

    fn size(self) -> Option<usize> {
        use ValueType::*;
        match self {
            None => Some(1),
            Number(nt) => ValueClass::size(nt),
            _ => Option::None,
        }
    }
}

impl From<ValueType> for Link {
    fn from(vt: ValueType) -> Link {
        use ValueType::*;
        let suffix = match vt {
            None => label("none"),
            Bytes => label("bytes"),
            Class => label("class"),
            Tuple => label("tuple"),
            Value => label("value"),
            Number(nt) => {
                return nt.into();
            }
            TCString(st) => {
                return st.into();
            }
        };

        ValueType::prefix().append(suffix).into()
    }
}

impl TryCastFrom<Link> for ValueType {
    fn can_cast_from(link: &Link) -> bool {
        match ValueType::from_path(&link.path()[..]) {
            Ok(_) => true,
            _ => false,
        }
    }

    fn opt_cast_from(link: Link) -> Option<ValueType> {
        ValueType::from_path(&link.path()[..]).ok()
    }
}

impl TryCastFrom<TCType> for ValueType {
    fn can_cast_from(class: &TCType) -> bool {
        if let TCType::Scalar(ScalarType::Value(_)) = class {
            true
        } else {
            false
        }
    }

    fn opt_cast_from(class: TCType) -> Option<ValueType> {
        if let TCType::Scalar(ScalarType::Value(vt)) = class {
            Some(vt)
        } else {
            None
        }
    }
}

impl fmt::Debug for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use ValueType::*;
        match self {
            None => write!(f, "type None"),
            Bytes => write!(f, "type Bytes"),
            Class => write!(f, "type Class"),
            Number(nt) => write!(f, "type Number: {}", nt),
            TCString(st) => write!(f, "type String: {}", st),
            Tuple => write!(f, "type Tuple"),
            Value => write!(f, "Value"),
        }
    }
}
