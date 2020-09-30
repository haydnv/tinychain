use std::fmt;

use crate::class::{Class, TCResult, TCType};
use crate::error;
use crate::scalar::{Scalar, ScalarClass, ScalarInstance, ScalarType, TryCastFrom, TryCastInto};

use super::link::{Link, TCPath};
use super::number::NumberType;
use super::string::{label, StringType};
use super::Value;

pub trait ValueInstance: ScalarInstance {
    type Class: ValueClass;

    fn get(&self, _path: TCPath, _key: Value) -> TCResult<Self> {
        Err(error::method_not_allowed(format!("GET {}", self.class())))
    }
}

pub trait ValueClass: ScalarClass {
    type Instance: ValueInstance;
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
    Bound,
    Bytes,
    Class,
    None,
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

    fn from_path(path: &TCPath) -> TCResult<Self> {
        let suffix = path.from_path(&Self::prefix())?;

        if suffix.is_empty() {
            Ok(ValueType::Value)
        } else if suffix.len() == 1 {
            match suffix[0].as_str() {
                "none" => Ok(ValueType::None),
                "bytes" => Ok(ValueType::Bytes),
                "tuple" => Ok(ValueType::Tuple),
                other => Err(error::not_found(other)),
            }
        } else {
            match suffix[0].as_str() {
                "number" => NumberType::from_path(path).map(ValueType::Number),
                "string" => StringType::from_path(path).map(ValueType::TCString),
                other => Err(error::not_found(other)),
            }
        }
    }

    fn prefix() -> TCPath {
        TCType::prefix().join(label("value").into())
    }
}

impl ScalarClass for ValueType {
    type Instance = Value;

    fn size(self) -> Option<usize> {
        use ValueType::*;
        match self {
            None => Some(1),
            Number(nt) => ScalarClass::size(nt),
            _ => Option::None,
        }
    }

    fn try_cast<S: Into<Scalar>>(&self, scalar: S) -> TCResult<Value> {
        match self {
            Self::None => Ok(Value::None),
            Self::Number(nt) => nt.try_cast(scalar).map(Value::Number),
            Self::TCString(st) => st.try_cast(scalar).map(Value::TCString),
            other => {
                let scalar: Scalar = scalar.into();
                scalar.try_cast_into(|v| {
                    error::not_implemented(format!("Cast into {} from {}", other, v))
                })
            }
        }
    }
}

impl ValueClass for ValueType {
    type Instance = Value;
}

impl From<ValueType> for Link {
    fn from(vt: ValueType) -> Link {
        let prefix = ValueType::prefix();

        use ValueType::*;
        match vt {
            None => prefix.join(label("none").into()).into(),
            Bound => prefix.join(label("bound").into()).into(),
            Bytes => prefix.join(label("bytes").into()).into(),
            Class => prefix.join(label("class").into()).into(),
            Number(nt) => nt.into(),
            TCString(st) => st.into(),
            Tuple => prefix.join(label("tuple").into()).into(),
            Value => prefix.join(label("value").into()).into(),
        }
    }
}

impl TryCastFrom<Link> for ValueType {
    fn can_cast_from(link: &Link) -> bool {
        match ValueType::from_path(link.path()) {
            Ok(_) => true,
            _ => false,
        }
    }

    fn opt_cast_from(link: Link) -> Option<ValueType> {
        ValueType::from_path(link.path()).ok()
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

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use ValueType::*;
        match self {
            None => write!(f, "type None"),
            Bound => write!(f, "type Bound"),
            Bytes => write!(f, "type Bytes"),
            Class => write!(f, "type Class"),
            Number(nt) => write!(f, "type Number: {}", nt),
            TCString(st) => write!(f, "type String: {}", st),
            Tuple => write!(f, "type Tuple"),
            Value => write!(f, "Value"),
        }
    }
}
