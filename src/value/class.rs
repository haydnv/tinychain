use std::convert::TryInto;
use std::fmt;

use serde::Serialize;

use crate::class::{Class, Instance, TCResult, TCType};
use crate::error;

use super::link::TCPath;
use super::{label, Link, Value};

pub type NumberType = super::number::class::NumberType;
pub type StringType = super::string::StringType;

pub trait ValueInstance: Instance + Serialize + Sized {
    type Class: ValueClass;
}

pub trait ValueClass: Class {
    type Instance: ValueInstance;

    fn get(
        path: &TCPath,
        value: <Self as ValueClass>::Instance,
    ) -> TCResult<<Self as ValueClass>::Instance>;

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
    Bound,
    Bytes,
    Class,
    None,
    Number(NumberType),
    TCString(StringType),
    Op,
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
                "op" => Ok(ValueType::Op),
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

impl ValueClass for ValueType {
    type Instance = Value;

    fn get(path: &TCPath, value: Value) -> TCResult<Value> {
        let suffix = path.from_path(&Self::prefix())?;

        if suffix.is_empty() {
            return Ok(value);
        }

        match suffix[0].as_str() {
            "none" if suffix.len() == 1 => Ok(Value::None),
            "bytes" if suffix.len() == 1 => Err(error::not_implemented("/sbin/value/bytes")),
            "number" => NumberType::get(path, value.try_into()?).map(Value::Number),
            "string" => StringType::get(path, value.try_into()?).map(Value::TCString),
            "op" => Err(error::not_implemented("/sbin/value/op")),
            "tuple" => Err(error::not_implemented("/sbin/value/tuple")),
            other => Err(error::not_found(other)),
        }
    }

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
        let prefix = ValueType::prefix();

        use ValueType::*;
        match vt {
            None => prefix.join(label("none").into()).into(),
            Bound => prefix.join(label("bound").into()).into(),
            Bytes => prefix.join(label("bytes").into()).into(),
            Class => prefix.join(label("class").into()).into(),
            Number(n) => n.into(),
            TCString(s) => s.into(),
            Op => prefix.join(label("op").into()).into(),
            Tuple => prefix.join(label("tuple").into()).into(),
            Value => prefix.join(label("value").into()).into(),
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
            Number(n) => write!(f, "type Number: {}", n),
            TCString(s) => write!(f, "type String: {}", s),
            Op => write!(f, "type Op"),
            Tuple => write!(f, "type Tuple"),
            Value => write!(f, "Value"),
        }
    }
}
