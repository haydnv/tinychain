use std::fmt;

use serde::Serialize;

use crate::class::{Class, Instance};

use super::Value;

pub type NumberType = super::number::class::NumberType;
pub type StringType = super::string::StringType;

pub trait ValueInstance: Instance + Serialize {
    type Class: ValueClass;
}

pub trait ValueClass: Class {
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
    Bytes,
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

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use ValueType::*;
        match self {
            None => write!(f, "type None"),
            Bytes => write!(f, "type Bytes"),
            Number(n) => write!(f, "type Number: {}", n),
            TCString(s) => write!(f, "type String: {}", s),
            Op => write!(f, "type Op"),
            Tuple => write!(f, "type Tuple"),
            Value => write!(f, "Value"),
        }
    }
}
