use std::fmt;
use std::ops::{Add, Mul};

use serde::{Deserialize, Serialize};

use crate::error;

use super::number::{Complex, Float, Int, Number, UInt};
use super::string::TCString;
use super::{TCResult, Value};

pub trait Class: Clone + Eq + fmt::Display {
    type Impl: Impl;
}

pub trait ValueClass: Class {
    type Impl: ValueImpl;

    fn size(self) -> Option<usize>;
}

pub trait NumberClass: Class + Into<NumberType> + Send + Sync {
    type Impl: NumberImpl + Into<Number>;

    fn size(self) -> usize;

    fn one(&self) -> <Self as NumberClass>::Impl {
        true.into()
    }

    fn zero(&self) -> <Self as NumberClass>::Impl {
        false.into()
    }
}

impl<T: NumberClass> ValueClass for T {
    type Impl = <Self as NumberClass>::Impl;

    fn size(self) -> Option<usize> {
        Some(NumberClass::size(self))
    }
}

pub trait Impl {
    type Class: Class;

    fn class(&self) -> Self::Class;

    fn is_a(&self, dtype: Self::Class) -> bool {
        self.class() == dtype
    }

    fn expect<M: fmt::Display>(&self, dtype: Self::Class, context_msg: M) -> TCResult<()> {
        if self.is_a(dtype.clone()) {
            Ok(())
        } else {
            Err(error::TCError::of(
                error::Code::BadRequest,
                format!(
                    "Expected {} but found {} {}",
                    self.class(),
                    dtype,
                    context_msg
                ),
            ))
        }
    }
}

pub trait ValueImpl: Impl + Serialize {
    type Class: ValueClass;
}

pub trait NumberImpl:
    ValueImpl + Add + Mul + Sized + PartialOrd + From<bool> + Into<Number> + CastInto<bool>
{
    type Abs: NumberImpl;
    type Class: NumberClass;

    fn abs(&self) -> Self::Abs;

    fn into_type(self, dtype: NumberType) -> Number {
        match dtype {
            NumberType::Bool => Number::Bool(self.cast_into()),
            _ => unimplemented!(),
        }
    }
}

pub trait CastFrom<T> {
    fn cast_from(value: T) -> Self;
}

pub trait CastInto<T> {
    fn cast_into(self) -> T;
}

impl<T, F: CastFrom<T>> CastInto<F> for T {
    fn cast_into(self) -> F {
        F::cast_from(self)
    }
}

#[derive(Clone, Copy, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub enum ComplexType {
    C32,
    C64,
}

impl Class for ComplexType {
    type Impl = Complex;
}

impl NumberClass for ComplexType {
    type Impl = Complex;

    fn size(self) -> usize {
        match self {
            Self::C32 => 8,
            Self::C64 => 16,
        }
    }
}

impl From<ComplexType> for NumberType {
    fn from(ct: ComplexType) -> NumberType {
        Self::Complex(ct)
    }
}

impl fmt::Display for ComplexType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::C32 => write!(f, "C32"),
            Self::C64 => write!(f, "C64"),
        }
    }
}

#[derive(Clone, Copy, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub enum FloatType {
    F32,
    F64,
}

impl Class for FloatType {
    type Impl = Float;
}

impl NumberClass for FloatType {
    type Impl = Float;

    fn size(self) -> usize {
        match self {
            FloatType::F32 => 8,
            FloatType::F64 => 16,
        }
    }
}

impl From<FloatType> for NumberType {
    fn from(ft: FloatType) -> NumberType {
        NumberType::Float(ft)
    }
}

impl fmt::Display for FloatType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use FloatType::*;
        match self {
            F32 => write!(f, "F32"),
            F64 => write!(f, "F64"),
        }
    }
}

#[derive(Clone, Copy, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub enum IntType {
    I16,
    I32,
    I64,
}

impl Class for IntType {
    type Impl = Int;
}

impl NumberClass for IntType {
    type Impl = Int;

    fn size(self) -> usize {
        match self {
            IntType::I16 => 2,
            IntType::I32 => 4,
            IntType::I64 => 8,
        }
    }
}

impl From<IntType> for NumberType {
    fn from(it: IntType) -> NumberType {
        NumberType::Int(it)
    }
}

impl fmt::Display for IntType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use IntType::*;
        match self {
            I16 => write!(f, "I16"),
            I32 => write!(f, "I32"),
            I64 => write!(f, "I64"),
        }
    }
}

#[derive(Clone, Copy, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub enum UIntType {
    U8,
    U16,
    U32,
    U64,
}

impl Class for UIntType {
    type Impl = UInt;
}

impl NumberClass for UIntType {
    type Impl = UInt;

    fn size(self) -> usize {
        match self {
            UIntType::U8 => 1,
            UIntType::U16 => 2,
            UIntType::U32 => 4,
            UIntType::U64 => 8,
        }
    }
}

impl From<UIntType> for NumberType {
    fn from(ut: UIntType) -> NumberType {
        NumberType::UInt(ut)
    }
}

impl fmt::Display for UIntType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use UIntType::*;
        match self {
            U8 => write!(f, "U8"),
            U16 => write!(f, "U16"),
            U32 => write!(f, "U32"),
            U64 => write!(f, "U64"),
        }
    }
}

#[derive(Clone, Copy, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub enum NumberType {
    Bool,
    Complex(ComplexType),
    Float(FloatType),
    Int(IntType),
    UInt(UIntType),
}

impl Class for NumberType {
    type Impl = Number;
}

impl NumberClass for NumberType {
    type Impl = Number;

    fn size(self) -> usize {
        use NumberType::*;
        match self {
            Bool => 1,
            Complex(ct) => NumberClass::size(ct),
            Float(ft) => NumberClass::size(ft),
            Int(it) => NumberClass::size(it),
            UInt(ut) => NumberClass::size(ut),
        }
    }
}

impl From<NumberType> for ValueType {
    fn from(nt: NumberType) -> ValueType {
        ValueType::Number(nt)
    }
}

impl fmt::Display for NumberType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use NumberType::*;
        match self {
            Bool => write!(f, "Bool"),
            Complex(c) => write!(f, "Complex: {}", c),
            Float(ft) => write!(f, "Float: {}", ft),
            Int(i) => write!(f, "Int: {}", i),
            UInt(u) => write!(f, "UInt: {}", u),
        }
    }
}

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub enum StringType {
    Id,
    Link,
    Ref,
    r#String,
}

impl Class for StringType {
    type Impl = TCString;
}

impl From<StringType> for ValueType {
    fn from(st: StringType) -> ValueType {
        ValueType::TCString(st)
    }
}

impl fmt::Display for StringType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use StringType::*;
        match self {
            Id => write!(f, "type Id"),
            Link => write!(f, "type Link"),
            Ref => write!(f, "type Ref"),
            r#String => write!(f, "type String"),
        }
    }
}

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub enum ValueType {
    Bytes,
    None,
    Number(NumberType),
    TCString(StringType),
    Op,
    Vector,
}

impl Class for ValueType {
    type Impl = Value;
}

impl ValueClass for ValueType {
    type Impl = Value;

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
            Vector => write!(f, "type Vector"),
        }
    }
}
