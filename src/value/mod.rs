use std::fmt;
use std::ops::Mul;
use std::pin::Pin;

use futures::stream::Stream;

use crate::error;

pub mod link;
pub mod op;
mod reference;

#[allow(clippy::module_inception)]
mod value;
mod version;

pub type Complex = value::Complex;
pub type Float = value::Float;
pub type Int = value::Int;
pub type UInt = value::UInt;
pub type Number = value::Number;
pub type TCStream<T> = Pin<Box<dyn Stream<Item = T> + Send + Sync + Unpin>>;
pub type TCRef = reference::TCRef;
pub type TCResult<T> = Result<T, error::TCError>;
pub type Value = value::Value;
pub type ValueId = value::ValueId;

pub trait DataType: Eq {
    type Impl: TypeImpl;
}

pub trait NumberDataType: Eq + DataType + Into<NumberType> {
    type Impl: NumberTypeImpl + Mul + PartialOrd + From<bool>;

    fn size(&self) -> usize;

    fn one(&self) -> <Self as NumberDataType>::Impl {
        true.into()
    }

    fn zero(&self) -> <Self as NumberDataType>::Impl {
        false.into()
    }
}

pub trait TypeImpl {
    type DType: DataType;

    fn dtype(&self) -> Self::DType;

    fn is_a(&self, dtype: Self::DType) -> bool {
        self.dtype() == dtype
    }
}

pub trait NumberTypeImpl: Mul + Sized + PartialOrd {
    type DType: NumberDataType;
}

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub enum ComplexType {
    C32,
    C64,
}

impl DataType for ComplexType {
    type Impl = Complex;
}

impl NumberDataType for ComplexType {
    type Impl = Complex;

    fn size(&self) -> usize {
        match self {
            ComplexType::C32 => 8,
            ComplexType::C64 => 16,
        }
    }
}

impl From<ComplexType> for NumberType {
    fn from(ct: ComplexType) -> NumberType {
        NumberType::Complex(ct)
    }
}

impl fmt::Display for ComplexType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use ComplexType::*;
        match self {
            C32 => write!(f, "C32"),
            C64 => write!(f, "C64"),
        }
    }
}

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub enum FloatType {
    F32,
    F64,
}

impl DataType for FloatType {
    type Impl = Float;
}

impl NumberDataType for FloatType {
    type Impl = Float;

    fn size(&self) -> usize {
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

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub enum IntType {
    I16,
    I32,
    I64,
}

impl DataType for IntType {
    type Impl = Int;
}

impl NumberDataType for IntType {
    type Impl = Int;

    fn size(&self) -> usize {
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

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub enum UIntType {
    U8,
    U16,
    U32,
    U64,
}

impl DataType for UIntType {
    type Impl = UInt;
}

impl NumberDataType for UIntType {
    type Impl = UInt;

    fn size(&self) -> usize {
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

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub enum NumberType {
    Bool,
    Complex(ComplexType),
    Float(FloatType),
    Int(IntType),
    UInt(UIntType),
}

impl NumberDataType for NumberType {
    type Impl = Number;

    fn size(&self) -> usize {
        use NumberType::*;
        match self {
            Bool => 1,
            Complex(ct) => ct.size(),
            Float(ft) => ft.size(),
            Int(it) => it.size(),
            UInt(ut) => ut.size(),
        }
    }
}

impl DataType for NumberType {
    type Impl = value::Number;
}

impl From<NumberType> for TCType {
    fn from(nt: NumberType) -> TCType {
        TCType::Number(nt)
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

impl DataType for StringType {
    type Impl = value::TCString;
}

impl From<StringType> for TCType {
    fn from(st: StringType) -> TCType {
        TCType::TCString(st)
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
pub enum TCType {
    Bytes,
    None,
    Number(NumberType),
    TCString(StringType),
    Op,
    Vector,
}

impl TCType {
    pub fn size(&self) -> Option<usize> {
        use TCType::*;
        match self {
            None => Some(1),
            Number(nt) => Some(nt.size()),
            _ => Option::None,
        }
    }
}

impl DataType for TCType {
    type Impl = value::Value;
}

impl fmt::Display for TCType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use TCType::*;
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
