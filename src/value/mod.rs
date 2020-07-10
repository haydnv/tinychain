use std::fmt;
use std::pin::Pin;

use futures::stream::Stream;

use crate::error;

pub mod link;
pub mod op;
mod reference;

#[allow(clippy::module_inception)]
mod value;
mod version;

pub type Number = value::Number;
pub type TCStream<T> = Pin<Box<dyn Stream<Item = T> + Send + Sync + Unpin>>;
pub type TCRef = reference::TCRef;
pub type TCResult<T> = Result<T, error::TCError>;
pub type Value = value::Value;
pub type ValueId = value::ValueId;

pub trait DataType: Eq + Into<TCType> {
    type Impl: TypeImpl;
}

pub trait TypeImpl: Into<Value> {
    type DType: DataType;

    fn dtype(&self) -> Self::DType;

    fn is_a(&self, dtype: Self::DType) -> bool {
        self.dtype() == dtype
    }
}

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub enum NumberType {
    Bool,
    Complex32,
    Complex64,
    Float32,
    Float64,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
}

impl NumberType {
    pub fn size(&self) -> usize {
        use NumberType::*;
        match self {
            Bool => 1,
            Complex32 => 8,
            Complex64 => 16,
            Float32 => 4,
            Float64 => 8,
            Int16 => 2,
            Int32 => 4,
            Int64 => 8,
            UInt8 => 1,
            UInt16 => 2,
            UInt32 => 4,
            UInt64 => 8,
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
            Bool => write!(f, "type Bool"),
            Complex32 => write!(f, "type Complex32"),
            Complex64 => write!(f, "type Complex64"),
            Float32 => write!(f, "type Float32"),
            Float64 => write!(f, "type Float64"),
            Int16 => write!(f, "type Int16"),
            Int32 => write!(f, "type Int32"),
            Int64 => write!(f, "type Int64"),
            UInt8 => write!(f, "type UInt8"),
            UInt16 => write!(f, "type UInt26"),
            UInt32 => write!(f, "type UInt32"),
            UInt64 => write!(f, "type UInt64"),
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
            Number(number) => Some(number.size()),
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
            Number(n) => write!(f, "{}", n),
            TCString(s) => write!(f, "{}", s),
            Op => write!(f, "type Op"),
            Vector => write!(f, "type Vector"),
        }
    }
}
