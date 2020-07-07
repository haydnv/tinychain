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

pub type TCStream<T> = Pin<Box<dyn Stream<Item = T> + Send + Sync + Unpin>>;
pub type TCRef = reference::TCRef;
pub type TCResult<T> = Result<T, error::TCError>;
pub type Value = value::Value;
pub type ValueId = value::ValueId;

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub enum TCType {
    None,
    Bool,
    Bytes,
    Complex32,
    Complex64,
    Float32,
    Float64,
    Id,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Link,
    Op,
    Ref,
    r#String,
    Vector,
}

impl TCType {
    pub fn is_numeric(&self) -> bool {
        use TCType::*;
        match self {
            Bool => true,
            Complex32 => true,
            Complex64 => true,
            Float32 => true,
            Float64 => true,
            Int16 => true,
            Int32 => true,
            Int64 => true,
            UInt8 => true,
            UInt16 => true,
            UInt32 => true,
            UInt64 => true,
            _ => false,
        }
    }

    pub fn size(&self) -> Option<usize> {
        use TCType::*;
        match self {
            None => Some(1),
            Bool => Some(1),
            Complex32 => Some(8),
            Complex64 => Some(16),
            Float32 => Some(4),
            Float64 => Some(8),
            Int16 => Some(2),
            Int32 => Some(4),
            Int64 => Some(8),
            UInt8 => Some(1),
            UInt16 => Some(2),
            UInt32 => Some(4),
            UInt64 => Some(8),
            _ => Option::None,
        }
    }
}

impl fmt::Display for TCType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use TCType::*;
        match self {
            None => write!(f, "type None"),
            Bool => write!(f, "type Bool"),
            Bytes => write!(f, "type Bytes"),
            Complex32 => write!(f, "type Complex32"),
            Complex64 => write!(f, "type Complex64"),
            Id => write!(f, "type Id"),
            Float32 => write!(f, "type Float32"),
            Float64 => write!(f, "type Float64"),
            Int16 => write!(f, "type Int16"),
            Int32 => write!(f, "type Int32"),
            Int64 => write!(f, "type Int64"),
            UInt8 => write!(f, "type UInt8"),
            UInt16 => write!(f, "type UInt26"),
            UInt32 => write!(f, "type UInt32"),
            UInt64 => write!(f, "type UInt64"),
            Link => write!(f, "type Link"),
            Op => write!(f, "type Op"),
            Ref => write!(f, "type Ref"),
            r#String => write!(f, "type String"),
            Vector => write!(f, "type Vector"),
        }
    }
}
