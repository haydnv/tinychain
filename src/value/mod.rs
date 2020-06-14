use std::fmt;
use std::pin::Pin;

use futures::stream::Stream;

use crate::error;

pub mod link;
mod op;
mod reference;

#[allow(clippy::module_inception)]
mod value;
mod version;

pub type TCStream<T> = Pin<Box<dyn Stream<Item = T> + Send + Sync + Unpin>>;
pub type TCRef = reference::TCRef;
pub type TCResult<T> = Result<T, error::TCError>;
pub type Value = value::Value;
pub type ValueId = value::ValueId;

#[derive(Clone, Hash, Eq, PartialEq)]
pub enum TCType {
    None,
    Bytes,
    Id,
    Int32,
    UInt64,
    Link,
    Op,
    Ref,
    r#String,
    Vector,
}

impl TCType {
    pub fn size(&self) -> Option<usize> {
        use TCType::*;
        match self {
            None => Some(1),
            Int32 => Some(4),
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
            Bytes => write!(f, "type Bytes"),
            Id => write!(f, "Id"),
            Int32 => write!(f, "type Int32"),
            UInt64 => write!(f, "type UInt64"),
            Link => write!(f, "type Link"),
            Op => write!(f, "type Op"),
            Ref => write!(f, "type Ref"),
            r#String => write!(f, "type String"),
            Vector => write!(f, "type Vector"),
        }
    }
}
