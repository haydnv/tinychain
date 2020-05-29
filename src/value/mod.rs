use crate::error;

pub mod link;
pub mod op;
mod reference;
mod value;
mod version;

pub type TCRef = reference::TCRef;
pub type TCResult<T> = Result<T, error::TCError>;
pub type Subject = op::Subject;
pub type Value = value::Value;
pub type ValueId = value::ValueId;
pub type Version = version::Version;
