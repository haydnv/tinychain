use crate::error;

pub mod link;
mod reference;

#[allow(clippy::module_inception)]
mod value;
mod version;

pub type TCRef = reference::TCRef;
pub type TCResult<T> = Result<T, error::TCError>;
pub type Value = value::Value;
pub type ValueId = value::ValueId;
pub type Version = version::Version;
