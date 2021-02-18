//! Provides generic datatypes used across multiple Tinychain sub-crates.

use std::fmt;

pub mod id;
pub mod map;
pub mod time;
pub mod tuple;

pub use id::*;
pub use map::*;
pub use time::*;
pub use tuple::*;

/// A generic class trait
pub trait Class: fmt::Display + Sized {
    /// The [`Instance`] type of this class.
    type Instance;
}

/// A generic native (i.e. implemented in Rust) class trait
pub trait NativeClass: Class {
    /// Given a fully qualified path, return this class, or a subclass.
    ///
    /// Example:
    /// ```no_run
    /// assert_eq!(
    ///     Number::from_path("/state/scalar/value/number/int/32"),
    ///     NumberType::Int(IntType::I32));
    /// ```
    fn from_path(path: &[PathSegment]) -> Option<Self>;

    /// Returns the fully-qualified path of this class.
    fn path(&self) -> TCPathBuf;
}

/// A generic instance trait
pub trait Instance: Send + Sync {
    /// The [`Class`] type of this instance
    type Class: Class;

    /// Returns the [`Class]` of this instance.
    fn class(&self) -> Self::Class;
}
