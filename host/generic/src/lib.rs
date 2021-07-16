//! Provides generic datatypes used across multiple Tinychain sub-crates.
//!
//! This library is a part of Tinychain: [http://github.com/haydnv/tinychain](http://github.com/haydnv/tinychain)

use std::fmt;
use std::pin::Pin;

use futures::Future;

use tc_error::*;

pub use id::*;
pub use map::*;
pub use stream::*;
pub use time::*;
pub use tuple::*;

mod id;
mod map;
mod stream;
mod time;
mod tuple;

/// A pinned future
pub type TCBoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// A pinned future which returns a [`TCResult`]
pub type TCBoxTryFuture<'a, T> = Pin<Box<dyn Future<Output = TCResult<T>> + Send + 'a>>;

/// A generic class trait
pub trait Class: fmt::Display + Sized {}

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
