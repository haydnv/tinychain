//! Provides generic data types used across multiple TinyChain sub-crates.
//!
//! This library is a part of TinyChain: [http://github.com/haydnv/tinychain](http://github.com/haydnv/tinychain)

use std::fmt;
use std::pin::Pin;

use futures::{Future, Stream};

use tc_error::*;

pub use pathlink::{
    label, path_label, Id, Label, Path as TCPath, PathBuf as TCPathBuf, PathLabel, PathSegment,
};

pub use map::*;
pub use time::*;
pub use tuple::*;

mod map;
mod time;
mod tuple;

/// A pinned future
pub type TCBoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// A pinned future which returns a [`TCResult`]
pub type TCBoxTryFuture<'a, T> = Pin<Box<dyn Future<Output = TCResult<T>> + Send + 'a>>;

/// A pinned [`Stream`]
pub type TCBoxStream<'a, T> = Pin<Box<dyn Stream<Item = T> + Send + Unpin + 'a>>;

/// A pinned `TryStream` with error type [`TCError`]
pub type TCBoxTryStream<'a, T> = Pin<Box<dyn Stream<Item = TCResult<T>> + Send + Unpin + 'a>>;

/// A thread-safe type
pub trait ThreadSafe: Clone + Send + Sync + 'static {}

impl<T: Clone + Send + Sync + 'static> ThreadSafe for T {}

/// A generic class trait
pub trait Class: fmt::Debug + Sized {}

/// A generic native (i.e. implemented in Rust) class trait
pub trait NativeClass: Class {
    /// Given a fully qualified path, return this class, or a subclass.
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
