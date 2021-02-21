//! Provides a generic scalar [`Value`] enum which supports collation.
//!
//! This library is part of Tinychain: [http://github.com/haydnv/tinychain](http://github.com/haydnv/tinychain)

pub mod link;
mod value;

pub use link::*;
pub use value::*;
