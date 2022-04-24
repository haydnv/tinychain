//! Provides a generic scalar [`Value`] enum which supports collation.
//!
//! This library is part of TinyChain: [http://github.com/haydnv/tinychain](http://github.com/haydnv/tinychain)

use std::cmp::Ordering;

use bytes::Bytes;
use collate::{Collate, Collator};
pub use number_general::NumberCollator;

use tcgeneric::Instance;

pub use link::*;
pub use slice::*;
pub use string::*;
pub use value::*;
pub use version::*;

mod link;
mod slice;
mod string;
mod value;
mod version;

/// [`Collate`] support for [`Value`]
#[derive(Default, Clone)]
pub struct ValueCollator {
    bytes: Collator<Bytes>,
    link: Collator<Link>,
    number: NumberCollator,
    string: StringCollator,
    version: Collator<Version>,
}

impl Collate for ValueCollator {
    type Value = Value;

    fn compare(&self, left: &Self::Value, right: &Self::Value) -> Ordering {
        match (left, right) {
            (Value::Bytes(l), Value::Bytes(r)) => self.bytes.compare(l, r),
            (Value::Link(l), Value::Link(r)) => self.link.compare(l, r),
            (Value::Number(l), Value::Number(r)) => self.number.compare(l, r),
            (Value::Version(l), Value::Version(r)) => self.version.compare(l, r),
            (Value::String(l), Value::String(r)) => self.string.compare(l, r),
            (Value::Tuple(l), Value::Tuple(r)) => self.compare_slice(l.as_slice(), r.as_slice()),
            (l, r) => l.class().cmp(&r.class()),
        }
    }
}
