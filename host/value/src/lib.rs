//! Provides a generic scalar [`Value`] enum which supports collation.
//!
//! This library is part of Tinychain: [http://github.com/haydnv/tinychain](http://github.com/haydnv/tinychain)

use std::cmp::Ordering;

use bytes::Bytes;
use collate::{Collate, Collator};
use number_general::NumberCollator;

use tcgeneric::Instance;

pub use link::*;
pub use slice::*;
pub use value::*;

mod link;
mod slice;
mod value;

/// [`Collate`] support for [`Value`]
#[derive(Default, Clone)]
pub struct ValueCollator {
    bytes: Collator<Bytes>,
    link: Collator<Link>,
    number: NumberCollator,
    string: Collator<String>,
}

impl Collate for ValueCollator {
    type Value = Value;

    fn compare(&self, left: &Self::Value, right: &Self::Value) -> Ordering {
        match (left, right) {
            (Value::Bytes(l), Value::Bytes(r)) => self.bytes.compare(l, r),
            (Value::Link(l), Value::Link(r)) => self.link.compare(l, r),
            (Value::Number(l), Value::Number(r)) => self.number.compare(l, r),
            (Value::String(l), Value::String(r)) => self.string.compare(l, r),
            (Value::Tuple(l), Value::Tuple(r)) => self.compare_slice(l.as_slice(), r.as_slice()),
            (l, r) => l.class().cmp(&r.class()),
        }
    }
}
