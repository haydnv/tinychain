//! A generic scalar [`Value`] enum which supports collation.

pub extern crate uuid;

use std::cmp::Ordering;

use collate::{Collate, Collator};

use tcgeneric::Instance;

pub use class::*;
pub use link::*;
pub use number::*;
pub use string::*;
pub use value::*;
pub use version::*;

mod class;

mod link {
    pub use pathlink::{Address, Host, Link, Protocol};
}

mod number {
    pub use number_general::{
        Boolean, BooleanType, Complex, ComplexType, DType, Float, FloatInstance, FloatType, Int, IntType,
        Number, NumberClass, NumberCollator, NumberInstance, NumberType, Trigonometry, UInt,
        UIntType,
    };
}

mod string;
mod value;
mod version;

/// [`Collate`] support for [`Value`]
#[derive(Default, Clone, Eq, PartialEq)]
pub struct ValueCollator {
    bytes: Collator<Vec<u8>>,
    link: Collator<Link>,
    number: NumberCollator,
    string: StringCollator,
    version: Collator<Version>,
}

impl Collate for ValueCollator {
    type Value = Value;

    fn cmp(&self, left: &Self::Value, right: &Self::Value) -> Ordering {
        match (left, right) {
            (Value::Bytes(l), Value::Bytes(r)) => self.bytes.cmp(l, r),
            (Value::Link(l), Value::Link(r)) => self.link.cmp(l, r),
            (Value::Number(l), Value::Number(r)) => self.number.cmp(l, r),
            (Value::Version(l), Value::Version(r)) => self.version.cmp(l, r),
            (Value::String(l), Value::String(r)) => self.string.cmp(l, r),
            (Value::Tuple(l), Value::Tuple(r)) => {
                for i in 0..Ord::min(l.len(), r.len()) {
                    match self.cmp(&l[i], &r[i]) {
                        Ordering::Equal => {}
                        ordering => return ordering,
                    }
                }

                Ordering::Equal
            }
            (l, r) => l.class().cmp(&r.class()),
        }
    }
}
