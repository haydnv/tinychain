use async_trait::async_trait;
use destream::{FromStream, ToStream, Decoder, Encoder};
use number_general::{Number, NumberType};

use generic::{Map, Tuple};

pub mod link;

pub use link::*;
use std::cmp::Ordering;

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum ValueType {
    Link,
    Map,
    Number(NumberType),
    String,
    Tuple,
    Value,
}

impl Ord for ValueType {
    fn cmp(&self, other: &Self) -> Ordering {
        use Ordering::*;

        match (self, other) {
            (l, r) if l == r => Equal,

            (Self::Value, _) => Greater,
            (_, Self::Value) => Less,

            (Self::Map, _) => Greater,
            (_, Self::Map) => Less,

            (Self::Tuple, _) => Greater,
            (_, Self::Tuple) => Less,

            (Self::Link, _) => Greater,
            (_, Self::Link) => Less,

            (Self::String, _) => Greater,
            (_, Self::String) => Less,

            (Self::Number(l), Self::Number(r)) => l.cmp(r),
        }
    }
}

impl PartialOrd for ValueType {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum Value {
    Link(Link),
    Map(Map<Self>),
    Number(Number),
    String(String),
    Tuple(Tuple<Self>),
}

#[async_trait]
impl FromStream for Value {
    async fn from_stream<D: Decoder>(_decoder: &mut D) -> Result<Self, <D as Decoder>::Error> {
        unimplemented!()
    }
}

impl<'en> ToStream<'en> for Value {
    fn to_stream<E: Encoder<'en>>(&'en self, _encoder: E) -> Result<<E as Encoder<'en>>::Ok, <E as Encoder<'en>>::Error> {
        unimplemented!()
    }
}