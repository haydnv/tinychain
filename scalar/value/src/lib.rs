use async_trait::async_trait;
use destream::{FromStream, ToStream, Decoder, Encoder};
use number_general::Number;

pub mod link;
pub mod string;

pub use link::*;
pub use string::*;

#[derive(Eq, PartialEq)]
pub enum Value {
    Link(Link),
    Number(Number),
    String(String),
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