use std::fmt;

use async_trait::async_trait;
use destream::{Decoder, Encoder, FromStream, IntoStream, ToStream};

use generic::*;

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum RefType {}

impl Class for RefType {
    type Instance = TCRef;
}

impl NativeClass for RefType {
    fn from_path(_path: &[PathSegment]) -> Option<Self> {
        unimplemented!()
    }

    fn path(&self) -> TCPathBuf {
        unimplemented!()
    }
}

impl fmt::Display for RefType {
    fn fmt(&self, _f: &mut fmt::Formatter) -> fmt::Result {
        unimplemented!()
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum TCRef {}

impl Instance for TCRef {
    type Class = RefType;

    fn class(&self) -> Self::Class {
        unimplemented!()
    }
}

#[async_trait]
impl FromStream for TCRef {
    async fn from_stream<D: Decoder>(_decoder: &mut D) -> Result<Self, <D as Decoder>::Error> {
        unimplemented!()
    }
}

impl<'en> ToStream<'en> for TCRef {
    fn to_stream<E: Encoder<'en>>(
        &'en self,
        _encoder: E,
    ) -> Result<<E as Encoder<'en>>::Ok, <E as Encoder<'en>>::Error> {
        unimplemented!()
    }
}

impl<'en> IntoStream<'en> for TCRef {
    fn into_stream<E: Encoder<'en>>(
        self,
        _encoder: E,
    ) -> Result<<E as Encoder<'en>>::Ok, <E as Encoder<'en>>::Error> {
        unimplemented!()
    }
}

impl fmt::Display for TCRef {
    fn fmt(&self, _f: &mut fmt::Formatter) -> fmt::Result {
        unimplemented!()
    }
}
