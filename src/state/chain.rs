use std::fmt;

use destream::en::{Encoder, IntoStream, ToStream};

use generic::*;

const PREFIX: PathLabel = path_label(&["state", "chain"]);

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum ChainType {}

impl Class for ChainType {
    type Instance = Chain;
}

impl NativeClass for ChainType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if path.len() == 3 && &path[0..2] == &PREFIX[..] {
            unimplemented!()
        } else {
            None
        }
    }

    fn path(&self) -> TCPathBuf {
        unimplemented!()
    }
}

impl fmt::Display for ChainType {
    fn fmt(&self, _f: &mut fmt::Formatter) -> fmt::Result {
        unimplemented!()
    }
}

#[derive(Clone)]
pub enum Chain {}

impl Instance for Chain {
    type Class = ChainType;

    fn class(&self) -> Self::Class {
        unimplemented!()
    }
}

impl<'en> ToStream<'en> for Chain {
    fn to_stream<E: Encoder<'en>>(&'en self, _e: E) -> Result<E::Ok, E::Error> {
        unimplemented!()
    }
}

impl<'en> IntoStream<'en> for Chain {
    fn into_stream<E: Encoder<'en>>(self, _e: E) -> Result<E::Ok, E::Error> {
        unimplemented!()
    }
}

impl fmt::Display for Chain {
    fn fmt(&self, _f: &mut fmt::Formatter) -> fmt::Result {
        unimplemented!()
    }
}
