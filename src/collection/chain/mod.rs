use std::fmt;

use crate::class::{Class, Instance, TCResult};
use crate::error;
use crate::value::{label, Link, TCPath};

use super::CollectionType;

mod block;
mod null;

pub type ChainBlock = block::ChainBlock;

#[derive(Clone, Eq, PartialEq)]
pub enum ChainType {
    Null,
}

impl Class for ChainType {
    type Instance = Chain;

    fn from_path(path: &TCPath) -> TCResult<Self> {
        let suffix = path.from_path(&Self::prefix())?;

        if suffix.is_empty() {
            Err(error::unsupported("You must specify a type of Chain"))
        } else {
            match suffix[0].as_str() {
                "null" if suffix.len() == 1 => Ok(ChainType::Null),
                other => Err(error::not_found(other)),
            }
        }
    }

    fn prefix() -> TCPath {
        CollectionType::prefix().join(label("chain").into())
    }
}

impl From<ChainType> for Link {
    fn from(ct: ChainType) -> Link {
        match ct {
            ChainType::Null => ChainType::prefix().join(label("null").into()).into(),
        }
    }
}

impl fmt::Display for ChainType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Null => write!(f, "type: Null Chain"),
        }
    }
}

#[derive(Clone)]
pub enum Chain {
    Null(null::NullChain),
}

impl Instance for Chain {
    type Class = ChainType;

    fn class(&self) -> Self::Class {
        match self {
            Self::Null(_) => ChainType::Null,
        }
    }
}
