use std::fmt;

use async_trait::async_trait;

use crate::class::{Class, Instance, TCResult};
use crate::collection::{CollectionBase, CollectionType};
use crate::error;
use crate::transaction::{Transact, TxnId};
use crate::value::{label, Link, TCPath};

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
            return Err(error::unsupported("You must specify a type of Chain"));
        }

        match suffix[0].as_str() {
            "null" if suffix.len() == 1 => Ok(ChainType::Null),
            other => Err(error::not_found(other)),
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

pub trait ChainInstance: Instance
where
    <Self as Instance>::Class: Into<ChainType>,
{
    fn object(&'_ self) -> &'_ CollectionBase;
}

#[derive(Clone)]
pub enum Chain {
    Null(Box<null::NullChain>),
}

impl Instance for Chain {
    type Class = ChainType;

    fn class(&self) -> Self::Class {
        match self {
            Self::Null(nc) => nc.class(),
        }
    }
}

impl ChainInstance for Chain {
    fn object(&self) -> &CollectionBase {
        match self {
            Self::Null(nc) => nc.object(),
        }
    }
}

#[async_trait]
impl Transact for Chain {
    async fn commit(&self, txn_id: &TxnId) {
        match self {
            Self::Null(nc) => nc.commit(txn_id).await,
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        match self {
            Self::Null(nc) => nc.rollback(txn_id).await,
        }
    }
}

impl From<null::NullChain> for Chain {
    fn from(nc: null::NullChain) -> Chain {
        Chain::Null(Box::new(nc))
    }
}
