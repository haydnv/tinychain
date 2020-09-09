use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use futures::TryFutureExt;

use crate::class::{Class, Instance, TCResult, TCStream};
use crate::error;
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::{label, Link, Op, TCPath, Value, ValueId};

use super::class::{CollectionClass, CollectionInstance};
use super::*;

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

#[async_trait]
impl CollectionClass for ChainType {
    type Instance = Chain;

    async fn get(txn: Arc<Txn>, path: &TCPath, schema: Value) -> TCResult<Chain> {
        let suffix = path.from_path(&Self::prefix())?;

        if suffix.is_empty() {
            return Err(error::unsupported("You must specify a type of Chain"));
        }

        match suffix[0].as_str() {
            "null" if suffix.len() == 1 => {
                let (collection_type, schema, ops): (TCPath, Value, Vec<(ValueId, Op)>) =
                    schema.try_into()?;
                null::NullChain::create(txn, &collection_type, schema, ops.into_iter().collect())
                    .map_ok(Chain::from)
                    .await
            }
            other => Err(error::not_found(other)),
        }
    }
}

impl From<ChainType> for CollectionType {
    fn from(ct: ChainType) -> CollectionType {
        CollectionType::Base(CollectionBaseType::Chain(ct))
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

pub trait ChainInstance: Instance + CollectionInstance
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

#[async_trait]
impl CollectionInstance for Chain {
    type Item = Value;
    type Slice = CollectionView;

    async fn get_item(
        &self,
        txn: Arc<Txn>,
        selector: Value,
    ) -> TCResult<CollectionItem<Self::Item, Self::Slice>> {
        match self {
            Self::Null(nc) => nc.get_item(txn, selector).await,
        }
    }

    async fn is_empty(&self, txn: Arc<Txn>) -> TCResult<bool> {
        match self {
            Self::Null(nc) => nc.is_empty(txn).await,
        }
    }

    async fn put_item(
        &self,
        txn: Arc<Txn>,
        selector: Value,
        value: CollectionItem<Self::Item, Self::Slice>,
    ) -> TCResult<()> {
        match self {
            Self::Null(nc) => nc.put_item(txn, selector, value).await,
        }
    }

    async fn to_stream(&self, txn: Arc<Txn>) -> TCResult<TCStream<Value>> {
        match self {
            Self::Null(nc) => nc.to_stream(txn).await,
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

impl From<Chain> for Collection {
    fn from(chain: Chain) -> Collection {
        Collection::Base(CollectionBase::Chain(chain))
    }
}
