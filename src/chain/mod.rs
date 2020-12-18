use std::fmt;

use async_trait::async_trait;
use futures::TryFutureExt;

use crate::class::{Class, Instance, NativeClass, TCType};
use crate::error;
use crate::general::{TCResult, TCStreamOld};
use crate::handler::*;
use crate::scalar::{label, Link, MethodType, PathSegment, TCPathBuf, Value};
use crate::transaction::{Transact, Txn, TxnId};

mod block;
mod null;

pub type ChainBlock = block::ChainBlock;

#[async_trait]
pub trait ChainClass: Class + Into<ChainType> + Send {
    type Instance: ChainInstance;

    async fn get(
        &self,
        txn: &Txn,
        dtype: TCType,
        schema: Value,
    ) -> TCResult<<Self as ChainClass>::Instance>;
}

#[derive(Clone, Eq, PartialEq)]
pub enum ChainType {
    Null,
}

impl Class for ChainType {
    type Instance = Chain;
}

impl NativeClass for ChainType {
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        let suffix = Self::prefix().try_suffix(path)?;

        if suffix.is_empty() {
            return Err(error::unsupported("You must specify a type of Chain"));
        }

        match suffix[0].as_str() {
            "null" if suffix.len() == 1 => Ok(ChainType::Null),
            other => Err(error::not_found(other)),
        }
    }

    fn prefix() -> TCPathBuf {
        TCType::prefix().append(label("chain"))
    }
}

impl From<ChainType> for Link {
    fn from(ct: ChainType) -> Link {
        match ct {
            ChainType::Null => ChainType::prefix().append(label("null")).into(),
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

#[async_trait]
impl ChainClass for ChainType {
    type Instance = Chain;

    async fn get(&self, txn: &Txn, dtype: TCType, schema: Value) -> TCResult<Chain> {
        match self {
            Self::Null => {
                null::NullChain::create(txn, dtype, schema)
                    .map_ok(Box::new)
                    .map_ok(Chain::Null)
                    .await
            }
        }
    }
}

#[async_trait]
pub trait ChainInstance: Instance {
    type Class: ChainClass;

    async fn to_stream(&self, txn: Txn) -> TCResult<TCStreamOld<Value>>;
}

#[derive(Clone)]
pub enum Chain {
    Null(Box<null::NullChain>),
}

impl Instance for Chain {
    type Class = ChainType;

    fn class(&self) -> <Self as Instance>::Class {
        match self {
            Self::Null(nc) => nc.class(),
        }
    }
}

#[async_trait]
impl ChainInstance for Chain {
    type Class = ChainType;

    async fn to_stream(&self, txn: Txn) -> TCResult<TCStreamOld<Value>> {
        match self {
            Self::Null(nc) => nc.to_stream(txn).await,
        }
    }
}

impl Route for Chain {
    fn route(
        &'_ self,
        method: MethodType,
        path: &'_ [PathSegment],
    ) -> Option<Box<dyn Handler + '_>> {
        match self {
            Self::Null(nc) => nc.route(method, path),
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

    async fn finalize(&self, txn_id: &TxnId) {
        match self {
            Self::Null(nc) => nc.finalize(txn_id).await,
        }
    }
}

impl From<null::NullChain> for Chain {
    fn from(nc: null::NullChain) -> Chain {
        Chain::Null(Box::new(nc))
    }
}

impl fmt::Display for Chain {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Null(_) => write!(f, "(null chain)"),
        }
    }
}
