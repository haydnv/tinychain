use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use futures::TryFutureExt;

use crate::auth::Auth;
use crate::class::{Class, Instance, State, TCResult, TCStream, TCType};
use crate::collection::CollectionBase;
use crate::error;
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::op::OpDef;
use crate::value::{label, Link, TCPath, Value, ValueId};

mod block;
mod null;

pub type ChainBlock = block::ChainBlock;

#[async_trait]
pub trait ChainClass: Class + Into<ChainType> + Send + Sync {
    type Instance: ChainInstance;

    async fn get(
        txn: Arc<Txn>,
        path: &TCPath,
        ctype: TCPath,
        schema: Value,
        ops: HashMap<ValueId, OpDef>,
    ) -> TCResult<<Self as ChainClass>::Instance>;
}

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
        TCType::prefix().join(label("chain").into())
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

#[async_trait]
impl ChainClass for ChainType {
    type Instance = Chain;

    async fn get(
        txn: Arc<Txn>,
        path: &TCPath,
        ctype: TCPath,
        schema: Value,
        ops: HashMap<ValueId, OpDef>,
    ) -> TCResult<Chain> {
        let suffix = path.from_path(&Self::prefix())?;

        if suffix.is_empty() {
            Err(error::unsupported("You must specify a type of Chain"))
        } else if suffix.len() > 1 {
            Err(error::not_found(suffix))
        } else {
            match suffix[0].as_str() {
                "null" => {
                    null::NullChain::create(txn, ctype, schema, ops)
                        .map_ok(Box::new)
                        .map_ok(Chain::Null)
                        .await
                }
                other => Err(error::not_found(other)),
            }
        }
    }
}

#[async_trait]
pub trait ChainInstance: Instance {
    type Class: ChainClass;

    async fn get(&self, txn: Arc<Txn>, path: &TCPath, key: Value, auth: Auth) -> TCResult<State>;

    async fn to_stream(&self, txn: Arc<Txn>) -> TCResult<TCStream<Value>>;
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

    async fn get(&self, txn: Arc<Txn>, path: &TCPath, key: Value, auth: Auth) -> TCResult<State> {
        match self {
            Self::Null(nc) => nc.get(txn, path, key, auth).await,
        }
    }

    async fn to_stream(&self, txn: Arc<Txn>) -> TCResult<TCStream<Value>> {
        match self {
            Self::Null(nc) => nc.to_stream(txn).await,
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

impl fmt::Display for Chain {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Null(_) => write!(f, "(null chain)"),
        }
    }
}
