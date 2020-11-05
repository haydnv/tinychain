use async_trait::async_trait;
use futures::stream::{self, Stream};

use crate::class::*;
use crate::collection::class::*;
use crate::collection::CollectionBase;
use crate::error;
use crate::request::Request;
use crate::scalar::{Scalar, ScalarClass, TCPath, Value, ValueId};
use crate::transaction::lock::{Mutable, TxnLock};
use crate::transaction::{Transact, Txn, TxnId};

use super::{ChainInstance, ChainType};

const ERR_COLLECTION_VIEW: &str = "Chain does not support CollectionView; \
consider making a copy of the Collection first";

#[derive(Clone)]
enum ChainState {
    Collection(CollectionBase),
    Scalar(TxnLock<Mutable<Scalar>>),
}

#[async_trait]
impl Transact for ChainState {
    async fn commit(&self, txn_id: &TxnId) {
        match self {
            Self::Collection(c) => c.commit(txn_id).await,
            Self::Scalar(s) => s.commit(txn_id).await,
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        match self {
            Self::Collection(c) => c.rollback(txn_id).await,
            Self::Scalar(s) => s.rollback(txn_id).await,
        }
    }
}

impl From<CollectionBase> for ChainState {
    fn from(cb: CollectionBase) -> ChainState {
        ChainState::Collection(cb)
    }
}

impl From<Scalar> for ChainState {
    fn from(s: Scalar) -> ChainState {
        ChainState::Scalar(TxnLock::new("Chain value", s.into()))
    }
}

#[derive(Clone)]
pub struct NullChain {
    state: ChainState,
}

impl NullChain {
    pub async fn create(txn: &Txn, dtype: TCType, schema: Value) -> TCResult<NullChain> {
        let state = match dtype {
            TCType::Collection(ct) => match ct {
                CollectionType::Base(ct) => {
                    let collection = ct.get(txn, schema).await?;
                    collection.into()
                }
                _ => return Err(error::unsupported(ERR_COLLECTION_VIEW)),
            },
            TCType::Scalar(st) => {
                let scalar = st.try_cast(schema)?;
                println!("NullChain::create({}) wraps scalar {}", st, scalar);
                scalar.into()
            }
            other => return Err(error::not_implemented(format!("Chain({})", other))),
        };

        Ok(NullChain { state })
    }
}

impl Instance for NullChain {
    type Class = ChainType;

    fn class(&self) -> Self::Class {
        ChainType::Null
    }
}

#[async_trait]
impl ChainInstance for NullChain {
    type Class = ChainType;

    async fn get(
        &self,
        _request: &Request,
        _txn: &Txn,
        _path: &TCPath,
        _key: Value,
    ) -> TCResult<State> {
        Err(error::not_implemented("NullChain::get"))
    }

    async fn put(
        &self,
        _request: &Request,
        _txn: &Txn,
        _path: TCPath,
        _key: Value,
        _new_value: State,
    ) -> TCResult<()> {
        Err(error::not_implemented("NullChain::put"))
    }

    async fn post<S: Stream<Item = (ValueId, Scalar)> + Send + Unpin>(
        &self,
        _request: &Request,
        _txn: &Txn,
        _path: TCPath,
        _data: S,
    ) -> TCResult<State> {
        Err(error::not_implemented("NullChain::post"))
    }

    async fn to_stream(&self, _txn: Txn) -> TCResult<TCStream<Value>> {
        Ok(Box::pin(stream::empty()))
    }
}

#[async_trait]
impl Transact for NullChain {
    async fn commit(&self, txn_id: &TxnId) {
        self.state.commit(txn_id).await;
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.state.rollback(txn_id).await;
    }
}
