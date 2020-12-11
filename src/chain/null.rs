use async_trait::async_trait;
use futures::stream;
use log::debug;

use crate::auth::Scope;
use crate::class::{Instance, State, TCResult, TCStream, TCType};
use crate::collection::class::*;
use crate::collection::CollectionBase;
use crate::error;
use crate::handler::*;
use crate::request::Request;
use crate::scalar::{Object, PathSegment, Scalar, ScalarClass, Value};
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

    async fn finalize(&self, txn_id: &TxnId) {
        match self {
            Self::Collection(c) => c.finalize(txn_id).await,
            Self::Scalar(s) => s.finalize(txn_id).await,
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
                debug!("NullChain::create({}) wraps scalar {}", st, scalar);
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

    async fn to_stream(&self, _txn: Txn) -> TCResult<TCStream<Value>> {
        Ok(Box::pin(stream::empty()))
    }
}

struct NullChainHandler {
    chain: NullChain,
}

impl Handler for NullChainHandler {
    fn subject(&self) -> TCType {
        self.chain.class().into()
    }

    fn scope() -> Scope {
        "/admin".parse().unwrap()
    }
}

impl From<NullChain> for NullChainHandler {
    fn from(chain: NullChain) -> Self {
        Self { chain }
    }
}

#[async_trait]
impl Public for NullChain {
    async fn get(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        key: Value,
    ) -> TCResult<State> {
        if path.is_empty() {
            NullChainHandler::from(self.clone())
                .get(request, txn, key)
                .await
        } else {
            Err(error::path_not_found(path))
        }
    }

    async fn put(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        key: Value,
        value: State,
    ) -> TCResult<()> {
        if path.is_empty() {
            NullChainHandler::from(self.clone())
                .put(request, txn, key, value)
                .await
        } else {
            Err(error::path_not_found(path))
        }
    }

    async fn post(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        data: Object,
    ) -> TCResult<State> {
        if path.is_empty() {
            NullChainHandler::from(self.clone())
                .post(request, txn, data)
                .await
        } else {
            Err(error::path_not_found(path))
        }
    }

    async fn delete(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        key: Value,
    ) -> TCResult<()> {
        if path.is_empty() {
            NullChainHandler::from(self.clone())
                .delete(request, txn, key)
                .await
        } else {
            Err(error::path_not_found(path))
        }
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

    async fn finalize(&self, txn_id: &TxnId) {
        self.state.finalize(txn_id).await;
    }
}
