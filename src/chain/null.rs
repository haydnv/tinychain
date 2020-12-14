use async_trait::async_trait;
use futures::stream;

use crate::auth::Scope;
use crate::class::{Instance, State, TCResult, TCStream, TCType};
use crate::collection::btree::BTreeFile;
use crate::collection::table::TableIndex;
use crate::collection::tensor::dense::{BlockListFile, DenseTensor};
use crate::collection::tensor::sparse::{SparseTable, SparseTensor};
use crate::error;
use crate::handler::*;
use crate::request::Request;
use crate::scalar::{Object, PathSegment, Scalar, Value};
use crate::transaction::lock::{Mutable, TxnLock};
use crate::transaction::{Transact, Txn, TxnId};

use super::{ChainInstance, ChainType};

#[derive(Clone)]
enum ChainState {
    Tree(BTreeFile),
    Table(TableIndex),
    DenseTensor(DenseTensor<BlockListFile>),
    SparseTensor(SparseTensor<SparseTable>),
    Scalar(TxnLock<Mutable<Scalar>>),
}

#[async_trait]
impl Transact for ChainState {
    async fn commit(&self, txn_id: &TxnId) {
        match self {
            Self::Tree(tree) => tree.commit(txn_id).await,
            Self::Table(table) => table.commit(txn_id).await,
            Self::DenseTensor(tensor) => tensor.commit(txn_id).await,
            Self::SparseTensor(tensor) => tensor.commit(txn_id).await,
            Self::Scalar(s) => s.commit(txn_id).await,
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        match self {
            Self::Tree(tree) => tree.rollback(txn_id).await,
            Self::Table(table) => table.rollback(txn_id).await,
            Self::DenseTensor(tensor) => tensor.rollback(txn_id).await,
            Self::SparseTensor(tensor) => tensor.rollback(txn_id).await,
            Self::Scalar(s) => s.rollback(txn_id).await,
        }
    }

    async fn finalize(&self, txn_id: &TxnId) {
        match self {
            Self::Tree(tree) => tree.finalize(txn_id).await,
            Self::Table(table) => table.finalize(txn_id).await,
            Self::DenseTensor(tensor) => tensor.finalize(txn_id).await,
            Self::SparseTensor(tensor) => tensor.finalize(txn_id).await,
            Self::Scalar(s) => s.finalize(txn_id).await,
        }
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
    pub async fn create(
        _txn: &Txn,
        _dtype: TCType,
        _schema: Value,
    ) -> TCResult<NullChain> {
        Err(error::not_implemented("NullChain::create"))
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

    fn scope(&self) -> Scope {
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
