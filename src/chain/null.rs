use async_trait::async_trait;
use futures::stream;

use crate::auth::Scope;
use crate::class::{Instance, TCResult, TCStream, TCType};
use crate::collection::btree::BTreeFile;
use crate::collection::table::TableIndex;
use crate::collection::tensor::dense::{BlockListFile, DenseTensor};
use crate::collection::tensor::sparse::{SparseTable, SparseTensor};
use crate::error;
use crate::handler::*;
use crate::scalar::{MethodType, PathSegment, Value};
use crate::transaction::{Transact, Txn, TxnId};

use super::{ChainInstance, ChainType};

#[derive(Clone)]
enum ChainState {
    Tree(BTreeFile),
    Table(TableIndex),
    DenseTensor(DenseTensor<BlockListFile>),
    SparseTensor(SparseTensor<SparseTable>),
}

#[async_trait]
impl Transact for ChainState {
    async fn commit(&self, txn_id: &TxnId) {
        match self {
            Self::Tree(tree) => tree.commit(txn_id).await,
            Self::Table(table) => table.commit(txn_id).await,
            Self::DenseTensor(tensor) => tensor.commit(txn_id).await,
            Self::SparseTensor(tensor) => tensor.commit(txn_id).await,
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        match self {
            Self::Tree(tree) => tree.rollback(txn_id).await,
            Self::Table(table) => table.rollback(txn_id).await,
            Self::DenseTensor(tensor) => tensor.rollback(txn_id).await,
            Self::SparseTensor(tensor) => tensor.rollback(txn_id).await,
        }
    }

    async fn finalize(&self, txn_id: &TxnId) {
        match self {
            Self::Tree(tree) => tree.finalize(txn_id).await,
            Self::Table(table) => table.finalize(txn_id).await,
            Self::DenseTensor(tensor) => tensor.finalize(txn_id).await,
            Self::SparseTensor(tensor) => tensor.finalize(txn_id).await,
        }
    }
}

#[derive(Clone)]
pub struct NullChain {
    state: ChainState,
}

impl NullChain {
    pub async fn create(_txn: &Txn, _dtype: TCType, _schema: Value) -> TCResult<NullChain> {
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

struct NullChainHandler<'a> {
    chain: &'a NullChain,
}

impl<'a> Handler for NullChainHandler<'a> {
    fn subject(&self) -> TCType {
        self.chain.class().into()
    }

    fn scope(&self) -> Option<Scope> {
        Some("/admin".parse().unwrap())
    }
}

impl Route for NullChain {
    fn route(
        &'_ self,
        _method: MethodType,
        path: &'_ [PathSegment],
    ) -> Option<Box<dyn Handler + '_>> {
        let chain = self;

        if path.is_empty() {
            Some(Box::new(NullChainHandler { chain }))
        } else {
            None
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
