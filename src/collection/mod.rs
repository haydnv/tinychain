use std::fmt;

use async_trait::async_trait;

use crate::class::{Instance, State, TCResult, TCStream};
use crate::handler::Public;
use crate::request::Request;
use crate::scalar::{Object, PathSegment, Scalar, Value};
use crate::transaction::{Transact, Txn, TxnId};

pub mod btree;
pub mod class;
pub mod schema;
pub mod table;
pub mod tensor;

pub use btree::*;
pub use class::*;
pub use table::*;
pub use tensor::*;

#[derive(Clone)]
pub enum Collection {
    BTree(BTree),
    Table(Table),
    Tensor(Tensor),
}

impl Instance for Collection {
    type Class = class::CollectionType;

    fn class(&self) -> Self::Class {
        match self {
            Self::BTree(btree) => class::CollectionType::BTree(btree.class()),
            Self::Table(table) => class::CollectionType::Table(table.class()),
            Self::Tensor(tensor) => class::CollectionType::Tensor(tensor.class()),
        }
    }
}

#[async_trait]
impl CollectionInstance for Collection {
    type Item = Scalar;

    async fn is_empty(&self, txn: &Txn) -> TCResult<bool> {
        match self {
            Self::BTree(btree) => CollectionInstance::is_empty(btree, txn).await,
            Self::Table(table) => table.is_empty(txn).await,
            Self::Tensor(tensor) => tensor.is_empty(txn).await,
        }
    }

    async fn to_stream(&self, txn: Txn) -> TCResult<TCStream<Scalar>> {
        match self {
            Self::BTree(btree) => btree.to_stream(txn).await,
            Self::Table(table) => table.to_stream(txn).await,
            Self::Tensor(tensor) => tensor.to_stream(txn).await,
        }
    }
}

#[async_trait]
impl Public for Collection {
    async fn get(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        selector: Value,
    ) -> TCResult<State> {
        match self {
            Self::BTree(btree) => btree.get(request, txn, path, selector).await,
            Self::Table(table) => Public::get(table, request, txn, path, selector).await,
            Self::Tensor(tensor) => tensor.get(request, txn, path, selector).await,
        }
    }

    async fn put(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        selector: Value,
        value: State,
    ) -> TCResult<()> {
        match self {
            Self::BTree(btree) => btree.put(request, txn, path, selector, value).await,
            Self::Table(table) => table.put(request, txn, path, selector, value).await,
            Self::Tensor(tensor) => tensor.put(request, txn, path, selector, value).await,
        }
    }

    async fn post(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        params: Object,
    ) -> TCResult<State> {
        match self {
            Self::BTree(btree) => btree.post(request, txn, path, params).await,
            Self::Table(table) => table.post(request, txn, path, params).await,
            Self::Tensor(tensor) => tensor.post(request, txn, path, params).await,
        }
    }

    async fn delete(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        selector: Value,
    ) -> TCResult<()> {
        match self {
            Self::BTree(btree) => Public::delete(btree, request, txn, path, selector).await,
            Self::Table(table) => table.delete(request, txn, path, selector).await,
            Self::Tensor(tensor) => tensor.delete(request, txn, path, selector).await,
        }
    }
}

#[async_trait]
impl Transact for Collection {
    async fn commit(&self, txn_id: &TxnId) {
        match self {
            Self::BTree(btree) => btree.commit(txn_id).await,
            Self::Table(table) => table.commit(txn_id).await,
            Self::Tensor(tensor) => tensor.commit(txn_id).await,
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        match self {
            Self::BTree(btree) => btree.rollback(txn_id).await,
            Self::Table(table) => table.rollback(txn_id).await,
            Self::Tensor(tensor) => tensor.rollback(txn_id).await,
        }
    }

    async fn finalize(&self, txn_id: &TxnId) {
        match self {
            Self::BTree(btree) => btree.finalize(txn_id).await,
            Self::Table(table) => table.finalize(txn_id).await,
            Self::Tensor(tensor) => tensor.finalize(txn_id).await,
        }
    }
}

impl From<BTree> for Collection {
    fn from(btree: BTree) -> Self {
        Collection::BTree(btree)
    }
}

impl From<Table> for Collection {
    fn from(table: Table) -> Self {
        Collection::Table(table)
    }
}

impl From<Tensor> for Collection {
    fn from(tensor: Tensor) -> Self {
        Collection::Tensor(tensor)
    }
}

impl fmt::Display for Collection {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BTree(btree) => fmt::Display::fmt(btree, f),
            Self::Table(table) => fmt::Display::fmt(table, f),
            Self::Tensor(tensor) => fmt::Display::fmt(tensor, f),
        }
    }
}
