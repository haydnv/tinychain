use std::convert::TryFrom;
use std::sync::Arc;

use async_trait::async_trait;

use crate::class::{Class, Instance, TCResult};
use crate::error;
use crate::transaction::{Transact, Txn};
use crate::value::link::TCPath;
use crate::value::Value;

use super::btree::BTreeType;
use super::{Collection, CollectionView};

#[async_trait]
pub trait CollectionClass: Class + Into<CollectionType> + Send + Sync {
    type Instance: CollectionInstance;

    async fn get(
        txn: Arc<Txn>,
        path: &TCPath,
        schema: Value,
    ) -> TCResult<<Self as CollectionClass>::Instance>;
}

#[async_trait]
trait CollectionBaseClass: CollectionClass + Into<CollectionBaseType> + Send + Sync {
    type Instance: CollectionBaseInstance;

    fn get(
        txn: Arc<Txn>,
        path: &TCPath,
        schema: Value,
    ) -> TCResult<<Self as CollectionClass>::Instance>;
}

trait CollectionViewClass: CollectionClass + Into<CollectionViewType> + Send + Sync {
    type Instance: CollectionViewInstance;
}

#[async_trait]
pub trait CollectionInstance: Instance + Into<Collection> + Transact + Send + Sync {
    type Selector: Clone + TryFrom<Value, Error = error::TCError> + Send + Sync + 'static;

    type Item: Clone + Into<Value> + TryFrom<Value, Error = error::TCError> + Send + Sync + 'static;

    type Slice: CollectionViewInstance;

    async fn get(&self, txn: Arc<Txn>, selector: Self::Selector) -> TCResult<Self::Slice>;

    async fn put(
        &self,
        txn: &Arc<Txn>,
        selector: &Self::Selector,
        value: Self::Item,
    ) -> TCResult<()>;
}

#[async_trait]
pub trait CollectionBaseInstance: CollectionInstance {
    type Schema: TryFrom<Value, Error = error::TCError>;

    fn create(txn: Arc<Txn>, schema: Self::Schema) -> TCResult<Self>;
}

pub trait CollectionViewInstance: CollectionInstance + Into<CollectionView> {}

pub enum CollectionType {
    Base(CollectionBaseType),
    View(CollectionViewType),
}

pub enum CollectionBaseType {
    BTree,
    Graph,
    Table,
    Tensor,
}

pub enum CollectionViewType {
    BTree(BTreeType),
    Graph,
    Table,
    Tensor,
}
