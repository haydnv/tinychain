use std::convert::TryFrom;
use std::sync::Arc;

use async_trait::async_trait;

use crate::class::{Class, TCResult};
use crate::error;
use crate::transaction::{Transact, Txn};
use crate::value::link::TCPath;
use crate::value::Value;

use super::Collection;

#[async_trait]
trait CollectionClass: Class + Into<CollectionType> + Send + Sync {
    type Instance: CollectionInstance;

    fn get(
        txn: Arc<Txn>,
        path: &TCPath,
        schema: Value,
    ) -> TCResult<<Self as CollectionClass>::Instance>;
}

#[async_trait]
pub trait CollectionInstance: Into<Collection> + Transact + Send + Sync {
    type Selector: Clone + TryFrom<Value, Error = error::TCError> + Send + Sync + 'static;

    type Item: Clone + Into<Value> + TryFrom<Value, Error = error::TCError> + Send + Sync + 'static;

    type Slice: CollectionInstance;

    async fn get(&self, txn: Arc<Txn>, selector: Self::Selector) -> TCResult<Self::Slice>;

    async fn put(
        &self,
        txn: &Arc<Txn>,
        selector: &Self::Selector,
        value: Self::Item,
    ) -> TCResult<()>;
}

pub enum CollectionType {
    BTree,
    Graph,
    Table,
    Tensor,
}
