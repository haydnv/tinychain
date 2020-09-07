use std::sync::Arc;

use async_trait::async_trait;

use crate::class::{Instance, TCResult, TCStream};
use crate::collection::class::CollectionInstance;
use crate::collection::{Collection, CollectionBase, CollectionItem, CollectionView};
use crate::error;
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::Value;

use super::{Chain, ChainInstance, ChainType};

const ERR_NULL_STREAM: &str = "NullChain does not support to_stream. \
Consider using a different Chain.";

#[derive(Clone)]
pub struct NullChain {
    collection: CollectionBase,
}

impl Instance for NullChain {
    type Class = ChainType;

    fn class(&self) -> Self::Class {
        ChainType::Null
    }
}

#[async_trait]
impl CollectionInstance for NullChain {
    type Item = Value;
    type Slice = CollectionView;

    async fn get(
        &self,
        txn: Arc<Txn>,
        selector: Value,
    ) -> TCResult<CollectionItem<Self::Item, Self::Slice>> {
        self.collection.get(txn, selector).await
    }

    async fn is_empty(&self, _txn: Arc<Txn>) -> TCResult<bool> {
        // NullChain itself is always empty by definition
        Ok(true)
    }

    async fn put(
        &self,
        txn: Arc<Txn>,
        selector: Value,
        value: CollectionItem<Self::Item, Self::Slice>,
    ) -> TCResult<()> {
        self.collection.put(txn, selector, value).await
    }

    async fn to_stream(&self, _txn: Arc<Txn>) -> TCResult<TCStream<Value>> {
        Err(error::unsupported(ERR_NULL_STREAM))
    }
}

impl ChainInstance for NullChain {
    fn object(&'_ self) -> &'_ CollectionBase {
        &self.collection
    }
}

#[async_trait]
impl Transact for NullChain {
    async fn commit(&self, txn_id: &TxnId) {
        self.collection.commit(txn_id).await;
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.collection.rollback(txn_id).await;
    }
}

impl From<NullChain> for Collection {
    fn from(nc: NullChain) -> Collection {
        Chain::from(nc).into()
    }
}
