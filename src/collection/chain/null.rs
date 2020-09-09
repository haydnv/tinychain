use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::class::{Instance, TCResult, TCStream};
use crate::collection::class::{CollectionClass, CollectionInstance};
use crate::collection::*;
use crate::error;
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::{Op, TCPath, Value, ValueId};

use super::{Chain, ChainInstance, ChainType};

const ERR_NULL_STREAM: &str = "NullChain does not support to_stream. \
Consider using a different Chain.";

#[derive(Clone)]
pub struct NullChain {
    collection: CollectionBase,
    ops: HashMap<ValueId, Op>,
}

impl NullChain {
    pub async fn create(
        txn: Arc<Txn>,
        ctype: &TCPath,
        schema: Value,
        ops: HashMap<ValueId, Op>,
    ) -> TCResult<NullChain> {
        let collection = CollectionBaseType::get(txn, ctype, schema).await?;
        Ok(NullChain { collection, ops })
    }
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

    async fn get_item(
        &self,
        _txn: Arc<Txn>,
        _selector: Value,
    ) -> TCResult<CollectionItem<Self::Item, Self::Slice>> {
        Err(error::not_implemented("NullChain::get"))
    }

    async fn is_empty(&self, _txn: Arc<Txn>) -> TCResult<bool> {
        // NullChain itself is always empty by definition
        Ok(true)
    }

    async fn put_item(
        &self,
        _txn: Arc<Txn>,
        _selector: Value,
        _value: CollectionItem<Self::Item, Self::Slice>,
    ) -> TCResult<()> {
        Err(error::not_implemented("NullChain::put"))
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
