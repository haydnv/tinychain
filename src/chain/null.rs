use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::class::{Instance, TCResult};
use crate::collection::class::CollectionClass;
use crate::collection::{CollectionBase, CollectionBaseType};
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::{Op, TCPath, Value, ValueId};

use super::{ChainInstance, ChainType};

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
