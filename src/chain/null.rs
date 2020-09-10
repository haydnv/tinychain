use std::sync::Arc;

use async_trait::async_trait;
use futures::TryFutureExt;

use crate::auth::Auth;
use crate::class::{Instance, State, TCResult};
use crate::collection::class::*;
use crate::collection::{CollectionBase, CollectionBaseType};
use crate::error;
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::{TCPath, Value};

use super::{ChainInstance, ChainType};

const ERR_NULL_STREAM: &str = "NullChain does not support to_stream. \
Consider using a different Chain.";

#[derive(Clone)]
pub struct NullChain {
    collection: CollectionBase,
}

impl NullChain {
    pub async fn create(
        txn: Arc<Txn>,
        ctype: &TCPath,
        schema: Value,
    ) -> TCResult<NullChain> {
        let collection = CollectionBaseType::get(txn, ctype, schema).await?;
        Ok(NullChain { collection })
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
    async fn get(&self, txn: Arc<Txn>, path: &TCPath, key: Value, _auth: Auth) -> TCResult<State> {
        if path.is_empty() {
            self.object().get_item(txn, key).map_ok(State::from).await
        } else if path.len() == 1 {
            Err(error::not_implemented("NullChain::get"))
        } else {
            Err(error::not_found(path))
        }
    }

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
