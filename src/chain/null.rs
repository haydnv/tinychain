use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream;
use futures::TryFutureExt;

use crate::auth::Auth;
use crate::class::{Instance, State, TCResult, TCStream};
use crate::collection::class::*;
use crate::collection::{CollectionBase, CollectionBaseType};
use crate::error;
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::op::OpDef;
use crate::value::{TCPath, Value, ValueId};

use super::{ChainInstance, ChainType};

#[derive(Clone)]
pub struct NullChain {
    collection: CollectionBase,
    get_ops: HashMap<ValueId, OpDef>,
}

impl NullChain {
    pub async fn create(
        txn: Arc<Txn>,
        ctype: TCPath,
        schema: Value,
        get_ops: HashMap<ValueId, OpDef>,
    ) -> TCResult<NullChain> {
        let collection = CollectionBaseType::get(txn, &ctype, schema).await?;
        Ok(NullChain {
            collection,
            get_ops,
        })
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

    async fn get(&self, txn: Arc<Txn>, path: &TCPath, key: Value, auth: Auth) -> TCResult<State> {
        if path.is_empty() {
            self.object().get_item(txn, key).map_ok(State::from).await
        } else if path.len() == 1 {
            if let Some(op) = self.get_ops.get(&path[0]) {
                if let OpDef::Get((in_ref, def)) = op {
                    let mut params = Vec::with_capacity(def.len() + 1);
                    params.push((in_ref.value_id().clone(), key));
                    params.extend(def.to_vec());

                    let capture = &params.last().unwrap().0.clone();
                    let mut txn_state = txn
                        .execute(stream::iter(params), &[capture.clone()], auth)
                        .await?;
                    txn_state
                        .remove(capture)
                        .ok_or_else(|| error::not_found(capture))
                } else {
                    Err(error::method_not_allowed(path))
                }
            } else {
                Err(error::not_found(path))
            }
        } else {
            Err(error::not_found(path))
        }
    }

    fn object(&'_ self) -> &'_ CollectionBase {
        &self.collection
    }

    async fn to_stream(&self, _txn: Arc<Txn>) -> TCResult<TCStream<Value>> {
        Ok(Box::pin(stream::empty()))
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
