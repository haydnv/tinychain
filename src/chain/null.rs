use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream;
use futures::TryFutureExt;

use crate::auth::Auth;
use crate::class::*;
use crate::collection::class::*;
use crate::collection::{CollectionBase, CollectionBaseType};
use crate::error;
use crate::transaction::lock::{Mutable, TxnLock};
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::class::ValueClass;
use crate::value::op::OpDef;
use crate::value::{Link, TCPath, Value, ValueId};

use super::{ChainInstance, ChainType};

const ERR_COLLECTION_VIEW: &str = "Chain does not support CollectionView; \
consider making a copy of the Collection first";

#[derive(Clone)]
enum ChainState {
    Collection(CollectionBase),
    Value(TxnLock<Mutable<Value>>),
}

#[async_trait]
impl Transact for ChainState {
    async fn commit(&self, txn_id: &TxnId) {
        match self {
            Self::Collection(c) => c.commit(txn_id).await,
            Self::Value(v) => v.commit(txn_id).await,
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        match self {
            Self::Collection(c) => c.rollback(txn_id).await,
            Self::Value(v) => v.rollback(txn_id).await,
        }
    }
}

impl From<CollectionBase> for ChainState {
    fn from(cb: CollectionBase) -> ChainState {
        ChainState::Collection(cb)
    }
}

impl From<Value> for ChainState {
    fn from(v: Value) -> ChainState {
        ChainState::Value(TxnLock::new("Chain value", v.into()))
    }
}

#[derive(Clone)]
pub struct NullChain {
    state: ChainState,
    get_ops: HashMap<ValueId, OpDef>,
}

impl NullChain {
    pub async fn create(
        txn: Arc<Txn>,
        dtype: TCPath,
        schema: Value,
        get_ops: HashMap<ValueId, OpDef>,
    ) -> TCResult<NullChain> {
        let dtype = TCType::from_path(&dtype)?;
        let state = match dtype {
            TCType::Collection(ct) => match ct {
                CollectionType::Base(ct) => {
                    // TODO: figure out a bettern way than "Link::from(ct).path()"
                    let collection =
                        CollectionBaseType::get(txn, Link::from(ct).path(), schema).await?;
                    collection.into()
                }
                _ => return Err(error::unsupported(ERR_COLLECTION_VIEW)),
            },
            TCType::Value(vt) => {
                let value = vt.default();
                value.into()
            }
            other => return Err(error::not_implemented(format!("Chain({})", other))),
        };

        Ok(NullChain { state, get_ops })
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
            match &self.state {
                ChainState::Collection(collection) => {
                    collection.get_item(txn, key).map_ok(State::from).await
                }
                ChainState::Value(value) if key == Value::None => {
                    let value = value.read(txn.id()).await?;
                    Ok(State::Value(value.clone()))
                }
                ChainState::Value(_) => Err(error::not_found(format!(
                    "Value has no such property {}",
                    key
                ))),
            }
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

    async fn to_stream(&self, _txn: Arc<Txn>) -> TCResult<TCStream<Value>> {
        Ok(Box::pin(stream::empty()))
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
}
