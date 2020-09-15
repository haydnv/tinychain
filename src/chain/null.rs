use std::collections::HashMap;
use std::convert::TryInto;
use std::sync::Arc;

use async_trait::async_trait;
use futures::TryFutureExt;
use futures::stream::{self, Stream, StreamExt};

use crate::auth::Auth;
use crate::class::*;
use crate::collection::class::*;
use crate::collection::{Collection, CollectionBase, CollectionBaseType};
use crate::error;
use crate::transaction::lock::{Mutable, TxnLock};
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::class::ValueClass;
use crate::value::op::OpDef;
use crate::value::{Link, TCPath, Value, ValueId, ValueType};

use super::{Chain, ChainInstance, ChainType};

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
    ops: HashMap<ValueId, OpDef>,
}

impl NullChain {
    pub async fn create(
        txn: Arc<Txn>,
        dtype: TCPath,
        schema: Value,
        ops: HashMap<ValueId, OpDef>,
    ) -> TCResult<NullChain> {
        let dtype = TCType::from_path(&dtype)?;
        let state = match dtype {
            TCType::Collection(ct) => match ct {
                CollectionType::Base(ct) => {
                    // TODO: figure out a better way than "Link::from(ct).path()"
                    let collection =
                        CollectionBaseType::get(txn, Link::from(ct).path(), schema).await?;
                    collection.into()
                }
                _ => return Err(error::unsupported(ERR_COLLECTION_VIEW)),
            },
            TCType::Value(vt) => {
                let value = ValueType::get(Link::from(vt).path(), schema)?;
                println!("NullChain::create({}) wraps value {}", vt, value);
                value.into()
            }
            other => return Err(error::not_implemented(format!("Chain({})", other))),
        };

        println!("new chain with {} ops", ops.len());
        Ok(NullChain { state, ops })
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
        println!("NullChain::get {}: {}", path, &key);

        if path.is_empty() {
            Ok(Chain::Null(Box::new(self.clone())).into())
        } else if path == "/object" {
            match &self.state {
                ChainState::Collection(collection) => Ok(Collection::Base(collection.clone()).into()),
                ChainState::Value(value) => value.read(txn.id()).map_ok(|v| State::Value(v.clone())).await,
            }
        } else if path.len() == 1 {
            println!(
                "looking up {} in collection of {} ops...",
                path,
                self.ops.len()
            );
            for name in self.ops.keys() {
                println!("available op: {}", name);
            }

            if let Some(op) = self.ops.get(&path[0]) {
                if let OpDef::Get((key_name, def)) = op {
                    let mut params = Vec::with_capacity(def.len() + 1);
                    params.push((key_name.clone(), key));
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

    async fn put(&self, txn: Arc<Txn>, path: TCPath, key: Value, new_value: State) -> TCResult<()> {
        match &self.state {
            ChainState::Collection(_) => Err(error::not_implemented("NullChain::put")),
            ChainState::Value(value) => {
                if path.is_empty() {
                    if key == Value::None {
                        let mut value = value.write(txn.id().clone()).await?;
                        new_value.expect(value.class().into(), format!("Chain wraps {}", value.class()))?;
                        *value = new_value.try_into()?;
                        Ok(())
                    } else {
                        Err(error::bad_request("Value has no such attribute", key))
                    }
                } else {
                    Err(error::bad_request("Value contains no such resource", path))
                }
            }
        }
    }

    async fn post<S: Stream<Item = (ValueId, Value)> + Send + Sync + Unpin>(
        &self,
        txn: Arc<Txn>,
        path: TCPath,
        data: S,
        capture: &[ValueId],
        auth: Auth,
    ) -> TCResult<Vec<TCStream<Value>>> {
        if path.is_empty() {
            Err(error::method_not_allowed("NullChain::post"))
        } else if path.len() == 1 {
            if let Some(OpDef::Post(def)) = self.ops.get(&path[0]) {
                println!("Chain::post {} def: {}", path, def.iter().map(|(name, op)| format!("{}: {}", name, op)).collect::<Vec<String>>().join(", "));
                let data = data.chain(stream::iter(def.to_vec()));
                txn.execute_and_stream(data, capture, auth).await
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
