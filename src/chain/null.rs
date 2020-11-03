use std::collections::HashMap;
use std::convert::TryInto;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::{self, Stream, StreamExt};
use futures::TryFutureExt;

use crate::class::*;
use crate::collection::class::*;
use crate::collection::{Collection, CollectionBase};
use crate::error;
use crate::request::Request;
use crate::scalar::{OpDef, Scalar, ScalarClass, TCPath, Value, ValueId};
use crate::transaction::lock::{Mutable, TxnLock};
use crate::transaction::{Transact, Txn, TxnId};

use super::{Chain, ChainInstance, ChainType};

const ERR_COLLECTION_VIEW: &str = "Chain does not support CollectionView; \
consider making a copy of the Collection first";

#[derive(Clone)]
enum ChainState {
    Collection(CollectionBase),
    Scalar(TxnLock<Mutable<Scalar>>),
}

#[async_trait]
impl Transact for ChainState {
    async fn commit(&self, txn_id: &TxnId) {
        match self {
            Self::Collection(c) => c.commit(txn_id).await,
            Self::Scalar(s) => s.commit(txn_id).await,
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        match self {
            Self::Collection(c) => c.rollback(txn_id).await,
            Self::Scalar(s) => s.rollback(txn_id).await,
        }
    }
}

impl From<CollectionBase> for ChainState {
    fn from(cb: CollectionBase) -> ChainState {
        ChainState::Collection(cb)
    }
}

impl From<Scalar> for ChainState {
    fn from(s: Scalar) -> ChainState {
        ChainState::Scalar(TxnLock::new("Chain value", s.into()))
    }
}

#[derive(Clone)]
pub struct NullChain {
    state: ChainState,
    ops: TxnLock<Mutable<HashMap<ValueId, OpDef>>>,
}

impl NullChain {
    pub async fn create(
        txn: Arc<Txn>,
        dtype: TCType,
        schema: Value,
        ops: HashMap<ValueId, OpDef>,
    ) -> TCResult<NullChain> {
        let state = match dtype {
            TCType::Collection(ct) => match ct {
                CollectionType::Base(ct) => {
                    let collection = ct.get(txn, schema).await?;
                    collection.into()
                }
                _ => return Err(error::unsupported(ERR_COLLECTION_VIEW)),
            },
            TCType::Scalar(st) => {
                let scalar = st.try_cast(schema)?;
                println!("NullChain::create({}) wraps scalar {}", st, scalar);
                scalar.into()
            }
            other => return Err(error::not_implemented(format!("Chain({})", other))),
        };

        println!("new chain with {} ops", ops.len());
        Ok(NullChain {
            state,
            ops: TxnLock::new("NullChain ops", ops.into()),
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

    async fn get(
        &self,
        request: Request,
        txn: Arc<Txn>,
        path: &TCPath,
        key: Value,
    ) -> TCResult<State> {
        println!("NullChain::get {}: {}", path, &key);

        if path.is_empty() {
            Ok(Chain::Null(Box::new(self.clone())).into())
        } else if path == "/object" {
            match &self.state {
                ChainState::Collection(collection) => {
                    Ok(Collection::Base(collection.clone()).into())
                }
                ChainState::Scalar(scalar) => {
                    scalar
                        .read(txn.id())
                        .map_ok(|s| State::Scalar(s.clone()))
                        .await
                }
            }
        } else if path.len() == 1 {
            if let Some(op) = self.ops.read(txn.id()).await?.get(&path[0]) {
                if let OpDef::Get((key_name, def)) = op {
                    let mut params = Vec::with_capacity(def.len() + 1);
                    params.push((key_name.clone(), Scalar::Value(key)));
                    params.extend(def.to_vec());
                    txn.execute(request, stream::iter(params)).await
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

    async fn put(
        &self,
        _request: &Request,
        txn: Arc<Txn>,
        path: TCPath,
        key: Value,
        new_value: State,
    ) -> TCResult<()> {
        if &path == "/object" {
            match &self.state {
                ChainState::Collection(_) => Err(error::not_implemented("NullChain::put")),
                ChainState::Scalar(scalar) => {
                    if key == Value::None {
                        let mut scalar = scalar.write(txn.id().clone()).await?;

                        new_value.expect(
                            scalar.class().into(),
                            format!("Chain wraps {}", scalar.class()),
                        )?;
                        *scalar = new_value.try_into()?;
                        Ok(())
                    } else {
                        Err(error::bad_request("Value has no such attribute", key))
                    }
                }
            }
        } else {
            Err(error::not_implemented(path))
        }
    }

    async fn post<S: Stream<Item = (ValueId, Scalar)> + Send + Unpin>(
        &self,
        request: Request,
        txn: Arc<Txn>,
        path: TCPath,
        data: S,
    ) -> TCResult<State> {
        if path.is_empty() {
            Err(error::method_not_allowed("NullChain::post"))
        } else if path.len() == 1 {
            if let Some(OpDef::Post(def)) = self.ops.read(txn.id()).await?.get(&path[0]) {
                let data = data.chain(stream::iter(def.to_vec()));
                txn.execute(request, data).await
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
        self.ops.commit(txn_id).await;
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.state.rollback(txn_id).await;
        self.ops.rollback(txn_id).await;
    }
}
