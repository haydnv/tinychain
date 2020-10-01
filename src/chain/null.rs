use std::collections::HashMap;
use std::convert::TryInto;
use std::sync::Arc;

use futures::future;
use futures::join;
use futures::stream::{self, Stream};
use futures::TryFutureExt;

use crate::auth::Auth;
use crate::class::*;
use crate::collection::class::*;
use crate::collection::{Collection, CollectionBase};
use crate::error;
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

impl Transact for ChainState {
    fn commit<'a>(&'a self, txn_id: &'a TxnId) -> TCBoxFuture<'a, ()> {
        match self {
            Self::Collection(c) => c.commit(txn_id),
            Self::Scalar(s) => s.commit(txn_id),
        }
    }

    fn rollback<'a>(&'a self, txn_id: &'a TxnId) -> TCBoxFuture<'a, ()> {
        match self {
            Self::Collection(c) => c.rollback(txn_id),
            Self::Scalar(s) => s.rollback(txn_id),
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

impl ChainInstance for NullChain {
    type Class = ChainType;

    fn get<'a>(
        &'a self,
        txn: Arc<Txn>,
        path: &'a TCPath,
        key: Value,
        _auth: Auth,
    ) -> TCBoxTryFuture<'a, State> {
        Box::pin(async move {
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
                    if let OpDef::Get((_key_name, _def)) = op {
                        Err(error::not_implemented("Chain methods"))
                    } else {
                        Err(error::method_not_allowed(path))
                    }
                } else {
                    Err(error::not_found(path))
                }
            } else {
                Err(error::not_found(path))
            }
        })
    }

    fn put<'a>(
        &'a self,
        txn: Arc<Txn>,
        path: TCPath,
        key: Value,
        new_value: State,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
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
        })
    }

    fn post<'a, S: Stream<Item = (ValueId, Scalar)> + Send + Sync + Unpin + 'a>(
        &'a self,
        txn: Arc<Txn>,
        path: TCPath,
        _data: S,
        _auth: Auth,
    ) -> TCBoxTryFuture<'a, State> {
        Box::pin(async move {
            if path.is_empty() {
                Err(error::method_not_allowed("NullChain::post"))
            } else if path.len() == 1 {
                if let Some(OpDef::Post(_def)) = self.ops.read(txn.id()).await?.get(&path[0]) {
                    Err(error::not_implemented("Chain methods"))
                } else {
                    Err(error::not_found(path))
                }
            } else {
                Err(error::not_found(path))
            }
        })
    }

    fn to_stream<'a>(&'a self, _txn: Arc<Txn>) -> TCBoxTryFuture<'a, TCStream<Value>> {
        Box::pin(future::ready({
            let stream: TCStream<Value> = Box::pin(stream::empty());
            Ok(stream)
        }))
    }
}

impl Transact for NullChain {
    fn commit<'a>(&'a self, txn_id: &'a TxnId) -> TCBoxFuture<'a, ()> {
        Box::pin(async move {
            join!(self.state.commit(txn_id), self.ops.commit(txn_id));
        })
    }

    fn rollback<'a>(&'a self, txn_id: &'a TxnId) -> TCBoxFuture<'a, ()> {
        Box::pin(async move {
            join!(self.state.rollback(txn_id), self.ops.rollback(txn_id));
        })
    }
}
