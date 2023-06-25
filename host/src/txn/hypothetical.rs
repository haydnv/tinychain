use std::collections::btree_map::{self, BTreeMap};
use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use bytes::Bytes;
use futures::stream::{FuturesUnordered, StreamExt};
use log::error;
use safecast::{TryCastFrom, TryCastInto};
use tokio::sync::RwLock;

use tc_error::*;
use tc_scalar::{Executor, Refer, Scope};
use tc_transact::public::{Handler, Route};
use tc_transact::{RPCClient, Transaction, TxnId};
use tc_value::{Link, Value};
use tcgeneric::{path_label, Id, Map, PathLabel, PathSegment};

use crate::route::{DeleteHandler, GetHandler, PutHandler};
use crate::state::State;
use crate::txn::{Actor, Txn};

pub const PATH: PathLabel = path_label(&["transact", "hypothetical"]);

#[derive(Clone)]
pub struct Hypothetical {
    actor: Actor,
    participants: Arc<RwLock<BTreeMap<TxnId, HashSet<Link>>>>,
}

impl Hypothetical {
    pub fn new() -> Self {
        Self {
            actor: Actor::new(Link::default().into()),
            participants: Arc::new(RwLock::new(BTreeMap::new())),
        }
    }

    pub async fn finalize(&self, txn_id: TxnId) {
        let mut participants = self.participants.write().await;
        let expired = participants
            .keys()
            .take_while(|id| *id <= &txn_id)
            .copied()
            .collect::<Vec<_>>();

        for txn_id in expired {
            participants.remove(&txn_id);
        }
    }

    pub async fn execute(&self, txn: &Txn, data: State) -> TCResult<State> {
        let txn = txn.clone().claim(&self.actor, PATH.into()).await?;
        let context = Map::<State>::default();

        let result = if Vec::<(Id, State)>::can_cast_from(&data) {
            let op_def: Vec<(Id, State)> = data.opt_cast_into().unwrap();
            let capture = if let Some((capture, _)) = op_def.last() {
                capture.clone()
            } else {
                return Ok(State::default());
            };

            let executor: Executor<State, State> = Executor::new(&txn, None, op_def);
            executor.capture(capture).await
        } else {
            data.resolve(&Scope::<State, State>::new(None, context), &txn)
                .await
        };

        {
            let mut participants = self.participants.write().await;
            if let Some(participants) = participants.remove(txn.id()) {
                let mut rollbacks: FuturesUnordered<_> = participants
                    .iter()
                    .map(|link| txn.delete(link, Value::default()))
                    .collect();

                while let Some(result) = rollbacks.next().await {
                    if let Err(cause) = result {
                        error!("error finalizing hypothetical transaction: {}", cause);
                    }
                }
            }
        }

        result
    }
}

impl<'a> Handler<'a, State> for &'a Hypothetical {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                if key.is_none() {
                    Ok(
                        Value::from(Bytes::copy_from_slice(self.actor.public_key().as_bytes()))
                            .into(),
                    )
                } else {
                    Err(TCError::not_found(key))
                }
            })
        }))
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key, value| {
            Box::pin(async move {
                if key.is_some() {
                    return Err(TCError::not_found(key));
                }

                let participant = value.try_cast_into(|v| {
                    TCError::unexpected(v, "a Link to a transaction participant")
                })?;

                let mut participants = self.participants.write().await;
                match participants.entry(*txn.id()) {
                    btree_map::Entry::Vacant(entry) => {
                        entry.insert(HashSet::new()).insert(participant);
                    }
                    btree_map::Entry::Occupied(mut entry) => {
                        entry.get_mut().insert(participant);
                    }
                }

                Ok(())
            })
        }))
    }

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(move |txn, key| {
            Box::pin(async move {
                if key.is_none() {
                    let mut participants = self.participants.write().await;
                    participants.remove(txn.id());
                    Ok(())
                } else {
                    Err(TCError::not_found(key))
                }
            })
        }))
    }
}

impl Route<State> for Hypothetical {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        if path.is_empty() {
            Some(Box::new(self))
        } else {
            None
        }
    }
}

impl fmt::Debug for Hypothetical {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a hypothetical transaction handler")
    }
}
