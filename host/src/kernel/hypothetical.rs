use std::collections::HashSet;
use std::fmt;

use bytes::Bytes;
use safecast::{TryCastFrom, TryCastInto};

use tc_error::*;
use tc_transact::lock::TxnLock;
use tc_transact::Transaction;
use tc_value::{Link, Value};
use tcgeneric::{path_label, Id, Map, PathLabel};

use crate::scalar::{Executor, Refer, Scope};
use crate::state::State;
use crate::txn::{Actor, Txn};

use crate::generic::PathSegment;
use crate::route::{DeleteHandler, GetHandler, Handler, PutHandler, Route};

pub const PATH: PathLabel = path_label(&["transact", "hypothetical"]);

#[derive(Clone)]
pub struct Hypothetical {
    actor: Actor,
    participants: TxnLock<HashSet<Link>>,
}

impl Hypothetical {
    pub fn new() -> Self {
        Self {
            actor: Actor::new(Link::default().into()),
            participants: TxnLock::new("hypothetical transaction participants", HashSet::new()),
        }
    }

    pub async fn execute(&self, txn: &Txn, data: State) -> TCResult<State> {
        let txn = txn.clone().claim(&self.actor, PATH.into()).await?;
        let context = Map::<State>::default();

        if Vec::<(Id, State)>::can_cast_from(&data) {
            let op_def: Vec<(Id, State)> = data.opt_cast_into().unwrap();
            let capture = if let Some((capture, _)) = op_def.last() {
                capture.clone()
            } else {
                return Ok(State::default());
            };

            let executor: Executor<State> = Executor::new(&txn, None, op_def);
            executor.capture(capture).await
        } else {
            data.resolve(&Scope::<State>::new(None, context), &txn)
                .await
        }
    }
}

impl<'a> Handler<'a> for &'a Hypothetical {
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
                    TCError::bad_request("invalid transaction participant link", v)
                })?;

                let mut participants = self.participants.write(*txn.id()).await?;
                participants.insert(participant);
                Ok(())
            })
        }))
    }

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                if key.is_none() {
                    Ok(())
                } else {
                    Err(TCError::not_found(key))
                }
            })
        }))
    }
}

impl Route for Hypothetical {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.is_empty() {
            Some(Box::new(self))
        } else {
            None
        }
    }
}

impl fmt::Display for Hypothetical {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a hypothetical transaction handler")
    }
}
