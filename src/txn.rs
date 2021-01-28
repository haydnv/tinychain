use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;

use futures::stream::{FuturesUnordered, StreamExt};

use error::*;
use generic::Id;
use transact::{Transaction, TxnId};

use crate::state::scalar::reference::Refer;
use crate::state::State;

#[derive(Clone)]
pub struct Txn {
    id: TxnId,
    state: HashMap<Id, State>,
}

impl Txn {
    pub fn new<I: IntoIterator<Item = (Id, State)>>(data: I, id: TxnId) -> Self {
        let state = data.into_iter().collect();
        Self { id, state }
    }

    pub async fn execute(&mut self, capture: Id) -> TCResult<State> {
        while self.resolve_id(&capture)?.is_ref() {
            let mut pending = Vec::with_capacity(self.state.len());
            let mut unvisited = Vec::with_capacity(self.state.len());
            unvisited.push(capture.clone());

            while let Some(id) = unvisited.pop() {
                let state = self.resolve_id(&capture)?;

                if state.is_ref() {
                    let mut deps = HashSet::new();
                    state.requires(&mut deps);

                    let mut ready = true;
                    for dep_id in deps.into_iter() {
                        if self.resolve_id(&dep_id)?.is_ref() {
                            ready = false;
                            unvisited.push(dep_id);
                        }
                    }

                    if ready {
                        pending.push(id);
                    }
                }
            }

            if pending.is_empty() && self.resolve_id(&capture)?.is_ref() {
                return Err(TCError::bad_request(
                    "Cannot resolve all dependencies of",
                    capture,
                ));
            }

            let mut providers = FuturesUnordered::from_iter(
                pending
                    .into_iter()
                    .map(|id| async { (id, Err(TCError::not_implemented("State::resolve"))) }),
            );

            while let Some((id, r)) = providers.next().await {
                match r {
                    Ok(state) => {
                        self.state.insert(id, state);
                    }
                    Err(cause) => return Err(cause.consume(format!("Error resolving {}", id))),
                }
            }
        }

        self.state
            .remove(&capture)
            .ok_or_else(|| TCError::not_found(capture))
    }

    pub fn resolve_id(&'_ self, id: &Id) -> TCResult<&'_ State> {
        self.state.get(id).ok_or_else(|| TCError::not_found(id))
    }
}

impl Transaction for Txn {
    fn id(&self) -> &TxnId {
        &self.id
    }
}
