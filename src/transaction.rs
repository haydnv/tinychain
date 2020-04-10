use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, RwLock};

use futures::future::try_join_all;
use rand::Rng;
use serde::de::DeserializeOwned;

use crate::context::*;
use crate::error;
use crate::host::Host;
use crate::state::TCState;
use crate::value::{Link, Op, TCValue, ValueId};

#[derive(Clone)]
pub struct TransactionId {
    timestamp: u128, // nanoseconds since Unix epoch
    nonce: u16,
}

impl TransactionId {
    fn new(timestamp: u128) -> TransactionId {
        let nonce: u16 = rand::thread_rng().gen();
        TransactionId { timestamp, nonce }
    }
}

impl Into<Link> for TransactionId {
    fn into(self) -> Link {
        Link::to(&format!("/{}-{}", self.timestamp, self.nonce)).unwrap()
    }
}

impl Into<String> for TransactionId {
    fn into(self) -> String {
        format!("{}-{}", self.timestamp, self.nonce)
    }
}

impl Into<Vec<u8>> for TransactionId {
    fn into(self) -> Vec<u8> {
        [
            &self.timestamp.to_be_bytes()[..],
            &self.nonce.to_be_bytes()[..],
        ]
        .concat()
    }
}

impl fmt::Display for TransactionId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}-{}", self.timestamp, self.nonce)
    }
}

pub struct Transaction {
    host: Arc<Host>,
    id: TransactionId,
    context: Link,
    state: RwLock<HashMap<ValueId, TCState>>,
    pending: RwLock<HashMap<ValueId, _Op>>,
}

struct _Op {
    action: Link,
    requires: Vec<(String, ValueId)>,
}

fn calc_deps(
    op: Op,
    state: &mut HashMap<ValueId, TCState>,
    pending: &mut HashMap<ValueId, _Op>,
) -> TCResult<()> {
    for (id, provider) in op.requires() {
        match provider {
            TCValue::Op(dep) => {
                calc_deps(dep.clone(), state, pending)?;

                pending.insert(
                    id,
                    _Op {
                        action: op.action(),
                        requires: dep
                            .requires()
                            .iter()
                            .cloned()
                            .map(|(id, _)| (id.clone(), id))
                            .collect(),
                    },
                );
            }
            value => {
                if state.contains_key(&id) {
                    return Err(error::bad_request("Duplicate values provided for", id));
                }

                state.insert(id, TCState::Value(value));
            }
        }
    }

    Ok(())
}

impl Transaction {
    pub fn of(host: Arc<Host>, op: Op) -> TCResult<Arc<Transaction>> {
        let id = TransactionId::new(host.time());
        let context: Link = id.clone().into();

        let mut state: HashMap<ValueId, TCState> = HashMap::new();
        let mut pending: HashMap<ValueId, _Op> = HashMap::new();
        calc_deps(op, &mut state, &mut pending)?;

        Ok(Arc::new(Transaction {
            host,
            id,
            context,
            state: RwLock::new(state),
            pending: RwLock::new(pending),
        }))
    }

    fn extend(
        self: Arc<Self>,
        context: Link,
        required: HashMap<String, TCState>,
    ) -> Arc<Transaction> {
        Arc::new(Transaction {
            host: self.host.clone(),
            id: self.id.clone(),
            context: self.context.append(&context),
            state: RwLock::new(required),
            pending: RwLock::new(HashMap::new()),
        })
    }

    pub fn context(self: Arc<Self>) -> Link {
        self.context.clone()
    }

    fn known(self: Arc<Self>) -> HashSet<ValueId> {
        self.state.read().unwrap().keys().cloned().collect()
    }

    fn enqueue(
        self: Arc<Self>,
        mut unvisited: Vec<ValueId>,
        queue: &mut Vec<ValueId>,
    ) -> TCResult<()> {
        let mut visited: HashSet<ValueId> = HashSet::new();
        let pending = self.pending.read().unwrap();
        while let Some(value_id) = unvisited.pop() {
            if visited.contains(&value_id) {
                continue;
            }

            if let Some(op) = pending.get(&value_id) {
                queue.push(value_id.clone());

                for (id, _) in &op.requires {
                    if pending.contains_key(&*id) {
                        unvisited.push(id.clone());
                    }
                }
            } else {
                println!("{} must already be known", value_id);
            }

            visited.insert(value_id);
        }

        Ok(())
    }

    pub async fn execute<'a>(
        self: Arc<Self>,
        capture: HashSet<ValueId>,
    ) -> TCResult<HashMap<ValueId, TCState>> {
        // TODO: add a TCValue::Ref type and it to support discrete namespaces for each child txn

        let unvisited: Vec<ValueId> = capture.clone().into_iter().collect();
        let mut queue: Vec<ValueId> = vec![];
        self.clone().enqueue(unvisited, &mut queue)?;

        while !queue.is_empty() {
            let known = self.clone().known();
            let mut ready: Vec<(ValueId, Link, Arc<Transaction>)> = vec![];
            while let Some(value_id) = queue.pop() {
                let op = if let Some(op) = self.pending.write().unwrap().remove(&value_id) {
                    op
                } else {
                    return Err(error::bad_request("No value was provided for {}", value_id));
                };

                if known.is_superset(&op.requires.iter().map(|(_, id)| id.clone()).collect()) {
                    let mut captured: HashMap<String, TCState> = HashMap::new();
                    let state = self.state.read().unwrap();
                    for (name, id) in op.requires {
                        if let Some(r) = state.get(&id) {
                            captured.insert(name, r.clone());
                        }
                    }

                    ready.push((
                        value_id.clone(),
                        op.action,
                        self.clone()
                            .extend(Link::to(&format!("/{}", value_id))?, captured),
                    ));
                } else {
                    queue.push(value_id);
                    break;
                }
            }

            if ready.is_empty() {
                return Err(error::bad_request(
                    "Transaction graph stalled before completing",
                    format!("{:?}", queue),
                ));
            }

            let results = try_join_all(
                ready
                    .iter()
                    .map(|(_, action, txn)| self.host.clone().post(txn.clone(), action)),
            )
            .await?;
            let mut state = self.state.write().unwrap();
            for i in 0..results.len() {
                state.insert(ready[i].0.to_string(), results[i].clone());
            }
        }

        let state = self.state.read().unwrap();
        let mut responses: HashMap<String, TCState> = HashMap::new();
        for value_id in capture {
            match state.get(&value_id) {
                Some(r) => {
                    responses.insert(value_id, r.clone());
                }
                None => {
                    return Err(error::bad_request(
                        "Tried to capture unknown value",
                        value_id,
                    ));
                }
            }
        }

        Ok(responses)
    }

    pub fn require<T: DeserializeOwned>(self: Arc<Self>, value_id: &str) -> TCResult<T> {
        match self.state.read().unwrap().get(value_id) {
            Some(response) => match response {
                TCState::Value(value) => Ok(serde_json::from_str(&serde_json::to_string(&value)?)?),
                other => Err(error::bad_request(
                    &format!("Required value {} is not serialiable", value_id),
                    other,
                )),
            },
            None => Err(error::bad_request(
                &format!("{}: no value was provided for", self.context),
                value_id,
            )),
        }
    }

    pub async fn get(self: Arc<Self>, path: Link) -> TCResult<TCState> {
        self.host.clone().get(self.clone(), path).await
    }

    pub async fn put(self: Arc<Self>, path: Link, state: TCState) -> TCResult<TCState> {
        self.host.clone().put(self.clone(), path, state).await
    }

    pub async fn post(
        self: Arc<Self>,
        path: &Link,
        args: Vec<(&str, TCValue)>,
    ) -> TCResult<TCState> {
        let txn = self.clone().extend(
            self.context.clone(),
            args.iter()
                .map(|(k, v)| ((*k).to_string(), TCState::Value(v.clone())))
                .collect(),
        );

        self.host.clone().post(txn, path).await
    }
}
