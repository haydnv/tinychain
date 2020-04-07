use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, RwLock};

use futures::future::try_join_all;
use rand::Rng;
use serde::de::DeserializeOwned;

use crate::context::*;
use crate::error;
use crate::host::Host;
use crate::value::{Link, Op, TCValue};

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

pub type ValueId = String;

pub struct Transaction {
    host: Arc<Host>,
    id: TransactionId,
    context: Link,
    nodes: HashMap<ValueId, TCValue>,
    state: RwLock<HashMap<String, TCResponse>>,
}

struct _Op {
    action: Link,
    requires: Vec<(String, ValueId)>,
}

impl Transaction {
    pub fn with(host: Arc<Host>, op: Op) -> TCResult<Arc<Transaction>> {
        let mut nodes: HashMap<ValueId, TCValue> = HashMap::new();
        for (value_id, provider) in op.requires() {
            if nodes.contains_key(&value_id) {
                return Err(error::bad_request("Duplicate value provided for", value_id));
            }

            nodes.insert(value_id.clone(), provider.clone());
        }

        let id = TransactionId::new(host.time());
        let context: Link = id.clone().into();
        Ok(Arc::new(Transaction {
            host,
            id,
            context,
            nodes,
            state: RwLock::new(HashMap::new()),
        }))
    }

    fn extend(
        self: Arc<Self>,
        context: Link,
        state: HashMap<String, TCResponse>,
    ) -> Arc<Transaction> {
        Arc::new(Transaction {
            host: self.host.clone(),
            id: self.id.clone(),
            context: self.context.append(context),
            nodes: HashMap::new(),
            state: RwLock::new(state),
        })
    }

    pub fn context(self: Arc<Self>) -> Link {
        self.context.clone()
    }

    pub fn id(self: Arc<Self>) -> TransactionId {
        self.id.clone()
    }

    fn known(self: Arc<Self>) -> HashSet<ValueId> {
        self.state
            .read()
            .unwrap()
            .keys()
            .cloned()
            .collect()
    }

    fn enqueue(
        self: Arc<Self>,
        mut unvisited: Vec<(ValueId, Op)>,
        queue: &mut Vec<(ValueId, _Op)>,
    ) -> TCResult<()> {
        let mut visited: HashSet<ValueId> = HashSet::new();
        let mut state = self.state.write().unwrap();
        while let Some((value_id, op)) = unvisited.pop() {
            if visited.contains(&value_id) {
                continue;
            } else {
                queue.push((value_id.clone(), _Op {
                    action: op.action(),
                    requires: op.requires().iter().map(|(id, _)| (id.clone(), id.clone())).collect()
                }));
                visited.insert(value_id.clone());
            }

            for (id, provider) in op.requires() {
                match provider {
                    TCValue::Op(dep) => {
                        unvisited.push((id, dep.clone()));
                    }
                    value => {
                        state.insert(id.clone(), TCResponse::Value(value.clone()));
                    }
                }
            }
        }

        Ok(())
    }

    pub async fn execute<'a>(
        self: Arc<Self>,
        capture: HashSet<ValueId>,
    ) -> TCResult<HashMap<ValueId, TCResponse>> {
        // TODO: add a TCValue::Ref type and it to support discrete namespaces for each child txn

        let mut unvisited: Vec<(ValueId, Op)> = vec![];
        for value_id in &capture {
            if let Some(provider) = self.nodes.get(value_id) {
                match provider {
                    TCValue::Op(op) => {
                        unvisited.push((value_id.clone(), op.clone()));
                    }
                    value => {
                        println!("captured {}", value_id);
                        self.state
                            .write()
                            .unwrap()
                            .insert(value_id.to_string(), TCResponse::Value(value.clone()));
                    }
                }
            } else {
                return Err(error::bad_request("No such value to capture", value_id));
            }
        }

        let mut queue: Vec<(ValueId, _Op)> = vec![];
        self.clone().enqueue(unvisited, &mut queue)?;

        while !queue.is_empty() {
            let known = self.clone().known();
            let mut ready: Vec<(ValueId, Link, Arc<Transaction>)> = vec![];
            while let Some((value_id, op)) = queue.pop() {
                if known.is_superset(&op.requires.iter().map(|(_, id)| id.clone()).collect()) {
                    let mut captured: HashMap<String, TCResponse> = HashMap::new();
                    let state = self.state.read().unwrap();
                    for (name, id) in op.requires {
                        if let Some(r) = state.get(&id) {
                            captured.insert(name, r.clone());
                        }
                    }

                    ready.push((
                        value_id.clone(),
                        op.action,
                        self.clone().extend(Link::to(&format!("/{}", value_id))?, captured),
                    ));
                } else {
                    println!("still need {:?}", op.requires);
                    queue.push((value_id, op));
                    break;
                }
            }

            if ready.is_empty() {
                return Err(error::bad_request(
                    "Transaction graph stalled before completing",
                    format!(
                        "{:?}",
                        queue.iter().map(|(id, _)| id).collect::<Vec<&ValueId>>()
                    ),
                ));
            }

            let results = try_join_all(
                ready
                    .iter()
                    .map(|(_, action, txn)| self.host.clone().post(txn.clone(), action.clone())),
            )
            .await?;
            let mut state = self.state.write().unwrap();
            for i in 0..results.len() {
                state.insert(ready[i].0.to_string(), results[i].clone());
            }
        }

        Ok(self
            .state
            .read()
            .unwrap()
            .iter()
            .filter(|(id, _)| capture.contains(*id))
            .map(|(id, r)| (id.clone(), r.clone()))
            .collect())
    }

    pub fn require<T: DeserializeOwned>(self: Arc<Self>, value_id: &str) -> TCResult<T> {
        match self.state.read().unwrap().get(value_id) {
            Some(response) => match response {
                TCResponse::Value(value) => {
                    Ok(serde_json::from_str(&serde_json::to_string(&value)?)?)
                }
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

    pub async fn get(self: Arc<Self>, path: Link) -> TCResult<TCResponse> {
        self.host.clone().get(self.clone(), path).await
    }

    pub async fn put(self: Arc<Self>, path: Link, value: TCValue) -> TCResult<()> {
        self.host.clone().put(self.clone(), path, value).await
    }
}
