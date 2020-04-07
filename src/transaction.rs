use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, RwLock};

use rand::Rng;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::context::*;
use crate::error;
use crate::host::Host;

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

#[derive(Clone, Deserialize, Serialize, Hash)]
pub struct Op {
    action: Link,
    requires: Vec<(String, ValueId)>,
}

#[derive(Clone, Deserialize, Serialize, Hash)]
pub enum Provider {
    Op(Op),
    Value(TCValue),
}

#[derive(Clone, Deserialize, Serialize, Hash)]
pub struct Request {
    values: Vec<(ValueId, Provider)>,
}

pub struct Transaction {
    host: Arc<Host>,
    id: TransactionId,
    context: Link,
    nodes: HashMap<ValueId, Provider>,
    state: RwLock<HashMap<ValueId, TCResponse>>,
}

impl Transaction {
    pub fn from_request(host: Arc<Host>, request: Request) -> TCResult<Arc<Transaction>> {
        let mut nodes: HashMap<ValueId, Provider> = HashMap::new();
        let mut edges: Vec<(ValueId, ValueId)> = vec![];
        for (value_id, provider) in request.values {
            if nodes.contains_key(&value_id) {
                return Err(error::bad_request("Duplicate value provided for", value_id));
            }

            nodes.insert(value_id.clone(), provider.clone());
            if let Provider::Op(op) = provider {
                for (_, dep) in op.requires {
                    edges.push((dep, value_id.clone()));
                }
            }
        }

        for (dep, _) in &edges {
            if !nodes.contains_key(dep) {
                return Err(error::bad_request("Required value not provided", dep));
            }
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

    pub fn context(self: Arc<Self>) -> Link {
        self.context.clone()
    }

    pub fn id(self: Arc<Self>) -> TransactionId {
        self.id.clone()
    }

    fn extend(
        self: Arc<Self>,
        context: Link,
        deps: HashMap<ValueId, TCResponse>,
    ) -> Arc<Transaction> {
        Arc::new(Transaction {
            host: self.host.clone(),
            id: self.id.clone(),
            context: self.context.append(context),
            nodes: HashMap::new(),
            state: RwLock::new(deps),
        })
    }

    fn enqueue(
        self: Arc<Self>,
        mut unvisited: Vec<ValueId>,
        mut visited: HashSet<ValueId>,
        queue: &mut Vec<(ValueId, Op)>,
    ) -> TCResult<()> {
        while let Some(next) = unvisited.pop() {
            visited.insert(next.clone());
            match self.nodes.get(&next) {
                Some(provider) => match provider {
                    Provider::Value(val) => {
                        self.state.write().unwrap().insert(
                            next.clone(),
                            TCResponse::Value(val.clone()),
                        );
                    }
                    Provider::Op(op) => {
                        queue.push((next.clone(), op.clone()));
                        for (_, value_id) in &op.requires {
                            unvisited.push(value_id.clone())
                        }
                    }
                },
                None => {
                    return Err(error::bad_request("Required value is not provided", next));
                }
            }
        }

        Ok(())
    }

    pub async fn execute(
        self: Arc<Self>,
        capture: HashSet<&str>,
    ) -> TCResult<HashMap<ValueId, TCResponse>> {
        // BFS backward from the captured values to enqueue futures
        // then resolve them synchronously (in order)

        let capture: Vec<ValueId> = capture.iter().map(|s| (*s).to_string()).collect();
        let mut queue: Vec<(ValueId, Op)> = vec![];
        self.clone().enqueue(capture, HashSet::new(), &mut queue)?;
        while let Some((value_id, op)) = queue.pop() {
            // TODO: compute deps
            let mut deps: HashMap<ValueId, TCResponse> = HashMap::new();

            for (name_in_this_context, name_in_child_txn) in op.requires {
                match self.state.read().unwrap().get(&name_in_this_context) {
                    Some(r) => {
                        deps.insert(name_in_child_txn, r.clone());
                    }
                    None => {
                        return Err(error::bad_request(
                            "Required value not provided",
                            name_in_this_context,
                        ));
                    }
                }
            }

            let txn = self.clone().extend(value_context(&value_id)?, deps);
            let response = self.host.clone().post(txn, op.action).await?;
            self.state.write().unwrap().insert(value_id, response);
        }

        Err(error::not_implemented())
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

fn value_context(value_id: &str) -> TCResult<Link> {
    match Link::to(value_id) {
        Ok(value_id) => Ok(value_id),
        Err(cause) => Err(error::bad_request(
            &format!("{} is not a valid value name", value_id),
            cause,
        )),
    }
}
