use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use rand::Rng;

use crate::cache::{Map, Set, Value};
use crate::context::{TCContext, TCOp, TCResult, TCState, TCValue};
use crate::error;
use crate::host::HostContext;

pub type Pending = (
    Vec<String>,
    Arc<dyn FnOnce(HashMap<String, TCState>) -> TCResult<Arc<TCState>> + Send + Sync>,
);

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

pub struct Transaction {
    id: TransactionId,
    parent: Arc<dyn TCContext>,
    known: Set<String>,
    queue: RwLock<Vec<Pending>>,
    resolved: Map<String, TCState>,
    state: Value<State>,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum State {
    Open,
    Closed,
    Resolved,
}

impl Transaction {
    fn of(id: TransactionId, parent: Arc<dyn TCContext>) -> Arc<Transaction> {
        Arc::new(Transaction {
            id,
            parent,
            known: Set::new(),
            queue: RwLock::new(vec![]),
            resolved: Map::new(),
            state: Value::of(State::Open),
        })
    }

    pub fn new(host: Arc<HostContext>) -> Arc<Transaction> {
        Self::of(TransactionId::new(host.time()), host)
    }

    pub async fn extend(self: Arc<Self>, name: String, context: String, op: TCOp) -> TCResult<()> {
        if self.state.get() != State::Open {
            return Err(error::internal(
                "Attempted to extend a transaction already in progress",
            ));
        }

        let txn = Self::of(self.id.clone(), self.clone());
        for (name, arg) in op.args() {
            txn.clone().provide(name, arg)?;
        }

        match &*self.parent.clone().get(context).await? {
            TCState::Table(table) => {
                let pending = table.clone().post(op.method())?;
                self.queue.write().unwrap().push(pending);
                self.known.insert(name);
            }
            TCState::Value(value) => {
                self.provide(name, value.clone())?;
            }
        }

        Ok(())
    }

    pub fn provide(self: Arc<Self>, name: String, value: TCValue) -> TCResult<()> {
        if self.state.get() != State::Open {
            return Err(error::internal(
                "Attempted to provide a value to a transaction already in progress",
            ));
        }

        if self.known.contains(&name) {
            Err(error::bad_request(
                "This transaction already contains a value called",
                name,
            ))
        } else {
            self.resolved.insert(name, Arc::new(TCState::Value(value)));
            Ok(())
        }
    }

    pub async fn resolve(&self, capture: Vec<&str>) -> TCResult<HashMap<String, TCValue>> {
        if self.state.get() != State::Open {
            return Err(error::internal(
                "Attempt to resolve the same transaction multiple times",
            ));
        }

        self.state.set(State::Closed);

        // TODO: resolve all child transactions

        self.state.set(State::Resolved);

        let mut results: HashMap<String, TCValue> = HashMap::new();
        for name in capture {
            let name = name.to_string();
            match self.resolved.get(&name) {
                Some(arc_ref) => match &*arc_ref {
                    TCState::Value(val) => {
                        results.insert(name, val.clone());
                    },
                    TCState::Table(_) => {
                        return Err(error::bad_request("The transaction completed successfully but some captured values could not be serialized", name))
                    }
                },
                None => {
                    return Err(error::bad_request(
                        "Attempted to read value not in transaction",
                        name,
                    ));
                }
            }
        }

        Ok(results)
    }
}

#[async_trait]
impl TCContext for Transaction {
    async fn get(self: Arc<Self>, name: String) -> TCResult<Arc<TCState>> {
        if self.resolved.contains_key(&name) {
            Ok(self.resolved.get(&name).unwrap())
        } else {
            self.parent.clone().get(name).await
        }
    }
}
