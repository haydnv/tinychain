use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use rand::Rng;

use crate::cache::{Map, Value};
use crate::context::*;
use crate::error;
use crate::host::HostContext;

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
    resolved: Map<String, TCState>,
    state: Value<State>,
    stack: RwLock<Vec<Arc<Transaction>>>,
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
            resolved: Map::new(),
            state: Value::of(State::Open),
            stack: RwLock::new(vec![]),
        })
    }

    pub fn new(host: Arc<HostContext>) -> Arc<Transaction> {
        Self::of(TransactionId::new(host.time()), host)
    }

    pub fn extend(self: Arc<Self>, _name: String, _context: String, _op: TCOp) -> TCResult<()> {
        if self.state.get() != State::Open {
            return Err(error::internal(
                "Attempted to extend a transaction already in progress",
            ));
        }

        let txn = Self::of(self.id.clone(), self.clone());
        self.stack.write().unwrap().push(txn);
        Ok(())
    }

    pub fn provide(self: Arc<Self>, name: String, value: TCValue) -> TCResult<()> {
        if self.state.get() != State::Open {
            return Err(error::internal(
                "Attempted to provide a value to a transaction already in progress",
            ));
        }

        if self.resolved.contains_key(&name) {
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

impl TCContext for Transaction {
    fn post(
        self: Arc<Self>,
        _method: String,
        args: HashMap<String, TCValue>,
    ) -> TCResult<Arc<Transaction>> {
        for (name, value) in args {
            self.clone().provide(name, value)?;
        }

        Ok(self)
    }
}
