use std::collections::HashMap;
use std::sync::Arc;

use rand::Rng;

use crate::cache::{Map, Set, Value};
use crate::context::{TCResult, TCState, TCValue};
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

pub struct Transaction {
    id: TransactionId,
    host: Arc<Host>,
    known: Set<String>,
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
    fn of(id: TransactionId, host: Arc<Host>) -> Arc<Transaction> {
        Arc::new(Transaction {
            id,
            host,
            known: Set::new(),
            resolved: Map::new(),
            state: Value::of(State::Open),
        })
    }

    pub fn new(host: Arc<Host>) -> Arc<Transaction> {
        Self::of(TransactionId::new(host.time()), host)
    }

    pub async fn include(
        self: Arc<Self>,
        name: String,
        context: String,
        args: HashMap<String, TCValue>,
    ) -> TCResult<()> {
        if self.state.get() != State::Open {
            return Err(error::internal(
                "Attempted to extend a transaction already in progress",
            ));
        }

        let txn = Self::of(self.id.clone(), self.host.clone());
        for (name, arg) in args {
            txn.clone().provide(name, arg)?;
        }
        self.resolved
            .insert(name, self.host.clone().post(context, self.clone()).await?);

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

    pub fn require(self: Arc<Self>, name: &str) -> TCResult<Arc<TCState>> {
        match self.resolved.get(&name.to_string()) {
            Some(state) => Ok(state),
            None => Err(error::bad_request("Required value was not provided", name)),
        }
    }

    pub async fn resolve(&self, capture: Vec<&str>) -> TCResult<HashMap<String, TCValue>> {
        if self.state.get() != State::Open {
            return Err(error::internal(
                "Attempt to resolve the same transaction multiple times",
            ));
        }

        self.state.set(State::Closed);

        // TODO: handle asyncronous I/O

        self.state.set(State::Resolved);

        let mut results: HashMap<String, TCValue> = HashMap::new();
        for name in capture {
            let name = name.to_string();
            match self.resolved.get(&name) {
                Some(arc_ref) => match &*arc_ref.clone() {
                    TCState::Value(val) => {
                        results.insert(name, val.clone());
                    },
                    _ => {
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
