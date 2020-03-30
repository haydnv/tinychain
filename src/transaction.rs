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

#[derive(Clone, Copy)]
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

    pub fn extend(self: Arc<Self>, _name: String, _context: String, _op: TCOp) {
        let txn = Self::of(self.id.clone(), self.clone());
        self.stack.write().unwrap().push(txn);
    }

    pub fn provide(self: Arc<Self>, _name: String, _value: TCValue) -> TCResult<()> {
        Ok(())
    }

    pub async fn resolve(&self, _capture: Vec<&str>) -> TCResult<HashMap<String, TCValue>> {
        Err(error::not_implemented())
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
