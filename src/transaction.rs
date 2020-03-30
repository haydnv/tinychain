use std::collections::HashMap;
use std::sync::Arc;

use rand::Rng;

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
}

impl Transaction {
    pub fn new(host: Arc<HostContext>) -> Arc<Transaction> {
        Arc::new(Transaction {
            id: TransactionId::new(host.time()),
            parent: host,
        })
    }

    pub fn extend(self: Arc<Self>, _name: String, _context: String, _op: TCOp) -> Arc<Transaction> {
        Arc::new(Transaction {
            id: self.id.clone(),
            parent: self.clone(),
        })
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
