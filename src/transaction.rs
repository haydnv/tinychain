use std::sync::Arc;

use rand::Rng;

use crate::context::*;
use crate::error;
use crate::host::HostContext;

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
    pub fn new(host: Arc<HostContext>) -> Transaction {
        Transaction {
            id: TransactionId::new(host.time()),
            parent: host,
        }
    }

    pub fn resolve(&self) -> TCResult<TCValue> {
        Err(error::not_implemented())
    }
}
