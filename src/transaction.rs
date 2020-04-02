use std::fmt;
use std::sync::Arc;

use rand::Rng;

use crate::cache::Map;
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

    pub fn to_bytes(&self) -> Vec<u8> {
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
    id: TransactionId,
    host: Arc<Host>,
    context: Link,
    env: Map<String, TCState>,
}

impl Transaction {
    fn of(id: TransactionId, host: Arc<Host>, context: Link) -> Arc<Transaction> {
        Arc::new(Transaction {
            id,
            host,
            context,
            env: Map::new(),
        })
    }

    pub fn new(host: Arc<Host>) -> TCResult<Arc<Transaction>> {
        let id = TransactionId::new(host.time());
        let context = Link::to(&format!("/transaction/{}", id))?;
        Ok(Self::of(id, host, context))
    }

    pub fn context(self: Arc<Self>) -> Link {
        self.context.clone()
    }

    pub fn extend(self: Arc<Self>, new_context: Link) -> Arc<Transaction> {
        Transaction::of(self.id.clone(), self.host.clone(), new_context)
    }

    pub fn id(self: Arc<Self>) -> TransactionId {
        self.id.clone()
    }

    pub fn provide(self: Arc<Self>, name: String, value: TCValue) -> TCResult<Arc<Transaction>> {
        if self.env.contains_key(&name) {
            Err(error::bad_request(
                "This transaction already contains a value called",
                name,
            ))
        } else {
            self.env.insert(name, Arc::new(TCState::Value(value)));
            Ok(self)
        }
    }

    pub fn require(self: Arc<Self>, name: &str) -> TCResult<Arc<TCState>> {
        match self.env.get(&name.to_string()) {
            Some(state) => Ok(state),
            None => Err(error::bad_request("Required value was not provided", name)),
        }
    }

    pub async fn get(self: Arc<Self>) -> TCResult<Arc<TCState>> {
        self.host.clone().get(self.clone(), self.context()).await
    }

    pub async fn put(self: Arc<Self>, value: TCValue) -> TCResult<()> {
        self.host
            .clone()
            .put(self.clone(), self.context(), value)
            .await
    }

    pub async fn post(
        self: Arc<Self>,
        path: Link,
        args: Vec<(&str, TCValue)>,
    ) -> TCResult<Arc<TCState>> {
        // for POST, maintain the same context, so that the method executes in the caller's context
        let txn = self.clone().extend(self.context());
        for (name, val) in args {
            txn.clone().provide(name.to_string(), val)?;
        }

        txn.host.clone().post(txn, path).await
    }
}
