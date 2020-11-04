use std::time::Duration;

use crate::auth::Token;
use crate::transaction::TxnId;

#[derive(Clone)]
pub struct Request {
    auth: Option<Token>,
    ttl: Duration,
    txn_id: Option<TxnId>,
}

impl Request {
    pub fn new(ttl: Duration, auth: Option<Token>, txn_id: Option<TxnId>) -> Self {
        Request { auth, ttl, txn_id }
    }

    pub fn auth(&'_ self) -> &'_ Option<Token> {
        &self.auth
    }

    pub fn ttl(&self) -> Duration {
        self.ttl
    }

    pub fn txn_id(&self) -> &'_ Option<TxnId> {
        &self.txn_id
    }
}
