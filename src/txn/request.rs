use std::time::Duration;

use auth::Token;

use super::TxnId;

#[derive(Clone)]
pub struct Request {
    auth: Option<Token>,
    ttl: Duration,
    txn_id: TxnId,
}

impl Request {
    pub fn new(ttl: Duration, auth: Option<Token>, txn_id: TxnId) -> Self {
        Request { auth, ttl, txn_id }
    }

    pub fn auth(&'_ self) -> &'_ Option<Token> {
        &self.auth
    }

    pub fn ttl(&self) -> Duration {
        self.ttl
    }

    pub fn txn_id(&self) -> &'_ TxnId {
        &self.txn_id
    }
}
