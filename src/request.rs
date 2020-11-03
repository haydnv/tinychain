use std::time::Duration;

use crate::auth::Token;

#[derive(Clone)]
pub struct Request {
    auth: Option<Token>,
    ttl: Duration,
}

impl Request {
    pub fn new(ttl: Duration, auth: Option<Token>) -> Self {
        Request { auth, ttl }
    }

    pub fn auth(&'_ self) -> &'_ Option<Token> {
        &self.auth
    }

    pub fn ttl(&self) -> Duration {
        self.ttl
    }
}
