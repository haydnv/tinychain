use crate::auth::Token;

#[derive(Clone)]
pub struct Request {
    auth: Option<Token>,
    ttl: u32,
}

impl Request {
    pub fn new(ttl: u32, auth: Option<Token>) -> Self {
        Request { auth, ttl }
    }

    pub fn auth(&'_ self) -> &'_ Option<Token> {
        &self.auth
    }
}
