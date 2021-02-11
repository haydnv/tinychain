use std::time::*;

use async_trait::async_trait;
use futures::TryFutureExt;

use generic::{path_label, NetworkTime, PathLabel, TCPathBuf};
use transact::TxnId;

use crate::gateway::Gateway;
use crate::scalar::{Link, Value};

pub type Actor = rjwt::Actor<Value>;
pub type Claims = rjwt::Claims<Value, Vec<Scope>>;
pub type Scope = TCPathBuf;
pub type Token = rjwt::Token<Value, Vec<Scope>>;

pub const SCOPE_ROOT: PathLabel = path_label(&[]);

pub trait TokenExt {
    fn new(
        host: Link,
        time: NetworkTime,
        ttl: Duration,
        actor_id: Value,
        scopes: Vec<Scope>,
    ) -> Token;
}

impl TokenExt for Token {
    fn new(
        host: Link,
        time: NetworkTime,
        ttl: Duration,
        actor_id: Value,
        scopes: Vec<Scope>,
    ) -> Token {
        let host = host.to_string();
        let time = UNIX_EPOCH + Duration::from_nanos(time.as_nanos() as u64);

        Token::new(host, time, ttl, actor_id, scopes)
    }
}

pub struct Request {
    pub token: Token,
    pub claims: Claims,
    pub txn_id: TxnId,
}

impl Request {
    pub fn new(txn_id: TxnId, token: Token, claims: Claims) -> Self {
        Self {
            token,
            claims,
            txn_id,
        }
    }

    pub fn token(&self) -> &Token {
        &self.token
    }
}

pub struct Resolver<'a> {
    gateway: &'a Gateway,
    txn_id: &'a TxnId,
}

impl<'a> Resolver<'a> {
    pub fn new(gateway: &'a Gateway, txn_id: &'a TxnId) -> Self {
        Self { gateway, txn_id }
    }
}

#[async_trait]
impl<'a> rjwt::Resolve for Resolver<'a> {
    type Host = Link;
    type ActorId = Value;
    type Claims = Vec<Scope>;

    fn host(&self) -> Link {
        self.gateway.root().clone().into()
    }

    async fn resolve(&self, host: &Link, actor_id: &Value) -> Result<Actor, rjwt::Error> {
        let public_key: String = self
            .gateway
            .fetch(&self.txn_id, host, actor_id)
            .map_err(|e| rjwt::Error::new(rjwt::ErrorKind::Fetch, e))
            .await?;

        let public_key = base64::decode(&public_key).map_err(|e| {
            rjwt::Error::new(
                rjwt::ErrorKind::Format,
                format!("invalid public key {} for {}: {}", &public_key, actor_id, e),
            )
        })?;
        Actor::with_public_key(actor_id.clone(), &public_key)
    }
}
