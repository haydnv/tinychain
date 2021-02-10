use std::time::*;

use async_trait::async_trait;
use futures::TryFutureExt;

use generic::{NetworkTime, TCPathBuf};
use transact::TxnId;

use crate::scalar::{Link, Scalar, Value};
use crate::state::State;

use super::Txn;

pub type Actor = rjwt::Actor<Value>;
pub type Scope = TCPathBuf;
pub type Token = rjwt::Token<Value, Vec<Scope>>;

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
    pub auth: Token,
    pub txn_id: TxnId,
}

impl Request {
    pub fn new(auth: Token, txn_id: TxnId) -> Self {
        Self { auth, txn_id }
    }

    pub fn contains(&self, _other: &Self) -> bool {
        // TODO
        true
    }
}

struct Resolver {
    txn: Txn,
}

#[async_trait]
impl rjwt::Resolve for Resolver {
    type Host = Link;
    type ActorId = Value;
    type Claims = Vec<Scope>;

    async fn resolve(&self, host: &Link, actor_id: &Value) -> Result<Actor, rjwt::Error> {
        let public_key = self
            .txn
            .get(host.clone(), actor_id.clone())
            .map_err(|e| rjwt::Error::new(rjwt::ErrorKind::Fetch, e))
            .await?;

        if let State::Scalar(Scalar::Value(Value::Bytes(public_key))) = public_key {
            Actor::with_public_key(actor_id.clone(), &public_key)
        } else {
            Err(rjwt::Error::new(
                rjwt::ErrorKind::Fetch,
                format!("invalid public key for {}: {}", actor_id, public_key),
            ))
        }
    }
}
