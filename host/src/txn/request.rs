use async_trait::async_trait;
use futures::TryFutureExt;

use generic::{path_label, PathLabel, TCPathBuf};
use transact::TxnId;

use crate::gateway::Gateway;
use crate::scalar::{Link, Value};

pub type Actor = rjwt::Actor<Value>;
pub type Claims = rjwt::Claims<Link, Value, Vec<Scope>>;
pub type Scope = TCPathBuf;
pub type Token = rjwt::Token<Link, Value, Vec<Scope>>;

pub const SCOPE_ROOT: PathLabel = path_label(&[]);

pub struct Request {
    pub token: String,
    pub claims: Claims,
    pub txn_id: TxnId,
}

impl Request {
    pub fn new(txn_id: TxnId, token: String, claims: Claims) -> Self {
        Self {
            token,
            claims,
            txn_id,
        }
    }

    pub fn scopes(&self) -> &Claims {
        &self.claims
    }

    pub fn token(&self) -> &str {
        &self.token
    }
}

pub struct Resolver<'a> {
    host: &'a Link,
    txn_id: &'a TxnId,
}

impl<'a> Resolver<'a> {
    pub fn new(host: &'a Link, txn_id: &'a TxnId) -> Self {
        Self { host, txn_id }
    }
}

#[async_trait]
impl<'a> rjwt::Resolve for Resolver<'a> {
    type Host = Link;
    type ActorId = Value;
    type Claims = Vec<Scope>;

    fn host(&self) -> Link {
        self.host.clone()
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
