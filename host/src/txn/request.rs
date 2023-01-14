//! Authorization. INCOMPLETE AND UNSTABLE.

use std::convert::TryInto;

use async_trait::async_trait;
use bytes::Bytes;
use futures::TryFutureExt;

use tc_error::*;
use tc_transact::TxnId;
use tc_value::{Link, Value};
use tcgeneric::{NetworkTime, TCPathBuf};

use crate::gateway::Gateway;

/// The type of [`rjwt::Actor`] used to sign auth tokens
pub type Actor = rjwt::Actor<Value>;

/// The type of [`rjwt::Claims`] communicated in auth tokens
pub type Claims = rjwt::Claims<Link, Value, Vec<Scope>>;

/// The type of scope communicated by [`Claims`]
pub type Scope = TCPathBuf;

/// The type of token used to authenticate requests between hosts
pub type Token = rjwt::Token<Link, Value, Vec<Scope>>;

/// A `Txn`'s authorization.
pub struct Request {
    token: String,
    claims: Claims,
    txn_id: TxnId,
}

impl Request {
    /// Construct a new `Request`.
    pub fn new(txn_id: TxnId, token: String, claims: Claims) -> Self {
        Self {
            token,
            claims,
            txn_id,
        }
    }

    pub fn expires(&self) -> TCResult<NetworkTime> {
        self.claims
            .expires()
            .try_into()
            .map_err(|e| TCError::bad_request("invalid auth token expiry", e))
    }

    /// Return this request's authorizations.
    pub fn scopes(&self) -> &Claims {
        &self.claims
    }

    /// Return this request's JSON web token (cf. the [`rjwt`] crate)
    pub fn token(&self) -> &str {
        &self.token
    }

    #[inline]
    pub fn txn_id(&self) -> &TxnId {
        &self.txn_id
    }
}

/// Struct responsible for resolving JWT auth identities (cf. the [`rjwt`] crate).
pub struct Resolver<'a> {
    gateway: &'a Gateway,
    host: &'a Link,
    txn_id: &'a TxnId,
}

impl<'a> Resolver<'a> {
    /// Construct a new `Resolver`.
    pub fn new(gateway: &'a Gateway, host: &'a Link, txn_id: &'a TxnId) -> Self {
        Self {
            gateway,
            host,
            txn_id,
        }
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
        let public_key: Bytes = self
            .gateway
            .fetch(&self.txn_id, host, actor_id)
            .map_err(|e| rjwt::Error::new(rjwt::ErrorKind::Fetch, e))
            .await?;

        Actor::with_public_key(actor_id.clone(), &public_key)
    }
}
