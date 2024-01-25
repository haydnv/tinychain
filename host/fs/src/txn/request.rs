//! Authorization. INCOMPLETE AND UNSTABLE.

use std::convert::TryInto;

use async_trait::async_trait;
use bytes::Bytes;
use futures::{FutureExt, TryFutureExt};
use safecast::TryCastInto;

use tc_error::*;
use tc_transact::public::StateInstance;
use tc_transact::TxnId;
use tc_value::{Link, Value};
use tcgeneric::{NetworkTime, TCPathBuf};

use crate::block::CacheBlock;

use super::{Gateway, Txn};

/// The type of [`rjwt::Actor`] used to sign auth tokens
pub type Actor = rjwt::Actor<Value>;

/// The type of [`rjwt::SignedToken`] used for authentication & authorization
pub type SignedToken = rjwt::SignedToken<Link, Value, Vec<Scope>>;

/// The type of scope communicated by [`Claims`]
pub type Scope = TCPathBuf;

/// The type of token used to authenticate requests between hosts
pub type Token = rjwt::Token<Link, Value, Vec<Scope>>;

/// A `Txn`'s authorization.
pub struct Request {
    token: SignedToken,
    txn_id: TxnId,
}

impl Request {
    /// Construct a new `Request`.
    pub fn new(txn_id: TxnId, token: SignedToken) -> Self {
        Self { token, txn_id }
    }

    pub fn expires(&self) -> TCResult<NetworkTime> {
        let expires = self.token.expires();

        expires
            .try_into()
            .map_err(|cause| bad_request!("invalid token expiration time").consume(cause))
    }

    /// Return this request's JSON web token (cf. the [`rjwt`] crate)
    pub fn token(&self) -> &SignedToken {
        &self.token
    }

    #[inline]
    pub fn txn_id(&self) -> &TxnId {
        &self.txn_id
    }
}

/// Struct responsible for resolving JWT auth identities (cf. the [`rjwt`] crate).
pub struct Resolver<'a, State> {
    gateway: &'a dyn Gateway<State = State>,
    txn_id: &'a TxnId,
}

impl<'a, State> Resolver<'a, State> {
    /// Construct a new `Resolver`.
    pub fn new(gateway: &'a dyn Gateway<State = State>, txn_id: &'a TxnId) -> Self {
        Self { gateway, txn_id }
    }
}

#[async_trait]
impl<'a, State> rjwt::Resolve for Resolver<'a, State>
where
    State: StateInstance<FE = CacheBlock, Txn = Txn<State>>,
{
    type HostId = Link;
    type ActorId = Value;
    type Claims = Vec<Scope>;

    async fn resolve(&self, host: &Link, actor_id: &Value) -> Result<Actor, rjwt::Error> {
        let public_key: Bytes = self
            .gateway
            .fetch(&self.txn_id, host.into(), actor_id)
            .map(|result| {
                result.and_then(|value| {
                    value.try_cast_into(|v| bad_request!("invalid public key: {v:?}"))
                })
            })
            .map_err(|e| rjwt::Error::fetch(e))
            .await?;

        Actor::with_public_key(actor_id.clone(), &public_key)
    }
}
