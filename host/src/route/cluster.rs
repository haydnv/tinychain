use bytes::Bytes;
use futures::{future, TryFutureExt};
use log::debug;
use safecast::{TryCastFrom, TryCastInto};

use tc_error::*;
use tc_transact::{Transact, Transaction};
use tcgeneric::{label, Id, TCPath, Tuple};

use crate::cluster::Cluster;
use crate::route::*;
use crate::scalar::{Link, Value};
use crate::state::State;
use crate::txn::Txn;

struct AuthorizeHandler<'a> {
    cluster: &'a Cluster,
}

impl<'a> Handler<'a> for AuthorizeHandler<'a> {
    fn get(self: Box<Self>) -> Option<GetHandler<'a>> {
        Some(Box::new(|txn, scope| {
            Box::pin(async move {
                let scope = scope
                    .try_cast_into(|v| TCError::bad_request("expected an auth scope, not", v))?;

                self.cluster
                    .authorize(&txn, &scope)
                    .map_ok(State::from)
                    .await
            })
        }))
    }
}

impl<'a> From<&'a Cluster> for AuthorizeHandler<'a> {
    fn from(cluster: &'a Cluster) -> Self {
        Self { cluster }
    }
}

pub struct ClusterHandler<'a> {
    cluster: &'a Cluster,
}

impl<'a> ClusterHandler<'a> {
    fn handle_get(self, key: Value) -> TCResult<State> {
        debug!("Cluster::get {}", key);

        if key.is_some() {
            let key: Id = key.try_cast_into(|v| TCError::bad_request("invalid ID", v))?;
            self.cluster
                .chain(&key)
                .cloned()
                .map(State::from)
                .ok_or_else(|| TCError::not_found(key))
        } else {
            let public_key = Bytes::from(self.cluster.public_key().to_vec());
            Ok(Value::from(public_key).into())
        }
    }

    async fn handle_put(self, txn: Txn, peer: Link) -> TCResult<()> {
        let owner = self.cluster.owner(txn.id()).await?;
        owner.mutate(peer).await
    }
}

impl<'a> Handler<'a> for ClusterHandler<'a> {
    fn get(self: Box<Self>) -> Option<GetHandler<'a>> {
        Some(Box::new(|_txn, key| {
            Box::pin(future::ready(self.handle_get(key)))
        }))
    }

    fn put(self: Box<Self>) -> Option<PutHandler<'a>> {
        Some(Box::new(|txn, key, value| {
            Box::pin(async move {
                if key.is_some() {
                    return Err(TCError::unsupported("a Cluster itself is immutable"));
                }

                let peer =
                    value.try_cast_into(|s| TCError::bad_request("expected a Link, not", s))?;

                self.handle_put(txn, peer).await
            })
        }))
    }

    fn post(self: Box<Self>) -> Option<PostHandler<'a>> {
        Some(Box::new(|txn, params| {
            Box::pin(async move {
                // TODO: authorize request using a scope

                if !params.is_empty() {
                    return Err(TCError::bad_request(
                        "unrecognized commit parameters",
                        params,
                    ));
                }

                self.cluster.commit(txn.id()).await;
                Ok(State::default())
            })
        }))
    }
}

impl<'a> From<&'a Cluster> for ClusterHandler<'a> {
    fn from(cluster: &'a Cluster) -> Self {
        Self { cluster }
    }
}

struct GrantHandler<'a> {
    cluster: &'a Cluster,
}

impl<'a> Handler<'a> for GrantHandler<'a> {
    fn post(self: Box<Self>) -> Option<PostHandler<'a>> {
        Some(Box::new(|txn, mut params| {
            Box::pin(async move {
                let scope = params
                    .remove(&label("scope").into())
                    .ok_or_else(|| TCError::bad_request("missing required parameter", "scope"))?
                    .try_cast_into(|s| TCError::bad_request("invalid auth scope", s))?;

                let op = params
                    .remove(&label("op").into())
                    .ok_or_else(|| TCError::bad_request("missing required parameter", "op"))?
                    .try_cast_into(|s| TCError::bad_request("grantee must be an OpRef, not", s))?;

                let context = if let Some(context) = params.remove(&label("context").into()) {
                    context.try_cast_into(|v| {
                        TCError::bad_request("expected a Map of parameters, not", v)
                    })?
                } else {
                    Map::default()
                };

                self.cluster.grant(txn, scope, op, context).await
            })
        }))
    }
}

impl<'a> From<&'a Cluster> for GrantHandler<'a> {
    fn from(cluster: &'a Cluster) -> Self {
        Self { cluster }
    }
}

struct InstallHandler<'a> {
    cluster: &'a Cluster,
}

impl<'a> Handler<'a> for InstallHandler<'a> {
    fn put(self: Box<Self>) -> Option<PutHandler<'a>> {
        Some(Box::new(|txn, link, scopes| {
            Box::pin(async move {
                let link = link.try_cast_into(|v| {
                    TCError::bad_request("install requires a Link to a Cluster, not", v)
                })?;

                let scopes = Tuple::try_cast_from(scopes, |v| {
                    TCError::bad_request("expected a list of authorization scopes, not", v)
                })?;

                self.cluster
                    .install(*txn.id(), link, scopes.into_iter().collect())
                    .await
            })
        }))
    }
}

impl<'a> From<&'a Cluster> for InstallHandler<'a> {
    fn from(cluster: &'a Cluster) -> Self {
        Self { cluster }
    }
}

struct ReplicaHandler<'a> {
    cluster: &'a Cluster,
}

impl<'a> Handler<'a> for ReplicaHandler<'a> {
    fn put(self: Box<Self>) -> Option<PutHandler<'a>> {
        Some(Box::new(|txn, key, link| {
            Box::pin(async move {
                assert!(key.is_none());

                let link = link.try_cast_into(|v| {
                    TCError::bad_request("expected a Link to a Cluster, not", v)
                })?;

                self.cluster.add_replica(*txn.id(), link).await
            })
        }))
    }
}

impl<'a> From<&'a Cluster> for ReplicaHandler<'a> {
    fn from(cluster: &'a Cluster) -> Self {
        Self { cluster }
    }
}

impl Route for Cluster {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        debug!("Cluster::route {}", TCPath::from(path));

        if path.is_empty() {
            Some(Box::new(ClusterHandler::from(self)))
        } else if let Some(chain) = self.chain(&path[0]) {
            chain.route(&path[1..])
        } else if let Some(class) = self.class(&path[0]) {
            class.route(&path[1..])
        } else if path.len() == 1 {
            match path[0].as_str() {
                "authorize" => Some(Box::new(AuthorizeHandler::from(self))),
                "grant" => Some(Box::new(GrantHandler::from(self))),
                "install" => Some(Box::new(InstallHandler::from(self))),
                "replica" => Some(Box::new(ReplicaHandler::from(self))),
                _ => None,
            }
        } else {
            None
        }
    }
}
