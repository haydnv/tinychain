use std::fmt;

use bytes::Bytes;
use futures::future::{self, TryFutureExt};
use futures::stream::{FuturesUnordered, StreamExt};
use log::*;
use safecast::{TryCastFrom, TryCastInto};

use tc_error::*;
use tc_transact::{Transact, Transaction};
use tc_value::{Link, LinkHost, Value};
use tcgeneric::{label, PathSegment, TCPath, TCPathBuf, Tuple};

use crate::cluster::{Cluster, Replica, REPLICAS};
use crate::object::InstanceClass;
use crate::state::State;
use crate::txn::Txn;

use super::{DeleteHandler, GetHandler, Handler, PostHandler, Public, PutHandler, Route};

mod class;
mod dir;
mod library;
mod service;

pub struct ClusterHandler<'a, T> {
    cluster: &'a Cluster<T>,
}

impl<'a, T> Handler<'a> for ClusterHandler<'a, T>
where
    T: Transact + Public + Send + Sync,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(move |_txn, key| {
            Box::pin(future::ready((|key: Value| {
                trace!("GET key {} from {}", key, self.cluster);

                if key.is_none() {
                    // return the public key of this replica
                    let public_key = Bytes::from(self.cluster.public_key().to_vec());
                    return Ok(Value::from(public_key).into());
                }

                // TODO: remove this code and use the public key of the gateway instead

                let key = TCPathBuf::try_cast_from(key, |v| {
                    TCError::bad_request("invalid key specification", v)
                })?;

                if key == TCPathBuf::default() {
                    // return the public key of this cluster
                    let public_key = Bytes::from(self.cluster.schema().public_key().to_vec());
                    Ok(Value::from(public_key).into())
                } else {
                    Err(TCError::not_found(key))
                }
            })(key)))
        }))
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key, value| {
            Box::pin(async move {
                if key.is_none() {
                    // this is a notification of a new participant in the current transaction

                    let participant = value.try_cast_into(|v| {
                        TCError::bad_request("expected a participant Link but found", v)
                    })?;

                    return self.cluster.mutate(&txn, participant).await;
                }

                // this is a request to install a new cluster

                let value = if InstanceClass::can_cast_from(&value) {
                    let class = InstanceClass::try_cast_from(value, |v| {
                        TCError::bad_request("invalid class definition", v)
                    })?;

                    let link = class.extends();

                    if link.host().is_some() && link.host() != self.cluster.schema().lead() {
                        return Err(TCError::not_implemented(
                            "install a Cluster with a different lead replica",
                        ));
                    }

                    if !link.path().starts_with(self.cluster.path()) {
                        return Err(bad_request!(
                            "cannot install {} at {}",
                            link,
                            self.cluster.link().path()
                        ));
                    }

                    State::Object(class.into())
                } else {
                    value
                };

                self.cluster.state().put(&txn, &[], key.into(), value).await
            })
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, params| {
            Box::pin(async move {
                // TODO: authorize request using a scope

                if !params.is_empty() {
                    return Err(TCError::bad_request(
                        "unrecognized commit parameters",
                        params,
                    ));
                }

                if let Some(owner) = txn.owner() {
                    if owner.host() == Some(txn.host()) && owner.path() == self.cluster.path() {
                        return Err(TCError::bad_request(
                            "cluster received a commit message for itself",
                            self.cluster,
                        ));
                    }
                } else {
                    return Err(TCError::bad_request(
                        "commit message for an ownerless transaction",
                        txn.id(),
                    ));
                }

                #[cfg(debug_assertions)]
                info!("{} got commit message for {}", self.cluster, txn.id());

                let result = if !txn.has_leader(self.cluster.path()) {
                    // in this case, the kernel did not claim leadership
                    // since a POST request is not necessarily a write
                    // but there's no need to notify the txn owner
                    // because it has already sent a commit message
                    // so just claim leadership on this host and replicate the commit
                    info!(
                        "{} will lead and distribute the commit of {}...",
                        self.cluster,
                        txn.id()
                    );

                    self.cluster.lead_and_distribute_commit(txn.clone()).await
                } else if txn.is_leader(self.cluster.path()) {
                    info!(
                        "{} will distribute the commit of {}...",
                        self.cluster,
                        txn.id()
                    );

                    self.cluster.distribute_commit(txn).await
                } else {
                    info!("{} will commit {}...", self.cluster, txn.id());
                    self.cluster.commit(*txn.id()).await;
                    Ok(())
                };

                if result.is_ok() {
                    info!("{} commit {} succeeded", self.cluster, txn.id());
                } else {
                    info!("{} commit {} failed", self.cluster, txn.id());
                }

                result.map(State::from)
            })
        }))
    }

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                key.expect_none()?;

                if txn.is_leader(self.cluster.path()) {
                    self.cluster.distribute_rollback(txn).await;
                } else {
                    self.cluster.rollback(txn.id()).await;
                }

                Ok(())
            })
        }))
    }
}

impl<'a, T> From<&'a Cluster<T>> for ClusterHandler<'a, T> {
    fn from(cluster: &'a Cluster<T>) -> Self {
        Self { cluster }
    }
}

struct ReplicaHandler<'a, T> {
    cluster: &'a Cluster<T>,
}

impl<'a, T> Handler<'a> for ReplicaHandler<'a, T>
where
    T: Replica + Send + Sync,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                key.expect_none()?;

                self.cluster
                    .replicas(*txn.id())
                    .map_ok(|replicas| Value::Tuple(replicas.iter().cloned().collect()))
                    .map_ok(State::from)
                    .await
            })
        }))
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key, link| {
            Box::pin(async move {
                key.expect_none()?;

                let link = link.try_cast_into(|v| {
                    TCError::bad_request("expected a Link to a Cluster, not", v)
                })?;

                self.cluster.add_replica(txn, link).await?;

                Ok(())
            })
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, mut params| {
            Box::pin(async move {
                let new_replica = params.require::<LinkHost>(&label("add").into())?;
                params.expect_empty()?;

                let txn_id = *txn.id();

                if self.cluster.add_replica(txn, new_replica.clone()).await? {
                    let replicas = self.cluster.replicas(txn_id).await?;

                    let mut requests = replicas
                        .iter()
                        .filter(|replica| *replica != txn.host() && *replica != &new_replica)
                        .map(|replica| {
                            txn.put(
                                self.cluster.schema().link_to(replica).append(REPLICAS),
                                Value::default(),
                                new_replica.clone(),
                            )
                        })
                        .collect::<FuturesUnordered<_>>();

                    while let Some(result) = requests.next().await {
                        if let Err(cause) = result {
                            warn!("failed to propagate add replica request: {}", cause);
                        }
                    }
                }

                let state = self.cluster.state().state(txn_id).await?;
                debug!("state of source replica {} is {}", self.cluster, state);
                Ok(state)
            })
        }))
    }

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, replicas| {
            Box::pin(async move {
                let replicas = Tuple::<LinkHost>::try_cast_from(replicas, |v| {
                    TCError::bad_request("expected a Link to a Cluster, not", v)
                })?;

                self.cluster.remove_replicas(*txn.id(), &replicas).await
            })
        }))
    }
}

impl<'a, T> From<&'a Cluster<T>> for ReplicaHandler<'a, T> {
    fn from(cluster: &'a Cluster<T>) -> Self {
        Self { cluster }
    }
}

impl<T> Route for Cluster<T>
where
    T: Replica + Route + Transact + fmt::Display + Send + Sync,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        trace!("Cluster::route {}", TCPath::from(path));

        match path {
            path if path.is_empty() => Some(Box::new(ClusterHandler::from(self))),
            path if path == &[REPLICAS] => Some(Box::new(ReplicaHandler::from(self))),
            path => self.state().route(path),
        }
    }
}

fn authorize_install(txn: &Txn, parent: &Link, entry_path: &TCPathBuf) -> TCResult<()> {
    debug!(
        "check authorization to install {} at {}",
        entry_path, parent
    );

    let replicate_from_this_host = if let Some(lead) = parent.host() {
        lead == txn.host()
    } else {
        true
    };

    if replicate_from_this_host {
        // this is a new install, make sure the request was signed with the cluster's private key

        let parent = if parent.host().is_some() {
            parent.clone()
        } else {
            (txn.host().clone(), parent.path().clone()).into()
        };

        let authorized = txn
            .request()
            .scopes()
            .get(&parent, &TCPathBuf::default().into())
            .ok_or_else(|| unauthorized!("install request for {}", parent))?;

        if authorized.iter().any(|scope| entry_path.starts_with(scope)) {
            Ok(())
        } else {
            Err(forbidden!("install a new Cluster at {}", entry_path))
        }
    } else {
        // this is a replica, it doesn't make sense to require a recently-signed token
        Ok(())
    }
}
