use bytes::Bytes;
use futures::future::try_join_all;
use futures::{future, TryFutureExt};
use log::{debug, info};
use safecast::{TryCastFrom, TryCastInto};

use tc_error::*;
use tc_transact::{Transact, Transaction};
use tc_value::{Link, Value};
use tcgeneric::{label, PathSegment, Tuple};

use crate::chain::BlockChain;
use crate::cluster::dir::Dir;
use crate::cluster::{Class, Cluster, Legacy, Library, Replica, Service, REPLICAS};
use crate::object::{InstanceClass, Object};
use crate::state::State;

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
                key.expect_none()?;
                let public_key = Bytes::from(self.cluster.public_key().to_vec());
                Ok(Value::from(public_key).into())
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
                    let participant = value
                        .try_cast_into(|v| TCError::bad_request("invalid participant link", v))?;

                    return self.cluster.mutate(&txn, participant).await;
                }

                let value = if InstanceClass::can_cast_from(&value) {
                    InstanceClass::try_cast_from(value, |v| {
                        TCError::bad_request("invalid class definition", v)
                    })
                    .map(Object::Class)
                    .map(State::Object)?
                } else {
                    value
                };

                self.cluster.state().put(&txn, &[], key, value).await
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

                if txn.is_owner(self.cluster.path()) {
                    return Err(TCError::internal(format!(
                        "{} got commit message for itself",
                        txn.link(self.cluster.link().path().clone())
                    )));
                }

                #[cfg(debug_assertions)]
                info!(
                    "{} got commit message for {}",
                    txn.link(self.cluster.link().path().clone()),
                    txn.id()
                );

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
                    self.cluster.commit(txn.id()).await;
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

                let replicas = self
                    .cluster
                    .replicas(*txn.id())
                    .map_ok(|replicas| Value::Tuple(replicas.iter().cloned().collect()))
                    .map_ok(State::from)
                    .await?;

                Ok(State::Tuple(
                    (self.cluster.link().clone().into(), replicas).into(),
                ))
            })
        }))
    }

    // TODO: delete
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

                self.cluster.add_replica(txn, link, true).await?;

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
                let new_replica = params.require::<Link>(&label("add").into())?;
                params.expect_empty()?;

                let txn_id = *txn.id();
                let self_link = txn.link(self.cluster.link().path().clone());
                if self
                    .cluster
                    .add_replica(txn, new_replica.clone(), false)
                    .await?
                {
                    let replicas = self.cluster.replicas(txn_id).await?;
                    let requests = replicas
                        .iter()
                        .cloned()
                        .filter(|replica| replica != &self_link && replica != &new_replica)
                        .map(|replica| {
                            txn.put(
                                replica.append(REPLICAS),
                                Value::default(),
                                new_replica.clone().into(),
                            )
                        });

                    try_join_all(requests).await?;
                }

                let state = self.cluster.state().state(txn_id).await?;
                debug!("state of source replica {} is {}", self_link, state);
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
                let replicas = Tuple::<Link>::try_cast_from(replicas, |v| {
                    TCError::bad_request("expected a Link to a Cluster, not", v)
                })?;

                self.cluster.remove_replicas(txn, &replicas).await
            })
        }))
    }
}

impl<'a, T> From<&'a Cluster<T>> for ReplicaHandler<'a, T> {
    fn from(cluster: &'a Cluster<T>) -> Self {
        Self { cluster }
    }
}

// TODO: replace with a single impl
macro_rules! route_cluster {
    ($t:ty) => {
        impl Route for Cluster<$t> {
            fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
                match path {
                    path if path.is_empty() => Some(Box::new(ClusterHandler::from(self))),
                    path if path == &[REPLICAS] => Some(Box::new(ReplicaHandler::from(self))),
                    path => self.state().route(path),
                }
            }
        }
    };
}

route_cluster!(BlockChain<Class>);
route_cluster!(Dir<Class>);
route_cluster!(BlockChain<Library>);
route_cluster!(Dir<Library>);
route_cluster!(BlockChain<Service>);
route_cluster!(Dir<Service>);
route_cluster!(Legacy);

impl Route for Legacy {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if let Some(chain) = self.chain(&path[0]) {
            debug!("Legacy cluster has a Chain at {}", &path[0]);
            chain.route(&path[1..])
        } else if let Some(class) = self.class(&path[0]) {
            debug!("Legacy cluster has a Class at {}", &path[0]);
            class.route(&path[1..])
        } else {
            None
        }
    }
}
