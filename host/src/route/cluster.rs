use bytes::Bytes;
use futures::{future, TryFutureExt};
use log::debug;
use safecast::{TryCastFrom, TryCastInto};

use tc_error::*;
use tc_transact::Transaction;
use tc_value::{Link, Value};
use tcgeneric::Tuple;

use crate::cluster::{Cluster, Legacy};
use crate::route::*;
use crate::state::State;

pub struct ClusterHandler<'a, C> {
    cluster: &'a C,
}

impl<'a, C> Handler<'a> for ClusterHandler<'a, C>
where
    C: Cluster,
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
                if key.is_some() {
                    return Err(TCError::unsupported("a Cluster itself is immutable"));
                }

                let participant =
                    value.try_cast_into(|s| TCError::bad_request("expected a Link, not", s))?;

                self.cluster.mutate(&txn, participant).await
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

                if txn.is_leader(self.cluster.path()) {
                    self.cluster.distribute_commit(txn).await?;
                } else {
                    self.cluster.commit(txn.id()).await;
                }

                Ok(State::default())
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
                    self.cluster.finalize(txn.id()).await;
                }

                Ok(())
            })
        }))
    }
}

impl<'a> From<&'a Legacy> for ClusterHandler<'a, Legacy> {
    fn from(cluster: &'a Legacy) -> Self {
        Self { cluster }
    }
}

struct ReplicaHandler<'a, C> {
    cluster: &'a C,
}

impl<'a, C> Handler<'a> for ReplicaHandler<'a, C>
where
    C: Cluster,
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

                self.cluster.add_replica(&txn, link).await
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

impl<'a> From<&'a Legacy> for ReplicaHandler<'a, Legacy> {
    fn from(cluster: &'a Legacy) -> Self {
        Self { cluster }
    }
}

impl Route for Legacy {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.is_empty() {
            Some(Box::new(ClusterHandler::from(self)))
        } else if let Some(chain) = self.chain(&path[0]) {
            debug!("Cluster has a Chain at {}", &path[0]);
            chain.route(&path[1..])
        } else if let Some(class) = self.class(&path[0]) {
            debug!("Cluster has a Class at {}", &path[0]);
            class.route(&path[1..])
        } else if path.len() == 1 {
            match path[0].as_str() {
                "replicas" => Some(Box::new(ReplicaHandler::from(self))),
                _ => None,
            }
        } else {
            None
        }
    }
}
