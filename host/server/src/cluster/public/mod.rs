use std::fmt;
use std::sync::Arc;

use futures::Future;
use log::debug;
use rjwt::VerifyingKey;
use safecast::{TryCastFrom, TryCastInto};

use tc_error::*;
use tc_transact::hash::AsyncHash;
use tc_transact::public::*;
use tc_transact::{Gateway, Transact, Transaction};
use tc_value::{Host, Link, Value};
use tcgeneric::{PathSegment, Tuple};

use crate::txn::Txn;
use crate::State;

use super::{Cluster, REPLICAS};

mod class;
mod dir;
mod library;
#[cfg(feature = "service")]
mod service;

struct ClusterHandler<T> {
    cluster: Cluster<T>,
}

impl<'a, T> Handler<'a, State> for ClusterHandler<T>
where
    T: Public<State> + Transact + Send + Sync + fmt::Debug + 'a,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key: Value| {
            Box::pin(async move {
                if txn.has_claims() {
                    self.cluster.state().get(txn, &[], key).await
                } else {
                    let keyring = self.cluster.keyring(*txn.id())?;

                    if key.is_none() {
                        let keyring = keyring
                            .values()
                            .map(|public_key| Value::Bytes((*public_key.as_bytes()).into()))
                            .map(State::from)
                            .collect();

                        Ok(State::Tuple(keyring))
                    } else {
                        let key = Arc::<[u8]>::try_from(key)?;

                        if keyring
                            .values()
                            .any(|public_key| public_key.as_bytes() == &key[..])
                        {
                            Ok(State::from(Value::from(key)))
                        } else {
                            Err(not_found!("{key:?} (of {} keys)", keyring.len()))
                        }
                    }
                }
            })
        }))
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key, value| {
            Box::pin(async move {
                if txn.locked_by()?.is_some() {
                    if key.is_some() {
                        Err(TCError::unexpected(key, "empty commit message"))
                    } else if value.is_some() {
                        Err(TCError::unexpected(value, "empty commit message"))
                    } else {
                        self.cluster.replicate_commit(txn).await
                    }
                } else {
                    let txn = self.cluster.claim(txn.clone())?;

                    self.cluster
                        .state
                        .put(&txn, &[], key.clone(), value.clone())
                        .await?;

                    maybe_replicate(&self.cluster, &txn, |txn, link| {
                        let key = key.clone();
                        let value = value.clone();
                        async move { txn.put(link, key, value).await }
                    })
                    .await
                }
            })
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, params| {
            Box::pin(async move {
                let txn = self.cluster.claim(txn.clone())?;
                self.cluster.state.post(&txn, &[], params).await
            })
        }))
    }

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b, Txn>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if txn.locked_by()?.is_some() {
                    if key.is_some() {
                        Err(TCError::unexpected(key, "empty rollback message"))
                    } else {
                        self.cluster.replicate_rollback(txn).await
                    }
                } else {
                    let txn = self.cluster.claim(txn.clone())?;
                    self.cluster.state.delete(&txn, &[], key.clone()).await?;

                    maybe_replicate(&self.cluster, &txn, |txn, link| {
                        let key = key.clone();
                        async move { txn.delete(link, key).await }
                    })
                    .await
                }
            })
        }))
    }
}

impl<T> From<Cluster<T>> for ClusterHandler<T> {
    fn from(cluster: Cluster<T>) -> Self {
        Self { cluster }
    }
}

struct ReplicaSetHandler<T> {
    cluster: Cluster<T>,
}

impl<'a, T> Handler<'a, State> for ReplicaSetHandler<T>
where
    T: AsyncHash + Send + Sync + fmt::Debug + 'a,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                debug!("GET {:?} replicas", self.cluster);

                let keyring = self.cluster.keyring(*txn.id())?;

                if key.is_some() {
                    let key = Arc::<[u8]>::try_from(key)?;

                    for (host, public_key) in keyring.iter() {
                        if public_key.as_bytes() == &key[..] {
                            return Ok(Value::Link(host.clone().into()).into());
                        }
                    }

                    Ok(Value::None.into())
                } else {
                    let keyring = keyring
                        .iter()
                        .map(|(host, public_key)| {
                            (
                                Value::from(host.clone()),
                                Value::Bytes(Arc::new(*public_key.as_bytes())),
                            )
                        })
                        .map(|(host, public_key)| {
                            State::Tuple(vec![host.into(), public_key.into()].into())
                        })
                        .collect();

                    Ok(State::Tuple(keyring))
                }
            })
        }))
    }

    // TODO: require a correct hash of the state of the cluster in order to join
    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key, value| {
            Box::pin(async move {
                debug!("PUT {:?} replica {key}: {value:?}", self.cluster);

                let new_replica_hash = Value::try_from(value).and_then(Arc::<[u8]>::try_from)?;
                let this_replica_hash = AsyncHash::hash(self.cluster.state(), *txn.id()).await?;

                if &new_replica_hash[..] != &this_replica_hash[..] {
                    return Err(bad_request!("the provided cluster state hash is incorrect"));
                }

                let mut keyring = self.cluster.keyring_mut(*txn.id())?;

                let (host, public_key): (Host, Arc<[u8]>) =
                    key.try_cast_into(|v| TCError::unexpected(v, "a host address and key"))?;

                let public_key = VerifyingKey::try_from(&*public_key)
                    .map_err(|cause| bad_request!("invalid public key: {cause}"))?;

                keyring.insert(host, public_key);

                Ok(())
            })
        }))
    }

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b, Txn>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                debug!("DELETE {:?} replicas {key}", self.cluster);

                let mut keyring = self.cluster.keyring_mut(*txn.id())?;

                let hosts = Tuple::<Value>::try_from(key)?;

                for host in hosts {
                    let host =
                        Host::try_cast_from(host, |v| TCError::unexpected(v, "a host address"))?;

                    keyring.remove(&host);
                }

                Ok(())
            })
        }))
    }
}

impl<T> From<Cluster<T>> for ReplicaSetHandler<T> {
    fn from(cluster: Cluster<T>) -> Self {
        Self { cluster }
    }
}

struct ReplicationHandler<'a, T> {
    cluster: Cluster<T>,
    path: &'a [PathSegment],
}

impl<'a, T> ReplicationHandler<'a, T> {
    fn new(cluster: Cluster<T>, path: &'a [PathSegment]) -> Self {
        Self { cluster, path }
    }
}

impl<'a, T> Handler<'a, State> for ReplicationHandler<'a, T>
where
    T: Public<State> + Transact + Send + Sync + fmt::Debug + 'a,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                let txn = self.cluster.claim(txn.clone())?;
                self.cluster.state().get(&txn, self.path, key).await
            })
        }))
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        let path = self.path;
        let cluster = self.cluster;

        Some(Box::new(move |txn, key, value| {
            Box::pin(async move {
                let txn = cluster.claim(txn.clone())?;

                cluster
                    .state()
                    .put(&txn, path, key.clone(), value.clone())
                    .await?;

                maybe_replicate(&cluster, &txn, |txn, link| {
                    let key = key.clone();
                    let value = value.clone();
                    async move { txn.put(link, key, value).await }
                })
                .await
            })
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, params| {
            Box::pin(async move {
                let txn = self.cluster.claim(txn.clone())?;
                self.cluster.state().post(&txn, self.path, params).await
            })
        }))
    }

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b, Txn>>
    where
        'b: 'a,
    {
        let path = self.path;
        let cluster = self.cluster;

        Some(Box::new(move |txn, key| {
            Box::pin(async move {
                let txn = cluster.claim(txn.clone())?;
                cluster.state().delete(&txn, path, key.clone()).await?;
                maybe_replicate(&cluster, &txn, |txn, link| {
                    let key = key.clone();

                    async move { txn.delete(link, key).await }
                })
                .await
            })
        }))
    }
}

async fn maybe_replicate<T, Op, Fut>(cluster: &Cluster<T>, txn: &Txn, op: Op) -> TCResult<()>
where
    T: Transact + Send + Sync + fmt::Debug,
    Op: Fn(Txn, Link) -> Fut,
    Fut: Future<Output = TCResult<()>>,
{
    let leader = txn
        .leader(cluster.path())?
        .ok_or_else(|| internal!("leaderless transaction"))?;

    if leader == cluster.public_key() {
        cluster.replicate_write(txn, &[], op).await?;
    }

    let owner = txn
        .owner()?
        .ok_or_else(|| internal!("ownerless transaction"))?;

    if owner == cluster.public_key() {
        let txn = cluster.lock(txn.clone())?;
        cluster.replicate_commit(&txn).await
    } else {
        Ok(())
    }
}

impl<T> Cluster<T>
where
    T: AsyncHash + Route<State> + Transact + Send + Sync + fmt::Debug,
{
    pub fn route_owned<'a>(
        self,
        path: &'a [PathSegment],
    ) -> Option<Box<dyn Handler<'a, State> + 'a>>
    where
        T: 'a,
    {
        if path.is_empty() {
            Some(Box::new(ClusterHandler::from(self)))
        } else if path == [REPLICAS] {
            Some(Box::new(ReplicaSetHandler::from(self)))
        } else {
            Some(Box::new(ReplicationHandler::new(self, path)))
        }
    }
}

impl<T> Route<State> for Cluster<T>
where
    T: AsyncHash + Route<State> + Transact + Clone + Send + Sync + fmt::Debug,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        self.clone().route_owned(path)
    }
}
