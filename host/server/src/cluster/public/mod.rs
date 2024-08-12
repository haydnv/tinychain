use std::fmt;
use std::sync::Arc;

use futures::future::{Future, TryFutureExt};
use futures::stream::{FuturesUnordered, TryStreamExt};
use log::debug;
use rjwt::VerifyingKey;
use safecast::{TryCastFrom, TryCastInto};

use tc_error::*;
use tc_transact::hash::AsyncHash;
use tc_transact::public::*;
use tc_transact::{Gateway, Transact, Transaction};
use tc_value::{Host, Link, Value};
use tcgeneric::{label, Id, Label, Map, PathSegment, TCPath, Tuple};

use crate::txn::Txn;
use crate::State;

use super::{Cluster, IsDir, REPLICAS};

const ACTION: Label = label("action");
const JOIN: Label = label("join");

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
    T: AsyncHash + Public<State> + IsDir + Transact + Send + Sync + fmt::Debug + 'a,
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
                    let keyring = self.cluster.keyring(*txn.id()).await?;

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
                            Err(not_found!(
                                "{:?} (of {} keys)",
                                Value::Bytes(key),
                                keyring.len()
                            ))
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
                    debug!("received commit message for {:?}", self.cluster);

                    return if key.is_some() || value.is_some() {
                        Err(TCError::unexpected((key, value), "empty commit message"))
                    } else if txn.leader(self.cluster.path())?.is_none() {
                        let txn = self.cluster.claim(txn.clone())?;
                        self.cluster.replicate_commit(&txn).await
                    } else {
                        self.cluster.replicate_commit(txn).await
                    };
                }

                let txn = self.cluster.claim(txn.clone())?;

                self.cluster
                    .state
                    .put(&txn, &[], key.clone(), value.clone())
                    .await?;

                debug!("write to {:?} succeeded", self.cluster);

                let (_leader, leader_pk) = txn
                    .leader(self.cluster.path())?
                    .ok_or_else(|| internal!("leaderless transaction"))?;

                if leader_pk == self.cluster.public_key() {
                    self.cluster
                        .replicate_write(&txn, &[], |txn, link| {
                            debug!("replicating write to {:?} to {}...", self.cluster, link);

                            let key = key.clone();
                            let value = value.clone();
                            async move { txn.put(link, key, value).await }
                        })
                        .await?;
                }

                if self.cluster.is_dir() {
                    let this_host = if let Some(host) = self.cluster.link().host() {
                        host
                    } else {
                        return Ok(());
                    };

                    let entry_name: PathSegment =
                        key.try_cast_into(|v| TCError::unexpected(v, "a directory entry name"))?;

                    let (this_replica_hash, this_replica_pk) = self
                        .cluster
                        .get_dir_item_key(*txn.id(), &entry_name)
                        .await?
                        .ok_or_else(|| {
                            internal!("there is no directory entry {entry_name} to replicate")
                        })?;

                    let (leader, leader_pk) = txn.leader(self.cluster.path())?.expect("leader");

                    if leader_pk == self.cluster.public_key() {
                        let replica_set: Vec<Host> = if let Some(keyring) = self
                            .cluster
                            .get_dir_item_keyring(*txn.id(), &entry_name)
                            .await?
                        {
                            keyring.keys().cloned().collect()
                        } else {
                            vec![]
                        };

                        let this_replica_path = self.cluster.path().clone().append(entry_name);

                        replica_set
                            .iter()
                            .map(|host| {
                                let replicas = replica_set
                                    .iter()
                                    .filter(|that_host| host != *that_host)
                                    .cloned()
                                    .map(Value::from)
                                    .collect();

                                let params: Map<State> = [
                                    (Id::from(ACTION), Value::String(JOIN.into())),
                                    (Id::from(REPLICAS), Value::Tuple(replicas)),
                                ]
                                .into_iter()
                                .collect();

                                txn.post(
                                    Link::from((
                                        host.clone(),
                                        this_replica_path.clone().append(REPLICAS),
                                    )),
                                    params,
                                )
                            })
                            .collect::<FuturesUnordered<_>>()
                            .try_fold(State::default(), |_, _| {
                                futures::future::ready(Ok(State::default()))
                            })
                            .await?;
                    } else {
                        let this_replica_hash = Value::Bytes(this_replica_hash.as_slice().into());
                        let this_replica_pk = Value::Bytes(Arc::new(*this_replica_pk.as_bytes()));

                        txn.put(
                            leader.append(entry_name).append(REPLICAS),
                            (this_host.clone(), this_replica_pk),
                            this_replica_hash,
                        )
                        .await?;
                    }
                }

                let owner = txn
                    .owner()?
                    .ok_or_else(|| internal!("ownerless transaction"))?;

                if owner == self.cluster.public_key() {
                    let txn = self.cluster.lock(txn.clone())?;
                    self.cluster.replicate_commit(&txn).await
                } else {
                    Ok(())
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
                    return if key.is_some() {
                        Err(TCError::unexpected(key, "empty rollback message"))
                    } else {
                        self.cluster.replicate_rollback(txn).await
                    };
                }

                let txn = self.cluster.claim(txn.clone())?;
                self.cluster.state.delete(&txn, &[], key.clone()).await?;

                maybe_replicate(&self.cluster, &txn, &[], |txn, link| {
                    let key = key.clone();
                    async move { txn.delete(link, key).await }
                })
                .await
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

                let keyring = self.cluster.keyring(*txn.id()).await?;

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

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key, value| {
            Box::pin(async move {
                debug!("PUT {:?} replica {key}: {value:?}", self.cluster);

                let new_replica_hash = Value::try_from(value)?;
                let this_replica_hash = AsyncHash::hash(self.cluster.state(), *txn.id())
                    .map_ok(|hash| Arc::<[u8]>::from(hash.as_slice()))
                    .map_ok(Value::Bytes)
                    .await?;

                if new_replica_hash != this_replica_hash {
                    return Err(bad_request!("the provided cluster state hash {new_replica_hash} differs from the hash of this cluster {this_replica_hash}"));
                }

                let mut keyring = self.cluster.keyring_mut(*txn.id()).await?;

                let (host, public_key): (Host, Arc<[u8]>) =
                    key.try_cast_into(|v| TCError::unexpected(v, "a host address and key"))?;

                if self.cluster.link().host() == Some(&host) {
                    return Err(bad_request!(
                        "cannot overwrite the public key of {:?}",
                        self.cluster
                    ));
                }

                let public_key = VerifyingKey::try_from(&*public_key)
                    .map_err(|cause| bad_request!("invalid public key: {cause}"))?;

                keyring.insert(host, public_key);

                Ok(())
            })
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, mut params| {
            Box::pin(async move {
                let action: Id = params.require(&*ACTION)?;
                if action != JOIN {
                    return Err(bad_request!("unrecognized action: {action}"));
                }

                let replicas: Tuple<Link> = params.require(&*REPLICAS)?;

                let this_host = self
                    .cluster
                    .link()
                    .host()
                    .cloned()
                    .ok_or_else(|| bad_request!("{:?} cannot join a cluster", self.cluster))?;

                let this_path = self.cluster.link().path();

                let public_key = Value::Bytes((*self.cluster.public_key().as_bytes()).into());
                let hash = AsyncHash::hash(self.cluster.state(), *txn.id()).await?;

                replicas
                    .into_iter()
                    .map(|mut that_host| {
                        that_host.extend(this_path.iter().cloned());

                        let this_host = this_host.clone();
                        let public_key = public_key.clone();

                        async move {
                            if that_host.host().is_none() {
                                Err(bad_request!(
                                    "{} received a join request with no host",
                                    this_host
                                ))
                            } else if that_host.host() == Some(&this_host) {
                                Err(bad_request!(
                                    "{} received a join request for itself",
                                    this_host
                                ))
                            } else if that_host.path() == this_path {
                                txn.put(
                                    that_host.append(REPLICAS),
                                    (this_host, public_key),
                                    Value::Bytes(hash.as_slice().into()),
                                )
                                .await
                            } else {
                                Err(bad_request!(
                                    "{} received a request to join {}",
                                    this_host,
                                    that_host
                                ))
                            }
                        }
                    })
                    .collect::<FuturesUnordered<_>>()
                    .try_fold(State::default(), |_, _| {
                        futures::future::ready(Ok(State::default()))
                    })
                    .await
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

                let mut keyring = self.cluster.keyring_mut(*txn.id()).await?;

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

                maybe_replicate(&cluster, &txn, path, |txn, link| {
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
                maybe_replicate(&cluster, &txn, path, |txn, link| {
                    let key = key.clone();

                    async move { txn.delete(link, key).await }
                })
                .await
            })
        }))
    }
}

async fn maybe_replicate<T, Op, Fut>(
    cluster: &Cluster<T>,
    txn: &Txn,
    path: &[PathSegment],
    op: Op,
) -> TCResult<()>
where
    T: Transact + Send + Sync + fmt::Debug,
    Op: Fn(Txn, Link) -> Fut,
    Fut: Future<Output = TCResult<()>>,
{
    let (_leader, leader_pk) = txn
        .leader(cluster.path())?
        .ok_or_else(|| internal!("leaderless transaction"))?;

    if leader_pk == cluster.public_key() {
        cluster.replicate_write(txn, path, op).await?;
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
    T: AsyncHash + Route<State> + IsDir + Transact + Send + Sync + fmt::Debug,
{
    pub fn route_owned<'a>(
        self,
        path: &'a [PathSegment],
    ) -> Option<Box<dyn Handler<'a, State> + 'a>>
    where
        T: 'a,
    {
        debug!("{:?} routing request to {}...", self, TCPath::from(path));

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
    T: AsyncHash + Route<State> + IsDir + Transact + Clone + Send + Sync + fmt::Debug,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        self.clone().route_owned(path)
    }
}
