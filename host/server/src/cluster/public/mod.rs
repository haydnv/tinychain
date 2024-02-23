use std::fmt;
use std::sync::Arc;

use futures::Future;

use tc_error::*;
use tc_transact::public::*;
use tc_transact::{Gateway, Transaction};
use tc_value::{Link, Value};
use tcgeneric::PathSegment;

use crate::txn::Txn;
use crate::State;

use super::Cluster;

mod class;
mod dir;

struct ClusterHandler<'a, T> {
    cluster: &'a Cluster<T>,
}

impl<'a, T> Handler<'a, State> for ClusterHandler<'a, T>
where
    T: Public<State> + Send + Sync,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key: Value| {
            if txn.has_claims() {
                self.cluster.state().get(txn, &[], key)
            } else {
                Box::pin(async move {
                    let mut keyring = self.cluster.keyring(*txn.id())?;

                    if key.is_none() {
                        let keyring = keyring
                            .map(|actor| Value::Bytes((*actor.public_key().as_bytes()).into()))
                            .map(State::from)
                            .collect();

                        Ok(State::Tuple(keyring))
                    } else {
                        let key = Arc::<[u8]>::try_from(key)?;

                        if keyring.any(|actor| actor.public_key().as_bytes() == &key[..]) {
                            Ok(State::from(Value::from(key)))
                        } else {
                            Err(not_found!("{key:?}"))
                        }
                    }
                })
            }
        }))
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key, value| {
            Box::pin(async move {
                if txn.is_locked()? {
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

                    maybe_replicate(self.cluster, &txn, |link| {
                        txn.put(link, key.clone(), value.clone())
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
                if txn.is_locked()? {
                    if key.is_some() {
                        Err(TCError::unexpected(key, "empty rollback message"))
                    } else {
                        self.cluster.replicate_rollback(txn).await
                    }
                } else {
                    let txn = self.cluster.claim(txn.clone())?;
                    self.cluster.state.delete(&txn, &[], key.clone()).await?;
                    maybe_replicate(self.cluster, &txn, |link| txn.delete(link, key.clone())).await
                }
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
    path: &'a [PathSegment],
}

impl<'a, T> ReplicaHandler<'a, T> {
    fn new(cluster: &'a Cluster<T>, path: &'a [PathSegment]) -> Self {
        Self { cluster, path }
    }
}

impl<'a, T> Handler<'a, State> for ReplicaHandler<'a, T>
where
    T: Public<State> + Send + Sync,
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

                maybe_replicate(cluster, &txn, |link| {
                    txn.put(link, key.clone(), value.clone())
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
                maybe_replicate(cluster, &txn, |link| txn.delete(link, key.clone())).await
            })
        }))
    }
}

async fn maybe_replicate<T, Op, Fut>(cluster: &Cluster<T>, txn: &Txn, op: Op) -> TCResult<()>
where
    Op: Fn(Link) -> Fut,
    Fut: Future<Output = TCResult<()>>,
{
    let leader = txn
        .leader(cluster.path())?
        .ok_or_else(|| internal!("leaderless transaction"))?;

    if leader == cluster.public_key() {
        cluster.replicate_write(*txn.id(), &[], op).await?;
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

impl<T> Route<State> for Cluster<T>
where
    T: Route<State> + Send + Sync + fmt::Debug,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        if path.is_empty() {
            Some(Box::new(ClusterHandler::from(self)))
        } else {
            Some(Box::new(ReplicaHandler::new(self, path)))
        }
    }
}
