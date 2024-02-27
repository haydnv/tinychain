use std::fmt;
use std::sync::Arc;

use futures::Future;
use rjwt::VerifyingKey;

use tc_error::*;
use tc_transact::public::*;
use tc_transact::{Gateway, Transact, Transaction};
use tc_value::{Link, Value};
use tcgeneric::PathSegment;

use crate::txn::Txn;
use crate::State;

use super::Cluster;

mod class;
mod dir;

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
                    let keyring: Vec<VerifyingKey> =
                        keyring.values().map(|actor| actor.public_key()).collect();

                    if key.is_none() {
                        let keyring = keyring
                            .into_iter()
                            .map(|public_key| Value::Bytes((*public_key.as_bytes()).into()))
                            .map(State::from)
                            .collect();

                        Ok(State::Tuple(keyring))
                    } else {
                        let key = Arc::<[u8]>::try_from(key)?;

                        if keyring
                            .iter()
                            .any(|public_key| public_key.as_bytes() == &key[..])
                        {
                            Ok(State::from(Value::from(key)))
                        } else {
                            Err(not_found!("{key:?} (keys are {keyring:?})"))
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

                    maybe_replicate(&self.cluster, &txn, |link| {
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
                if txn.locked_by()?.is_some() {
                    if key.is_some() {
                        Err(TCError::unexpected(key, "empty rollback message"))
                    } else {
                        self.cluster.replicate_rollback(txn).await
                    }
                } else {
                    let txn = self.cluster.claim(txn.clone())?;
                    self.cluster.state.delete(&txn, &[], key.clone()).await?;
                    maybe_replicate(&self.cluster, &txn, |link| txn.delete(link, key.clone())).await
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

struct ReplicaHandler<'a, T> {
    cluster: Cluster<T>,
    path: &'a [PathSegment],
}

impl<'a, T> ReplicaHandler<'a, T> {
    fn new(cluster: Cluster<T>, path: &'a [PathSegment]) -> Self {
        Self { cluster, path }
    }
}

impl<'a, T> Handler<'a, State> for ReplicaHandler<'a, T>
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

                maybe_replicate(&cluster, &txn, |link| {
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
                maybe_replicate(&cluster, &txn, |link| txn.delete(link, key.clone())).await
            })
        }))
    }
}

async fn maybe_replicate<T, Op, Fut>(cluster: &Cluster<T>, txn: &Txn, op: Op) -> TCResult<()>
where
    T: Transact + Send + Sync + fmt::Debug,
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

impl<T> Cluster<T>
where
    T: Route<State> + Transact + Send + Sync + fmt::Debug,
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
        } else {
            Some(Box::new(ReplicaHandler::new(self, path)))
        }
    }
}

impl<T> Route<State> for Cluster<T>
where
    T: Route<State> + Transact + Clone + Send + Sync + fmt::Debug,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        self.clone().route_owned(path)
    }
}
