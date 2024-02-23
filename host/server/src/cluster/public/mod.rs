use std::fmt;

use tc_error::*;
use tc_transact::public::*;
use tc_transact::{Gateway, Transaction};
use tc_value::Value;
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
                    if key.is_none() {
                        let keyring = self.cluster.keyring(*txn.id())?;

                        let keyring = keyring
                            .map(|actor| Value::Bytes((*actor.public_key().as_bytes()).into()))
                            .map(State::from)
                            .collect();

                        Ok(State::Tuple(keyring))
                    } else {
                        Err(not_found!("{key:?}"))
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
            let txn = self.cluster.claim(txn.clone());

            Box::pin(async move { self.cluster.state.put(&txn, &[], key, value).await })
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, params| {
            self.cluster.state.post(txn, &[], params)
        }))
    }

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b, Txn>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            let txn = self.cluster.claim(txn.clone());

            Box::pin(async move { self.cluster.state.delete(&txn, &[], key).await })
        }))
    }
}

impl<'a, T> From<&'a Cluster<T>> for ClusterHandler<'a, T> {
    fn from(cluster: &'a Cluster<T>) -> Self {
        Self { cluster }
    }
}

struct ReplicaHandler<'a, T> {
    path: &'a [PathSegment],
    cluster: &'a Cluster<T>,
    handler: Box<dyn Handler<'a, State> + 'a>,
}

impl<'a, T> ReplicaHandler<'a, T> {
    fn new(
        cluster: &'a Cluster<T>,
        path: &'a [PathSegment],
        handler: Box<dyn Handler<'a, State> + 'a>,
    ) -> Self {
        Self {
            path,
            cluster,
            handler,
        }
    }
}

impl<'a, T> Handler<'a, State> for ReplicaHandler<'a, T>
where
    T: Send + Sync,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        self.handler.get()
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        let path = self.path;
        let cluster = self.cluster;
        let handler = self.handler.put()?;

        Some(Box::new(move |txn, key, value| {
            debug_assert!(txn.leader(cluster.path()).is_some());

            Box::pin(async move {
                (handler)(&txn, key.clone(), value.clone()).await?;

                if txn.leader(cluster.path()) == Some(cluster.public_key()) {
                    cluster
                        .replicate_write(*txn.id(), path, |link| {
                            Gateway::<State>::put(&txn, link, key.clone(), value.clone())
                        })
                        .await?;
                }

                if txn.owner().expect("owner") == cluster.public_key() {
                    cluster.replicate_commit(*txn.id()).await
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
        self.handler.post()
    }

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b, Txn>>
    where
        'b: 'a,
    {
        let path = self.path;
        let cluster = self.cluster;
        let handler = self.handler.delete()?;

        Some(Box::new(move |txn, key| {
            debug_assert!(txn.leader(cluster.path()).is_some());

            Box::pin(async move {
                (handler)(&txn, key.clone()).await?;

                if txn.leader(cluster.path()).expect("leader") == cluster.public_key() {
                    cluster
                        .replicate_write(*txn.id(), path, |link| {
                            Gateway::<State>::delete(&txn, link, key.clone())
                        })
                        .await?;
                }

                if txn.owner().expect("owner") == cluster.public_key() {
                    cluster.replicate_commit(*txn.id()).await
                } else {
                    Ok(())
                }
            })
        }))
    }
}

impl<T> Route<State> for Cluster<T>
where
    T: Route<State> + Send + Sync + fmt::Debug,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        if path.is_empty() {
            Some(Box::new(ClusterHandler::from(self)))
        } else if let Some(handler) = self.state().route(path) {
            Some(Box::new(ReplicaHandler::new(self, path, handler)))
        } else {
            None
        }
    }
}
