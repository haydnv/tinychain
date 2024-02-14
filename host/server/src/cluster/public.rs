use tc_error::*;
use tc_transact::public::*;
use tc_transact::{RPCClient, Transaction};
use tc_value::Value;
use tcgeneric::{PathSegment, ThreadSafe};

use crate::txn::Txn;

use super::Cluster;

struct ClusterHandler<'a, T> {
    cluster: &'a Cluster<T>,
}

impl<'a, FE, State, T> Handler<'a, State> for ClusterHandler<'a, T>
where
    State: StateInstance<FE = FE, Txn = Txn<State, FE>>,
    T: Send + Sync,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn<State, FE>, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key: Value| {
            Box::pin(async move {
                if key.is_none() {
                    let public_key = self.cluster.public_key().as_bytes();
                    let public_key = Value::Bytes((*public_key).into());
                    Ok(public_key.into())
                } else {
                    Err(TCError::not_found(key))
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

struct ReplicaHandler<'a, State, T> {
    path: &'a [PathSegment],
    cluster: &'a Cluster<T>,
    handler: Box<dyn Handler<'a, State> + 'a>,
}

impl<'a, State, T> ReplicaHandler<'a, State, T> {
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

impl<'a, FE, State, T> Handler<'a, State> for ReplicaHandler<'a, State, T>
where
    FE: ThreadSafe + Clone,
    State: StateInstance<FE = FE, Txn = Txn<State, FE>>,
    T: Send + Sync,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        self.handler.get()
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b, State::Txn, State>>
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
                            RPCClient::<State>::put(&txn, link, key.clone(), value.clone())
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

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        self.handler.post()
    }

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b, State::Txn>>
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
                            RPCClient::<State>::delete(&txn, link, key.clone())
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

impl<FE, State, T> Route<State> for Cluster<T>
where
    FE: ThreadSafe + Clone,
    State: StateInstance<FE = FE, Txn = Txn<State, FE>>,
    T: Route<State>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        if path.is_empty() {
            Some(Box::new(ClusterHandler::from(self)))
        } else if let Some(handler) = self.subject().route(path) {
            Some(Box::new(ReplicaHandler::new(self, path, handler)))
        } else {
            None
        }
    }
}
