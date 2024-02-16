use tc_error::*;
use tc_transact::public::*;
use tc_transact::{Gateway, Transaction};
use tc_value::Value;
use tcgeneric::PathSegment;

use crate::txn::Txn;
use crate::State;

use super::Cluster;

struct ClusterHandler<'a, T> {
    cluster: &'a Cluster<T>,
    handler: Option<Box<dyn Handler<'a, State> + 'a>>,
}

impl<'a, T> ClusterHandler<'a, T> {
    fn new(cluster: &'a Cluster<T>, handler: Option<Box<dyn Handler<'a, State> + 'a>>) -> Self {
        Self { cluster, handler }
    }
}

impl<'a, T> Handler<'a, State> for ClusterHandler<'a, T>
where
    T: Send + Sync,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key: Value| {
            Box::pin(async move {
                if key.is_none() {
                    let public_key = self.cluster.public_key();
                    let public_key = Value::Bytes((*public_key.as_bytes()).into());
                    Ok(public_key.into())
                } else {
                    Err(TCError::not_found(key))
                }
            })
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        self.handler.and_then(|handler| handler.post())
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
    T: Route<State>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        let subject_handler = self.subject().route(path);

        if path.is_empty() {
            Some(Box::new(ClusterHandler::new(self, subject_handler)))
        } else if let Some(handler) = subject_handler {
            Some(Box::new(ReplicaHandler::new(self, path, handler)))
        } else {
            None
        }
    }
}
