use bytes::Bytes;
use futures::{future, Future};
use safecast::TryCastInto;

use error::*;
use generic::Id;
use transact::{Transact, Transaction};

use crate::route::*;
use crate::scalar::{Link, Value};
use crate::state::State;
use crate::txn::Txn;

use super::Cluster;

pub struct ClusterHandler<'a> {
    cluster: &'a Cluster,
}

impl<'a> ClusterHandler<'a> {
    fn handle_get(self, key: Value) -> TCResult<State> {
        if key.is_some() {
            let key: Id = key.try_cast_into(|v| TCError::bad_request("invalid ID", v))?;
            self.cluster
                .chains
                .get(&key)
                .cloned()
                .map(State::from)
                .ok_or_else(|| TCError::not_found(key))
        } else {
            let public_key = Bytes::from(self.cluster.public_key().to_vec());
            Ok(Value::from(public_key).into())
        }
    }

    async fn handle_put(self, txn: Txn, peer: Link) -> TCResult<()> {
        let owned = self.cluster.owned.read().await;
        let owner = owned
            .get(txn.id())
            .ok_or_else(|| TCError::bad_request("cluster does not own transaction", txn.id()))?;

        owner.mutate(peer).await
    }
}

impl<'a> Handler<'a> for ClusterHandler<'a> {
    fn get(self: Box<Self>) -> Option<GetHandler<'a>> {
        Some(Box::new(|_txn, key| {
            Box::pin(future::ready(self.handle_get(key)))
        }))
    }

    fn put(self: Box<Self>) -> Option<PutHandler<'a>> {
        Some(Box::new(|txn, key, value| {
            Box::pin(async move {
                if key.is_some() {
                    return Err(TCError::unsupported("a Cluster itself is immutable"));
                }

                let peer =
                    value.try_cast_into(|s| TCError::bad_request("expected a Link, not", s))?;

                self.handle_put(txn, peer).await
            })
        }))
    }

    fn post(self: Box<Self>) -> Option<PostHandler<'a>> {
        Some(Box::new(|txn, params| {
            Box::pin(async move {
                // TODO: authorize request using a scope

                if !params.is_empty() {
                    return Err(TCError::bad_request(
                        "unrecognized commit parameters",
                        params,
                    ));
                }

                self.cluster.commit(txn.id()).await;
                Ok(State::default())
            })
        }))
    }
}

impl<'a> From<&'a Cluster> for ClusterHandler<'a> {
    fn from(cluster: &'a Cluster) -> Self {
        Self { cluster }
    }
}

pub struct ChainHandler<'a> {
    cluster: &'a Cluster,
    handler: Box<dyn Handler<'a> + 'a>,
}

impl<'a> ChainHandler<'a> {
    pub fn new(cluster: &'a Cluster, handler: Box<dyn Handler<'a> + 'a>) -> Self {
        Self { cluster, handler }
    }

    async fn wrap_handler<R, Fut: Future<Output = TCResult<R>>, F: FnOnce(Txn) -> Fut>(
        cluster: &'a Cluster,
        txn: Txn,
        handler: F,
    ) -> TCResult<R> {
        if let Some(owner_link) = txn.owner() {
            txn.put(
                owner_link.clone(),
                Value::default(),
                Link::from(cluster.path.clone()).into(),
            )
            .await?;

            handler(txn).await
        } else {
            let txn = cluster.claim(&txn).await?;
            let state = handler(txn.clone()).await?;

            let owner = cluster
                .owned
                .write()
                .await
                .remove(&txn.id())
                .expect("transaction owner");

            owner.commit(&txn).await?;
            cluster.commit(txn.id()).await;

            Ok(state)
        }
    }
}

impl<'a> Handler<'a> for ChainHandler<'a> {
    fn get(self: Box<Self>) -> Option<GetHandler<'a>> {
        let cluster = self.cluster;

        self.handler.get().map(|get_handler| {
            let wrapped: GetHandler = Box::new(move |txn, key| {
                Box::pin(Self::wrap_handler(cluster, txn.clone(), |txn| {
                    get_handler(txn, key)
                }))
            });

            wrapped
        })
    }

    fn put(self: Box<Self>) -> Option<PutHandler<'a>> {
        let cluster = self.cluster;

        self.handler.put().map(|put_handler| {
            let wrapped: PutHandler = Box::new(move |txn, key, value| {
                Box::pin(Self::wrap_handler(cluster, txn.clone(), |txn| {
                    put_handler(txn, key, value)
                }))
            });

            wrapped
        })
    }

    fn post(self: Box<Self>) -> Option<PostHandler<'a>> {
        let cluster = self.cluster;

        self.handler.post().map(|post_handler| {
            let wrapped: PostHandler = Box::new(move |txn, params| {
                Box::pin(Self::wrap_handler(cluster, txn.clone(), |txn| {
                    post_handler(txn, params)
                }))
            });

            wrapped
        })
    }
}
