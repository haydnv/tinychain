use bytes::Bytes;
use futures::future;
use safecast::TryCastInto;

use tc_error::*;
use tc_generic::Id;
use tc_transact::{Transact, Transaction};

use crate::cluster::Cluster;
use crate::route::*;
use crate::scalar::{Link, Value};
use crate::state::State;
use crate::txn::Txn;

pub struct ClusterHandler<'a> {
    cluster: &'a Cluster,
}

impl<'a> ClusterHandler<'a> {
    fn handle_get(self, key: Value) -> TCResult<State> {
        if key.is_some() {
            let key: Id = key.try_cast_into(|v| TCError::bad_request("invalid ID", v))?;
            self.cluster
                .chains()
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
        let owner = self.cluster.owner(txn.id()).await?;
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

impl Route for Cluster {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.is_empty() {
            return Some(Box::new(ClusterHandler::from(self)));
        } else if let Some(chain) = self.chains().get(&path[0]) {
            chain.route(&path[1..])
        } else {
            None
        }
    }
}
