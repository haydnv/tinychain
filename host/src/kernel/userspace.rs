use std::convert::TryInto;
use std::fmt;
use std::pin::Pin;

use async_trait::async_trait;
use futures::future::Future;
use log::debug;

use tc_error::*;
use tc_value::{Link, LinkHost, Value};
use tcgeneric::{label, Label, Map, PathSegment, TCPath, TCPathBuf};

use crate::cluster::{Cluster, Dir, Legacy, Replica};
use crate::object::InstanceExt;
use crate::route::{Public, Route};
use crate::state::State;
use crate::txn::Txn;

use super::{hypothetical, Dispatch, Hosted, Hypothetical};

/// The library directory
pub const LIB: Label = label("lib");

/// The host userspace, responsible for dispatching requests to stateful services
pub struct UserSpace {
    hosted: Hosted, // TODO: delete
    hypothetical: Hypothetical,
    library: Cluster<Dir>,
}

impl UserSpace {
    /// Construct a new `Kernel` to host the given [`Cluster`]s.
    pub fn new<I>(address: LinkHost, library: Dir, clusters: I) -> Self
    where
        I: IntoIterator<Item = InstanceExt<Cluster<Legacy>>>,
    {
        Self {
            hosted: clusters.into_iter().collect(),
            hypothetical: Hypothetical::new(),
            library: Cluster::with_state(Link::new(address, TCPathBuf::from(LIB)), library),
        }
    }

    pub fn handles(&self, path: &[PathSegment]) -> bool {
        path.starts_with(&[LIB.into()])
    }

    /// Return a list of hosted clusters
    // TODO: delete
    pub fn hosted(&self) -> impl Iterator<Item = &InstanceExt<Cluster<Legacy>>> {
        self.hosted.clusters()
    }
}

#[async_trait]
impl Dispatch for UserSpace {
    async fn get(&self, txn: &Txn, path: &[PathSegment], key: Value) -> TCResult<State> {
        if path.starts_with(&[LIB.into()]) {
            let path = &path[1..];
            debug!("GET {}: {}", TCPath::from(path), key);
            self.library.get(&txn, path, key).await
        } else if path == &hypothetical::PATH[..] {
            self.hypothetical.get(txn, &path[..], key).await
        } else if let Some((suffix, cluster)) = self.hosted.get(path) {
            // TODO: delete this clause
            debug!(
                "GET {}: {} from cluster {}",
                TCPath::from(suffix),
                key,
                cluster
            );
            cluster.get(&txn, suffix, key).await
        } else {
            Err(TCError::not_found(&path[0]))
        }
    }

    async fn put(&self, txn: &Txn, path: &[PathSegment], key: Value, value: State) -> TCResult<()> {
        if path == &hypothetical::PATH[..] {
            self.hypothetical.put(txn, &path[..], key, value).await
        } else if path.starts_with(&[LIB.into()]) {
            let path = &path[1..];

            debug!("PUT {}: {} <- {}", TCPath::from(path), key, value);

            let txn = maybe_claim_leadership(&self.library, txn).await?;

            execute(txn, &self.library, |txn, cluster| async move {
                self.library
                    .put(&txn, path, key.clone(), value.clone())
                    .await?;

                let self_link = txn.link(cluster.path().to_vec().into());
                if path.is_empty() && key.is_none() && value.is_none() {
                    // it's a synchronization message
                    return Ok(());
                } else if !txn.is_leader(cluster.path()) {
                    debug!(
                        "{} successfully replicated PUT {}",
                        self_link,
                        TCPath::from(path)
                    );

                    return Ok(());
                }

                debug!(
                    "{} is leading replication of PUT {}",
                    self_link,
                    TCPath::from(path)
                );

                let write = |replica_link: Link| {
                    let mut target = replica_link.clone();
                    target.extend(path.to_vec());

                    debug!("replicate PUT to {}", target);
                    txn.put(target, key.clone(), value.clone())
                };

                cluster.replicate_write(txn.clone(), write).await
            })
            .await
        } else if let Some((suffix, cluster)) = self.hosted.get(path) {
            // TODO: delete this clause

            debug!(
                "PUT {}: {} <- {} to cluster {}",
                TCPath::from(suffix),
                key,
                value,
                cluster
            );

            let txn = maybe_claim_leadership(cluster, txn).await?;

            execute_legacy(txn, cluster, |txn, cluster| async move {
                cluster
                    .put(&txn, suffix, key.clone(), value.clone())
                    .await?;

                let self_link = txn.link(cluster.path().to_vec().into());
                if suffix.is_empty() {
                    // it's a synchronization message
                    return Ok(());
                } else if !txn.is_leader(cluster.path()) {
                    debug!(
                        "{} successfully replicated PUT {}",
                        self_link,
                        TCPath::from(suffix)
                    );
                    return Ok(());
                }

                debug!(
                    "{} is leading replication of PUT {}",
                    self_link,
                    TCPath::from(suffix)
                );

                let write = |replica_link: Link| {
                    let mut target = replica_link.clone();
                    target.extend(suffix.to_vec());

                    debug!("replicate PUT to {}", target);
                    txn.put(target, key.clone(), value.clone())
                };

                cluster.replicate_write(txn.clone(), write).await
            })
            .await
        } else {
            Err(TCError::not_found(&path[0]))
        }
    }

    async fn post(&self, txn: &Txn, path: &[PathSegment], data: State) -> TCResult<State> {
        if path == &hypothetical::PATH[..] {
            self.hypothetical.execute(txn, data).await
        } else if let Some((suffix, cluster)) = self.hosted.get(path) {
            // TODO: delete this clause
            let params: Map<State> = data.try_into()?;

            debug!(
                "POST {}: {} to cluster {}",
                TCPath::from(suffix),
                params,
                cluster
            );

            let txn = maybe_claim_leadership(cluster, txn).await?;
            if suffix.is_empty() && params.is_empty() {
                // it's a "commit" instruction
                cluster.post(&txn, suffix, params).await
            } else {
                execute_legacy(txn, cluster, |txn, cluster| async move {
                    cluster.post(&txn, suffix, params).await
                })
                .await
            }
        } else {
            Err(TCError::not_found(TCPath::from(path)))
        }
    }

    async fn delete(&self, txn: &Txn, path: &[PathSegment], key: Value) -> TCResult<()> {
        if path == &hypothetical::PATH[..] {
            self.hypothetical.delete(txn, &path[2..], key).await
        } else if path.starts_with(&[LIB.into()]) {
            let path = &path[1..];

            if path.is_empty() && key.is_none() {
                // it's a rollback message
                return self.library.delete(&txn, path, key).await;
            }

            debug!("DELETE {}: {}", TCPath::from(path), key);

            let txn = maybe_claim_leadership(&self.library, txn).await?;
            execute(txn, &self.library, |txn, cluster| async move {
                cluster.delete(&txn, path, key.clone()).await?;

                let txn = if !txn.has_leader(cluster.path()) {
                    cluster.lead(txn).await?
                } else {
                    txn
                };

                if path.is_empty() {
                    // it's a synchronization message
                    return Ok(());
                } else if !txn.is_leader(cluster.path()) {
                    return Ok(());
                }

                let write = |replica_link: Link| {
                    let mut target = replica_link.clone();
                    target.extend(path.to_vec());

                    debug!("replicate DELETE to {}", target);
                    txn.delete(target, key.clone())
                };

                cluster.replicate_write(txn.clone(), write).await
            })
            .await
        } else if let Some((suffix, cluster)) = self.hosted.get(path) {
            // TODO: delete this clause
            if suffix.is_empty() && key.is_none() {
                // it's a rollback message
                return cluster.delete(&txn, suffix, key).await;
            }

            debug!(
                "DELETE {}: {} from cluster {}",
                TCPath::from(suffix),
                key,
                cluster
            );

            let txn = maybe_claim_leadership(cluster, txn).await?;
            execute_legacy(txn, cluster, |txn, cluster| async move {
                cluster.delete(&txn, suffix, key.clone()).await?;

                let txn = if !txn.has_leader(cluster.path()) {
                    cluster.lead(txn).await?
                } else {
                    txn
                };

                let self_link = txn.link(cluster.path().to_vec().into());
                if suffix.is_empty() {
                    // it's a synchronization message
                    return Ok(());
                } else if !txn.is_leader(cluster.path()) {
                    debug!(
                        "{} successfully replicated DELETE {}",
                        self_link,
                        TCPath::from(suffix)
                    );

                    return Ok(());
                }

                debug!(
                    "{} is leading replication of DELETE {}...",
                    self_link,
                    TCPath::from(suffix)
                );

                let write = |replica_link: Link| {
                    let mut target = replica_link.clone();
                    target.extend(suffix.to_vec());

                    debug!("replicate DELETE to {}", target);
                    txn.delete(target, key.clone())
                };

                cluster.replicate_write(txn.clone(), write).await
            })
            .await
        } else {
            Err(TCError::not_found(TCPath::from(path)))
        }
    }
}

fn execute<'a, T, R, Fut, F>(
    txn: Txn,
    cluster: &'a Cluster<T>,
    handler: F,
) -> Pin<Box<dyn Future<Output = TCResult<R>> + Send + 'a>>
where
    T: Replica + Send + Sync + fmt::Display,
    R: Send + Sync,
    Fut: Future<Output = TCResult<R>> + Send,
    F: FnOnce(Txn, &'a Cluster<T>) -> Fut + Send + 'a,
    Cluster<T>: Route,
{
    Box::pin(async move {
        if let Some(owner) = txn.owner() {
            if owner.path() == cluster.path() {
                debug!("{} owns this transaction, no need to notify", cluster);
            } else if txn.is_leader(cluster.path()) {
                let self_link = txn.link(cluster.path().to_vec().into());
                txn.put(owner.clone(), Value::None, self_link.into())
                    .await?;
            } else {
                let self_link = txn.link(cluster.path().to_vec().into());
                debug!(
                    "{} is not leading this transaction, no need to notify owner",
                    self_link
                );
            }

            handler(txn.clone(), cluster).await
        } else {
            // Claim and execute the transaction
            let txn = cluster.claim(&txn).await?;
            let result = handler(txn.clone(), cluster).await;

            match &result {
                Ok(_) => {
                    debug!("commit {}", cluster);
                    cluster.distribute_commit(&txn).await?;
                }
                Err(cause) => {
                    debug!("rollback {} due to {}", cluster, cause);
                    cluster.distribute_rollback(&txn).await;
                }
            }

            result
        }
    })
}

// TODO: delete
fn execute_legacy<
    'a,
    R: Send + Sync,
    Fut: Future<Output = TCResult<R>> + Send,
    F: FnOnce(Txn, &'a InstanceExt<Cluster<Legacy>>) -> Fut + Send + 'a,
>(
    txn: Txn,
    cluster: &'a InstanceExt<Cluster<Legacy>>,
    handler: F,
) -> Pin<Box<dyn Future<Output = TCResult<R>> + Send + 'a>> {
    Box::pin(async move {
        if let Some(owner) = txn.owner() {
            if owner.path() == cluster.path() {
                debug!("{} owns this transaction, no need to notify", cluster);
            } else if txn.is_leader(cluster.path()) {
                let self_link = txn.link(cluster.path().to_vec().into());
                txn.put(owner.clone(), Value::None, self_link.into())
                    .await?;
            } else {
                let self_link = txn.link(cluster.path().to_vec().into());
                debug!(
                    "{} is not leading this transaction, no need to notify owner",
                    self_link
                );
            }

            handler(txn.clone(), cluster).await
        } else {
            // Claim and execute the transaction
            let txn = cluster.claim(&txn).await?;
            let result = handler(txn.clone(), cluster).await;

            match &result {
                Ok(_) => {
                    debug!("commit {}", cluster);
                    cluster.distribute_commit(&txn).await?;
                }
                Err(cause) => {
                    debug!("rollback {} due to {}", cluster, cause);
                    cluster.distribute_rollback(&txn).await;
                }
            }

            result
        }
    })
}

async fn maybe_claim_leadership<T>(cluster: &Cluster<T>, txn: &Txn) -> TCResult<Txn> {
    if txn.has_owner() && !txn.has_leader(cluster.path()) {
        cluster.lead(txn.clone()).await
    } else {
        Ok(txn.clone())
    }
}
