use std::convert::TryInto;
use std::fmt;
use std::pin::Pin;

use async_trait::async_trait;
use futures::future::Future;
use log::debug;

use tc_error::*;
use tc_transact::{Transact, Transaction};
use tc_value::{Link, Value};
use tcgeneric::{path_label, Map, PathLabel, PathSegment, TCPath};

use crate::cluster::{Cluster, Dir, DirEntry, Legacy, Replica};
use crate::object::InstanceExt;
use crate::route::{Public, Route};
use crate::state::State;
use crate::txn::Txn;

use super::{hypothetical, Dispatch, Hosted, Hypothetical};

/// The type of the library directory
pub type Library = Cluster<Dir<crate::cluster::Library>>;

/// The library directory path
pub const LIB: PathLabel = path_label(&["lib"]);

/// The host userspace, responsible for dispatching requests to stateful services
pub struct UserSpace {
    hosted: Hosted, // TODO: delete
    hypothetical: Hypothetical,
    library: Library,
}

impl UserSpace {
    /// Construct a new `Kernel` to host the given [`Cluster`]s.
    pub fn new<I>(library: Library, clusters: I) -> Self
    where
        I: IntoIterator<Item = InstanceExt<Cluster<Legacy>>>,
    {
        Self {
            hosted: clusters.into_iter().collect(),
            hypothetical: Hypothetical::new(),
            library,
        }
    }

    pub fn handles(&self, path: &[PathSegment]) -> bool {
        if path.is_empty() {
            return false;
        }

        &path[..1] == &LIB[..]
            || &path[..] == &hypothetical::PATH[..]
            || self.hosted.contains(&path[0])
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
        if path == &hypothetical::PATH[..] {
            self.hypothetical.get(txn, &path[..], key).await
        } else if let Some((suffix, cluster)) = self.hosted.get(path) {
            // TODO: delete this clause
            debug!("GET {}: {} from {}", TCPath::from(suffix), key, cluster);
            cluster.get(&txn, suffix, key).await
        } else if &path[..1] == &LIB[..] {
            let (suffix, cluster) = self.library.lookup(*txn.id(), &path[1..])?;
            debug!("GET {}: {} from {}", TCPath::from(suffix), key, cluster);
            cluster.get(&txn, suffix, key).await
        } else {
            Err(TCError::not_found(TCPath::from(path)))
        }
    }

    async fn put(&self, txn: &Txn, path: &[PathSegment], key: Value, value: State) -> TCResult<()> {
        if path == &hypothetical::PATH[..] {
            self.hypothetical.put(txn, &path[..], key, value).await
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

                let self_link = txn.link(cluster.link().path().clone());
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
        } else if &path[..1] == &LIB[..] {
            debug!("PUT {}: {} <- {}", TCPath::from(path), key, value);
            let (suffix, cluster) = self.library.lookup(*txn.id(), &path[1..])?;

            if suffix.is_empty() && key.is_none() {
                // it's a synchronization message
                return cluster.put(txn, suffix, key, value).await;
            }

            match cluster {
                DirEntry::Dir(cluster) => execute_put(&cluster, txn, suffix, key, value).await,
                DirEntry::Item(cluster) => execute_put(&cluster, txn, suffix, key, value).await,
            }
        } else {
            Err(TCError::not_found(TCPath::from(path)))
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
        } else if &path[..1] == &LIB[..] {
            let params: Map<State> = data.try_into()?;
            let (suffix, cluster) = self.library.lookup(*txn.id(), &path[1..])?;
            debug!("POST {}: {} to {}", TCPath::from(suffix), params, cluster);

            if suffix.is_empty() && params.is_empty() {
                // it's a commit message
                return cluster.post(txn, suffix, params).await;
            }

            match cluster {
                DirEntry::Dir(cluster) => execute_post(&cluster, txn, suffix, params).await,
                DirEntry::Item(cluster) => execute_post(&cluster, txn, suffix, params).await,
            }
        } else {
            Err(TCError::not_found(TCPath::from(path)))
        }
    }

    async fn delete(&self, txn: &Txn, path: &[PathSegment], key: Value) -> TCResult<()> {
        if path == &hypothetical::PATH[..] {
            self.hypothetical.delete(txn, &path[2..], key).await
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

                let self_link = txn.link(cluster.link().path().clone());
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
        } else if &path[..1] == &LIB[..] {
            let (suffix, cluster) = self.library.lookup(*txn.id(), &path[1..])?;

            if suffix.is_empty() && key.is_none() {
                // it's a rollback message
                return self.library.delete(&txn, path, key).await;
            }

            debug!("DELETE {}: {}", TCPath::from(path), key);

            match cluster {
                DirEntry::Dir(cluster) => execute_delete(&cluster, txn, suffix, key).await,
                DirEntry::Item(cluster) => execute_delete(&cluster, txn, suffix, key).await,
            }
        } else {
            Err(TCError::not_found(TCPath::from(path)))
        }
    }
}

async fn maybe_claim_leadership<T>(cluster: &Cluster<T>, txn: &Txn) -> TCResult<Txn> {
    if txn.has_owner() && !txn.has_leader(cluster.path()) {
        cluster.lead(txn.clone()).await
    } else {
        Ok(txn.clone())
    }
}

async fn execute_post<T>(
    cluster: &Cluster<T>,
    txn: &Txn,
    path: &[PathSegment],
    params: Map<State>,
) -> TCResult<State>
where
    T: Replica + Transact + Send + Sync + fmt::Display,
    Cluster<T>: Route,
{
    let txn = maybe_claim_leadership(cluster, txn).await?;

    execute(txn, cluster, |txn, cluster| async move {
        cluster.post(&txn, path, params).await
    })
    .await
}

async fn execute_put<T>(
    cluster: &Cluster<T>,
    txn: &Txn,
    path: &[PathSegment],
    key: Value,
    value: State,
) -> TCResult<()>
where
    T: Replica + Transact + Send + Sync + fmt::Display,
    Cluster<T>: Route,
{
    let txn = maybe_claim_leadership(cluster, txn).await?;

    execute(txn, cluster, |txn, cluster| async move {
        cluster.put(&txn, path, key.clone(), value.clone()).await?;

        if !txn.is_leader(cluster.path()) {
            debug!(
                "{} successfully replicated PUT {}",
                cluster,
                TCPath::from(path)
            );

            return Ok(());
        }

        debug!(
            "{} is leading replication of PUT {}",
            cluster,
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
}

async fn execute_delete<T>(
    cluster: &Cluster<T>,
    txn: &Txn,
    path: &[PathSegment],
    key: Value,
) -> TCResult<()>
where
    T: Replica + Transact + Send + Sync + fmt::Display,
    Cluster<T>: Route,
{
    let txn = maybe_claim_leadership(cluster, txn).await?;

    execute(txn, cluster, |txn, cluster| async move {
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
}

fn execute<'a, T, R, Fut, F>(
    txn: Txn,
    cluster: &'a Cluster<T>,
    handler: F,
) -> Pin<Box<dyn Future<Output = TCResult<R>> + Send + 'a>>
where
    T: Replica + Transact + Send + Sync + fmt::Display,
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
                let self_link = txn.link(cluster.link().path().clone());
                txn.put(owner.clone(), Value::None, self_link.into())
                    .await?;
            } else {
                let self_link = txn.link(cluster.link().path().clone());
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
                let self_link = txn.link(cluster.link().path().clone());
                txn.put(owner.clone(), Value::None, self_link.into())
                    .await?;
            } else {
                let self_link = txn.link(cluster.link().path().clone());
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
