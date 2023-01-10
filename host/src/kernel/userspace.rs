use std::convert::TryInto;
use std::fmt;
use std::pin::Pin;

use async_trait::async_trait;
use futures::future::Future;
use log::{debug, info};

use tc_error::*;
use tc_transact::{Transact, Transaction};
use tc_value::{Link, Value};
use tcgeneric::{path_label, Map, PathLabel, PathSegment, TCPath};

use crate::chain::BlockChain;
use crate::cluster::{Cluster, Dir, DirEntry, Replica};
use crate::route::{Public, Route};
use crate::state::State;
use crate::txn::Txn;

use super::{hypothetical, Dispatch, Hypothetical};

/// The type of the class directory
pub type Class = Cluster<Dir<crate::cluster::Class>>;

/// The type of the library directory
pub type Library = Cluster<Dir<crate::cluster::Library>>;

/// The type of the service directory
pub type Service = Cluster<Dir<crate::cluster::Service>>;

/// The class directory path
pub const CLASS: PathLabel = path_label(&["class"]);

/// The library directory path
pub const LIB: PathLabel = path_label(&["lib"]);

/// The service directory path
pub const SERVICE: PathLabel = path_label(&["service"]);

/// The host userspace, responsible for dispatching requests to stateful services
pub struct UserSpace {
    hypothetical: Hypothetical,
    class: Class,
    library: Library,
    service: Service,
}

impl UserSpace {
    /// Construct a new `Kernel` to host the given [`Cluster`]s.
    pub fn new(class: Class, library: Library, service: Service) -> Self {
        Self {
            hypothetical: Hypothetical::new(),
            class,
            library,
            service,
        }
    }

    pub fn handles(&self, path: &[PathSegment]) -> bool {
        if path.is_empty() {
            return false;
        }

        &path[..1] == &SERVICE[..]
            || &path[..1] == &CLASS[..]
            || &path[..1] == &LIB[..]
            || &path[..] == &hypothetical::PATH[..]
    }
}

#[async_trait]
impl<T: Transact + Clone + Send + Sync + 'static> Dispatch for Cluster<Dir<T>>
where
    Cluster<BlockChain<T>>: Route,
    BlockChain<T>: Replica,
    Dir<T>: Replica,
    Self: Route,
{
    async fn get(&self, txn: &Txn, path: &[PathSegment], key: Value) -> TCResult<State> {
        let (suffix, cluster) = self.lookup(*txn.id(), path)?;
        debug!("GET {}: {} from {}", TCPath::from(suffix), key, cluster);
        Public::get(&cluster, txn, suffix, key).await
    }

    async fn put(&self, txn: &Txn, path: &[PathSegment], key: Value, value: State) -> TCResult<()> {
        info!("PUT {}: {} <- {}", TCPath::from(path), key, value);
        let (suffix, cluster) = self.lookup(*txn.id(), path)?;
        debug!(
            "cluster is {}, endpoint is {}",
            cluster,
            TCPath::from(suffix)
        );

        if suffix.is_empty() && key.is_none() {
            // it's a synchronization message
            Public::put(&cluster, txn, suffix, key, value).await
        } else {
            match cluster {
                DirEntry::Dir(cluster) => execute_put(&cluster, txn, suffix, key, value).await,
                DirEntry::Item(cluster) => execute_put(&cluster, txn, suffix, key, value).await,
            }
        }
    }

    async fn post(&self, txn: &Txn, path: &[PathSegment], data: State) -> TCResult<State> {
        let params: Map<State> = data.try_into()?;
        let (suffix, cluster) = self.lookup(*txn.id(), path)?;
        debug!("POST {}: {} to {}", TCPath::from(suffix), params, cluster);

        if suffix.is_empty() && params.is_empty() {
            // it's a commit message
            Public::post(&cluster, txn, suffix, params).await
        } else {
            match cluster {
                DirEntry::Dir(cluster) => execute_post(&cluster, txn, suffix, params).await,
                DirEntry::Item(cluster) => execute_post(&cluster, txn, suffix, params).await,
            }
        }
    }

    async fn delete(&self, txn: &Txn, path: &[PathSegment], key: Value) -> TCResult<()> {
        info!("DELETE {}: {}", TCPath::from(path), key);

        let (suffix, cluster) = self.lookup(*txn.id(), path)?;

        if suffix.is_empty() && key.is_none() {
            // it's a rollback message
            Public::delete(&cluster, txn, path, key).await
        } else {
            match cluster {
                DirEntry::Dir(cluster) => execute_delete(&cluster, txn, suffix, key).await,
                DirEntry::Item(cluster) => execute_delete(&cluster, txn, suffix, key).await,
            }
        }
    }
}

// TODO: consolidate redundant if..else clauses
#[async_trait]
impl Dispatch for UserSpace {
    async fn get(&self, txn: &Txn, path: &[PathSegment], key: Value) -> TCResult<State> {
        if path == &hypothetical::PATH[..] {
            self.hypothetical.get(txn, &path[..], key).await
        } else if &path[..1] == &LIB[..] {
            Dispatch::get(&self.library, txn, &path[1..], key).await
        } else if &path[..1] == &CLASS[..] {
            Dispatch::get(&self.class, txn, &path[1..], key).await
        } else if &path[..1] == &SERVICE[..] {
            Dispatch::get(&self.service, txn, &path[1..], key).await
        } else {
            Err(TCError::not_found(TCPath::from(path)))
        }
    }

    async fn put(&self, txn: &Txn, path: &[PathSegment], key: Value, value: State) -> TCResult<()> {
        if path == &hypothetical::PATH[..] {
            self.hypothetical.put(txn, &path[..], key, value).await
        } else if &path[..1] == &LIB[..] {
            Dispatch::put(&self.library, txn, &path[1..], key, value).await
        } else if &path[..1] == &CLASS[..] {
            Dispatch::put(&self.class, txn, &path[1..], key, value).await
        } else if &path[..1] == &SERVICE[..] {
            Dispatch::put(&self.service, txn, &path[1..], key, value).await
        } else {
            Err(TCError::not_found(TCPath::from(path)))
        }
    }

    async fn post(&self, txn: &Txn, path: &[PathSegment], data: State) -> TCResult<State> {
        if path == &hypothetical::PATH[..] {
            self.hypothetical.execute(txn, data).await
        } else if &path[..1] == &LIB[..] {
            Dispatch::post(&self.library, txn, &path[1..], data).await
        } else if &path[..1] == &CLASS[..] {
            Dispatch::post(&self.class, txn, &path[1..], data).await
        } else if &path[..1] == &SERVICE[..] {
            Dispatch::post(&self.service, txn, &path[1..], data).await
        } else {
            Err(TCError::not_found(TCPath::from(path)))
        }
    }

    async fn delete(&self, txn: &Txn, path: &[PathSegment], key: Value) -> TCResult<()> {
        if path == &hypothetical::PATH[..] {
            self.hypothetical.delete(txn, &path[2..], key).await
        } else if &path[..1] == &LIB[..] {
            Dispatch::delete(&self.library, txn, &path[1..], key).await
        } else if &path[..1] == &CLASS[..] {
            Dispatch::delete(&self.class, txn, &path[1..], key).await
        } else if &path[..1] == &SERVICE[..] {
            Dispatch::delete(&self.service, txn, &path[1..], key).await
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
            info!(
                "{} successfully replicated PUT {}",
                cluster,
                TCPath::from(path)
            );

            return Ok(());
        }

        info!(
            "{} is leading replication of PUT {}",
            cluster,
            TCPath::from(path)
        );

        let write = |replica_link: Link| {
            let mut target = replica_link.clone();
            target.extend(path.to_vec());
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
                txn.put(owner.clone(), Value::default(), self_link.into())
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
                    info!("rollback {} due to {}", cluster, cause);
                    cluster.distribute_rollback(&txn).await;
                }
            }

            result
        }
    })
}
