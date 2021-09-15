//! The host kernel, responsible for dispatching requests to the local host

use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::pin::Pin;

use futures::future::Future;
use log::debug;
use safecast::*;
use tc_error::*;
use tc_value::{Link, Value};
use tcgeneric::*;

use crate::cluster::Cluster;
use crate::object::InstanceExt;
use crate::route::{Public, Static};
use crate::scalar::{OpRefType, Scalar, ScalarType};
use crate::state::{State, StateType};
use crate::txn::Txn;

use hosted::Hosted;
use hypothetical::Hypothetical;

mod hosted;
mod hypothetical;

/// The host kernel, responsible for dispatching requests to the local host
pub struct Kernel {
    hosted: Hosted,
    hypothetical: Hypothetical,
}

impl Kernel {
    /// Construct a new `Kernel` to host the given [`Cluster`]s.
    pub fn new<I: IntoIterator<Item = InstanceExt<Cluster>>>(clusters: I) -> Self {
        Self {
            hosted: clusters.into_iter().collect(),
            hypothetical: Hypothetical::new(),
        }
    }

    /// Return a list of hosted clusters
    pub fn hosted(&self) -> impl Iterator<Item = &InstanceExt<Cluster>> {
        self.hosted.clusters()
    }

    /// Route a GET request.
    pub async fn get(&self, txn: &Txn, path: &[PathSegment], key: Value) -> TCResult<State> {
        if path.is_empty() {
            Err(TCError::not_found(format!(
                "{} at {}",
                key,
                TCPath::from(path)
            )))
        } else if let Some(class) = ScalarType::from_path(path) {
            let err = format!("cannot cast into an instance of {} from {}", class, key);
            Scalar::from(key)
                .into_type(class)
                .map(State::Scalar)
                .ok_or_else(|| TCError::unsupported(err))
        } else if path == &hypothetical::PATH[..] {
            self.hypothetical.get(txn, &path[..], key).await
        } else if let Some((suffix, cluster)) = self.hosted.get(path) {
            debug!(
                "GET {}: {} from cluster {}",
                TCPath::from(suffix),
                key,
                cluster
            );

            cluster.get(&txn, suffix, key).await
        } else {
            Static.get(txn, path, key).await
        }
    }

    /// Route a PUT request.
    pub async fn put(
        &self,
        txn: &Txn,
        path: &[PathSegment],
        key: Value,
        value: State,
    ) -> TCResult<()> {
        if path.is_empty() {
            if key.is_none() {
                if Link::can_cast_from(&value) {
                    // It's a synchronization message for a hypothetical transaction
                    return Ok(());
                }
            }

            Err(TCError::method_not_allowed(
                OpRefType::Put,
                self,
                TCPath::from(path),
            ))
        } else if path == &hypothetical::PATH[..] {
            self.hypothetical.put(txn, &path[..], key, value).await
        } else if let Some(class) = StateType::from_path(path) {
            Err(TCError::method_not_allowed(
                OpRefType::Put,
                class,
                TCPath::from(path),
            ))
        } else if let Some((suffix, cluster)) = self.hosted.get(path) {
            debug!(
                "PUT {}: {} <- {} to cluster {}",
                TCPath::from(suffix),
                key,
                value,
                cluster
            );

            let txn = maybe_claim_leadership(cluster, txn).await?;

            execute(txn, cluster, |txn, cluster| async move {
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
            Static.put(txn, path, key, value).await
        }
    }

    /// Route a POST request.
    pub async fn post(&self, txn: &Txn, path: &[PathSegment], data: State) -> TCResult<State> {
        if path.is_empty() {
            if Map::try_from(data)?.is_empty() {
                // it's a "commit" instruction for a hypothetical transaction
                Ok(State::default())
            } else {
                Err(TCError::method_not_allowed(
                    OpRefType::Post,
                    self,
                    TCPath::from(path),
                ))
            }
        } else if path == &hypothetical::PATH[..] {
            self.hypothetical.execute(txn, data).await
        } else if let Some(class) = StateType::from_path(path) {
            let params = data.try_into()?;
            class.post(txn, path, params).await
        } else if let Some((suffix, cluster)) = self.hosted.get(path) {
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
                execute(txn, cluster, |txn, cluster| async move {
                    cluster.post(&txn, suffix, params).await
                })
                .await
            }
        } else {
            let params = data.try_into()?;
            Static.post(txn, path, params).await
        }
    }

    /// Route a DELETE request.
    pub async fn delete(&self, txn: &Txn, path: &[PathSegment], key: Value) -> TCResult<()> {
        if path.is_empty() {
            Err(TCError::method_not_allowed(
                OpRefType::Delete,
                self,
                TCPath::from(path),
            ))
        } else if path == &hypothetical::PATH[..] {
            self.hypothetical.delete(txn, &path[2..], key).await
        } else if let Some(class) = StateType::from_path(path) {
            Err(TCError::method_not_allowed(
                OpRefType::Post,
                class,
                TCPath::default(),
            ))
        } else if let Some((suffix, cluster)) = self.hosted.get(path) {
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
            execute(txn, cluster, |txn, cluster| async move {
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
            Static.delete(txn, path, key).await
        }
    }
}

impl fmt::Display for Kernel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("host kernel")
    }
}

fn execute<
    'a,
    R: Send,
    Fut: Future<Output = TCResult<R>> + Send,
    F: FnOnce(Txn, &'a InstanceExt<Cluster>) -> Fut + Send + 'a,
>(
    txn: Txn,
    cluster: &'a InstanceExt<Cluster>,
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

            if result.is_ok() {
                debug!("commit {}", cluster);
                cluster.distribute_commit(&txn).await?;
            } else {
                debug!("rollback {}", cluster);
                cluster.distribute_rollback(&txn).await;
            }

            result
        }
    })
}

async fn maybe_claim_leadership(cluster: &Cluster, txn: &Txn) -> TCResult<Txn> {
    if txn.has_owner() && !txn.has_leader(cluster.path()) {
        cluster.lead(txn.clone()).await
    } else {
        Ok(txn.clone())
    }
}
