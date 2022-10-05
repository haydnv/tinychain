//! Maintains the consistency of the network by coordinating transaction commits.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::iter::FromIterator;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{join_all, try_join_all, Future, FutureExt};
use futures::stream::StreamExt;
use log::{debug, warn};
use safecast::TryCastFrom;
use tokio::sync::RwLock;

use tc_error::*;
use tc_transact::lock::{TxnLock, TxnLockReadGuard};
use tc_transact::{Transact, Transaction};
use tc_value::{Link, Value};
use tcgeneric::*;

use crate::chain::{Chain, ChainInstance};
use crate::object::InstanceClass;
use crate::scalar::Scalar;
use crate::state::{State, ToState};
use crate::txn::{Actor, Txn, TxnId};

use owner::Owner;

use futures::stream::FuturesUnordered;
pub use load::instantiate;

mod load; // TODO: delete
mod owner;

/// The name of the endpoint which serves a [`Link`] to each of this [`Cluster`]'s replicas.
pub const REPLICAS: Label = label("replicas");

/// The [`Class`] of a [`Cluster`].
pub struct ClusterType;

impl Class for ClusterType {}

impl fmt::Display for ClusterType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("Cluster")
    }
}

/// Methods responsible for maintaining consensus per-transaction across the network.
#[async_trait]
pub trait Cluster {
    /// Borrow the canonical [`Link`] to this cluster (probably not on this host).
    fn link(&self) -> &Link;

    /// Borrow the path of this cluster, relative to this host.
    fn path(&'_ self) -> &'_ [PathSegment];

    /// Borrow the public key of this cluster.
    fn public_key(&self) -> &[u8];

    /// Claim ownership of the given [`Txn`].
    async fn claim(&self, txn: &Txn) -> TCResult<Txn>;

    /// Claim leadership of the given [`Txn`].
    async fn lead(&self, txn: Txn) -> TCResult<Txn>;

    /// Commit the given [`Txn`] for all members of this [`Cluster`].
    async fn distribute_commit(&self, txn: &Txn) -> TCResult<()>;

    /// Roll back the given [`Txn`] for all members of this [`Cluster`].
    async fn distribute_rollback(&self, txn: &Txn);

    /// Get the set of current replicas of this [`Cluster`].
    async fn replicas(&self, txn_id: TxnId) -> TCResult<TxnLockReadGuard<HashSet<Link>>>;

    /// Add a replica to this [`Cluster`].
    async fn add_replica(&self, txn: &Txn, replica: Link) -> TCResult<()>;

    /// Register a participant in the given [`Txn`].
    async fn mutate(&self, txn: &Txn, participant: Link) -> TCResult<()>;

    /// Remove one or more replicas from this [`Cluster`].
    async fn remove_replicas(&self, txn: &Txn, to_remove: &[Link]) -> TCResult<()>;

    /// Replicate the given `write` operation across all replicas of this [`Cluster`].
    async fn replicate_write<F, W>(&self, txn: Txn, write: W) -> TCResult<()>
    where
        F: Future<Output = TCResult<()>> + Send,
        W: Fn(Link) -> F + Send;

    /// Write any mutations in the current transaction to the write-ahead log.
    /// TODO: move this out of [`Cluster`] and make it private to [`Legacy`]
    async fn write_ahead(&self, txn_id: &TxnId);
}

/// The data structure responsible for maintaining consensus per-transaction.
/// TODO: delete and replace with `Service`
pub struct Legacy {
    link: Link,
    actor: Arc<Actor>,
    chains: Map<Chain>,
    classes: Map<InstanceClass>,
    owned: RwLock<HashMap<TxnId, Owner>>,
    replicas: TxnLock<HashSet<Link>>,
}

impl Legacy {
    /// Borrow one of this cluster's [`Chain`]s.
    pub fn chain(&self, name: &Id) -> Option<&Chain> {
        self.chains.get(name)
    }

    /// Borrow an [`InstanceClass`], if there is one defined with the given name.
    pub fn class(&self, name: &Id) -> Option<&InstanceClass> {
        self.classes.get(name)
    }

    /// Return the names of the members of this cluster.
    pub fn ns(&self) -> impl Iterator<Item = &Id> {
        self.chains.keys().chain(self.classes.keys())
    }

    async fn replicate(&self, txn: &Txn) -> TCResult<()> {
        let replication = self.chains.iter().map(|(name, chain)| {
            let mut path = self.link.path().to_vec();
            path.push(name.clone());

            chain.replicate(txn, self.link.clone().append(name.clone()))
        });

        try_join_all(replication).await?;

        Ok(())
    }
}

#[async_trait]
impl Cluster for Legacy {
    fn link(&self) -> &Link {
        &self.link
    }

    fn path(&'_ self) -> &'_ [PathSegment] {
        self.link.path()
    }

    fn public_key(&self) -> &[u8] {
        self.actor.public_key().as_bytes()
    }

    async fn claim(&self, txn: &Txn) -> TCResult<Txn> {
        debug_assert!(!txn.has_owner(), "tried to claim an owned transaction");

        let mut owned = self.owned.write().await;

        let txn = txn
            .clone()
            .claim(&self.actor, self.link.path().clone())
            .await?;

        owned.insert(*txn.id(), Owner::new());

        Ok(txn)
    }

    async fn lead(&self, txn: Txn) -> TCResult<Txn> {
        txn.lead(&self.actor, self.link.path().clone()).await
    }

    async fn distribute_commit(&self, txn: &Txn) -> TCResult<()> {
        let replicas = self.replicas.read(*txn.id()).await?;

        if let Some(owner) = self.owned.read().await.get(txn.id()) {
            owner.commit(txn).await?;
        }

        self.write_ahead(txn.id()).await;

        let self_link = txn.link(self.link.path().clone());
        let mut replica_commits: FuturesUnordered<_> = replicas
                .iter()
                .filter(|replica| *replica != &self_link)
                .map(|replica| {
                    debug!("commit replica {}...", replica);
                    txn.post(replica.clone(), State::Map(Map::default()))
                })
                .collect();

        while let Some(result) = replica_commits.next().await {
            match result {
                Ok(_) => {}
                Err(cause) => log::error!("commit failure: {}", cause),
            }
        }

        self.commit(txn.id()).await;

        Ok(())
    }

    async fn distribute_rollback(&self, txn: &Txn) {
        let replicas = self.replicas.read(*txn.id()).await;

        if let Some(owner) = self.owned.read().await.get(txn.id()) {
            owner.rollback(txn).await;
        }

        if let Ok(replicas) = replicas {
            let self_link = txn.link(self.link.path().clone());

            join_all(
                replicas
                    .iter()
                    .filter(|replica| *replica != &self_link)
                    .map(|replica| txn.delete(replica.clone(), Value::None)),
            )
            .await;
        }

        self.finalize(txn.id()).await;
    }

    async fn replicas(&self, txn_id: TxnId) -> TCResult<TxnLockReadGuard<HashSet<Link>>> {
        self.replicas.read(txn_id).await
    }

    async fn add_replica(&self, txn: &Txn, replica: Link) -> TCResult<()> {
        let self_link = txn.link(self.link.path().clone());

        debug!("cluster at {} adding replica {}...", self_link, replica);

        if replica == self_link {
            if self.link.host().is_none() || self.link == self_link {
                debug!("{} cannot replicate itself", self);
                return Ok(());
            }

            debug!(
                "{} replica at {} got add request for self: {}",
                self, replica, self_link
            );

            let replicas = txn
                .get(self.link.clone().append(REPLICAS.into()), Value::None)
                .await?;

            if replicas.is_some() {
                let replicas = Tuple::<Link>::try_cast_from(replicas, |s| {
                    TCError::bad_request("invalid replica set", s)
                })?;

                debug!("{} has replicas: {}", self, replicas);

                let mut replicas: HashSet<Link> = HashSet::from_iter(replicas);
                replicas.remove(&self_link);

                try_join_all(replicas.iter().map(|replica| {
                    txn.put(
                        replica.clone().append(REPLICAS.into()),
                        Value::None,
                        self_link.clone().into(),
                    )
                }))
                .await?;

                (*self.replicas.write(*txn.id()).await?).extend(replicas);
            } else {
                warn!("{} has no other replicas", self);
            }

            self.replicate(txn).await?;
        } else {
            debug!("add replica {}", replica);
            (*self.replicas.write(*txn.id()).await?).insert(replica);
        }

        Ok(())
    }

    async fn mutate(&self, txn: &Txn, participant: Link) -> TCResult<()> {
        if participant.path() == self.link.path() {
            log::warn!(
                "got participant message within Cluster {}",
                self.link.path()
            );

            return Ok(());
        }

        let owned = self.owned.write().await;
        let owner = owned.get(txn.id()).ok_or_else(|| {
            TCError::bad_request(
                format!(
                    "{} does not own transaction",
                    txn.link(self.link.path().clone())
                ),
                txn.id(),
            )
        })?;

        owner.mutate(participant).await;
        Ok(())
    }

    async fn remove_replicas(&self, txn: &Txn, to_remove: &[Link]) -> TCResult<()> {
        let self_link = txn.link(self.link.path().clone());
        let mut replicas = self.replicas.write(*txn.id()).await?;

        for replica in to_remove {
            if replica == &self_link {
                panic!("{} received remove replica request for itself", self);
            }

            replicas.remove(replica);
        }

        Ok(())
    }

    async fn replicate_write<F, W>(&self, txn: Txn, write: W) -> TCResult<()>
    where
        F: Future<Output = TCResult<()>> + Send,
        W: Fn(Link) -> F + Send,
    {
        let replicas = self.replicas.read(*txn.id()).await?;
        debug!("replicating write to {} replicas", replicas.len() - 1);

        let max_failures = (replicas.len() - 1) / 2;
        let mut failed = HashSet::with_capacity(replicas.len());
        let mut succeeded = HashSet::with_capacity(replicas.len());

        {
            let self_link = txn.link(self.path().to_vec().into());
            let mut results: FuturesUnordered<_> = replicas
                .iter()
                .filter(|replica| *replica != &self_link)
                .map(|link| write(link.clone()).map(move |result| (link, result)))
                .collect();

            while let Some((replica, result)) = results.next().await {
                match result {
                    Err(cause) if cause.code() == ErrorType::Conflict => return Err(cause),
                    Err(ref cause) => {
                        debug!("replica at {} failed: {}", replica, cause);
                        failed.insert(replica.clone());
                    }
                    Ok(()) => {
                        debug!("replica at {} succeeded", replica);
                        succeeded.insert(replica);
                    }
                };

                if failed.len() > max_failures {
                    assert!(result.is_err());
                    return result;
                }
            }
        }

        if !failed.is_empty() {
            let failed = Value::from_iter(failed);
            try_join_all(succeeded.into_iter().map(|replica| {
                txn.delete(replica.clone().append(REPLICAS.into()), failed.clone())
            }))
            .await?;
        }

        Ok(())
    }

    async fn write_ahead(&self, txn_id: &TxnId) {
        join_all(self.chains.values().map(|chain| chain.write_ahead(txn_id))).await;
    }
}

impl Instance for Legacy {
    type Class = ClusterType;

    fn class(&self) -> Self::Class {
        ClusterType
    }
}

#[async_trait]
impl Transact for Legacy {
    async fn commit(&self, txn_id: &TxnId) {
        join_all(self.chains.iter().map(|(name, chain)| {
            debug!("cluster {} committing chain {}", self.link, name);
            chain.commit(txn_id)
        }))
        .await;

        self.replicas.commit(txn_id).await;
    }

    async fn finalize(&self, txn_id: &TxnId) {
        join_all(self.chains.values().map(|chain| chain.finalize(txn_id))).await;
        self.owned.write().await.remove(txn_id);
        self.replicas.finalize(txn_id).await
    }
}

impl ToState for Legacy {
    fn to_state(&self) -> State {
        State::Scalar(Scalar::Cluster(self.link.path().clone().into()))
    }
}

impl Eq for Legacy {}

impl PartialEq for Legacy {
    fn eq(&self, other: &Self) -> bool {
        self.path() == other.path()
    }
}

impl Hash for Legacy {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.path().hash(h)
    }
}

impl fmt::Display for Legacy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Cluster {}", self.link.path())
    }
}
