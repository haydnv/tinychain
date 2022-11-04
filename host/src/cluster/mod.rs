//! Maintains the consistency of the network by coordinating transaction commits.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::iter::FromIterator;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{join_all, try_join_all, Future, FutureExt};
use futures::stream::StreamExt;
use log::{debug, warn};
use safecast::TryCastFrom;
use tokio::sync::RwLock;

use tc_error::*;
use tc_transact::lock::{TxnLock, TxnLockCommitGuard, TxnLockReadGuard};
use tc_transact::{Transact, Transaction};
use tc_value::{Link, Value, Version as VersionNumber};
use tcgeneric::*;

use crate::chain::{Chain, ChainInstance};
use crate::collection::CollectionBase;
use crate::object::InstanceClass;
use crate::scalar::Scalar;
use crate::state::{State, ToState};
use crate::txn::{Actor, Txn, TxnId};

use owner::Owner;

use futures::stream::FuturesUnordered;

pub use library::{Dir, Library};
pub use load::instantiate;

pub mod library;
mod load; // TODO: delete
mod owner;

/// The name of the endpoint which serves a [`Link`] to each of this [`Cluster`]'s replicas.
pub const REPLICAS: Label = label("replicas");

/// A state which supports replication in a [`Cluster`]
#[async_trait]
pub trait Replica: Transact {
    async fn replicate(&self, txn: &Txn, source: &Link) -> TCResult<()>;
}

/// The [`Class`] of a [`Cluster`].
pub enum ClusterType {
    Dir,
    Legacy,
}

impl Class for ClusterType {}

impl fmt::Display for ClusterType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("Cluster")
    }
}

/// The data structure responsible for maintaining consensus per-transaction across the network.
pub struct Cluster<T> {
    link: Link,
    actor: Arc<Actor>,
    owned: RwLock<HashMap<TxnId, Owner>>,
    replicas: TxnLock<HashSet<Link>>,
    state: T,
}

impl<T> Cluster<T> {
    /// Create a new [`Cluster`] to manage replication of the given `state`.
    pub fn with_state(link: Link, state: T) -> Self {
        let replicas = [&link].iter().map(|link| (*link).clone()).collect();

        Self {
            replicas: TxnLock::new(format!("replicas of {}", link), replicas),
            actor: Arc::new(Actor::new(Value::None)),
            owned: RwLock::new(HashMap::new()),
            link,
            state,
        }
    }

    /// Borrow the canonical [`Link`] to this `Cluster` (probably not on this host).
    pub fn link(&self) -> &Link {
        &self.link
    }

    /// Borrow the state managed by this `Cluster`
    pub fn state(&self) -> &T {
        &self.state
    }

    /// Borrow the path of this `Cluster`, relative to this host.
    pub fn path(&'_ self) -> &'_ [PathSegment] {
        self.link.path()
    }

    /// Borrow the public key of this `Cluster`.
    pub fn public_key(&self) -> &[u8] {
        self.actor.public_key().as_bytes()
    }

    /// Claim ownership of the given [`Txn`].
    pub async fn claim(&self, txn: &Txn) -> TCResult<Txn> {
        debug_assert!(!txn.has_owner(), "tried to claim an owned transaction");

        let mut owned = self.owned.write().await;

        let txn = txn
            .clone()
            .claim(&self.actor, self.link.path().clone())
            .await?;

        owned.insert(*txn.id(), Owner::new());

        Ok(txn)
    }

    /// Claim leadership of the given [`Txn`].
    pub async fn lead(&self, txn: Txn) -> TCResult<Txn> {
        txn.lead(&self.actor, self.link.path().clone()).await
    }
}

impl<T> Cluster<T>
where
    T: Transact + Send + Sync,
{
    /// Register a participant in the given [`Txn`].
    pub async fn mutate(&self, txn: &Txn, participant: Link) -> TCResult<()> {
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

    /// Commit the given [`Txn`] for all members of this `Cluster`.
    pub async fn distribute_commit(&self, txn: &Txn) -> TCResult<()> {
        debug!("distribute commit of {}", self);

        {
            let replicas = self.replicas.read(*txn.id()).await?;

            if let Some(owner) = self.owned.read().await.get(txn.id()) {
                owner.commit(txn).await?;
            }

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
        }

        self.commit(txn.id()).await;

        Ok(())
    }

    /// Roll back the given [`Txn`] for all members of this `Cluster`.
    pub async fn distribute_rollback(&self, txn: &Txn) {
        debug!("distribute rollback of {}", self);

        {
            let replicas = self.replicas.read(*txn.id()).await;

            if let Some(owner) = self.owned.read().await.get(txn.id()) {
                owner.rollback(txn).await;
            }

            if let Ok(replicas) = replicas {
                let self_link = txn.link(self.link.path().clone());

                let replicas = replicas
                    .iter()
                    .filter(|replica| *replica != &self_link)
                    .map(|replica| {
                        debug!("roll back replica {} of {}", replica, self);
                        txn.delete(replica.clone(), Value::None)
                    });

                join_all(replicas).await;
            }
        }

        self.finalize(txn.id()).await;
    }
}

impl<T> Cluster<T>
where
    T: Replica,
{
    /// Get the set of current replicas of this `Cluster`.
    pub async fn replicas(&self, txn_id: TxnId) -> TCResult<TxnLockReadGuard<HashSet<Link>>> {
        self.replicas.read(txn_id).await
    }

    /// Add a replica to this `Cluster`.
    pub async fn add_replica(&self, txn: &Txn, replica: Link) -> TCResult<()> {
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

            self.state.replicate(txn, &self.link).await?;
        } else {
            debug!("add replica {}", replica);
            (*self.replicas.write(*txn.id()).await?).insert(replica);
        }

        Ok(())
    }

    /// Remove one or more replicas from this `Cluster`.
    pub async fn remove_replicas(&self, txn: &Txn, to_remove: &[Link]) -> TCResult<()> {
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

    /// Replicate the given `write` operation across all replicas of this `Cluster`.
    pub async fn replicate_write<F, W>(&self, txn: Txn, write: W) -> TCResult<()>
    where
        F: Future<Output = TCResult<()>> + Send,
        W: Fn(Link) -> F + Send,
    {
        let replicas = self.replicas.read(*txn.id()).await?;
        assert!(!replicas.is_empty());
        if replicas.len() == 1 {
            return Ok(());
        }

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
}

impl Cluster<Dir> {
    pub async fn create_dir(&self, txn: &Txn, name: PathSegment) -> TCResult<()> {
        self.state().create_dir(txn, &self.link, name).await
    }

    pub async fn create_lib(
        &self,
        txn: &Txn,
        name: PathSegment,
        number: VersionNumber,
        lib: Map<Scalar>,
    ) -> TCResult<()> {
        self.state()
            .create_lib(txn, &self.link, name, number, lib)
            .await
    }
}

impl Instance for Cluster<Dir> {
    type Class = ClusterType;

    fn class(&self) -> Self::Class {
        ClusterType::Dir
    }
}

impl Instance for Cluster<Legacy> {
    type Class = ClusterType;

    fn class(&self) -> Self::Class {
        ClusterType::Legacy
    }
}

#[async_trait]
impl<T> Transact for Cluster<T>
where
    T: Transact + Send + Sync,
{
    type Commit = TxnLockCommitGuard<HashSet<Link>>;

    async fn commit(&self, txn_id: &TxnId) -> Self::Commit {
        let guard = self.replicas.commit(txn_id).await;
        self.state.commit(txn_id).await;
        guard
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.state.finalize(txn_id).await;
        self.owned.write().await.remove(txn_id);
        self.replicas.finalize(txn_id).await
    }
}

impl<T> ToState for Cluster<T> {
    fn to_state(&self) -> State {
        State::Scalar(Scalar::Cluster(self.link.path().clone().into()))
    }
}

impl<T> fmt::Display for Cluster<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Cluster {}", self.link.path())
    }
}

// TODO: delete and replace with `Service`
pub struct Legacy {
    chains: Map<Chain<CollectionBase>>,
    classes: Map<InstanceClass>,
}

impl Legacy {
    /// Borrow one of this cluster's [`Chain`]s.
    pub fn chain(&self, name: &Id) -> Option<&Chain<CollectionBase>> {
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

    async fn write_ahead(&self, txn_id: &TxnId) {
        join_all(self.chains.values().map(|chain| chain.write_ahead(txn_id))).await;
    }
}

#[async_trait]
impl Replica for Legacy {
    async fn replicate(&self, txn: &Txn, source: &Link) -> TCResult<()> {
        let replication = self
            .chains
            .iter()
            .map(|(name, chain)| chain.replicate(txn, source.clone().append(name.clone())));

        try_join_all(replication).await?;

        Ok(())
    }
}

#[async_trait]
impl Transact for Legacy {
    type Commit = ();

    async fn commit(&self, txn_id: &TxnId) {
        self.write_ahead(txn_id).await;

        join_all(self.chains.values().map(|chain| chain.commit(txn_id))).await;
    }

    async fn finalize(&self, txn_id: &TxnId) {
        join_all(self.chains.values().map(|chain| chain.finalize(txn_id))).await;
    }
}
