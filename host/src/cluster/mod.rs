//! Maintains the consistency of the network by coordinating transaction commits.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::iter::FromIterator;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{join_all, try_join_all, Future, FutureExt, TryFutureExt};
use futures::stream::{FuturesUnordered, StreamExt};
use log::{debug, info};
use safecast::TryCastFrom;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use tc_error::*;
use tc_transact::fs::Persist;
use tc_transact::lock::{TxnLock, TxnLockCommitGuard, TxnLockReadGuard};
use tc_transact::{Transact, Transaction};
use tc_value::{Link, LinkHost, Value};
use tcgeneric::*;

use crate::chain::{BlockChain, Chain};
use crate::collection::CollectionBase;
use crate::fs;
use crate::object::InstanceClass;
use crate::scalar::Scalar;
use crate::state::{State, ToState};
use crate::txn::{Actor, Txn, TxnId};

use owner::Owner;

pub use dir::{Dir, DirEntry, DirItem};
pub use library::Library;
pub use load::instantiate;

pub mod dir;
pub mod library;
mod load; // TODO: delete
mod owner;

/// The name of the endpoint which serves a [`Link`] to each of this [`Cluster`]'s replicas.
pub const REPLICAS: Label = label("replicas");

/// A state which supports replication in a [`Cluster`]
#[async_trait]
pub trait Replica {
    async fn state(&self, txn_id: TxnId) -> TCResult<State>;

    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()>;
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

struct Inner<T> {
    link: Link,
    actor: Arc<Actor>,
    owned: RwLock<HashMap<TxnId, Owner>>,
    replicas: TxnLock<HashSet<Link>>,
    state: T,
}

/// The data structure responsible for maintaining consensus per-transaction across the network.
#[derive(Clone)]
pub struct Cluster<T> {
    inner: Arc<Inner<T>>,
}

impl<T> Cluster<T> {
    /// Create a new [`Cluster`] to manage replication of the given `state`.
    // TODO: set visibility to private
    pub fn with_state(self_link: Link, cluster: Link, state: T) -> Self {
        assert!(self_link.host().is_some());

        let cluster = if self_link == cluster {
            cluster.path().clone().into()
        } else {
            cluster
        };

        let mut replicas = HashSet::new();
        replicas.insert(self_link);

        Self {
            inner: Arc::new(Inner {
                replicas: TxnLock::new(format!("replicas of {}", cluster), replicas),
                actor: Arc::new(Actor::new(Value::None)),
                owned: RwLock::new(HashMap::new()),
                link: cluster,
                state,
            }),
        }
    }

    /// Borrow the canonical [`Link`] to this `Cluster` (probably not on this host).
    pub fn link(&self) -> &Link {
        &self.inner.link
    }

    /// Borrow the state managed by this `Cluster`
    pub fn state(&self) -> &T {
        &self.inner.state
    }

    /// Borrow the path of this `Cluster`, relative to this host.
    pub fn path(&'_ self) -> &'_ [PathSegment] {
        self.inner.link.path()
    }

    /// Borrow the public key of this `Cluster`.
    pub fn public_key(&self) -> &[u8] {
        self.inner.actor.public_key().as_bytes()
    }

    /// Claim ownership of the given [`Txn`].
    pub async fn claim(&self, txn: &Txn) -> TCResult<Txn> {
        debug_assert!(!txn.has_owner(), "tried to claim an owned transaction");

        let mut owned = self.inner.owned.write().await;

        let txn = txn
            .clone()
            .claim(&self.inner.actor, self.link().path().clone())
            .await?;

        owned.insert(*txn.id(), Owner::new());

        Ok(txn)
    }

    /// Claim leadership of the given [`Txn`].
    pub async fn lead(&self, txn: Txn) -> TCResult<Txn> {
        txn.lead(&self.inner.actor, self.inner.link.path().clone())
            .await
    }
}

impl<T> Cluster<T>
where
    T: Transact + Send + Sync,
{
    /// Register a participant in the given [`Txn`].
    pub async fn mutate(&self, txn: &Txn, participant: Link) -> TCResult<()> {
        if participant.path() == self.path() {
            log::warn!(
                "got participant message within Cluster {}",
                self.link().path()
            );

            return Ok(());
        }

        let owned = self.inner.owned.write().await;
        let owner = owned.get(txn.id()).ok_or_else(|| {
            TCError::bad_request(
                format!(
                    "{} does not own transaction",
                    txn.link(self.link().path().clone())
                ),
                txn.id(),
            )
        })?;

        owner.mutate(participant).await;
        Ok(())
    }

    /// Claim leadership of this [`Txn`], then commit all replicas.
    pub async fn lead_and_distribute_commit(&self, txn: Txn) -> TCResult<()> {
        let owner = txn
            .owner()
            .cloned()
            .ok_or_else(|| TCError::internal("ownerless transaction"))?;

        let self_link = txn.link(self.link().path().clone());

        if owner.path() == self_link.path() {
            return self.distribute_commit(&txn).await;
        } else if txn.leader(self.path()).is_some() {
            return self.distribute_commit(&txn).await;
        }

        let txn = txn
            .lead(&self.inner.actor, self.link().path().clone())
            .await?;

        self.distribute_commit(&txn).await
    }

    /// Commit the given [`Txn`] for all members of this `Cluster`.
    pub async fn distribute_commit(&self, txn: &Txn) -> TCResult<()> {
        let self_link = txn.link(self.inner.link.path().clone());

        let replicas = self.commit(txn.id()).await;

        info!(
            "{} will distribute commit {} of {} to replica set {}...",
            self_link,
            txn.id(),
            self,
            replicas.iter().collect::<Tuple<&Link>>(),
        );

        if let Some(owner) = self.inner.owned.read().await.get(txn.id()) {
            owner.commit(txn).await?;
        }

        {
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

        info!("{} distributed commit {} of {}", self_link, txn.id(), self);

        Ok(())
    }

    /// Roll back the given [`Txn`] for all members of this `Cluster`.
    pub async fn distribute_rollback(&self, txn: &Txn) {
        debug!("distribute rollback of {}", self);

        {
            let replicas = self.inner.replicas.read(*txn.id()).await;

            if let Some(owner) = self.inner.owned.read().await.get(txn.id()) {
                owner.rollback(txn).await;
            }

            if let Ok(replicas) = replicas {
                let self_link = txn.link(self.inner.link.path().clone());

                let replicas = replicas
                    .iter()
                    .filter(|replica| *replica != &self_link)
                    .map(|replica| {
                        debug!("roll back replica {} of {}", replica, self);
                        txn.delete(replica.clone(), Value::default())
                    });

                join_all(replicas).await;
            }
        }

        self.finalize(txn.id()).await;
    }
}

impl<T> Cluster<T>
where
    T: Replica + Send + Sync,
{
    /// Get the set of current replicas of this `Cluster`.
    pub async fn replicas(&self, txn_id: TxnId) -> TCResult<TxnLockReadGuard<HashSet<Link>>> {
        self.inner.replicas.read(txn_id).await
    }

    /// Add a replica to this `Cluster`.
    pub async fn add_replica(&self, txn: &Txn, replica: Link) -> TCResult<()> {
        if replica.path() != self.path() {
            return Err(TCError::unsupported(format!(
                "tried to replicate {} from {}",
                self, replica
            )));
        }

        let self_link = txn.link(self.link().path().clone());

        debug!("cluster at {} adding replica {}...", self_link, replica);

        if replica == self_link {
            // make sure there's a different source to replicate from
            if self.link() == &self_link {
                if let Some(host) = self.link().host() {
                    debug!("{} at {} cannot replicate itself", self, host);
                    return Ok(());
                }
            } else if self.link().host().is_none() {
                debug!("{} cannot replicate itself", self);
                return Ok(());
            }

            // handle the case that this is a new replica and its state needs to be sync'd

            debug!(
                "{} replica at {} got add request for self: {}",
                self, replica, self_link
            );

            let replicas = txn
                .get(self.link().clone().append(REPLICAS), Value::default())
                .await?;

            if replicas.is_some() {
                let replicas = Tuple::<Link>::try_cast_from(replicas, |s| {
                    TCError::bad_request("invalid replica set", s)
                })?;

                debug!("{} has replicas: {}", self, replicas);

                let replicas: HashSet<Link> = HashSet::from_iter(replicas);
                if !replicas.contains(&self_link) {
                    try_join_all(replicas.iter().map(|replica| {
                        txn.put(
                            replica.clone().append(REPLICAS),
                            Value::default(),
                            self_link.clone().into(),
                        )
                    }))
                    .await?;
                }

                (*self.inner.replicas.write(*txn.id()).await?).extend(replicas);
            } else {
                // this case is most likely adding the first replica to a load balancer
                txn.put(
                    self.link().clone().append(REPLICAS),
                    Value::default(),
                    self_link.into(),
                )
                .await?;
            }

            self.inner.state.replicate(txn, self.link().clone()).await?;
        } else {
            debug!("add replica {}", replica);
            (*self.inner.replicas.write(*txn.id()).await?).insert(replica);
        }

        Ok(())
    }

    /// Claim leadership of the given [`Txn`] and add a replica to this `Cluster`.
    pub async fn lead_and_add_replica(&self, txn: Txn) -> TCResult<()> {
        let owner = txn
            .owner()
            .cloned()
            .ok_or_else(|| TCError::internal("ownerless transaction"))?;

        let self_link = txn.link(self.link().path().clone());

        if owner.path() == self_link.path() {
            return self.add_replica(&txn, self_link).await;
        } else if txn.leader(self.path()).is_some() {
            return self.add_replica(&txn, self_link).await;
        }

        let txn = txn
            .lead(&self.inner.actor, self.link().path().clone())
            .await?;

        info!(
            "{} claimed leadership of txn {} with owner {}",
            self_link,
            txn.id(),
            owner
        );

        txn.put(owner.clone(), Value::None, self_link.clone().into())
            .await?;

        self.add_replica(&txn, self_link).await
    }

    /// Remove one or more replicas from this `Cluster`.
    pub async fn remove_replicas(&self, txn: &Txn, to_remove: &[Link]) -> TCResult<()> {
        let self_link = txn.link(self.link().path().clone());
        let mut replicas = self.inner.replicas.write(*txn.id()).await?;

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
        // this lock needs to be held for the duration of the write
        // to make sure that any future write to this Cluster is replicated to the new replica
        let replicas = self.inner.replicas.read(*txn.id()).await?;
        assert!(!replicas.is_empty());
        if replicas.len() == 1 {
            return Ok(());
        }

        let max_failures = (replicas.len() - 1) / 2;
        let mut failed = HashSet::with_capacity(replicas.len());
        let mut succeeded = HashSet::with_capacity(replicas.len());

        {
            let self_link = txn.link(self.link().path().clone());
            let mut results: FuturesUnordered<_> = replicas
                .iter()
                .filter(|replica| *replica != &self_link)
                .map(|link| {
                    debug!("replicate write to {}", link);
                    write(link.clone()).map(move |result| (link, result))
                })
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
            try_join_all(
                succeeded
                    .into_iter()
                    .map(|replica| txn.delete(replica.clone().append(REPLICAS), failed.clone())),
            )
            .await?;
        }

        Ok(())
    }
}

// TODO: only impl Persist for Cluster once
#[async_trait]
impl Persist<fs::Dir> for Cluster<Dir<Library>> {
    type Txn = Txn;
    type Schema = Link;

    async fn create(txn: &Txn, link: Link, store: fs::Store) -> TCResult<Self> {
        let self_link = txn.link(link.path().clone());
        Dir::create(txn, link.clone(), store)
            .map_ok(|state| Self::with_state(self_link, link, state))
            .await
    }

    async fn load(txn: &Txn, link: Link, store: fs::Store) -> TCResult<Self> {
        let self_link = txn.link(link.path().clone());
        Dir::load(txn, link.clone(), store)
            .map_ok(|state| Self::with_state(self_link, link, state))
            .await
    }

    fn dir(&self) -> <fs::Dir as tc_transact::fs::Dir>::Inner {
        Dir::dir(&self.inner.state)
    }
}

#[async_trait]
impl Persist<fs::Dir> for Cluster<BlockChain<Library>> {
    type Txn = Txn;
    type Schema = Link;

    async fn create(txn: &Txn, link: Link, store: fs::Store) -> TCResult<Self> {
        let self_link = txn.link(link.path().clone());

        BlockChain::create(txn, (), store)
            .map_ok(|state| Self::with_state(self_link, link, state))
            .await
    }

    async fn load(txn: &Txn, link: Link, store: fs::Store) -> TCResult<Self> {
        let self_link = txn.link(link.path().clone());

        BlockChain::load(txn, (), store)
            .map_ok(|state| Self::with_state(self_link, link, state))
            .await
    }

    fn dir(&self) -> <fs::Dir as tc_transact::fs::Dir>::Inner {
        BlockChain::dir(&self.inner.state)
    }
}

impl<T> Cluster<Dir<T>>
where
    T: Clone,
{
    pub fn lookup<'a>(
        &self,
        txn_id: TxnId,
        path: &'a [PathSegment],
    ) -> TCResult<(&'a [PathSegment], DirEntry<T>)> {
        match self.state().lookup(txn_id, path)? {
            Some((path, entry)) => Ok((path, entry)),
            None => Ok((path, DirEntry::Dir(self.clone()))),
        }
    }
}

impl<T> Instance for Cluster<Dir<T>>
where
    T: Send + Sync,
{
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
        debug!("commit {}", self);

        let guard = self.inner.replicas.commit(txn_id).await;
        self.inner.state.commit(txn_id).await;
        guard
    }

    async fn finalize(&self, txn_id: &TxnId) {
        debug!("finalize {}", self);

        self.inner.state.finalize(txn_id).await;
        self.inner.owned.write().await.remove(txn_id);
        self.inner.replicas.finalize(txn_id).await
    }
}

impl<T> ToState for Cluster<T> {
    fn to_state(&self) -> State {
        State::Scalar(Scalar::Cluster(self.link().path().clone().into()))
    }
}

impl<T> fmt::Display for Cluster<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "cluster at {}", self.link().path())
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
}

#[async_trait]
impl Replica for Legacy {
    async fn state(&self, _txn_id: TxnId) -> TCResult<State> {
        let map = self
            .chains
            .iter()
            .map(|(name, chain)| (name.clone(), State::Chain(chain.clone())))
            .collect();

        Ok(State::Map(map))
    }

    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()> {
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
        join_all(self.chains.values().map(|chain| chain.commit(txn_id))).await;
    }

    async fn finalize(&self, txn_id: &TxnId) {
        join_all(self.chains.values().map(|chain| chain.finalize(txn_id))).await;
    }
}

impl fmt::Display for Legacy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a legacy cluster")
    }
}

#[derive(Deserialize, Serialize)]
enum ItemType {
    Dir,
    Lib,
}

#[derive(Deserialize, Serialize)]
struct Item {
    r#type: ItemType,
    host: Option<LinkHost>,
    children: HashSet<PathSegment>,
}
