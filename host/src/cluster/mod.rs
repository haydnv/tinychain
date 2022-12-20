//! Maintains the consistency of the network by coordinating transaction commits.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::iter::FromIterator;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{join_all, try_join_all, Future, FutureExt, TryFutureExt};
use futures::join;
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

pub use class::Class;
pub use dir::{Dir, DirEntry, DirItem};
pub use library::Library;
pub use load::instantiate;
pub use service::Service;

pub mod class;
pub mod dir;
pub mod library;
pub mod service;

mod leader;
mod load; // TODO: delete

/// The name of the endpoint which serves a [`Link`] to each of this [`Cluster`]'s replicas.
pub const REPLICAS: Label = label("replicas");

/// A state which supports replication in a [`Cluster`]
#[async_trait]
pub trait Replica {
    async fn state(&self, txn_id: TxnId) -> TCResult<State>;

    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()>;
}

// TODO: delete
/// The [`Class`] of a [`Cluster`].
pub enum ClusterType {
    Dir,
    Legacy,
}

impl tcgeneric::Class for ClusterType {}

impl fmt::Display for ClusterType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("Cluster")
    }
}

/// The data structure responsible for maintaining consensus per-transaction across the network.
#[derive(Clone)]
pub struct Cluster<T> {
    link: Arc<Link>,
    actor: Arc<Actor>,
    led: Arc<RwLock<HashMap<TxnId, leader::Leader>>>,
    replicas: TxnLock<HashSet<Link>>,
    state: T,
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
            replicas: TxnLock::new(format!("replicas of {}", cluster), replicas),
            actor: Arc::new(Actor::new(Value::None)),
            led: Arc::new(RwLock::new(HashMap::new())),
            link: Arc::new(cluster),
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

        let mut led = self.led.write().await;

        let txn = txn
            .clone()
            .claim(&self.actor, self.link().path().clone())
            .await?;

        led.insert(*txn.id(), leader::Leader::new());

        Ok(txn)
    }

    /// Claim leadership of the given [`Txn`].
    pub async fn lead(&self, txn: Txn) -> TCResult<Txn> {
        if txn.is_leader(self.path()) {
            Ok(txn)
        } else {
            let mut led = self.led.write().await;

            let txn = txn.lead(&self.actor, self.link.path().clone()).await?;

            led.insert(*txn.id(), leader::Leader::new());
            Ok(txn)
        }
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

        let led = self.led.write().await;
        let leader = led.get(txn.id()).ok_or_else(|| {
            TCError::bad_request(
                format!(
                    "{} does not own transaction",
                    txn.link(self.link().path().clone())
                ),
                txn.id(),
            )
        })?;

        leader.mutate(participant).await;
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

        let txn = txn.lead(&self.actor, self.link().path().clone()).await?;

        self.distribute_commit(&txn).await
    }

    /// Commit the given [`Txn`] for all members of this `Cluster`.
    pub async fn distribute_commit(&self, txn: &Txn) -> TCResult<()> {
        let self_link = txn.link(self.link.path().clone());
        debug!("{} will distribute commit {}", self_link, txn.id());

        let replicas = {
            // just drop this commit guard immediately
            // so a remote dependency can't pause commits to this Cluster
            let guard = self.commit(txn.id()).await;
            HashSet::clone(&*guard)
        };

        let dep_commits: TCBoxTryFuture<()> = {
            let led = self.led.read().await;
            let leader: Option<leader::Leader> = led.get(txn.id()).cloned();
            if let Some(leader) = leader {
                Box::pin(async move { leader.commit(txn).await })
            } else {
                Box::pin(futures::future::ready(Ok(())))
            }
        };

        info!(
            "{} will distribute commit {} of {} to replica set {}...",
            self_link,
            txn.id(),
            self,
            replicas.iter().collect::<Tuple<&Link>>(),
        );

        let replica_commits = replicas
            .into_iter()
            .filter(|replica| replica != &self_link)
            .map(|replica| {
                debug!("commit replica {}...", replica);
                txn.post(replica, State::Map(Map::default()))
            })
            .collect::<Vec<_>>();

        let (_deps, replicas) = join!(dep_commits, join_all(replica_commits));

        for result in replicas {
            match result {
                Ok(_) => {}
                Err(cause) => log::error!("commit failure: {}", cause),
            }
        }

        info!("{} distributed commit {} of {}", self_link, txn.id(), self);

        Ok(())
    }

    /// Roll back the given [`Txn`] for all members of this `Cluster`.
    pub async fn distribute_rollback(&self, txn: &Txn) {
        let self_link = txn.link(self.link.path().clone());
        debug!("{} will distribute rollback of {}", self_link, txn.id());

        {
            if let Some(leader) = self.led.read().await.get(txn.id()) {
                leader.rollback(txn).await;
            }

            let replicas = self.replicas.read(*txn.id()).await;

            if let Ok(replicas) = replicas {
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
        self.replicas.read(txn_id).await
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

                (*self.replicas.write(*txn.id()).await?).extend(replicas);
            } else {
                // this case is most likely adding the first replica to a load balancer
                txn.put(
                    self.link().clone().append(REPLICAS),
                    Value::default(),
                    self_link.into(),
                )
                .await?;
            }

            self.state.replicate(txn, self.link().clone()).await?;
        } else {
            debug!("add replica {}", replica);
            (*self.replicas.write(*txn.id()).await?).insert(replica);
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

        let txn = txn.lead(&self.actor, self.link().path().clone()).await?;

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
        // this lock needs to be held for the duration of the write
        // to make sure that any future write to this Cluster is replicated to the new replica
        let replicas = self.replicas.read(*txn.id()).await?;
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
impl Persist<fs::Dir> for Cluster<BlockChain<Class>> {
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
        BlockChain::dir(&self.state)
    }
}

// TODO: only impl Persist for Cluster once
#[async_trait]
impl Persist<fs::Dir> for Cluster<Dir<Class>> {
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
        Dir::dir(&self.state)
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
        Dir::dir(&self.state)
    }
}

// TODO: only impl Persist for Cluster once
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
        BlockChain::dir(&self.state)
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

        let guard = self.replicas.commit(txn_id).await;
        self.state.commit(txn_id).await;
        guard
    }

    async fn finalize(&self, txn_id: &TxnId) {
        debug!("finalize {}", self);

        self.state.finalize(txn_id).await;
        self.led.write().await.remove(txn_id);
        self.replicas.finalize(txn_id).await
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
