//! Maintains the consistency of the network by coordinating transaction commits.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fmt;
use std::iter::FromIterator;
use std::ops::Deref;
use std::sync::Arc;

use async_trait::async_trait;
use destream::de::Error;
use futures::future::{join_all, Future, FutureExt, TryFutureExt};
use futures::stream::{FuturesUnordered, StreamExt};
use log::{debug, info, trace, warn};
use safecast::TryCastInto;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use tc_error::*;
use tc_transact::fs::Persist;
use tc_transact::lock::{TxnLock, TxnLockVersionGuard};
use tc_transact::{Transact, Transaction};
use tc_value::{Host, Link, Value};
use tcgeneric::*;

use crate::chain::{BlockChain, Recover};
use crate::fs;
use crate::state::State;
use crate::txn::{Actor, Txn, TxnId};

pub use class::Class;
pub use dir::{Dir, DirEntry, DirItem};
pub use library::Library;
pub use service::Service;

pub mod class;
pub mod dir;
pub mod library;
pub mod service;

mod leader;

/// The name of the endpoint which serves a [`Link`] to each of this [`Cluster`]'s replicas.
pub const REPLICAS: Label = label("replicas");

/// A state which supports replication in a [`Cluster`]
#[async_trait]
pub trait Replica {
    async fn state(&self, txn_id: TxnId) -> TCResult<State>;

    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()>;
}

/// The static configuration of a [`Cluster`]
#[derive(Clone)]
pub struct Schema {
    path: TCPathBuf,
    host: Host,
    lead: Option<Host>,
    actor: Arc<Actor>, // TODO: remove this and use the public key of the gateway instead
}

impl Schema {
    /// Construct a new [`Schema`].
    pub fn new(host: Host, path: TCPathBuf, lead: Option<Host>, actor: Arc<Actor>) -> Self {
        Self {
            path,
            host,
            lead,
            actor,
        }
    }

    /// Borrow the [`Host`] of the lead replica, if any.
    pub fn lead(&self) -> Option<&Host> {
        self.lead.as_ref()
    }

    /// Construct a canonical [`Link`]
    pub fn link(&self) -> Link {
        if let Some(lead) = &self.lead {
            self.link_to(lead)
        } else {
            self.path.clone().into()
        }
    }

    /// The path to a [`Cluster`]
    pub fn path(&self) -> &TCPathBuf {
        &self.path
    }

    /// Construct a [`Link`] to this [`Replica`]
    pub fn self_link(&self) -> Link {
        self.link_to(&self.host)
    }

    /// Borrow the public key of this [`Cluster`]
    pub fn public_key(&self) -> &[u8] {
        self.actor.public_key().as_bytes()
    }

    fn extend<N: Into<PathSegment>>(&self, name: N) -> Self {
        Self {
            path: self.path.clone().append(name),
            host: self.host.clone(),
            lead: self.lead.clone(),
            actor: self.actor.clone(),
        }
    }

    // TODO: make private
    pub(crate) fn link_to(&self, replica: &Host) -> Link {
        (replica.clone(), self.path.clone()).into()
    }
}

/// The data structure responsible for maintaining consensus per-transaction across the network.
#[derive(Clone)]
pub struct Cluster<T> {
    schema: Schema,
    actor: Arc<Actor>,
    replicas: TxnLock<BTreeSet<Host>>,
    led: Arc<RwLock<BTreeMap<TxnId, leader::Leader>>>,
    state: T,
}

impl<T> Cluster<T> {
    fn with_state(schema: Schema, txn_id: TxnId, state: T) -> Self {
        let mut replicas = BTreeSet::new();
        replicas.insert(schema.host.clone());

        Self {
            schema,
            actor: Arc::new(Actor::new(Value::None)),
            replicas: TxnLock::new(replicas),
            led: Arc::new(RwLock::new(BTreeMap::new())),
            state,
        }
    }

    /// Borrow the [`Schema`] of this [`Cluster`].
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Return a canonical [`Link`] to this [`Cluster`] (probably not on this host).
    pub fn link(&self) -> Link {
        self.schema.link()
    }

    /// Borrow the state managed by this `Cluster`
    // TODO: can this be deleted?
    pub fn state(&self) -> &T {
        &self.state
    }

    /// Borrow the path of this `Cluster`, relative to this host.
    pub fn path(&self) -> &[PathSegment] {
        self.schema.path.deref()
    }

    /// Borrow the public key of this replica.
    pub fn public_key(&self) -> &[u8] {
        self.actor.public_key().as_bytes()
    }

    /// Claim ownership of the given [`Txn`].
    pub async fn claim(&self, txn: &Txn) -> TCResult<Txn> {
        debug_assert!(!txn.has_owner(), "tried to claim an owned transaction");

        let mut led = self.led.write().await;

        let txn = txn
            .clone()
            .claim(&self.actor, self.schema.path.clone())
            .await?;

        led.insert(*txn.id(), leader::Leader::new());

        Ok(txn)
    }

    /// Claim leadership of the given [`Txn`].
    pub async fn lead(&self, txn: Txn) -> TCResult<Txn> {
        if txn.has_leader(self.path()) {
            Ok(txn)
        } else {
            let mut led = self.led.write().await;
            let txn = txn.lead(&self.actor, self.schema.path.clone()).await?;
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
                self.schema.path
            );

            return Ok(());
        }

        let led = self.led.write().await;
        let leader = led
            .get(txn.id())
            .ok_or_else(|| bad_request!("{:?} does not own transaction {}", self, txn.id()))?;

        leader.mutate(participant).await;
        Ok(())
    }

    /// Claim leadership of this [`Txn`], then commit all replicas.
    pub async fn lead_and_distribute_commit(&self, txn: Txn) -> TCResult<()> {
        if txn.has_leader(self.path()) {
            self.distribute_commit(&txn).await
        } else {
            let txn = txn.lead(&self.actor, self.schema.path.clone()).await?;
            self.distribute_commit(&txn).await
        }
    }

    #[cfg(not(debug_assertions))]
    async fn distribute_commit_concurrent(&self, txn: &Txn) -> TCResult<Vec<TCResult<State>>> {
        let replicas = self.replicas.read(*txn.id()).await?;

        let dep_commits: TCBoxFuture<()> = {
            let led = self.led.read().await;
            let leader: Option<leader::Leader> = led.get(txn.id()).cloned();
            if let Some(leader) = leader {
                Box::pin(async move { leader.commit(txn).await })
            } else {
                Box::pin(futures::future::ready(()))
            }
        };

        info!(
            "{} will distribute commit {} of {} to replica set {}...",
            self,
            txn.id(),
            self.schema.path,
            replicas.iter().collect::<Tuple<&Host>>(),
        );

        let replica_commits = replicas
            .iter()
            .filter(|replica| *replica != &self.schema.host)
            .map(|replica| self.schema.link_to(replica))
            .map(|replica| {
                debug!("commit replica {}...", replica);
                txn.post(replica, State::Map(Map::default()))
            })
            .collect::<Vec<_>>();

        let (_deps, results) = futures::join!(dep_commits, join_all(replica_commits));

        Ok(results)
    }

    #[cfg(debug_assertions)]
    async fn distribute_commit_debug(&self, txn: &Txn) -> TCResult<Vec<TCResult<State>>> {
        let replicas = self.replicas.read(*txn.id()).await?;

        {
            let leader = {
                let led = self.led.read().await;
                led.get(txn.id()).cloned()
            };

            if let Some(leader) = leader {
                leader.commit(txn).await;
            }
        }

        info!(
            "{:?} will distribute commit {} of {} to replica set {}...",
            self,
            txn.id(),
            self.schema.path,
            replicas.iter().collect::<Tuple<&Host>>(),
        );

        let mut results = Vec::with_capacity(replicas.len());
        for replica in &*replicas {
            if replica == &self.schema.host {
                continue;
            }

            let link = self.schema.link_to(replica);
            let result = txn.post(link, State::Map(Map::default())).await;

            if let Err(cause) = &result {
                warn!("replica at {} failed: {}", replica, cause);
            }

            results.push(result);
        }

        Ok(results)
    }

    /// Commit the given [`Txn`] for all members of this `Cluster`.
    pub async fn distribute_commit(&self, txn: &Txn) -> TCResult<()> {
        debug!("{:?} will distribute commit {}", self, txn.id());

        #[cfg(debug_assertions)]
        let results = self.distribute_commit_debug(txn).await?;

        #[cfg(not(debug_assertions))]
        let results = self.distribute_commit_concurrent(txn).await?;

        let num_replicas = results.len() + 1; // +1 for this replica, which is committed below

        let mut succeeded = 0;
        for result in results {
            match result {
                Ok(_) => succeeded += 1,
                Err(cause) => log::error!("commit failure: {}", cause),
            }
        }

        // note: this first condition will always pass when num_replicas == 1
        if succeeded >= num_replicas / 2 {
            self.commit(*txn.id()).await;
            info!("{:?} distributed commit {} of {:?}", self, txn.id(), self);
            Ok(())
        } else if succeeded == 0 {
            Err(bad_gateway!(
                "{:?} failed to replicate commit {}",
                self,
                txn.id()
            ))
        } else {
            // in this case, the transaction failed to replicate
            // but as a result it's not possible to remove the bad replicas

            panic!(
                "commit failed--only {} out of {} were committed",
                succeeded, num_replicas,
            );
        }
    }

    /// Roll back the given [`Txn`] for all members of this `Cluster`.
    pub async fn distribute_rollback(&self, txn: &Txn) {
        debug!("{:?} will distribute rollback of {}", self, txn.id());

        {
            if let Some(leader) = self.led.read().await.get(txn.id()) {
                leader.rollback(txn).await;
            }

            let replicas = self.replicas.read(*txn.id()).await;

            if let Ok(replicas) = replicas {
                let replicas = replicas
                    .iter()
                    .filter(|replica| *replica != &self.schema.host)
                    .map(|replica| self.schema.link_to(replica))
                    .map(|replica| {
                        debug!("roll back replica {} of {:?}", replica, self);
                        txn.delete(replica, Value::default())
                    });

                join_all(replicas).await;
            }
        }

        self.rollback(txn.id()).await;
    }
}

impl<T> Cluster<T>
where
    T: Replica + Send + Sync,
{
    /// Get the set of current replicas of this `Cluster`.
    pub async fn replicas(&self, txn_id: TxnId) -> TCResult<impl Deref<Target = BTreeSet<Host>>> {
        self.replicas.read(txn_id).map_err(TCError::from).await
    }

    /// Add a replica to this `Cluster`.
    ///
    /// Returns `true` if a new replica was added.
    pub async fn add_replica(&self, txn: &Txn, replica: Host) -> TCResult<bool> {
        let txn_id = *txn.id();

        debug!("{:?} adding replica {}...", self, replica);

        if replica == self.schema.host {
            // make sure there's a different source to replicate from
            if self.schema.lead.is_none() || self.schema.lead == Some(replica) {
                debug!("{:?} cannot replicate itself", self);
                return Ok(false);
            }

            // handle the case that this is a new replica and its state needs to be sync'd
            debug!("{:?} got an add request for itself", self);

            // notify the other replicas about this host and synchronize the cluster state
            self.state.replicate(txn, self.link()).await?;

            // synchronize this host's replica set
            let replicas = txn
                .get(self.link().append(REPLICAS), Value::default())
                .await?;

            let replicas = replicas.try_into_tuple(|s| TCError::unexpected(s, "a replica set"))?;

            let replicas = replicas
                .into_iter()
                .map(|state| state.try_cast_into(|s| TCError::unexpected(s, "a replica host")))
                .collect::<TCResult<BTreeSet<Host>>>()?;

            if !replicas.contains(&self.schema.host) {
                return Err(unexpected!(
                    "failed to update {:?} with new replica {}",
                    self,
                    &self.schema.host,
                ));
            }

            let mut replica_set = self.replicas.write(txn_id).await?;
            replica_set.extend(replicas);

            Ok(false)
        } else {
            let mut replicas = self.replicas.write(txn_id).await?;
            Ok(replicas.insert(replica))
        }
    }

    /// Claim leadership of the given [`Txn`] and add a replica to this `Cluster`.
    pub async fn lead_and_add_replica(&self, txn: Txn) -> TCResult<bool> {
        if txn.has_leader(self.path()) {
            self.add_replica(&txn, self.schema.host.clone()).await
        } else {
            let txn = txn.lead(&self.actor, self.schema.path.clone()).await?;
            self.add_replica(&txn, self.schema.host.clone()).await
        }
    }

    /// Remove one or more replicas from this `Cluster`.
    pub async fn remove_replicas<'a, R: IntoIterator<Item = &'a Host>>(
        &'a self,
        txn_id: TxnId,
        to_remove: R,
    ) -> TCResult<()> {
        let mut replicas = self.replicas.write(txn_id).await?;

        for replica in to_remove {
            if replica == &self.schema.host {
                return Err(bad_request!(
                    "{:?} received a remove request for itself",
                    self
                ));
            } else {
                replicas.remove(replica);
            }
        }

        Ok(())
    }

    /// Replicate the given `write` operation across all other replicas of this `Cluster`.
    pub async fn replicate_write<F, W>(&self, txn: Txn, write: W) -> TCResult<()>
    where
        F: Future<Output = TCResult<()>> + Send,
        W: Fn(Link) -> F + Send, // TODO: this should accept a &Link
    {
        // this lock needs to be held for the duration of the write
        // to make sure that any future write to this Cluster is replicated to any new replicas
        let replicas = self.replicas.read(*txn.id()).await?;
        assert!(!replicas.is_empty());
        if replicas.len() == 1 {
            return Ok(());
        }

        let max_failures = (replicas.len() - 1) / 2;
        let mut failed = HashSet::with_capacity(replicas.len());
        let mut succeeded = HashSet::with_capacity(replicas.len());

        {
            let mut results: FuturesUnordered<_> = replicas
                .iter()
                .filter(|replica| *replica != &self.schema.host)
                .map(|replica| {
                    let link = self.schema.link_to(replica);
                    info!("replicate write to {}", link);
                    write(link).map(move |result| (replica, result))
                })
                .collect();

            while let Some((replica, result)) = results.next().await {
                match result {
                    Err(cause) if cause.code() == ErrorKind::Conflict => return Err(cause),
                    Err(ref cause) => {
                        warn!("replica {} failed: {}", replica, cause);
                        failed.insert(replica.clone());
                    }
                    Ok(()) => {
                        info!("replica {} succeeded", replica);
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

            let mut cleanup = succeeded
                .into_iter()
                .map(|replica| {
                    let link = self.schema.link_to(replica).append(REPLICAS);
                    txn.delete(link, failed.clone()).map(move |r| (replica, r))
                })
                .collect::<FuturesUnordered<_>>();

            while let Some((replica, result)) = cleanup.next().await {
                if let Err(cause) = result {
                    warn!(
                        "attempt to remove failed replicas from {} itself failed: {}",
                        replica, cause
                    );
                }
            }
        }

        Ok(())
    }
}

#[async_trait]
impl<T: Persist<fs::CacheBlock, Txn = Txn>> Persist<fs::CacheBlock> for Cluster<BlockChain<T>>
where
    BlockChain<T>: Persist<fs::CacheBlock, Schema = (), Txn = Txn>,
{
    type Txn = Txn;
    type Schema = Schema;

    async fn create(txn_id: TxnId, schema: Self::Schema, store: fs::Dir) -> TCResult<Self> {
        BlockChain::create(txn_id, (), store)
            .map_ok(|state| Self::with_state(schema, txn_id, state))
            .await
    }

    async fn load(txn_id: TxnId, schema: Self::Schema, store: fs::Dir) -> TCResult<Self> {
        BlockChain::load(txn_id, (), store)
            .map_ok(|state| Self::with_state(schema, txn_id, state))
            .await
    }

    fn dir(&self) -> tc_transact::fs::Inner<fs::CacheBlock> {
        Persist::dir(&self.state)
    }
}

#[async_trait]
impl<T: Persist<fs::CacheBlock, Txn = Txn>> Persist<fs::CacheBlock> for Cluster<Dir<T>>
where
    Dir<T>: Persist<fs::CacheBlock, Schema = Schema, Txn = Txn>,
{
    type Txn = Txn;
    type Schema = Schema;

    async fn create(txn_id: TxnId, schema: Self::Schema, store: fs::Dir) -> TCResult<Self> {
        Dir::create(txn_id, schema.clone(), store)
            .map_ok(|state| Self::with_state(schema, txn_id, state))
            .await
    }

    async fn load(txn_id: TxnId, schema: Self::Schema, store: fs::Dir) -> TCResult<Self> {
        Dir::load(txn_id, schema.clone(), store)
            .map_ok(|state| Self::with_state(schema, txn_id, state))
            .await
    }

    fn dir(&self) -> tc_transact::fs::Inner<fs::CacheBlock> {
        Persist::dir(&self.state)
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

#[async_trait]
impl<T> Transact for Cluster<T>
where
    T: Transact + Send + Sync,
{
    type Commit = TxnLockVersionGuard<BTreeSet<Host>>;

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        debug!("commit {:?} at {}", self, txn_id);

        let replicas = self.replicas.read_and_commit(txn_id).await;
        trace!("committed replica set");

        self.state.commit(txn_id).await;
        trace!("committed cluster state");

        replicas
    }

    async fn rollback(&self, txn_id: &TxnId) {
        debug!("roll back {:?} at {}", self, txn_id);

        self.state.rollback(txn_id).await;
        self.led.write().await.remove(txn_id);
        self.replicas.rollback(txn_id);
    }

    async fn finalize(&self, txn_id: &TxnId) {
        debug!("finalize {:?} at {}", self, txn_id);

        self.state.finalize(txn_id).await;
        self.led.write().await.remove(txn_id);
        self.replicas.finalize(*txn_id);
    }
}

#[async_trait]
impl<T: Recover + Send + Sync> Recover for Cluster<T> {
    async fn recover(&self, txn: &Txn) -> TCResult<()> {
        self.state.recover(txn).await
    }
}

impl<T> fmt::Debug for Cluster<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "cluster at {}{}", self.schema.host, self.schema.path)
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
    host: Option<Host>,
    children: HashSet<PathSegment>,
}
