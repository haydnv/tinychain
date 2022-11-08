//! Maintains the consistency of the network by coordinating transaction commits.

use std::collections::{HashMap, HashSet};
use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::iter::FromIterator;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{join_all, try_join_all, Future, FutureExt, TryFutureExt};
use futures::stream::{FuturesUnordered, StreamExt};
use log::{debug, warn};
use safecast::TryCastFrom;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use tc_error::*;
use tc_transact::fs::Persist;
use tc_transact::lock::{TxnLock, TxnLockCommitGuard, TxnLockReadGuard, TxnMapRead};
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

use crate::fs;
pub use dir::{Dir, DirEntry, DirItem};
pub use library::Library;
pub use load::instantiate;

pub mod dir;
pub mod library;
mod load; // TODO: delete
mod owner;

/// The name of the endpoint which serves a [`Link`] to each of this [`Cluster`]'s replicas.
pub const REPLICAS: Label = label("replicas");

#[derive(Deserialize, Serialize)]
pub struct Config<T>
where
    T: Persist<fs::Dir>,
    <T as Persist<fs::Dir>>::Schema: DeserializeOwned + Serialize,
{
    link: Link,
    config: <T as Persist<fs::Dir>>::Schema,
}

impl<T> Clone for Config<T>
where
    T: Persist<fs::Dir>,
    <T as Persist<fs::Dir>>::Schema: DeserializeOwned + Serialize,
{
    fn clone(&self) -> Self {
        Self {
            link: self.link.clone(),
            config: self.config.clone(),
        }
    }
}

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
    pub fn with_state(link: Link, state: T) -> Self {
        let replicas = [&link].iter().map(|link| (*link).clone()).collect();

        Self {
            inner: Arc::new(Inner {
                replicas: TxnLock::new(format!("replicas of {}", link), replicas),
                actor: Arc::new(Actor::new(Value::None)),
                owned: RwLock::new(HashMap::new()),
                link,
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

    /// Commit the given [`Txn`] for all members of this `Cluster`.
    pub async fn distribute_commit(&self, txn: &Txn) -> TCResult<()> {
        debug!("distribute commit of {}", self);

        {
            let replicas = self.inner.replicas.read(*txn.id()).await?;

            if let Some(owner) = self.inner.owned.read().await.get(txn.id()) {
                owner.commit(txn).await?;
            }

            let self_link = txn.link(self.inner.link.path().clone());
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
    T: Replica + Send + Sync,
{
    /// Get the set of current replicas of this `Cluster`.
    pub async fn replicas(&self, txn_id: TxnId) -> TCResult<TxnLockReadGuard<HashSet<Link>>> {
        self.inner.replicas.read(txn_id).await
    }

    /// Add a replica to this `Cluster`.
    pub async fn add_replica(&self, txn: &Txn, replica: Link) -> TCResult<()> {
        let self_link = txn.link(self.link().path().clone());

        debug!("cluster at {} adding replica {}...", self_link, replica);

        if replica == self_link {
            if self.link().host().is_none() || self.link() == &self_link {
                debug!("{} cannot replicate itself", self);
                return Ok(());
            }

            debug!(
                "{} replica at {} got add request for self: {}",
                self, replica, self_link
            );

            let replicas = txn
                .get(self.link().clone().append(REPLICAS.into()), Value::None)
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

                (*self.inner.replicas.write(*txn.id()).await?).extend(replicas);
            } else {
                warn!("{} has no other replicas", self);
            }

            self.inner.state.replicate(txn, self.link()).await?;
        } else {
            debug!("add replica {}", replica);
            (*self.inner.replicas.write(*txn.id()).await?).insert(replica);
        }

        Ok(())
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

#[async_trait]
impl<T> Persist<fs::Dir> for Cluster<T>
where
    T: Persist<fs::Dir, Txn = Txn>,
    <T as Persist<fs::Dir>>::Schema: DeserializeOwned + Serialize,
    <T as Persist<fs::Dir>>::Store: TryFrom<fs::Store, Error = TCError>,
{
    type Schema = Config<T>;
    type Store = fs::Store;
    type Txn = Txn;

    async fn create(txn: &Self::Txn, schema: Self::Schema, store: Self::Store) -> TCResult<Self> {
        let Config { link, config } = schema;
        let store = store.try_into()?;

        T::create(txn, config, store)
            .map_ok(|state| Self::with_state(link, state))
            .await
    }

    async fn load(txn: &Self::Txn, schema: Self::Schema, store: Self::Store) -> TCResult<Self> {
        let Config { link, config } = schema;
        let store = store.try_into()?;

        T::load(txn, config, store)
            .map_ok(|state| Self::with_state(link, state))
            .await
    }
}

impl<T> Cluster<Dir<T>>
where
    Self: Clone,
    DirEntry<T>: Clone,
{
    pub async fn create_dir(&self, txn: &Txn, name: PathSegment) -> TCResult<()> {
        self.state().create_dir(txn, self.link(), name).await
    }

    pub fn lookup<'a>(
        &self,
        txn_id: TxnId,
        path: &'a [PathSegment],
    ) -> TCResult<(&'a [PathSegment], DirEntry<T>)> {
        if path.is_empty() {
            return Ok((path, DirEntry::Dir(self.clone())));
        }

        // IMPORTANT! Only use synchronous lock acquisition!
        // async lock acquisition here could cause deadlocks
        // and make services in this `Dir` impossible to update
        let contents = self.state().contents(txn_id)?;
        match contents.get(&path[0]) {
            None => Ok((path, DirEntry::Dir(self.clone()))),
            Some(DirEntry::Item(item)) => Ok((&path[1..], DirEntry::Item(item))),
            Some(DirEntry::Dir(dir)) => dir.lookup(txn_id, &path[1..]),
        }
    }
}

impl<T> Cluster<Dir<T>>
where
    T: DirItem,
    T: Persist<fs::Dir, Schema = ()>,
    DirEntry<T>: Clone,
{
    pub async fn create_item(
        &self,
        txn: &Txn,
        name: PathSegment,
        number: VersionNumber,
        state: State,
    ) -> TCResult<()> {
        self.state()
            .create_item(txn, self.link(), name, number, state)
            .await
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
        let guard = self.inner.replicas.commit(txn_id).await;
        self.inner.state.commit(txn_id).await;
        guard
    }

    async fn finalize(&self, txn_id: &TxnId) {
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
        write!(f, "Cluster {}", self.link().path())
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
