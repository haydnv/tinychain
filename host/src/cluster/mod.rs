//! Maintains the consistency of the network by coordinating transaction commits.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::iter::FromIterator;
use std::ops::Deref;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{join_all, try_join_all, Future, FutureExt};
use futures::{join, StreamExt};
use log::{debug, info, warn};
use safecast::TryCastFrom;
use uplock::RwLock;

use tc_error::*;
use tc_transact::lock::{Mutable, TxnLock};
use tc_transact::{Transact, Transaction};
use tcgeneric::*;

use crate::chain::{Chain, ChainInstance};
use crate::object::InstanceClass;
use crate::scalar::{Link, OpDef, Value};
use crate::state::State;
use crate::txn::{Actor, Scope, Txn, TxnId};

use owner::Owner;

use futures::stream::FuturesUnordered;
pub use load::instantiate;

mod load;
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

/// The data structure responsible for maintaining consensus per-transaction.
pub struct Cluster {
    link: Link,
    actor: Arc<Actor>,
    chains: Map<Chain>,
    classes: Map<InstanceClass>,
    confirmed: RwLock<TxnId>,
    owned: RwLock<HashMap<TxnId, Owner>>,
    installed: TxnLock<Mutable<HashMap<Link, HashSet<Scope>>>>,
    replicas: TxnLock<Mutable<HashSet<Link>>>,
}

impl Cluster {
    /// Borrow one of this cluster's [`Chain`]s.
    pub fn chain(&self, name: &Id) -> Option<&Chain> {
        self.chains.get(name)
    }

    /// Borrow an [`InstanceClass`], if there is one defined with the given name.
    pub fn class(&self, name: &Id) -> Option<&InstanceClass> {
        self.classes.get(name)
    }

    /// Borrow the public key of this cluster.
    pub fn public_key(&self) -> &[u8] {
        self.actor.public_key().as_bytes()
    }

    /// Return the canonical [`Link`] to this cluster (probably not on this host).
    pub fn link(&self) -> &Link {
        &self.link
    }

    /// Return the path of this cluster, relative to this host.
    pub fn path(&'_ self) -> &'_ [PathSegment] {
        self.link.path()
    }

    /// Return the names of the members of this cluster.
    pub fn ns(&self) -> impl Iterator<Item = &Id> {
        self.chains.keys().chain(self.classes.keys())
    }

    /// Iterate over a list of replicas of this cluster.
    pub async fn replicas(&self, txn_id: &TxnId) -> TCResult<HashSet<Link>> {
        let replicas = self.replicas.read(txn_id).await?;
        Ok(replicas.deref().clone())
    }

    /// Claim ownership of the given [`Txn`].
    pub async fn claim(&self, txn: &Txn) -> TCResult<Txn> {
        self.validate_txn_id(txn.id()).await?;

        let mut owned = self.owned.write().await;
        if owned.contains_key(txn.id()) {
            return Err(TCError::bad_request("received an unclaimed transaction, but there is a record of an owner for this transaction at cluster", self.link.path()));
        }

        let txn = txn
            .clone()
            .claim(&self.actor, self.link.path().clone())
            .await?;

        owned.insert(*txn.id(), Owner::new());
        Ok(txn)
    }

    /// Return `Unauthorized` if the request does not have the given `scope` from a trusted issuer.
    pub async fn authorize(&self, txn: &Txn, scope: &Scope) -> TCResult<()> {
        debug!("authorize scope {}...", scope);

        let installed = self.installed.read(txn.id()).await?;
        debug!("{} authorized callers installed", installed.len());

        for (host, actor_id, scopes) in txn.request().scopes().iter() {
            debug!(
                "token has scopes {} issued by {}: {}",
                Tuple::<Scope>::from_iter(scopes.to_vec()),
                host,
                actor_id
            );

            if actor_id.is_none() {
                if let Some(authorized) = installed.get(host) {
                    if authorized.contains(scope) {
                        if scopes.contains(scope) {
                            return Ok(());
                        }
                    }
                }
            }
        }

        Err(TCError::unauthorized(format!(
            "no trusted caller authorized the required scope \"{}\"",
            scope
        )))
    }

    /// Grant the given `scope` to the `txn` and use it to resolve the given `OpRef`.
    pub async fn grant(
        &self,
        txn: Txn,
        scope: Scope,
        op: OpDef,
        context: Map<State>,
    ) -> TCResult<State> {
        debug!("Cluster received grant request for scope {}", scope);

        // TODO: require `SCOPE_EXECUTE` in order to grant a scope
        let txn = txn
            .grant(&self.actor, self.link.path().clone(), vec![scope])
            .await?;

        OpDef::call(op.into_form(), txn, context).await
    }

    /// Trust the `Cluster` at the given [`Link`] to issue the given auth [`Scope`]s.
    pub async fn install(
        &self,
        txn_id: TxnId,
        other: Link,
        scopes: HashSet<Scope>,
    ) -> TCResult<()> {
        info!(
            "{} will now trust {} to issue scopes [{}]",
            self,
            other,
            scopes
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<String>>()
                .join(", ")
        );

        let mut installed = self.installed.write(txn_id).await?;
        installed.insert(other, scopes);
        Ok(())
    }

    /// Claim leadership of the given [`Txn`].
    pub async fn lead(&self, txn: Txn) -> TCResult<Txn> {
        self.validate_txn_id(txn.id()).await?;
        txn.lead(&self.actor, self.link.path().clone()).await
    }

    /// Add a replica to this cluster.
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

            self.replicate(txn).await?;
        } else {
            debug!("add replica {}", replica);
            (*self.replicas.write(*txn.id()).await?).insert(replica);
        }

        Ok(())
    }

    /// Remove a replica from this cluster.
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

    async fn validate_txn_id(&self, txn_id: &TxnId) -> TCResult<()> {
        let last_commit = self.confirmed.read().await;
        if txn_id <= &*last_commit {
            Err(TCError::unsupported(format!(
                "cluster at {} cannot claim transaction {} because the last commit is at {}",
                self.link, txn_id, *last_commit
            )))
        } else {
            Ok(())
        }
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

    pub async fn replicate_write<F: Future<Output = TCResult<()>>, W: Fn(Link) -> F>(
        &self,
        txn: Txn,
        write: W,
    ) -> TCResult<()> {
        let mut replicas = self.replicas(txn.id()).await?;
        replicas.remove(&txn.link(self.link().path().clone()));
        debug!("replicating write to {} replicas", replicas.len());

        let max_failures = replicas.len() / 2;
        let mut failed = HashSet::with_capacity(replicas.len());
        let mut succeeded = HashSet::with_capacity(replicas.len());

        {
            let mut results = FuturesUnordered::from_iter(
                replicas
                    .into_iter()
                    .map(|link| write(link.clone()).map(|result| (link, result))),
            );

            while let Some((replica, result)) = results.next().await {
                match result {
                    Err(cause) if cause.code() == ErrorType::Conflict => return Err(cause),
                    Err(ref cause) => {
                        debug!("replica at {} failed: {}", replica, cause);
                        failed.insert(replica);
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
                    .map(|replica| txn.delete(replica.append(REPLICAS.into()), failed.clone())),
            )
            .await?;
        }

        Ok(())
    }

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

    pub async fn distribute_commit(&self, txn: Txn) -> TCResult<()> {
        let replicas = self.replicas.read(txn.id()).await?;

        if let Some(owner) = self.owned.read().await.get(txn.id()) {
            owner.commit(&txn).await?;
        }

        self.write_ahead(txn.id()).await;

        let self_link = txn.link(self.link.path().clone());
        let mut replica_commits = FuturesUnordered::from_iter(
            replicas
                .iter()
                .filter(|replica| *replica != &self_link)
                .map(|replica| {
                    debug!("commit replica {}...", replica);
                    txn.post(replica.clone(), State::Map(Map::default()))
                }),
        );

        while let Some(result) = replica_commits.next().await {
            match result {
                Ok(_) => {}
                Err(cause) => log::error!("commit failure: {}", cause),
            }
        }

        self.commit(txn.id()).await;

        Ok(())
    }

    pub async fn distribute_rollback(&self, txn: Txn) {
        let replicas = self.replicas.read(txn.id()).await;

        if let Some(owner) = self.owned.read().await.get(txn.id()) {
            if let Err(cause) = owner.rollback(&txn).await {
                warn!("failed to rollback transaction: {}", cause);
            }
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

    pub async fn write_ahead(&self, txn_id: &TxnId) {
        join_all(self.chains.values().map(|chain| chain.write_ahead(txn_id))).await;
    }
}

impl Eq for Cluster {}

impl PartialEq for Cluster {
    fn eq(&self, other: &Self) -> bool {
        self.path() == other.path()
    }
}

impl Hash for Cluster {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.path().hash(h)
    }
}

impl Instance for Cluster {
    type Class = ClusterType;

    fn class(&self) -> Self::Class {
        ClusterType
    }
}

#[async_trait]
impl Transact for Cluster {
    async fn commit(&self, txn_id: &TxnId) {
        let mut confirmed = self.confirmed.write().await;
        {
            debug!(
                "replicas at commit: {}",
                Value::from_iter(self.replicas.read(txn_id).await.unwrap().iter().cloned())
            );
        }

        join_all(self.chains.values().map(|chain| chain.commit(txn_id))).await;
        join!(self.installed.commit(txn_id), self.replicas.commit(txn_id));

        {
            debug!(
                "replicas after commit: {}",
                Value::from_iter(self.replicas.read(txn_id).await.unwrap().iter().cloned())
            );
        }

        if txn_id > &*confirmed {
            *confirmed = *txn_id;
        }
    }

    async fn finalize(&self, txn_id: &TxnId) {
        join_all(self.chains.values().map(|chain| chain.finalize(txn_id))).await;
        self.owned.write().await.remove(txn_id);
        join!(
            self.installed.finalize(txn_id),
            self.replicas.finalize(txn_id)
        );
    }
}

impl fmt::Display for Cluster {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Cluster {}", self.link.path())
    }
}
