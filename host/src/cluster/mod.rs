//! Maintains the consistency of the network by coordinating transaction commits.
//!
//! INCOMPLETE AND UNSTABLE.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::iter::FromIterator;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::join_all;
use log::{debug, info};
use uplock::RwLock;

use tc_error::*;
use tc_transact::lock::{Mutable, TxnLock};
use tc_transact::{Transact, Transaction};
use tcgeneric::*;

use crate::chain::Chain;
use crate::object::InstanceClass;
use crate::scalar::{Link, OpDef};
use crate::state::State;
use crate::txn::{Actor, Scope, Txn, TxnId};

mod load;
mod owner;

use owner::Owner;

pub use load::instantiate;

/// The [`Class`] of a [`Cluster`].
pub struct ClusterType;

impl Class for ClusterType {
    type Instance = Cluster;
}

impl fmt::Display for ClusterType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("Cluster")
    }
}

/// The data structure responsible for maintaining consensus per-transaction.
pub struct Cluster {
    actor: Arc<Actor>,
    path: TCPathBuf,
    chains: Map<Chain>,
    classes: Map<InstanceClass>,
    confirmed: RwLock<TxnId>,
    owned: RwLock<HashMap<TxnId, Owner>>,
    installed: TxnLock<Mutable<HashMap<Link, HashSet<Scope>>>>,
}

impl Cluster {
    /// Claim ownership of the given [`Txn`].
    pub async fn claim(&self, txn: &Txn) -> TCResult<Txn> {
        let last_commit = self.confirmed.read().await;
        if txn.id() <= &*last_commit {
            return Err(TCError::unsupported(format!(
                "cluster at {} cannot claim transaction {} because the last commit is at {}",
                &self.path,
                txn.id(),
                &*last_commit
            )));
        }

        let mut owned = self.owned.write().await;
        if owned.contains_key(txn.id()) {
            return Err(TCError::bad_request("received an unclaimed transaction, but there is a record of an owner for this transaction at cluster", &self.path));
        }

        let txn = txn.clone().claim(&self.actor, self.path.clone()).await?;
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

    /// Grant the given `scope` to the `txn` and use it to resolve the given [`OpRef`].
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
            .grant(&self.actor, self.path.clone(), vec![scope])
            .await?;

        OpDef::call(op.into_def(), txn, context).await
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

    /// Return the [`Owner`] of the given transaction.
    pub async fn owner(&self, txn_id: &TxnId) -> TCResult<Owner> {
        self.owned
            .read()
            .await
            .get(txn_id)
            .cloned()
            .ok_or_else(|| TCError::bad_request("cluster does not own transaction", txn_id))
    }
}

impl Eq for Cluster {}

impl PartialEq for Cluster {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
    }
}

impl Hash for Cluster {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.path.hash(h)
    }
}

impl Cluster {
    /// Borrow one of this `Cluster`'s [`Chain`]s.
    pub fn chain(&self, name: &Id) -> Option<&Chain> {
        self.chains.get(name)
    }

    /// Borrow an [`InstanceClass`], if there is one defined with the given name.
    pub fn class(&self, name: &Id) -> Option<&InstanceClass> {
        self.classes.get(name)
    }

    /// Borrow the public key of this `Cluster`.
    pub fn public_key(&self) -> &[u8] {
        self.actor.public_key().as_bytes()
    }

    /// Return the path of this cluster, relative to this host.
    pub fn path(&'_ self) -> &'_ [PathSegment] {
        &self.path
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

        join_all(self.chains.values().map(|chain| chain.commit(txn_id))).await;

        *confirmed = *txn_id;

        self.installed.commit(txn_id).await;
    }

    async fn finalize(&self, txn_id: &TxnId) {
        join_all(self.chains.values().map(|chain| chain.finalize(txn_id))).await;
        self.owned.write().await.remove(txn_id);
        self.installed.finalize(txn_id).await;
    }
}

impl fmt::Display for Cluster {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Cluster at {}", self.path)
    }
}
