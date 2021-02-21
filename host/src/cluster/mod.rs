//! Maintains the consistency of the network by coordinating transaction commits.
//!
//! INCOMPLETE AND UNSTABLE.

use std::collections::HashMap;
use std::convert::TryInto;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{join_all, TryFutureExt};
use safecast::TryCastInto;
use uplock::RwLock;

use tc_error::*;
use tc_transact::fs::{Dir, Persist};
use tc_transact::{Transact, Transaction, TxnId};
use tcgeneric::*;

use crate::chain::{Chain, ChainType, SyncChain};
use crate::fs;
use crate::object::{InstanceClass, InstanceExt};
use crate::scalar::{Link, OpRef, Scalar, TCRef, Value};
use crate::txn::{Actor, Txn};

mod owner;

use owner::Owner;

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

/// The data structure responsible for Paxos synchronization per-transaction.
#[derive(Clone)]
pub struct Cluster {
    actor: Arc<Actor>,
    path: TCPathBuf,
    chains: Map<Chain>,
    confirmed: RwLock<TxnId>,
    owned: RwLock<HashMap<TxnId, Owner>>,
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
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.path.hash(state)
    }
}

impl Cluster {
    /// Borrow the [`Chain`]s which make up the mutable state of this `Cluster`.
    pub fn chains(&self) -> &Map<Chain> {
        &self.chains
    }

    /// Borrow the public key of this `Cluster`.
    pub fn public_key(&self) -> &[u8] {
        self.actor.public_key().as_bytes()
    }

    /// Load a cluster from the filesystem, or instantiate a new one.
    pub async fn instantiate(
        class: InstanceClass,
        data_dir: fs::Dir,
        txn_id: TxnId,
    ) -> TCResult<InstanceExt<Cluster>> {
        let (path, proto) = class.into_inner();
        let path = path.ok_or_else(|| {
            TCError::unsupported("cluster config must specify the path of the cluster to host")
        })?;

        let path = path.into_path();

        let mut chain_schema = HashMap::new();
        let mut cluster_proto = HashMap::new();
        for (id, scalar) in proto.into_iter() {
            match scalar {
                Scalar::Ref(tc_ref) => {
                    let (ct, schema) = if let TCRef::Op(OpRef::Get((path, schema))) = *tc_ref {
                        let path: TCPathBuf = path.try_into()?;
                        let schema: Value = schema.try_into()?;

                        if let Some(ct) = ChainType::from_path(&path) {
                            (ct, schema)
                        } else {
                            return Err(TCError::bad_request(
                                "Cluster requires its mutable data to be wrapped in a chain, not",
                                path,
                            ));
                        }
                    } else {
                        return Err(TCError::bad_request("expected a Chain but found", tc_ref));
                    };

                    chain_schema.insert(id, (ct, schema));
                }
                Scalar::Op(op_def) => {
                    cluster_proto.insert(id, Scalar::Op(op_def));
                }
                other => return Err(TCError::bad_request(
                    "Cluster member must be a Chain (for mutable data), or an immutable OpDef, not",
                    other,
                )),
            }
        }

        let dir = if let Some(dir) = data_dir.find(&txn_id, &path).await? {
            match dir {
                fs::DirEntry::Dir(dir) => dir,
                _ => {
                    return Err(TCError::bad_request("there is already a file at", &path));
                }
            }
        } else {
            create_dir(data_dir, txn_id, &path).await?
        };

        let mut chains = HashMap::<Id, Chain>::new();
        for (id, (class, schema)) in chain_schema.into_iter() {
            let dir = dir.create_dir(txn_id, id.clone()).await?;
            let chain = match class {
                ChainType::Sync => {
                    let schema = schema
                        .try_cast_into(|v| TCError::bad_request("invalid Chain schema", v))?;

                    SyncChain::load(schema, dir, txn_id)
                        .map_ok(Chain::Sync)
                        .await?
                }
            };

            chains.insert(id, chain);
        }

        let actor_id = Value::from(Link::default());
        let cluster = Cluster {
            actor: Arc::new(Actor::new(actor_id)),
            path: path.clone(),
            chains: chains.into(),
            confirmed: RwLock::new(txn_id),
            owned: RwLock::new(HashMap::new()),
        };

        let class = InstanceClass::new(Some(path.into()), cluster_proto.into());

        Ok(InstanceExt::new(cluster, class))
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
        join_all(self.chains.values().map(|chain| chain.commit(txn_id))).await;

        let mut confirmed = self.confirmed.write().await;
        *confirmed = *txn_id;
    }

    async fn finalize(&self, txn_id: &TxnId) {
        join_all(self.chains.values().map(|chain| chain.finalize(txn_id))).await;
    }
}

impl fmt::Display for Cluster {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Cluster at {}", self.path)
    }
}

async fn create_dir(data_dir: fs::Dir, txn_id: TxnId, path: &[PathSegment]) -> TCResult<fs::Dir> {
    let mut dir = data_dir;
    for name in path {
        dir = dir.create_dir(txn_id, name.clone()).await?;
    }

    Ok(dir)
}
