use std::collections::{BTreeMap, HashMap, HashSet};
use std::convert::TryInto;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::future::{self, join_all, try_join_all, TryFutureExt};
use futures_locks::RwLock;
use safecast::TryCastFrom;

use error::*;
use generic::*;
use safecast::TryCastInto;
use transact::fs::{Dir, Persist};
use transact::lock::{Mutable, TxnLock};
use transact::{Transact, Transaction, TxnId};

use crate::chain::{Chain, ChainType, SyncChain};
use crate::fs;
use crate::object::{InstanceClass, InstanceExt};
use crate::route::Public;
use crate::scalar::{Link, OpRef, Scalar, TCRef, Value};
use crate::state::State;
use crate::txn::{Actor, Txn};

pub const PATH: Label = label("cluster");

pub struct ClusterType;

impl Class for ClusterType {
    type Instance = Cluster;
}

impl fmt::Display for ClusterType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("Cluster")
    }
}

#[derive(Clone)]
pub struct Cluster {
    actor: Arc<Actor>,
    path: TCPathBuf,
    chains: Map<Chain>,
    mutated: TxnLock<Mutable<HashSet<Link>>>,
    confirmed: RwLock<TxnId>,
    owned: RwLock<BTreeMap<TxnId, Txn>>,
}

impl Cluster {
    async fn maybe_claim_txn(&self, txn: Txn) -> TCResult<Txn> {
        if txn.owner().is_none() {
            txn.claim(&self.actor, self.path.clone()).await
        } else {
            Ok(txn)
        }
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
            mutated: TxnLock::new(HashSet::new().into()),
            confirmed: RwLock::new(txn_id),
            owned: RwLock::new(BTreeMap::new()),
        };

        let class = InstanceClass::new(Some(path.into()), cluster_proto.into());

        Ok(InstanceExt::new(cluster, class))
    }

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
impl Public for Cluster {
    async fn get(&self, txn: &Txn, path: &[PathSegment], key: Value) -> TCResult<State> {
        if path.is_empty() && key.is_none() {
            let public_key = Bytes::from(self.actor.public_key().as_bytes().to_vec());
            return Ok(State::from(Value::from(public_key)));
        }

        nonempty_path(path)?;

        if let Some(chain) = self.chains.get(&path[0]) {
            let txn = self.maybe_claim_txn(txn.clone()).await?;
            return chain.get(&txn, &path[1..], key).await;
        }

        not_found(path)
    }

    async fn put(&self, txn: &Txn, path: &[PathSegment], key: Value, value: State) -> TCResult<()> {
        if path.is_empty() && key.is_none() {
            let peer = Value::try_cast_from(value, |v| TCError::bad_request("expected a Link", v))?;
            let peer = Link::try_cast_from(peer, |v| TCError::bad_request("expected a Link", v))?;
            let mut mutated = self.mutated.write(*txn.id()).await?;
            mutated.insert(peer);
            return Ok(());
        }

        nonempty_path(path)?;

        if let Some(chain) = self.chains.get(&path[0]) {
            let txn = self.maybe_claim_txn(txn.clone()).await?;
            return chain.put(&txn, &path[1..], key, value).await;
        }

        not_found(path)
    }

    async fn post(&self, txn: &Txn, path: &[PathSegment], params: Map<State>) -> TCResult<State> {
        if path.is_empty() && params.is_empty() {
            // TODO: authorize request using a scope
            txn.commit(txn.id()).await;
            return Ok(State::default());
        }

        nonempty_path(path)?;

        if let Some(chain) = self.chains.get(&path[0]) {
            let txn = self.maybe_claim_txn(txn.clone()).await?;
            return chain.post(&txn, &path[1..], params).await;
        }

        not_found(path)
    }
}

#[async_trait]
impl Transact for Cluster {
    async fn commit(&self, txn_id: &TxnId) {
        {
            let mut owned = self.owned.write().await;
            if let Some(txn) = owned.remove(txn_id) {
                let mut mutated = self.mutated.write(*txn_id).await.unwrap();
                let mutated = mutated.drain();

                // TODO: update Transact to allow returning a TCResult
                try_join_all(
                    mutated
                        .into_iter()
                        .map(|link| txn.post(link, Map::<State>::default().into())),
                )
                .await
                .unwrap();
            }
        }

        join_all(self.chains.values().map(|chain| chain.commit(txn_id))).await;

        let confirmed = *txn_id;
        self.confirmed
            .with_write(move |mut id| future::ready(*id = confirmed))
            .await;
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

#[inline]
fn nonempty_path(path: &[PathSegment]) -> TCResult<()> {
    if path.is_empty() {
        Err(TCError::method_not_allowed(TCPath::from(path)))
    } else {
        Ok(())
    }
}

fn not_found<T>(path: &[PathSegment]) -> TCResult<T> {
    Err(TCError::not_found(TCPath::from(path)))
}
