use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;

use crate::auth::Auth;
use crate::block::Dir;
use crate::chain::{Chain, ChainInstance};
use crate::class::{State, TCResult};
use crate::error;
use crate::gateway::Gateway;
use crate::transaction::lock::{Mutate, TxnLock};
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::link::{LinkHost, PathSegment, TCPath};
use crate::value::Value;

#[derive(Clone)]
enum ClusterReplica {
    Director(HashSet<LinkHost>), // set of all hosts replicating this cluster
    Actor(LinkHost),             // link to the director
}

impl Default for ClusterReplica {
    fn default() -> ClusterReplica {
        ClusterReplica::Director(HashSet::new())
    }
}

#[derive(Clone)]
struct ClusterState {
    replica: ClusterReplica,
    data: HashMap<PathSegment, Chain>,
}

#[async_trait]
impl Mutate for ClusterState {
    type Pending = Self;

    fn diverge(&self, _txn_id: &TxnId) -> Self::Pending {
        self.clone()
    }

    async fn converge(&mut self, new_value: Self::Pending) {
        *self = new_value
    }
}

#[derive(Clone)]
pub struct Cluster {
    path: TCPath,
    data_dir: Arc<Dir>,
    workspace: Arc<Dir>,
    state: TxnLock<ClusterState>,
}

impl Cluster {
    pub fn create(path: TCPath, data_dir: Arc<Dir>, workspace: Arc<Dir>) -> TCResult<Cluster> {
        let replica = ClusterReplica::default();
        let state = TxnLock::new(
            format!("State of Cluster at {}", &path),
            ClusterState {
                replica,
                data: HashMap::new(),
            },
        );

        Ok(Cluster {
            path,
            data_dir,
            workspace,
            state,
        })
    }

    pub async fn get(
        &self,
        gateway: Arc<Gateway>,
        txn: Option<Arc<Txn>>,
        path: TCPath,
        key: Value,
        auth: Auth,
    ) -> TCResult<State> {
        if path.is_empty() {
            Ok(self.clone().into())
        } else {
            let txn = if let Some(txn) = txn {
                txn
            } else {
                Txn::new(gateway.clone(), self.workspace.clone()).await?
            };

            let state = self.state.read(txn.id()).await?;
            if let Some(chain) = state.data.get(&path[0]) {
                println!(
                    "Cluster::get chain {}/{}: {}",
                    &path[0],
                    path.slice_from(1),
                    &key
                );
                chain.get(txn, &path.slice_from(1), key, auth).await
            } else {
                println!("Cluster has no chain at {}", path[0]);
                Err(error::not_found(path))
            }
        }
    }

    pub async fn put(
        self,
        txn: Arc<Txn>,
        name: PathSegment,
        chain: Chain,
        _auth: &Auth,
    ) -> TCResult<Self> {
        println!("Cluster will now host a chain called {}", name);
        txn.mutate(self.clone().into()).await;
        let mut state = self.state.write(txn.id().clone()).await?;
        state.data.insert(name, chain);
        Ok(self)
    }
}

#[async_trait]
impl Transact for Cluster {
    async fn commit(&self, txn_id: &TxnId) {
        println!("Cluster::commit!");
        self.state.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        println!("Cluster::rollback!");
        self.state.rollback(txn_id).await
    }
}

impl fmt::Display for Cluster {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Cluster at {}", &self.path)
    }
}
