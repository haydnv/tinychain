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
use crate::transaction::lock::{Mutable, TxnLock};
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::link::{LinkHost, PathSegment, TCPath};
use crate::value::Value;

enum ClusterReplica {
    Director(HashSet<LinkHost>), // set of all hosts replicating this cluster
    Actor(LinkHost),             // link to the director
}

impl Default for ClusterReplica {
    fn default() -> ClusterReplica {
        ClusterReplica::Director(HashSet::new())
    }
}

struct ClusterState {
    replica: ClusterReplica,
    data: TxnLock<Mutable<HashMap<PathSegment, Chain>>>,
}

#[derive(Clone)]
pub struct Cluster {
    path: TCPath,
    data_dir: Arc<Dir>,
    workspace: Arc<Dir>,
    state: Arc<ClusterState>,
}

impl Cluster {
    pub fn create(path: TCPath, data_dir: Arc<Dir>, workspace: Arc<Dir>) -> TCResult<Cluster> {
        let replica = ClusterReplica::default();
        let state = Arc::new(ClusterState {
            replica,
            data: TxnLock::new(format!("Cluster {} data", &path), HashMap::new().into()),
        });

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

            let data = self.state.data.read(txn.id()).await?;
            println!("Cluster hosts {} chains at txn {}", data.len(), txn.id());
            if let Some(chain) = data.get(&path[0]) {
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
        let mut data = self.state.data.write(txn.id().clone()).await?;
        data.insert(name, chain);
        Ok(self)
    }
}

#[async_trait]
impl Transact for Cluster {
    async fn commit(&self, txn_id: &TxnId) {
        println!("Cluster::commit!");
        self.state.data.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        println!("Cluster::rollback!");
        self.state.data.rollback(txn_id).await
    }
}

impl fmt::Display for Cluster {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Cluster at {}", &self.path)
    }
}
