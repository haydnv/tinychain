use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use std::fmt;
use std::sync::Arc;

use futures::stream::Stream;

use crate::auth::Auth;
use crate::block::Dir;
use crate::chain::{Chain, ChainInstance};
use crate::class::{State, TCBoxFuture, TCResult};
use crate::error;
use crate::gateway::Gateway;
use crate::scalar::*;
use crate::transaction::lock::{Mutate, TxnLock};
use crate::transaction::{Transact, Txn, TxnId};

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
    chains: HashMap<PathSegment, Chain>,
}

impl Mutate for ClusterState {
    type Pending = Self;

    fn diverge(&self, _txn_id: &TxnId) -> Self::Pending {
        self.clone()
    }

    fn converge<'a>(&'a mut self, new_value: Self::Pending) -> TCBoxFuture<'a, ()> {
        Box::pin(async move { *self = new_value })
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
                chains: HashMap::new(),
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
            if let Some(chain) = state.chains.get(&path[0]) {
                println!(
                    "Cluster::get chain {}{}: {}",
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
        gateway: Arc<Gateway>,
        txn: Option<Arc<Txn>>,
        path: &TCPath,
        key: Value,
        state: State,
        _auth: &Auth,
    ) -> TCResult<()> {
        let txn = if let Some(txn) = txn {
            txn
        } else {
            Txn::new(gateway.clone(), self.workspace.clone()).await?
        };

        let result = if path == &self.path {
            let name: ValueId = key.try_into()?;
            let chain: Chain = state.try_into()?;

            txn.mutate(self.clone().into()).await;

            println!("Cluster will now host a chain called {}", name);
            let mut state = self.state.write(txn.id().clone()).await?;
            state.chains.insert(name, chain);
            Ok(())
        } else {
            let suffix = path.from_path(&self.path)?;
            if path.is_empty() {
                Err(error::not_found(path))
            } else {
                println!("Cluster::put {}: {} <- {}", path, key, state);
                let cluster_state = self.state.read(txn.id()).await?;
                if let Some(chain) = cluster_state.chains.get(&suffix[0]) {
                    txn.mutate(chain.clone().into()).await;
                    chain
                        .put(txn.clone(), suffix.slice_from(1), key, state)
                        .await
                } else {
                    Err(error::not_found(suffix))
                }
            }
        };

        result?;
        txn.commit().await;
        Ok(())
    }

    pub async fn post<S: Stream<Item = (ValueId, Scalar)> + Send + Sync + Unpin>(
        self,
        txn: Arc<Txn>,
        path: TCPath,
        _data: S,
        _auth: Auth,
    ) -> TCResult<State> {
        if path.is_empty() {
            Err(error::method_not_allowed("Cluster::post"))
        } else if let Some(_chain) = self.state.read(txn.id()).await?.chains.get(&path[0]) {
            Err(error::not_implemented(format!(
                "Cluster::post to chain {}",
                &path[0]
            )))
        } else {
            Err(error::not_found(path))
        }
    }
}

impl Transact for Cluster {
    fn commit<'a>(&'a self, txn_id: &'a TxnId) -> TCBoxFuture<'a, ()> {
        println!("Cluster::commit!");
        self.state.commit(txn_id)
    }

    fn rollback<'a>(&'a self, txn_id: &'a TxnId) -> TCBoxFuture<'a, ()> {
        println!("Cluster::rollback!");
        self.state.rollback(txn_id)
    }
}

impl fmt::Display for Cluster {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Cluster at {}", &self.path)
    }
}
