use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::Stream;

use crate::block::Dir;
use crate::chain::Chain;
use crate::class::{State, TCResult};
use crate::error;
use crate::gateway::Gateway;
use crate::request::Request;
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
        _request: &Request,
        _gateway: &Gateway,
        _txn: &Txn,
        _path: TCPath,
        _key: Value,
    ) -> TCResult<State> {
        Err(error::not_implemented("Cluster::get"))
    }

    pub async fn put(
        &self,
        _request: &Request,
        _gateway: &Gateway,
        _txn: &Txn,
        _path: &TCPath,
        _key: Value,
        _state: State,
    ) -> TCResult<()> {
        Err(error::not_implemented("Gateway::put"))
    }

    pub async fn post<S: Stream<Item = (ValueId, Scalar)> + Send + Sync + Unpin>(
        self,
        _request: &Request,
        _txn: &Txn,
        _path: TCPath,
        _data: S,
    ) -> TCResult<State> {
        Err(error::not_implemented("Gateway::post"))
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
