use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use log::debug;

use crate::block::Dir;
use crate::class::State;
use crate::error;
use crate::general::Map;
use crate::handler::Public;
use crate::request::Request;
use crate::scalar::*;
use crate::transaction::lock::{Mutate, TxnLock};
use crate::transaction::{Transact, Txn, TxnId};
use crate::TCResult;

const ERR_ID: &str = "Invalid Id for Cluster member";

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
    path: TCPathBuf,
    data_dir: Arc<Dir>,
    workspace: Arc<Dir>,
    state: TxnLock<ClusterState>,
}

impl Cluster {
    pub fn create(path: TCPathBuf, data_dir: Arc<Dir>, workspace: Arc<Dir>) -> TCResult<Cluster> {
        let replica = ClusterReplica::default();
        let state = TxnLock::new(
            format!("State of Cluster at {}", path),
            ClusterState { replica },
        );

        Ok(Cluster {
            path,
            data_dir,
            workspace,
            state,
        })
    }
}

#[async_trait]
impl Public for Cluster {
    async fn get(
        &self,
        _request: &Request,
        _txn: &Txn,
        _path: &[PathSegment],
        _key: Value,
    ) -> TCResult<State> {
        Err(error::not_implemented("Cluster::get"))
    }

    async fn put(
        &self,
        _request: &Request,
        _txn: &Txn,
        _path: &[PathSegment],
        _key: Value,
        _value: State,
    ) -> TCResult<()> {
        Err(error::not_implemented("Cluster::put"))
    }

    async fn post(
        &self,
        _request: &Request,
        _txn: &Txn,
        _path: &[PathSegment],
        _params: Map<Scalar>,
    ) -> TCResult<State> {
        Err(error::not_implemented("Cluster::post"))
    }

    async fn delete(
        &self,
        _request: &Request,
        _txn: &Txn,
        _path: &[PathSegment],
        _key: Value,
    ) -> TCResult<()> {
        Err(error::not_implemented("Cluster::delete"))
    }
}

#[async_trait]
impl Transact for Cluster {
    async fn commit(&self, txn_id: &TxnId) {
        debug!("Cluster::commit!");
        self.state.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        debug!("Cluster::rollback!");
        self.state.rollback(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        debug!("Cluster::finalize!");
        self.state.finalize(txn_id).await
    }
}

impl fmt::Display for Cluster {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Cluster at {}", self.path)
    }
}
