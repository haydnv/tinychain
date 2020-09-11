use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;

use crate::auth::Auth;
use crate::chain::Chain;
use crate::class::TCResult;
use crate::transaction::lock::{Mutable, TxnLock};
use crate::transaction::{Transact, TxnId};
use crate::value::link::{LinkHost, PathSegment, TCPath};

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
    state: Arc<ClusterState>,
}

impl Cluster {
    pub fn create(path: TCPath) -> TCResult<Cluster> {
        let replica = ClusterReplica::default();
        let state = Arc::new(ClusterState {
            replica,
            data: TxnLock::new(format!("Cluster {} data", &path), HashMap::new().into()),
        });

        Ok(Cluster { path, state })
    }

    pub async fn put(
        self,
        txn_id: TxnId,
        name: PathSegment,
        chain: Chain,
        _auth: &Auth,
    ) -> TCResult<Self> {
        let mut data = self.state.data.write(txn_id).await?;
        data.insert(name, chain);
        Ok(self)
    }
}

#[async_trait]
impl Transact for Cluster {
    async fn commit(&self, txn_id: &TxnId) {
        self.state.data.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.state.data.rollback(txn_id).await
    }
}

impl fmt::Display for Cluster {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Cluster at {}", &self.path)
    }
}
