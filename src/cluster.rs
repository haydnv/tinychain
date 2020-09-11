use std::collections::HashSet;
use std::sync::Arc;

use crate::class::TCResult;
use crate::value::link::{LinkHost, TCPath};

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
}

#[derive(Clone)]
pub struct Cluster {
    path: TCPath,
    state: Arc<ClusterState>,
}

impl Cluster {
    pub fn create(path: TCPath) -> TCResult<Cluster> {
        let replica = ClusterReplica::default();
        let state = Arc::new(ClusterState { replica });
        Ok(Cluster { path, state })
    }
}
