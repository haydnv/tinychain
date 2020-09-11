use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use crate::auth::Auth;
use crate::chain::Chain;
use crate::class::TCResult;
use crate::error;
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
    data: HashMap<PathSegment, Chain>,
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
            data: HashMap::new(),
        });
        Ok(Cluster { path, state })
    }

    pub fn put(self, _name: PathSegment, _chain: Chain, _auth: &Auth) -> TCResult<Self> {
        Err(error::not_implemented("Cluster::put"))
    }
}

impl fmt::Display for Cluster {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Cluster at {}", &self.path)
    }
}
