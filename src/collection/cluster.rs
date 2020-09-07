use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::chain::Chain;
use crate::class::TCResult;
use crate::error;
use crate::transaction::Txn;
use crate::value::link::{LinkHost, TCPath};
use crate::value::ValueId;

use super::schema::GraphSchema;
use super::Graph;

enum ClusterReplica {
    Director(HashSet<LinkHost>), // set of all hosts replicating this cluster
    Actor(LinkHost),             // link to the director
}

#[derive(Clone)]
pub struct Cluster {
    path: TCPath,
    chain: HashMap<ValueId, Chain>,
    state: Graph,
}

impl Cluster {
    pub fn create(_txn: Arc<Txn>, _schema: GraphSchema) -> TCResult<Cluster> {
        Err(error::not_implemented("Cluster::create"))
    }
}
