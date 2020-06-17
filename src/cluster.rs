use std::sync::Arc;

use crate::state::Dir;
use crate::transaction::TxnId;

pub struct Cluster {
    context: Arc<Dir>,
}

impl Cluster {
    pub fn new(_txn_id: &TxnId, context: Arc<Dir>) -> Arc<Cluster> {
        Arc::new(Cluster { context })
    }
}
