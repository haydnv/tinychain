use std::sync::Arc;

use crate::block::Dir;
use crate::error::TCResult;
use crate::gateway::Gateway;

use super::{Txn, TxnId};

pub struct TxnServer {
    workspace: Arc<Dir>,
}

impl TxnServer {
    pub fn new(workspace: Arc<Dir>) -> Self {
        Self { workspace }
    }

    pub async fn new_txn(
        &self,
        gateway: Arc<Gateway>,
        txn_id: Option<TxnId>,
    ) -> TCResult<Arc<Txn>> {
        let txn_id = txn_id.unwrap_or_else(|| TxnId::new(Gateway::time()));
        Txn::new(gateway, self.workspace.clone(), txn_id).await
    }
}
