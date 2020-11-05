use std::collections::HashMap;
use std::sync::Arc;
use std::thread;

use tokio::sync::mpsc;

use crate::block::Dir;
use crate::error::TCResult;
use crate::gateway::Gateway;
use crate::lock::RwLock;

use super::{Txn, TxnId};

pub struct TxnServer {
    workspace: Arc<Dir>,
    sender: mpsc::UnboundedSender<TxnId>,
    txn_pool: RwLock<HashMap<TxnId, Txn>>,
}

impl TxnServer {
    pub fn new(workspace: Arc<Dir>) -> Self {
        let txn_pool: RwLock<HashMap<TxnId, Txn>> = RwLock::new(HashMap::new());
        let (sender, mut receiver) = mpsc::unbounded_channel();

        let txn_pool_clone = txn_pool.clone();
        thread::spawn(move || {
            use futures::executor::block_on;

            while let Some(txn_id) = block_on(receiver.recv()) {
                if let Some(txn) = { block_on(txn_pool_clone.write()).remove(&txn_id) } {
                    block_on(txn.finalize())
                }
            }
        });

        Self {
            workspace,
            sender,
            txn_pool,
        }
    }

    pub async fn new_txn(
        &self,
        gateway: Arc<Gateway>,
        txn_id: Option<TxnId>,
    ) -> TCResult<Arc<Txn>> {
        let txn_id = txn_id.unwrap_or_else(|| TxnId::new(Gateway::time()));
        Txn::new(gateway, self.workspace.clone(), txn_id, self.sender.clone()).await
    }
}
