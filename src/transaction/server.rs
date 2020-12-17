use std::collections::HashMap;
use std::sync::Arc;
use std::thread;

use tokio::sync::mpsc;

use crate::block::Dir;
use crate::gateway::Gateway;
use crate::general::TCResult;
use crate::lock::RwLock;

use super::{Transact, Txn, TxnId};

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
        let workspace_clone = workspace.clone();
        thread::spawn(move || {
            use futures::executor::block_on;

            while let Some(txn_id) = block_on(receiver.recv()) {
                if let Some(txn) = { block_on(txn_pool_clone.write()).remove(&txn_id) } {
                    block_on(workspace_clone.delete(txn_id, txn_id.to_path())).unwrap();
                    block_on(txn.finalize());
                    block_on(workspace_clone.finalize(&txn_id));
                }
            }
        });

        Self {
            workspace,
            sender,
            txn_pool,
        }
    }

    pub async fn new_txn(&self, gateway: Arc<Gateway>, txn_id: Option<TxnId>) -> TCResult<Txn> {
        let txn_id = txn_id.unwrap_or_else(|| TxnId::new(Gateway::time()));
        let dir = self
            .workspace
            .get_or_create_dir(&txn_id, &[txn_id.to_path()])
            .await?;
        let txn = Txn::new(gateway, dir, txn_id, self.sender.clone()).await?;
        self.txn_pool.write().await.insert(txn_id, txn.clone());
        Ok(txn)
    }
}
