use std::collections::hash_map::{Entry, HashMap};
use std::sync::Arc;

use futures_locks::RwLock;

use error::*;

use crate::fs::{CacheDir, DirView};
use crate::gateway::Gateway;

use super::{Request, Txn, TxnId};

pub struct TxnServer {
    active: RwLock<HashMap<TxnId, Txn>>,
    workspace: RwLock<CacheDir>,
}

impl TxnServer {
    pub async fn new(workspace: RwLock<CacheDir>) -> Self {
        Self {
            active: RwLock::new(HashMap::new()),
            workspace: workspace,
        }
    }

    pub async fn new_txn(&self, gateway: Arc<Gateway>, request: Request) -> TCResult<Txn> {
        let mut active = self.active.write().await;

        match active.entry(request.txn_id) {
            Entry::Occupied(entry) => {
                let txn = entry.get();
                if request.contains(txn.request()) {
                    Ok(txn.clone())
                } else {
                    Err(TCError::conflict())
                }
            }
            Entry::Vacant(entry) => {
                let mut workspace = self.workspace.write().await;
                let txn_dir = workspace.create_dir(entry.key().to_id()).await?;
                let txn_dir = RwLock::new(DirView::new(*entry.key(), txn_dir));
                let txn = Txn::new(gateway, txn_dir, request);
                entry.insert(txn.clone());
                Ok(txn)
            }
        }
    }
}
