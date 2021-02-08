use std::collections::hash_map::{Entry, HashMap};
use std::path::PathBuf;
use std::sync::Arc;

use futures_locks::RwLock;

use error::*;

use crate::fs::Root;
use crate::gateway::Gateway;

use super::{Request, Txn, TxnId};

pub struct TxnServer {
    active: RwLock<HashMap<TxnId, Txn>>,
    workspace: Root,
}

impl TxnServer {
    pub async fn new(workspace: PathBuf, cache_size: usize) -> Self {
        let workspace = Root::load(workspace, cache_size).await.unwrap();

        Self {
            active: RwLock::new(HashMap::new()),
            workspace,
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
                let txn_dir = self.workspace.version(*entry.key()).await?;
                let txn = Txn::new(gateway, txn_dir, request);
                entry.insert(txn.clone());
                Ok(txn)
            }
        }
    }
}
