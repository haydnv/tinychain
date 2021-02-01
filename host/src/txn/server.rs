use std::collections::hash_map::{Entry, HashMap};
use std::path::PathBuf;
use std::sync::Arc;

use futures_locks::RwLock;

use error::*;
use transact::fs;

use super::{FileEntry, Request, Txn, TxnId};

pub struct TxnServer {
    active: RwLock<HashMap<TxnId, Txn>>,
    workspace: Arc<fs::Dir<FileEntry>>,
}

impl TxnServer {
    pub async fn new(workspace: PathBuf) -> Self {
        let workspace = fs::mount(workspace).await;
        let workspace = fs::Dir::create(workspace, "txn");

        Self {
            active: RwLock::new(HashMap::new()),
            workspace,
        }
    }

    pub async fn new_txn(&self, request: Request) -> TCResult<Txn> {
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
                let txn_dir = self
                    .workspace
                    .create_dir(request.txn_id, &[request.txn_id.to_id()])
                    .await?;
                let txn = Txn::new(txn_dir, request);
                entry.insert(txn.clone());
                Ok(txn)
            }
        }
    }
}
