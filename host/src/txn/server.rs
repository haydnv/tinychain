use std::collections::hash_map::{Entry, HashMap};
use std::sync::Arc;

use futures_locks::RwLock;

use error::*;

use crate::fs;
use crate::gateway::Gateway;

use super::{Request, Txn, TxnId};

#[derive(Clone)]
pub struct TxnServer {
    active: RwLock<HashMap<TxnId, Txn>>,
    workspace: fs::Dir,
}

impl TxnServer {
    pub async fn new(workspace: fs::Dir) -> Self {
        let active = RwLock::new(HashMap::new());
        Self { active, workspace }
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
                let txn = Txn::new(gateway, self.workspace.clone(), request);
                entry.insert(txn.clone());
                Ok(txn)
            }
        }
    }
}
