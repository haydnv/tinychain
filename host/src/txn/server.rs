use std::collections::hash_map::{Entry, HashMap};
use std::sync::Arc;
use std::thread;

use futures_locks::RwLock;
use tokio::sync::mpsc;

use error::*;
use transact::Transact;

use crate::fs;
use crate::gateway::Gateway;

use super::request::*;
use super::{Txn, TxnId};

#[derive(Clone)]
pub struct TxnServer {
    active: RwLock<HashMap<TxnId, Txn>>,
    sender: mpsc::UnboundedSender<TxnId>,
    workspace: fs::Dir,
}

impl TxnServer {
    pub async fn new(workspace: fs::Dir) -> Self {
        let (sender, mut receiver) = mpsc::unbounded_channel();

        let active = RwLock::new(HashMap::new());
        let active_clone = active.clone();
        let workspace_clone = workspace.clone();
        thread::spawn(move || {
            use futures::executor::block_on;

            while let Some(txn_id) = block_on(receiver.recv()) {
                let txn: Option<Txn> = { block_on(active_clone.write()).remove(&txn_id) };
                if let Some(txn) = txn {
                    // TODO: implement delete
                    // block_on(workspace_clone.delete(txn_id, txn_id.to_path())).unwrap();
                    block_on(txn.finalize(&txn_id));
                    block_on(workspace_clone.finalize(&txn_id));
                }
            }
        });

        Self {
            active,
            sender,
            workspace,
        }
    }

    pub async fn new_txn(
        &self,
        gateway: Arc<Gateway>,
        txn_id: TxnId,
        token: (String, Claims),
    ) -> TCResult<Txn> {
        let mut active = self.active.write().await;

        match active.entry(txn_id) {
            Entry::Occupied(entry) => {
                let txn = entry.get();
                // TODO: authorize access to this Txn
                Ok(txn.clone())
            }
            Entry::Vacant(entry) => {
                let request = Request::new(txn_id, token.0, token.1);
                let txn = Txn::new(
                    self.sender.clone(),
                    gateway,
                    self.workspace.clone(),
                    request,
                );
                entry.insert(txn.clone());
                Ok(txn)
            }
        }
    }
}
