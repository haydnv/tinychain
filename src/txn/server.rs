use std::collections::hash_map::{Entry, HashMap};

use futures_locks::RwLock;

use error::*;

use super::{Request, Txn, TxnId};

pub struct TxnServer {
    active: RwLock<HashMap<TxnId, Txn>>,
}

impl TxnServer {
    pub fn new() -> Self {
        Self {
            active: RwLock::new(HashMap::new()),
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
                let txn = Txn::new(request);
                entry.insert(txn.clone());
                Ok(txn)
            }
        }
    }
}
