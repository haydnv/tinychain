//! A server to keep track of active transactions.

use std::collections::hash_map::{Entry, HashMap};
use std::sync::Arc;

use futures::join;
use futures_locks::RwLock;

use error::*;
use transact::{Transact, Transaction};

use crate::fs;
use crate::gateway::Gateway;

use super::request::*;
use super::{Txn, TxnId};

/// Server to keep track of the transactions currently active for this host.
#[derive(Clone)]
pub struct TxnServer {
    active: RwLock<HashMap<TxnId, Txn>>,
    workspace: fs::Dir,
}

impl TxnServer {
    /// Construct a new `TxnServer`.
    pub async fn new(workspace: fs::Dir) -> Self {
        let active = RwLock::new(HashMap::new());

        spawn_cleanup_thread(workspace.clone(), active.clone());

        Self { active, workspace }
    }

    /// Return the active `Txn` with the given [`TxnId`], or initiate a new [`Txn`].
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
                let txn = Txn::new(gateway, self.workspace.clone(), request);
                entry.insert(txn.clone());
                Ok(txn)
            }
        }
    }
}

fn spawn_cleanup_thread(workspace: fs::Dir, active: RwLock<HashMap<TxnId, Txn>>) {
    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(1));

    tokio::spawn(async move {
        loop {
            interval.tick().await;
            cleanup(&workspace, &active).await;
        }
    });
}

async fn cleanup(workspace: &fs::Dir, active: &RwLock<HashMap<TxnId, Txn>>) {
    let expired = {
        let mut txn_pool = active.write().await;
        let mut expired_ids = Vec::with_capacity(txn_pool.len());
        for (txn_id, txn) in txn_pool.iter() {
            if txn.ref_count() == 1 && Gateway::time() > txn.request.expires() {
                expired_ids.push(*txn_id);
            }
        }

        expired_ids
            .into_iter()
            .map(move |txn_id| txn_pool.remove(&txn_id).unwrap())
    };

    for txn in expired.into_iter() {
        // TODO: implement delete
        // workspace.delete(txn_id, txn_id.to_path()).await;
        join!(txn.finalize(txn.id()), workspace.finalize(txn.id()));
    }
}
