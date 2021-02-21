//! A server to keep track of active transactions.

use std::collections::hash_map::{Entry, HashMap};
use std::sync::Arc;

use futures::TryFutureExt;
use uplock::RwLock;
use uuid::Uuid;

use tc_error::*;
use tc_transact::fs::Dir;
use tc_transact::Transact;
use tcgeneric::PathSegment;

use crate::fs;
use crate::gateway::Gateway;

use super::request::*;
use super::{Active, Txn, TxnId};
use std::convert::TryInto;

/// Server to keep track of the transactions currently active for this host.
#[derive(Clone)]
pub struct TxnServer {
    active: RwLock<HashMap<TxnId, Arc<Active>>>,
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
        let expires = token.1.expires().try_into()?;
        let dir = self.txn_dir(&txn_id).await?;
        let request = Request::new(txn_id, token.0, token.1);
        let mut active = self.active.write().await;

        match active.entry(txn_id) {
            Entry::Occupied(entry) => {
                let active = entry.get();
                Ok(Txn::new(active.clone(), gateway, dir, request))
            }
            Entry::Vacant(entry) => {
                let active = Arc::new(Active::new(&txn_id, expires));
                let txn = Txn::new(active.clone(), gateway, self.workspace.clone(), request);
                entry.insert(active);
                Ok(txn)
            }
        }
    }

    pub async fn shutdown(self) -> TCResult<()> {
        tokio::spawn(async move {
            let result = loop {
                if self.active.read().await.is_empty() {
                    break TCResult::Ok(());
                } else {
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                }
            };

            result
        })
        .map_err(|e| TCError::internal(format!("failed to schedule graceful shutdown: {}", e)))
        .await?
    }

    async fn txn_dir(&self, txn_id: &TxnId) -> TCResult<fs::Dir> {
        let existing_ids = self.workspace.entry_ids(txn_id).await?;
        let id = loop {
            let id: PathSegment = Uuid::new_v4().into();
            if !existing_ids.contains(&id) {
                break id;
            }
        };

        self.workspace.create_dir(*txn_id, id).await
    }
}

fn spawn_cleanup_thread(workspace: fs::Dir, active: RwLock<HashMap<TxnId, Arc<Active>>>) {
    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(1));

    tokio::spawn(async move {
        loop {
            interval.tick().await;
            cleanup(&workspace, &active).await;
        }
    });
}

async fn cleanup(workspace: &fs::Dir, txn_pool: &RwLock<HashMap<TxnId, Arc<Active>>>) {
    let expired = {
        let now = Gateway::time();
        let mut txn_pool = txn_pool.write().await;
        let mut expired = Vec::with_capacity(txn_pool.len());
        for (txn_id, txn) in txn_pool.iter() {
            if txn.expires() < &now {
                expired.push(*txn_id);
            }
        }

        for txn_id in &expired {
            txn_pool.remove(txn_id);
        }

        expired
    };

    for txn_id in expired.into_iter() {
        // TODO: implement delete
        // workspace.delete(txn_id, txn_id.to_path()).await;
        workspace.finalize(&txn_id).await;
    }
}
