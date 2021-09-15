//! A server to keep track of active transactions.

use std::collections::hash_map::{Entry, HashMap};
use std::convert::TryInto;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use futures::future::TryFutureExt;
use futures::stream::{FuturesUnordered, StreamExt};
use log::{debug, error};
use tokio::sync::RwLock;

use tc_error::*;
use tc_transact::fs::Dir;
use tc_transact::Transact;

use crate::fs;
use crate::gateway::Gateway;

use super::request::*;
use super::{Active, Txn, TxnId};

const GRACE: Duration = Duration::from_secs(3);
const INTERVAL: Duration = Duration::from_secs(1);

/// Server to keep track of the transactions currently active for this host.
#[derive(Clone)]
pub struct TxnServer {
    active: Arc<RwLock<HashMap<TxnId, Arc<Active>>>>,
    workspace: std::path::PathBuf,
    cache: fs::Cache,
}

impl TxnServer {
    /// Construct a new `TxnServer`.
    pub async fn new(workspace: std::path::PathBuf, cache: fs::Cache) -> Self {
        let active = Arc::new(RwLock::new(HashMap::new()));

        spawn_cleanup_thread(workspace.clone(), active.clone());

        Self {
            active,
            workspace,
            cache,
        }
    }

    /// Return the active `Txn` with the given [`TxnId`], or initiate a new [`Txn`].
    pub async fn new_txn(
        &self,
        gateway: Arc<Gateway>,
        txn_id: TxnId,
        token: (String, Claims),
    ) -> TCResult<Txn> {
        debug!("TxnServer::new_txn");

        let expires = token.1.expires().try_into()?;
        let request = Request::new(txn_id, token.0, token.1);
        let mut active = self.active.write().await;

        match active.entry(txn_id) {
            Entry::Occupied(entry) => {
                let active = entry.get();
                let dir = active.workspace.create_dir_tmp(txn_id).await?;
                Ok(Txn::new(active.clone(), gateway, dir, request))
            }
            Entry::Vacant(entry) => {
                let workspace = self.txn_dir(txn_id);
                let dir = workspace.create_dir_tmp(txn_id).await?;
                let active = Arc::new(Active::new(&txn_id, workspace, expires));
                let txn = Txn::new(active.clone(), gateway, dir, request);
                entry.insert(active);
                Ok(txn)
            }
        }
    }

    /// Gracefully shut down this `TxnServer` by allowing all active transactions to drain.
    pub async fn shutdown(self) -> TCResult<()> {
        debug!("TxnServer::shutdown");

        tokio::spawn(async move {
            let result = loop {
                if self.active.read().await.is_empty() {
                    break TCResult::Ok(());
                } else {
                    debug!("TxnServer::shutdown pending active transactions");
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                }
            };

            result
        })
        .map_err(|e| TCError::internal(format!("failed to schedule graceful shutdown: {}", e)))
        .await?
    }

    fn txn_dir(&self, txn_id: TxnId) -> fs::Dir {
        let mut path = self.workspace.clone();
        path.push(txn_id.to_string());
        fs::Dir::new(path, self.cache.clone())
    }
}

fn spawn_cleanup_thread(workspace: PathBuf, active: Arc<RwLock<HashMap<TxnId, Arc<Active>>>>) {
    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(INTERVAL.as_secs()));

    tokio::spawn(async move {
        loop {
            interval.tick().await;
            cleanup(&workspace, &active).await;
        }
    });
}

async fn cleanup(workspace: &PathBuf, txn_pool: &RwLock<HashMap<TxnId, Arc<Active>>>) {
    let now = Gateway::time();
    let mut txn_pool = txn_pool.write().await;
    let expired: Vec<TxnId> = txn_pool
        .iter()
        .filter_map(|(txn_id, active)| {
            if active.expires() + GRACE < now {
                Some(txn_id)
            } else {
                None
            }
        })
        .cloned()
        .collect();

    let mut cleanup: FuturesUnordered<_> = expired
        .into_iter()
        .filter_map(|txn_id| txn_pool.remove(&txn_id).map(|active| (txn_id, active)))
        .map(|(txn_id, active)| async move {
            debug!("clean up txn {}", txn_id);

            // call finalize to clean up the cache
            active.workspace.finalize(&txn_id).await;

            let mut path = workspace.clone();
            path.push(txn_id.to_string());

            if path.exists() {
                tokio::fs::remove_dir_all(path).await
            } else {
                Ok(())
            }
        })
        .collect();

    while let Some(result) = cleanup.next().await {
        if let Err(cause) = result {
            error!("error finalizing transaction: {}", cause);
        }
    }
}
