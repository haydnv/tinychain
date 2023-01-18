//! A server to keep track of active transactions.

use std::collections::hash_map::{Entry, HashMap};
use std::convert::TryFrom;
use std::sync::Arc;
use std::time::Duration;

use freqfs::DirLock;
use futures::future::TryFutureExt;
use log::{debug, trace};
use tokio::sync::RwLock;

use tc_error::*;
use tc_transact::fs::Dir;
use tcgeneric::NetworkTime;

use crate::fs;
use crate::gateway::Gateway;

use super::request::*;
use super::{Active, Txn, TxnId};

const GRACE: Duration = Duration::from_secs(3);
const INTERVAL: Duration = Duration::from_millis(100);

/// Server to keep track of the transactions currently active for this host.
#[derive(Clone)]
pub struct TxnServer {
    active: Arc<RwLock<HashMap<TxnId, Arc<Active>>>>,
    workspace: DirLock<fs::CacheBlock>,
}

impl TxnServer {
    /// Construct a new `TxnServer`.
    pub async fn new(workspace: DirLock<fs::CacheBlock>) -> Self {
        let active = Arc::new(RwLock::new(HashMap::new()));
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
        debug!("TxnServer::new_txn");

        let expires = NetworkTime::try_from(token.1.expires())?;
        let request = Request::new(txn_id, token.0, token.1);
        let mut active = self.active.write().await;

        match active.entry(txn_id) {
            Entry::Occupied(entry) => {
                trace!("txn {} is already known", txn_id);
                let active = entry.get();
                let dir = active.workspace.create_dir_unique(txn_id).await?;
                Ok(Txn::new(active.clone(), gateway, dir, request))
            }
            Entry::Vacant(entry) => {
                trace!("creating new workspace for txn {}...", txn_id);
                let workspace = self.txn_dir(txn_id).await?;
                let dir = workspace.create_dir_unique(txn_id).await?;
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
        .map_err(|e| unexpected!("failed to schedule graceful shutdown: {}", e))
        .await?
    }

    async fn txn_dir(&self, txn_id: TxnId) -> TCResult<fs::Dir> {
        let mut workspace = self.workspace.write().await;
        let cache = workspace
            .create_dir(txn_id.to_string())
            .map_err(fs::io_err)?;

        Ok(fs::Dir::new(cache, txn_id))
    }
}

fn spawn_cleanup_thread(
    workspace: DirLock<fs::CacheBlock>,
    active: Arc<RwLock<HashMap<TxnId, Arc<Active>>>>,
) {
    let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(
        INTERVAL.as_millis() as u64,
    ));

    tokio::spawn(async move {
        loop {
            interval.tick().await;
            cleanup(&workspace, &active).await;
        }
    });
}

async fn cleanup(
    workspace: &DirLock<fs::CacheBlock>,
    txn_pool: &RwLock<HashMap<TxnId, Arc<Active>>>,
) {
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

    let mut workspace = workspace.write().await;
    for txn_id in expired.into_iter() {
        if let Some(_active) = txn_pool.remove(&txn_id) {
            debug!("clean up txn {}", txn_id);
            workspace.delete(txn_id.to_string());
        }
    }
}
