//! A server to keep track of active transactions.

use std::collections::hash_map::{Entry, HashMap};
use std::convert::TryFrom;
use std::sync::Arc;

use freqfs::DirLock;
use futures::future::TryFutureExt;
use log::{debug, trace};
use tokio::sync::RwLock;

use tc_error::*;
use tcgeneric::NetworkTime;

use crate::block::CacheBlock;

use super::request::*;
use super::{Active, Gateway, Txn, TxnId};

/// Server to keep track of the transactions currently active for this host.
#[derive(Clone)]
pub struct TxnServer {
    active: Arc<RwLock<HashMap<TxnId, Arc<Active>>>>,
    workspace: DirLock<CacheBlock>,
}

impl TxnServer {
    /// Construct a new `TxnServer`.
    pub async fn new(workspace: DirLock<CacheBlock>) -> Self {
        Self {
            active: Arc::new(RwLock::new(HashMap::new())),
            workspace,
        }
    }

    /// Return the active `Txn` with the given [`TxnId`], or initiate a new [`Txn`].
    pub async fn new_txn<State>(
        &self,
        gateway: Arc<Box<dyn Gateway<State = State>>>,
        txn_id: TxnId,
        token: (String, Claims),
    ) -> TCResult<Txn<State>> {
        debug!("TxnServer::new_txn");

        let expires = NetworkTime::try_from(token.1.expires())?;
        let request = Request::new(txn_id, token.0, token.1);
        let mut active = self.active.write().await;

        match active.entry(txn_id) {
            Entry::Occupied(entry) => {
                trace!("txn {} is already known", txn_id);
                Ok(Txn::new(entry.get().clone(), gateway, request))
            }
            Entry::Vacant(entry) => {
                trace!("creating new workspace for txn {}...", txn_id);
                let workspace = self.txn_dir(txn_id).await?;
                let active = Arc::new(Active::new(&txn_id, workspace.clone(), expires));
                let txn = Txn::new(active.clone(), gateway, request);
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

    pub async fn finalize_expired<G>(&self, gateway: &G, now: NetworkTime)
    where
        G: Gateway,
    {
        let mut active = self.active.write().await;

        let expired = active
            .iter()
            .filter(|(_, active)| active.expires() <= &now)
            .map(|(txn_id, _)| txn_id)
            .copied()
            .collect::<Vec<TxnId>>();

        for txn_id in expired {
            assert!(active.remove(&txn_id).is_some());

            gateway.finalize(txn_id).await;

            let workspace = {
                let workspace = self.workspace.read().await;
                workspace.get_dir(&txn_id).cloned()
            };

            if let Some(workspace) = workspace {
                workspace
                    .write()
                    .await
                    .truncate_and_sync()
                    .await
                    .expect("finalize txn workspace");
            }
        }
    }

    async fn txn_dir(&self, txn_id: TxnId) -> TCResult<DirLock<CacheBlock>> {
        let mut workspace = self.workspace.write().await;

        workspace
            .create_dir(txn_id.to_string())
            .map_err(TCError::from)
    }
}
