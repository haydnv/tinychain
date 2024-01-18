//! A server to keep track of active transactions.

use std::cmp::Ordering;
use std::collections::hash_map::{Entry, HashMap};
use std::convert::TryFrom;
use std::sync::Arc;

use freqfs::DirLock;
use futures::future::TryFutureExt;
use log::*;
use tokio::sync::{mpsc, RwLock};

use tc_error::*;
use tcgeneric::NetworkTime;

use crate::block::CacheBlock;

use super::request::*;
use super::{Gateway, Txn, TxnId};

/// Server to keep track of the transactions currently active for this host.
#[derive(Clone)]
pub struct TxnServer {
    active: Arc<RwLock<HashMap<TxnId, NetworkTime>>>,
    workspace: DirLock<CacheBlock>,
    tx: mpsc::UnboundedSender<(TxnId, NetworkTime)>,
}

impl TxnServer {
    /// Construct a new `TxnServer`.
    pub async fn new(workspace: DirLock<CacheBlock>) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();

        let server = Self {
            active: Arc::new(RwLock::new(HashMap::new())),
            workspace,
            tx,
        };

        spawn_receiver_thread(server.clone(), rx);

        server
    }

    /// Return the active `Txn` with the given [`TxnId`], or initiate a new [`Txn`].
    pub fn new_txn<State>(
        &self,
        gateway: Arc<dyn Gateway<State = State>>,
        txn_id: TxnId,
        token: (String, Claims),
    ) -> TCResult<Txn<State>> {
        debug!("TxnServer::new_txn");

        let expires = NetworkTime::try_from(token.1.expires())?;
        let request = Request::new(txn_id, token.0, token.1);

        self.tx
            .send((txn_id, expires))
            .map_err(|cause| internal!("transaction queue failure: {cause}"))?;

        Ok(Txn::new(self.workspace.clone(), gateway, request))
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
        .map_err(|e| internal!("failed to schedule graceful shutdown: {}", e))
        .await?
    }

    pub async fn finalize_expired<G>(&self, gateway: &G, now: NetworkTime)
    where
        G: Gateway,
    {
        let expired = {
            let mut active = self.active.write().await;

            let expired = active
                .iter()
                .filter_map(|(txn_id, expires)| if expires <= &now { Some(txn_id) } else { None })
                .copied()
                .collect::<Vec<TxnId>>();

            for txn_id in &expired {
                active.remove(txn_id);
            }

            expired
        };

        if expired.is_empty() {
            return;
        }

        debug!("TxnServer::finalize_expired");

        for txn_id in expired.iter().copied() {
            gateway.finalize(txn_id).await;
        }

        let mut workspace = self.workspace.write().await;

        for txn_id in expired {
            workspace.delete(&txn_id).await;
        }

        workspace.sync().await.expect("sync workspace dir");
    }
}

fn spawn_receiver_thread(server: TxnServer, mut rx: mpsc::UnboundedReceiver<(TxnId, NetworkTime)>) {
    let new_txn = |active: &mut HashMap<TxnId, NetworkTime>, txn_id, expires| {
        match active.entry(txn_id) {
            Entry::Occupied(mut entry) => {
                trace!("txn {} is already known", txn_id);

                match entry.get().cmp(&expires) {
                    Ordering::Greater | Ordering::Equal => {
                        // no-op
                    }
                    Ordering::Less => {
                        warn!("expiration time of txn {txn_id} extended");
                        entry.insert(expires);
                    }
                }
            }
            Entry::Vacant(entry) => {
                entry.insert(expires);
            }
        }
    };

    tokio::spawn(async move {
        while let Some((txn_id, expires)) = rx.recv().await {
            let mut active = server.active.write().await;
            new_txn(&mut *active, txn_id, expires);

            while let Ok((txn_id, expires)) = rx.try_recv() {
                new_txn(&mut *active, txn_id, expires);
            }
        }
    });
}
