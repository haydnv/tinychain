//! A server to keep track of active transactions.

use std::cmp::Ordering;
use std::convert::TryFrom;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use ds_ext::OrdHashSet;
use freqfs::DirLock;
use futures::future::TryFutureExt;
use log::*;
use tokio::sync::{mpsc, RwLock};

use tc_error::*;
use tcgeneric::NetworkTime;

use crate::block::CacheBlock;

use super::request::*;
use super::{Gateway, Txn, TxnId};

// allow the end-user request to time out gracefully before garbage collection
const GRACE: Duration = Duration::from_secs(1);

struct Active {
    txn_id: TxnId,
    expires: NetworkTime,
}

impl Active {
    fn new(txn_id: TxnId, expires: NetworkTime) -> Self {
        Self { txn_id, expires }
    }
}

impl Eq for Active {}

impl PartialEq<Self> for Active {
    fn eq(&self, other: &Self) -> bool {
        self.txn_id == other.txn_id
    }
}

impl Ord for Active {
    fn cmp(&self, other: &Self) -> Ordering {
        self.expires
            .cmp(&other.expires)
            .then(self.txn_id.cmp(&other.txn_id))
    }
}

impl Hash for Active {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.txn_id.hash(hasher)
    }
}

impl PartialOrd for Active {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Server to keep track of the transactions currently active for this host.
#[derive(Clone)]
pub struct TxnServer {
    active: Arc<RwLock<OrdHashSet<Active>>>,
    tx: mpsc::UnboundedSender<Active>,
    workspace: DirLock<CacheBlock>,
}

impl TxnServer {
    /// Construct a new `TxnServer`.
    pub async fn new(workspace: DirLock<CacheBlock>) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();

        let server = Self {
            active: Arc::new(RwLock::new(OrdHashSet::new())),
            workspace,
            tx,
        };

        spawn_receiver_thread(server.clone(), rx);

        server
    }

    /// Return the active `Txn` with the given [`TxnId`], or initiate a new [`Txn`].
    pub fn new_txn(
        &self,
        gateway: Arc<dyn Gateway>,
        txn_id: TxnId,
        token: SignedToken,
    ) -> TCResult<Txn> {
        debug!("TxnServer::new_txn");

        let expires = NetworkTime::try_from(token.expires())?;
        let request = Request::new(txn_id, token);

        self.tx
            .send(Active::new(txn_id, expires))
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
        let now = SystemTime::from(now);

        let expired = {
            let mut active = self.active.write().await;
            let mut expired = Vec::with_capacity(active.len());

            while let Some(txn) = active.first() {
                if SystemTime::from(txn.expires) + GRACE < now {
                    let txn = active.pop_first().expect("expired txn id");
                    expired.push(txn.txn_id);
                } else {
                    break;
                }
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

fn spawn_receiver_thread(server: TxnServer, mut rx: mpsc::UnboundedReceiver<Active>) {
    tokio::spawn(async move {
        while let Some(txn) = rx.recv().await {
            let mut active = server.active.write().await;
            active.insert(txn);

            while let Ok(txn) = rx.try_recv() {
                active.insert(txn);
            }
        }
    });
}
