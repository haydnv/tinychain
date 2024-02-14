use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use ds_ext::OrdHashSet;
use freqfs::{DirLock, FileSave};
use log::debug;
use tokio::sync::mpsc;

use crate::gateway::Gateway;
use tc_transact::TxnId;
use tcgeneric::NetworkTime;

use crate::kernel::Kernel;
use crate::txn::{LazyDir, Txn};

// allow an end-user's request to time out gracefully before garbage collection
const GRACE: Duration = Duration::from_secs(1);

#[derive(Debug)]
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
pub struct TxnServer<FE> {
    workspace: DirLock<FE>,
    gateway: Arc<Gateway>,
    active: Arc<RwLock<OrdHashSet<Active>>>,
    tx: mpsc::UnboundedSender<Active>,
    ttl: Duration,
}

impl<FE> Clone for TxnServer<FE> {
    fn clone(&self) -> Self {
        Self {
            workspace: self.workspace.clone(),
            gateway: self.gateway.clone(),
            active: self.active.clone(),
            tx: self.tx.clone(),
            ttl: self.ttl,
        }
    }
}

impl<FE: for<'a> FileSave<'a>> TxnServer<FE> {
    pub fn create(workspace: DirLock<FE>, gateway: Arc<Gateway>, ttl: Duration) -> Self {
        let active = Arc::new(RwLock::new(OrdHashSet::new()));

        let (tx, rx) = mpsc::unbounded_channel();
        spawn_receiver_thread(active.clone(), rx);

        Self {
            workspace,
            gateway,
            active,
            tx,
            ttl,
        }
    }

    pub(crate) async fn finalize<State>(&self, kernel: &Kernel<State, FE>, now: NetworkTime) {
        let expired = {
            let mut active = self.active.write().expect("active transactions");
            let mut expired = Vec::with_capacity(active.len());

            while let Some(txn) = active.first() {
                if txn.expires + GRACE < now {
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
            kernel.finalize(txn_id).await;
        }

        let mut workspace = self.workspace.write().await;

        for txn_id in expired {
            workspace.delete(&txn_id).await;
        }

        workspace.sync().await.expect("sync workspace dir");
    }
}

impl<FE: Send + Sync> TxnServer<FE> {
    pub fn new_txn(&self, now: NetworkTime) -> Txn<FE> {
        let txn_id = TxnId::new(now);
        let expiry = txn_id.time() + self.ttl;
        let workspace = LazyDir::from(self.workspace.clone()).create_dir(txn_id.to_id());
        let gateway = self.gateway.clone();

        self.tx
            .send(Active::new(txn_id, expiry))
            .expect("active txn");

        Txn::new(txn_id, expiry, workspace, gateway, None)
    }
}

fn spawn_receiver_thread(
    active: Arc<RwLock<OrdHashSet<Active>>>,
    mut rx: mpsc::UnboundedReceiver<Active>,
) {
    tokio::spawn(async move {
        while let Some(txn) = rx.recv().await {
            let mut active = active.write().expect("active transaction list");
            active.insert(txn);

            while let Ok(txn) = rx.try_recv() {
                active.insert(txn);
            }
        }
    });
}
