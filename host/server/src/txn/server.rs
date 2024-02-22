use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use ds_ext::OrdHashSet;
use freqfs::DirLock;
use log::debug;
use tokio::sync::mpsc;

use tc_error::*;
use tc_state::CacheBlock;
use tc_transact::TxnId;
use tcgeneric::NetworkTime;

use crate::kernel::Kernel;
use crate::RPCClient;

use super::{LazyDir, Txn};

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
pub struct TxnServer {
    rpc_client: Arc<dyn RPCClient>,
    workspace: DirLock<CacheBlock>,
    active: Arc<RwLock<OrdHashSet<Active>>>,
    tx: mpsc::UnboundedSender<Active>,
    ttl: Duration,
}

impl Clone for TxnServer {
    fn clone(&self) -> Self {
        Self {
            rpc_client: self.rpc_client.clone(),
            workspace: self.workspace.clone(),
            active: self.active.clone(),
            tx: self.tx.clone(),
            ttl: self.ttl,
        }
    }
}

impl TxnServer {
    pub fn create(
        workspace: DirLock<CacheBlock>,
        rpc_client: Arc<dyn RPCClient>,
        ttl: Duration,
    ) -> Self {
        let active = Arc::new(RwLock::new(OrdHashSet::new()));

        let (tx, rx) = mpsc::unbounded_channel();
        spawn_receiver_thread(active.clone(), rx);

        Self {
            rpc_client,
            workspace,
            active,
            tx,
            ttl,
        }
    }
}

impl TxnServer {
    pub(crate) async fn finalize(&self, kernel: &Kernel, now: NetworkTime) {
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

impl TxnServer {
    pub fn create_txn(&self, now: NetworkTime) -> Txn {
        self.get_txn(TxnId::new(now))
    }

    pub fn get_txn(&self, txn_id: TxnId) -> Txn {
        let expiry = txn_id.time() + self.ttl;
        let workspace = LazyDir::from(self.workspace.clone()).create_dir(txn_id.to_id());
        let rpc_client = self.rpc_client.clone();

        self.tx
            .send(Active::new(txn_id, expiry))
            .expect("active txn");

        Txn::new(txn_id, expiry, workspace, rpc_client, None)
    }

    pub async fn verify_txn(
        &self,
        txn_id: TxnId,
        now: NetworkTime,
        token: String,
    ) -> TCResult<Txn> {
        let token = self.rpc_client.verify(token, now.into()).await?;
        let expiry = token.expires().try_into()?;

        let workspace = LazyDir::from(self.workspace.clone()).create_dir(txn_id.to_id());
        let rpc_client = self.rpc_client.clone();

        self.tx
            .send(Active::new(txn_id, expiry))
            .expect("active txn");

        Ok(Txn::new(txn_id, expiry, workspace, rpc_client, Some(token)))
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
