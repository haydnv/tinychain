use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use async_trait::async_trait;
use ds_ext::OrdHashSet;
use freqfs::DirLock;
use futures::TryFutureExt;
use log::debug;
use rjwt::{Actor, Error, Resolve};
use tokio::sync::mpsc;

use tc_error::*;
use tc_state::CacheBlock;
use tc_transact::TxnId;
use tc_value::{Host, Link, Value};
use tcgeneric::NetworkTime;

use crate::claim::Claim;
use crate::client::Client;
use crate::kernel::Kernel;
use crate::RPCClient;

use super::{LazyDir, Txn};

// allow an end-user's request to time out gracefully before garbage collection
const GRACE: Duration = Duration::from_secs(1);

struct Resolver<'a> {
    client: &'a Client,
    txn_id: TxnId,
}

impl<'a> Resolver<'a> {
    fn new(client: &'a Client, txn_id: TxnId) -> Self {
        Self { client, txn_id }
    }
}

#[async_trait]
impl<'a> Resolve for Resolver<'a> {
    type HostId = Link;
    type ActorId = Value;
    type Claims = Claim;

    async fn resolve(
        &self,
        host: &Self::HostId,
        actor_id: &Self::ActorId,
    ) -> Result<Actor<Self::ActorId>, Error> {
        self.client
            .fetch(self.txn_id, host.into(), actor_id.clone())
            .map_err(Error::fetch)
            .await
    }
}

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
    client: Client,
    workspace: DirLock<CacheBlock>,
    active: Arc<RwLock<OrdHashSet<Active>>>,
    tx: mpsc::UnboundedSender<Active>,
    ttl: Duration,
}

impl Clone for TxnServer {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            workspace: self.workspace.clone(),
            active: self.active.clone(),
            tx: self.tx.clone(),
            ttl: self.ttl,
        }
    }
}

impl TxnServer {
    pub fn create(client: Client, workspace: DirLock<CacheBlock>, ttl: Duration) -> Self {
        let active = Arc::new(RwLock::new(OrdHashSet::new()));

        let (tx, rx) = mpsc::unbounded_channel();
        spawn_receiver_thread(active.clone(), rx);

        Self {
            client,
            workspace,
            active,
            tx,
            ttl,
        }
    }

    pub(crate) fn address(&self) -> &Host {
        self.client.host()
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

        for txn_id in expired.iter() {
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
    pub fn create_txn(&self, now: NetworkTime) -> TCResult<Txn> {
        self.get_txn(TxnId::new(now))
    }

    pub fn get_txn(&self, txn_id: TxnId) -> TCResult<Txn> {
        let expiry = txn_id.time() + self.ttl;
        let workspace = LazyDir::from(self.workspace.clone()).create_dir(txn_id.to_id());
        let client = self.client.clone();

        self.tx
            .send(Active::new(txn_id, expiry))
            .expect("active txn");

        Txn::new(txn_id, expiry, workspace, client, None)
    }

    pub async fn verify_txn(
        &self,
        txn_id: TxnId,
        now: NetworkTime,
        token: String,
    ) -> TCResult<Txn> {
        let resolver = Resolver::new(&self.client, txn_id);
        let token = resolver.verify(token, now.into()).await?;
        let expiry = token.expires().try_into()?;

        let workspace = LazyDir::from(self.workspace.clone()).create_dir(txn_id.to_id());
        let client = self.client.clone();

        self.tx
            .send(Active::new(txn_id, expiry))
            .expect("active txn");

        Txn::new(txn_id, expiry, workspace, client, Some(token))
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
