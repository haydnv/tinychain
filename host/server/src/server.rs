use std::collections::BTreeSet;
use std::sync::Arc;

use mdns_sd::ServiceDaemon;
use tokio::time::{Duration, MissedTickBehavior};

use tc_error::*;
use tc_transact::TxnId;
use tc_value::Host;
use tcgeneric::{NetworkTime, PathSegment};

use crate::kernel::Kernel;
use crate::txn::{Txn, TxnServer};
use crate::Endpoint;

const GC_INTERVAL: Duration = Duration::from_millis(100);

pub struct Server {
    kernel: Arc<Kernel>,
    txn_server: TxnServer,
    mdns: ServiceDaemon,
}

impl Server {
    pub(crate) fn new(kernel: Arc<Kernel>, txn_server: TxnServer) -> mdns_sd::Result<Self> {
        spawn_cleanup_thread(kernel.clone(), txn_server.clone());

        let mdns = ServiceDaemon::new()?;

        Ok(Self {
            kernel,
            txn_server,
            mdns,
        })
    }

    pub(crate) fn mdns(&self) -> &ServiceDaemon {
        &self.mdns
    }

    pub fn create_txn(&self) -> TCResult<Txn> {
        self.txn_server.create_txn(NetworkTime::now())
    }

    pub async fn verify_txn(&self, txn_id: TxnId, token: String) -> TCResult<Txn> {
        self.txn_server
            .verify_txn(txn_id, NetworkTime::now(), token)
            .await
    }

    pub async fn get_txn(&self, txn_id: Option<TxnId>, token: Option<String>) -> TCResult<Txn> {
        if let Some(token) = token {
            let now = NetworkTime::now();
            let txn_id = txn_id.unwrap_or_else(|| TxnId::new(now));
            self.txn_server.verify_txn(txn_id, now, token).await
        } else {
            let txn_id = txn_id.unwrap_or_else(|| TxnId::new(NetworkTime::now()));
            self.txn_server.get_txn(txn_id)
        }
    }

    pub(crate) async fn replicate_and_join(&self, peers: BTreeSet<Host>) -> Result<(), bool> {
        self.kernel
            .replicate_and_join(self.txn_server.clone(), peers)
            .await
    }
}

impl Server {
    pub fn authorize_claim_and_route<'a>(
        &'a self,
        path: &'a [PathSegment],
        txn: &'a Txn,
    ) -> TCResult<Endpoint<'a>> {
        self.kernel.route(path, txn)
    }
}

fn spawn_cleanup_thread(kernel: Arc<Kernel>, txn_server: TxnServer) {
    let mut interval = tokio::time::interval(GC_INTERVAL);
    interval.set_missed_tick_behavior(MissedTickBehavior::Skip);

    tokio::spawn(async move {
        loop {
            interval.tick().await;
            txn_server.finalize(&kernel, NetworkTime::now()).await;
        }
    });
}
