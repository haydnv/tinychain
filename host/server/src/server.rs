use std::sync::Arc;

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
}

impl Server {
    pub(crate) fn new(kernel: Arc<Kernel>, txn_server: TxnServer) -> mdns_sd::Result<Self> {
        spawn_cleanup_thread(kernel.clone(), txn_server.clone());

        Ok(Self { kernel, txn_server })
    }

    pub(crate) fn kernel(&self) -> Arc<Kernel> {
        self.kernel.clone()
    }

    pub(crate) fn txn_server(&self) -> TxnServer {
        self.txn_server.clone()
    }

    pub fn address(&self) -> &Host {
        self.txn_server.address()
    }

    pub fn create_txn(&self) -> TCResult<Txn> {
        self.txn_server.create_txn(NetworkTime::now())
    }

    pub async fn verify_txn(&self, txn_id: TxnId, token: String) -> TCResult<Txn> {
        self.txn_server
            .verify_txn(txn_id, NetworkTime::now(), token)
            .await
    }

    pub async fn get_txn(&self, txn_id: TxnId, token: Option<String>) -> TCResult<Txn> {
        if let Some(token) = token {
            let now = NetworkTime::now();
            self.txn_server.verify_txn(txn_id, now, token).await
        } else {
            self.txn_server.get_txn(txn_id)
        }
    }
}

impl Server {
    pub async fn authorize_claim_and_route<'a>(
        &'a self,
        path: &'a [PathSegment],
        txn: &'a Txn,
    ) -> TCResult<Endpoint<'a>> {
        self.kernel.route(path, txn).await
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
