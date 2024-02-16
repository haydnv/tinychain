use std::sync::Arc;

use tokio::time::{Duration, MissedTickBehavior};

use tc_error::*;
use tcgeneric::{NetworkTime, PathSegment};

use crate::kernel::Kernel;
use crate::txn::{Txn, TxnServer};
use crate::{Endpoint, SignedToken};

const GC_INTERVAL: Duration = Duration::from_millis(100);

pub struct Server {
    kernel: Arc<Kernel>,
    txn_server: TxnServer,
}

impl Server {
    pub(crate) fn new(kernel: Arc<Kernel>, txn_server: TxnServer) -> Self {
        spawn_cleanup_thread(kernel.clone(), txn_server.clone());

        Self { kernel, txn_server }
    }

    pub fn get_txn(&self, token: Option<SignedToken>) -> TCResult<Txn> {
        if let Some(token) = token {
            todo!("construct Txn from existing token")
        } else {
            Ok(self.txn_server.new_txn(NetworkTime::now()))
        }
    }
}

impl Server {
    pub fn authorize_claim_and_route<'a>(
        &'a self,
        path: &'a [PathSegment],
        txn: Txn,
    ) -> TCResult<(Txn, Endpoint<'a>)> {
        self.kernel.authorize_claim_and_route(path, txn)
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
