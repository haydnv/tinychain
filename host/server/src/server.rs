use std::sync::Arc;

use freqfs::FileSave;
use tokio::time::{Duration, MissedTickBehavior};

use crate::gateway::Gateway;
use crate::txn::TxnServer;

const GC_INTERVAL: Duration = Duration::from_millis(100);

pub struct Server<FE> {
    gateway: Arc<Gateway<FE>>,
    txn_server: TxnServer<FE>,
}

impl<FE: for<'a> FileSave<'a>> Server<FE> {
    pub(crate) fn new(gateway: Gateway<FE>, txn_server: TxnServer<FE>) -> Self {
        let gateway = Arc::new(gateway);

        spawn_cleanup_thread(gateway.clone(), txn_server.clone());

        Self {
            gateway,
            txn_server,
        }
    }
}

fn spawn_cleanup_thread<FE>(gateway: Arc<Gateway<FE>>, txn_server: TxnServer<FE>)
where
    FE: for<'a> FileSave<'a>,
{
    let mut interval = tokio::time::interval(GC_INTERVAL);
    interval.set_missed_tick_behavior(MissedTickBehavior::Skip);

    tokio::spawn(async move {
        loop {
            interval.tick().await;
            txn_server.finalize(&gateway).await;
        }
    });
}
