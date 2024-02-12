use std::sync::Arc;

use freqfs::FileSave;
use tokio::time::{Duration, MissedTickBehavior};

use tcgeneric::NetworkTime;

use crate::kernel::Kernel;
use crate::txn::TxnServer;

const GC_INTERVAL: Duration = Duration::from_millis(100);

pub struct Server<FE> {
    kernel: Arc<Kernel<FE>>,
    txn_server: TxnServer<FE>,
}

impl<FE: for<'a> FileSave<'a>> Server<FE> {
    pub(crate) fn new(kernel: Arc<Kernel<FE>>, txn_server: TxnServer<FE>) -> Self {
        spawn_cleanup_thread(kernel.clone(), txn_server.clone());

        Self { kernel, txn_server }
    }
}

fn spawn_cleanup_thread<FE>(kernel: Arc<Kernel<FE>>, txn_server: TxnServer<FE>)
where
    FE: for<'a> FileSave<'a>,
{
    let mut interval = tokio::time::interval(GC_INTERVAL);
    interval.set_missed_tick_behavior(MissedTickBehavior::Skip);

    tokio::spawn(async move {
        loop {
            interval.tick().await;
            txn_server.finalize(&kernel, NetworkTime::now()).await;
        }
    });
}
