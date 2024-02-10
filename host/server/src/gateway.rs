use std::sync::Arc;

use freqfs::FileSave;
use tokio::time::{Duration, MissedTickBehavior};

use tc_transact::TxnId;
use tcgeneric::{NetworkTime, TCBoxFuture, ThreadSafe};

use crate::kernel::Kernel;
use crate::txn::TxnServer;

const GC_INTERVAL: Duration = Duration::from_millis(100);

pub struct Gateway<FE> {
    kernel: Kernel<FE>,
    txn_server: TxnServer<FE>,
}

impl<FE> Gateway<FE> {
    pub fn new(txn_server: TxnServer<FE>, kernel: Kernel<FE>) -> Arc<Self>
    where
        FE: for<'a> FileSave<'a>,
    {
        let gateway = Arc::new(Self { kernel, txn_server });

        spawn_cleanup_thread(gateway.clone());

        gateway
    }

    pub fn now(&self) -> NetworkTime {
        NetworkTime::now()
    }

    pub(crate) fn finalize(&self, txn_id: TxnId) -> TCBoxFuture<()>
    where
        FE: ThreadSafe,
    {
        Box::pin(self.kernel.finalize(txn_id))
    }
}

fn spawn_cleanup_thread<FE>(gateway: Arc<Gateway<FE>>)
where
    FE: for<'a> FileSave<'a>,
{
    let mut interval = tokio::time::interval(GC_INTERVAL);
    interval.set_missed_tick_behavior(MissedTickBehavior::Skip);

    tokio::spawn(async move {
        loop {
            interval.tick().await;
            gateway.txn_server.finalize(&gateway).await;
        }
    });
}
