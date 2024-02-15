use std::sync::Arc;

use freqfs::FileSave;
use tokio::time::{Duration, MissedTickBehavior};

use tc_error::{TCError, TCResult};
use tc_scalar::{Refer, Scalar};
use tc_transact::public::StateInstance;
use tcgeneric::{NetworkTime, PathSegment, ThreadSafe};

use crate::kernel::Kernel;
use crate::txn::{Txn, TxnServer};
use crate::{Endpoint, SignedToken};

const GC_INTERVAL: Duration = Duration::from_millis(100);

pub struct Server<State, FE> {
    kernel: Arc<Kernel<State, FE>>,
    txn_server: TxnServer<FE>,
}

impl<State, FE> Server<State, FE> {
    pub(crate) fn new(kernel: Arc<Kernel<State, FE>>, txn_server: TxnServer<FE>) -> Self
    where
        State: ThreadSafe,
        FE: for<'a> FileSave<'a>,
    {
        spawn_cleanup_thread(kernel.clone(), txn_server.clone());

        Self { kernel, txn_server }
    }

    pub fn get_txn(&self, token: Option<SignedToken>) -> TCResult<Txn<State, FE>>
    where
        FE: Send + Sync,
    {
        if let Some(token) = token {
            todo!("construct Txn from existing token")
        } else {
            Ok(self.txn_server.new_txn(NetworkTime::now()))
        }
    }
}

impl<State, FE> Server<State, FE>
where
    State: StateInstance<FE = FE, Txn = Txn<State, FE>> + Refer<State> + From<Scalar>,
    FE: ThreadSafe + Clone,
    Scalar: TryFrom<State, Error = TCError>,
{
    pub fn authorize_claim_and_route<'a>(
        &'a self,
        path: &'a [PathSegment],
        txn: Txn<State, FE>,
    ) -> TCResult<(Txn<State, FE>, Endpoint<'a, State>)> {
        self.kernel.authorize_claim_and_route(path, txn)
    }
}

fn spawn_cleanup_thread<State, FE>(kernel: Arc<Kernel<State, FE>>, txn_server: TxnServer<FE>)
where
    State: ThreadSafe,
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
