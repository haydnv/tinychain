use std::sync::Arc;

use freqfs::FileSave;
use tokio::time::{Duration, MissedTickBehavior};
use umask::Mode;

use tc_error::TCResult;
use tc_transact::public::{Handler, StateInstance};
use tcgeneric::{NetworkTime, PathSegment, ThreadSafe};

use crate::kernel::Kernel;
use crate::txn::{Txn, TxnServer};
use crate::SignedToken;

const GC_INTERVAL: Duration = Duration::from_millis(100);

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum RequestType {
    Read,
    Write,
    Execute,
}

impl From<RequestType> for Mode {
    fn from(rtype: RequestType) -> Self {
        match rtype {
            RequestType::Read => umask::USER_READ,
            RequestType::Write => umask::USER_WRITE,
            RequestType::Execute => umask::USER_EXEC,
        }
    }
}

pub struct Server<FE> {
    kernel: Arc<Kernel<FE>>,
    txn_server: TxnServer<FE>,
}

impl<FE> Server<FE> {
    pub(crate) fn new(kernel: Arc<Kernel<FE>>, txn_server: TxnServer<FE>) -> Self
    where
        FE: for<'a> FileSave<'a>,
    {
        spawn_cleanup_thread(kernel.clone(), txn_server.clone());

        Self { kernel, txn_server }
    }

    pub fn get_txn(&self, token: Option<SignedToken>) -> TCResult<Txn<FE>>
    where
        FE: Send + Sync,
    {
        if let Some(token) = token {
            todo!("construct Txn from existing token")
        } else {
            Ok(self.txn_server.new_txn(NetworkTime::now()))
        }
    }

    pub fn authorize_and_route<'a, State>(
        &'a self,
        request_type: RequestType,
        path: &'a [PathSegment],
        txn: &Txn<FE>,
    ) -> TCResult<Box<dyn Handler<'a, State> + 'a>>
    where
        FE: ThreadSafe + Clone,
        State: StateInstance<FE = FE, Txn = Txn<FE>>,
    {
        self.kernel
            .authorize_and_route(request_type.into(), path, txn)
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
