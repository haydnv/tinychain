use tc_transact::TxnId;
use tcgeneric::{NetworkTime, TCBoxFuture, ThreadSafe};

use crate::kernel::Kernel;

pub struct Gateway<FE> {
    kernel: Kernel<FE>,
}

impl<FE> Gateway<FE> {
    pub fn new(kernel: Kernel<FE>) -> Self {
        Self { kernel }
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
