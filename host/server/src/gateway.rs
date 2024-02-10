use tc_transact::TxnId;
use tcgeneric::{NetworkTime, TCBoxFuture};

use crate::kernel::Kernel;

pub struct Gateway {
    kernel: Kernel,
}

impl Gateway {
    pub fn new(kernel: Kernel) -> Self {
        Self { kernel }
    }

    pub fn now(&self) -> NetworkTime {
        NetworkTime::now()
    }

    pub(crate) fn finalize(&self, txn_id: TxnId) -> TCBoxFuture<()> {
        Box::pin(self.kernel.finalize(txn_id))
    }
}
