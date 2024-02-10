use tc_transact::TxnId;

pub struct Kernel;

impl Kernel {
    pub(crate) async fn finalize(&self, _txn_id: TxnId) {
        todo!("Kernel::finalize")
    }
}
