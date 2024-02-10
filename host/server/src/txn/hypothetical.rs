use async_trait::async_trait;

use tc_transact::{Transact, TxnId};
use tcgeneric::{path_label, PathLabel};

pub struct Hypothetical {}

impl Hypothetical {
    pub const PATH: PathLabel = path_label(&["txn", "hypothetical"]);

    pub fn new() -> Self {
        Self {}
    }
}

#[async_trait]
impl Transact for Hypothetical {
    type Commit = ();

    async fn commit(&self, _txn_id: TxnId) -> Self::Commit {
        // no-op
    }

    async fn rollback(&self, _txn_id: &TxnId) {
        // no-op
    }

    async fn finalize(&self, _txn_id: &TxnId) {
        // no-op
    }
}
