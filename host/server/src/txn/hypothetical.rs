use async_trait::async_trait;

use tc_transact::public::{Handler, Route, StateInstance};
use tc_transact::{Transact, TxnId};
use tcgeneric::{path_label, PathLabel, PathSegment};

use super::Txn;

pub struct Hypothetical {}

impl Hypothetical {
    pub const PATH: PathLabel = path_label(&["txn", "hypothetical"]);

    pub fn new() -> Self {
        Self {}
    }
}

impl<FE, State: StateInstance<FE = FE, Txn = Txn<State, FE>>> Route<State> for Hypothetical {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        None
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
