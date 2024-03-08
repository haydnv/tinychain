use async_trait::async_trait;

use tc_error::*;
use tc_scalar::Executor;
use tc_transact::hash::{default_hash, AsyncHash, Output, Sha256};
use tc_transact::public::{Handler, PostHandler, Route};
use tc_transact::{Transact, TxnId};
use tcgeneric::{path_label, Id, PathLabel, PathSegment};

use crate::State;

use super::Txn;

#[derive(Clone, Debug)]
pub struct Hypothetical {}

impl Hypothetical {
    pub const PATH: PathLabel = path_label(&["transact", "hypothetical"]);

    pub fn new() -> Self {
        Self {}
    }
}

#[async_trait]
impl AsyncHash for Hypothetical {
    async fn hash(&self, _txn_id: TxnId) -> TCResult<Output<Sha256>> {
        Ok(default_hash::<Sha256>())
    }
}

struct OpHandler;

impl<'a> Handler<'a, State> for OpHandler {
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, mut params| {
            Box::pin(async move {
                let op_def: Vec<(Id, State)> = params.require("op")?;
                params.expect_empty()?;

                let capture = if let Some((capture, _)) = op_def.last() {
                    capture.clone()
                } else {
                    return Ok(State::default());
                };

                let executor: Executor<State, State> = Executor::new(txn, None, op_def);
                executor.capture(capture).await
            })
        }))
    }
}

impl Route<State> for Hypothetical {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        if path.is_empty() {
            Some(Box::new(OpHandler))
        } else {
            None
        }
    }
}

#[async_trait]
impl Transact for Hypothetical {
    type Commit = ();

    async fn commit(&self, _txn_id: TxnId) -> Self::Commit {
        unreachable!("cannot commit a hypothetical transaction")
    }

    async fn rollback(&self, _txn_id: &TxnId) {
        // no-op
    }

    async fn finalize(&self, _txn_id: &TxnId) {
        // no-op
    }
}
