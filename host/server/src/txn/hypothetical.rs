use async_trait::async_trait;

use tc_error::TCError;
use tc_scalar::{Executor, OpDef, Refer, Scalar, Scope};
use tc_transact::public::{Handler, PostHandler, Route, StateInstance};
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

struct OpHandler;

impl<'a, State, FE> Handler<'a, State> for OpHandler
where
    State: StateInstance<FE = FE, Txn = Txn<State, FE>> + Refer<State> + From<Scalar>,
    FE: Send + Sync,
    Scalar: TryFrom<State, Error = TCError>,
{
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, mut params| {
            Box::pin(async move {
                let op_def: State = params.require("op")?;
                params.expect_empty()?;

                let op_def = Scalar::try_from(op_def)?;
                let op_def = match op_def {
                    Scalar::Op(OpDef::Post(op_def)) => Ok(op_def),
                    other => Err(TCError::unexpected(other, "a POST Op")),
                }?;

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

impl<FE, State> Route<State> for Hypothetical
where
    FE: Send + Sync,
    State: StateInstance<FE = FE, Txn = Txn<State, FE>> + Refer<State> + From<Scalar>,
    Scalar: TryFrom<State, Error = TCError>,
{
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
        // no-op
    }

    async fn rollback(&self, _txn_id: &TxnId) {
        // no-op
    }

    async fn finalize(&self, _txn_id: &TxnId) {
        // no-op
    }
}
