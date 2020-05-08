use std::convert::TryFrom;
use std::sync::Arc;

use async_trait::async_trait;

use crate::state::State;
use crate::transaction::Transaction;
use crate::value::{Args, PathSegment, TCResult, TCValue};

mod actor;

pub type Actor = actor::Actor;

#[async_trait]
pub trait TCObject: Into<TCValue> + TryFrom<TCValue> {
    async fn new(txn: Arc<Transaction>, id: TCValue) -> TCResult<Self>;

    fn id(&self) -> TCValue;

    async fn post(
        &self,
        txn: Arc<Transaction>,
        method: PathSegment,
        mut args: Args,
    ) -> TCResult<State>;
}
