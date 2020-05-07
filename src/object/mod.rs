use std::convert::TryFrom;
use std::sync::Arc;

use async_trait::async_trait;

use crate::transaction::Transaction;
use crate::value::{TCResult, TCValue};

#[async_trait]
pub trait TCObject: Into<TCValue> + TryFrom<TCValue> {
    async fn new(txn: Arc<Transaction>, id: TCValue) -> TCResult<Self>;

    fn id(&self) -> TCValue;
}
