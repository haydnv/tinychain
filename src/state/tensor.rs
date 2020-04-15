use std::sync::Arc;

use async_trait::async_trait;

use crate::context::*;
use crate::error;
use crate::state::TCState;
use crate::transaction::Transaction;
use crate::value::{Link, TCValue};

#[derive(Debug, Hash)]
pub struct Tensor {}

#[async_trait]
impl TCContext for Tensor {
    async fn get(self: &Arc<Self>, _txn: Arc<Transaction>, _slice: TCValue) -> TCResult<TCState> {
        Err(error::not_implemented())
    }

    async fn put(
        self: &Arc<Self>,
        _txn: Arc<Transaction>,
        _slice: TCValue,
        _values: TCState,
    ) -> TCResult<TCState> {
        Err(error::not_implemented())
    }
}

#[async_trait]
impl TCExecutable for Tensor {
    async fn post(self: &Arc<Self>, _txn: Arc<Transaction>, _method: &Link) -> TCResult<TCState> {
        Err(error::not_implemented())
    }
}
