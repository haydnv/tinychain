use std::sync::Arc;

use async_trait::async_trait;

use crate::context::*;
use crate::error;
use crate::transaction::Transaction;
use crate::state::TCState;
use crate::value::Link;

#[derive(Hash)]
pub struct Graph {}

#[async_trait]
impl TCContext for Graph {
    async fn get(self: Arc<Self>, _txn: Arc<Transaction>, _node: Link) -> TCResult<TCState> {
        Err(error::not_implemented())
    }

    async fn put(self: Arc<Self>, _txn: Arc<Transaction>, _state: TCState) -> TCResult<()> {
        Err(error::not_implemented())
    }
}

#[async_trait]
impl TCExecutable for Graph {
    async fn post(self: Arc<Self>, _txn: Arc<Transaction>, _method: Link) -> TCResult<TCState> {
        Err(error::not_implemented())
    }
}
