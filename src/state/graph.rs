use std::sync::Arc;

use async_trait::async_trait;

use crate::context::*;
use crate::error;
use crate::transaction::Transaction;
use crate::value::{Link, TCValue};

#[derive(Hash)]
pub struct Graph {}

#[async_trait]
impl TCContext for Graph {
    async fn get(self: Arc<Self>, _txn: Arc<Transaction>, _node: Link) -> TCResult<TCResponse> {
        Err(error::not_implemented())
    }

    async fn put(self: Arc<Self>, _txn: Arc<Transaction>, _value: TCValue) -> TCResult<()> {
        Err(error::not_implemented())
    }
}

#[async_trait]
impl TCExecutable for Graph {
    async fn post(self: Arc<Self>, _txn: Arc<Transaction>, _method: Link) -> TCResult<TCResponse> {
        Err(error::not_implemented())
    }
}
