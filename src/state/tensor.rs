use std::sync::Arc;

use async_trait::async_trait;

use crate::context::*;
use crate::error;
use crate::transaction::Transaction;

#[derive(Hash)]
pub struct Tensor {}

#[async_trait]
impl TCContext for Tensor {
    async fn get(self: Arc<Self>, _txn: Arc<Transaction>, _node: Link) -> TCResult<Arc<TCState>> {
        Err(error::not_implemented())
    }

    async fn put(self: Arc<Self>, _txn: Arc<Transaction>, _value: TCValue) -> TCResult<()> {
        Err(error::not_implemented())
    }
}

#[async_trait]
impl TCExecutable for Tensor {
    async fn post(self: Arc<Self>, _txn: Arc<Transaction>, _method: Link) -> TCResult<Arc<TCState>> {
        Err(error::not_implemented())
    }
}

pub struct TensorContext {}

impl TensorContext {
    pub fn new() -> Arc<TensorContext> {
        Arc::new(TensorContext {})
    }
}

#[async_trait]
impl TCExecutable for TensorContext {
    async fn post(self: Arc<Self>, _txn: Arc<Transaction>, _method: Link) -> TCResult<Arc<TCState>> {
        Err(error::not_implemented())
    }
}
