use std::sync::Arc;

use async_trait::async_trait;

use crate::context::*;
use crate::error;
use crate::state::block::BlockContext;
use crate::transaction::Transaction;

#[derive(Hash)]
pub struct Chain {}

impl Chain {
    fn new() -> Arc<Chain> {
        Arc::new(Chain {})
    }
}

#[async_trait]
impl TCContext for Chain {
    async fn post(
        self: Arc<Self>,
        _txn: Arc<Transaction>,
        _method: &str,
    ) -> TCResult<Arc<TCState>> {
        Err(error::not_implemented())
    }
}

pub struct ChainContext {
    block_context: Arc<BlockContext>,
}

impl ChainContext {
    pub fn new(block_context: Arc<BlockContext>) -> Arc<ChainContext> {
        Arc::new(ChainContext { block_context })
    }
}

#[async_trait]
impl TCContext for ChainContext {
    async fn post(self: Arc<Self>, _txn: Arc<Transaction>, method: &str) -> TCResult<Arc<TCState>> {
        if method != "new" {
            return Err(error::bad_request(
                "ChainContext has no such method",
                method,
            ));
        }

        Ok(TCState::from_chain(Chain::new()))
    }
}
