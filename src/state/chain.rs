use std::sync::Arc;

use async_trait::async_trait;

use crate::context::*;
use crate::error;
use crate::state::TCState;
use crate::state::block::Block;
use crate::transaction::Transaction;
use crate::value::Link;

#[derive(Hash)]
pub struct Chain {
    block_count: u64,
    latest_block: Arc<Block>,
}

impl Chain {
    fn new(first_block: Arc<Block>) -> Arc<Chain> {
        Arc::new(Chain {
            block_count: 1,
            latest_block: first_block,
        })
    }
}

#[async_trait]
impl TCContext for Chain {
    async fn get(self: Arc<Self>, _txn: Arc<Transaction>, _path: Link) -> TCResult<TCState> {
        Err(error::not_implemented())
    }

    async fn put(self: Arc<Self>, _txn: Arc<Transaction>, _state: TCState) -> TCResult<()> {
        Err(error::not_implemented())
    }
}

#[async_trait]
impl TCExecutable for Chain {
    async fn post(self: Arc<Self>, _txn: Arc<Transaction>, _method: Link) -> TCResult<TCState> {
        Err(error::not_implemented())
    }
}

pub struct ChainContext {}

impl ChainContext {
    pub fn new() -> Arc<ChainContext> {
        Arc::new(ChainContext {})
    }
}

#[async_trait]
impl TCContext for ChainContext {
    async fn get(self: Arc<Self>, _txn: Arc<Transaction>, _path: Link) -> TCResult<TCState> {
        // TODO: check if the chain already exists
        // if so, load it
        // otherwise, return a NOT FOUND error
        Err(error::not_implemented())
    }

    async fn put(self: Arc<Self>, _txn: Arc<Transaction>, _state: TCState) -> TCResult<()> {
        // TODO: check if the chain already exists
        // if so, return an error
        // otherwise, create a new empty chain at the specified path and return it
        Err(error::not_implemented())
    }
}
