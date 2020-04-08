use std::sync::Arc;

use async_trait::async_trait;

use crate::context::*;
use crate::error;
use crate::fs;
use crate::state::TCState;
use crate::transaction::Transaction;
use crate::value::Link;

#[derive(Hash)]
pub struct Chain {
    mount_point: Arc<fs::Dir>,
    latest_block: u64,
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

pub struct ChainContext {
    mount_point: Arc<fs::Dir>,
}

impl ChainContext {
    pub fn new(mount_point: Arc<fs::Dir>) -> Arc<ChainContext> {
        Arc::new(ChainContext { mount_point })
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
