use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::context::*;
use crate::error;
use crate::state::block::Block;
use crate::transaction::Transaction;

#[derive(Eq, PartialEq, Ord, PartialOrd)]
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
    async fn put(self: Arc<Self>, txn: Arc<Transaction>, value: TCValue) -> TCResult<()> {
        self.latest_block.clone().put(txn, value).await
    }

    async fn post(
        self: Arc<Self>,
        _txn: Arc<Transaction>,
        _method: Link,
    ) -> TCResult<Arc<TCState>> {
        Err(error::not_implemented())
    }
}

pub struct ChainContext {}

impl ChainContext {
    pub fn new() -> Arc<ChainContext> {
        Arc::new(ChainContext {})
    }
}

#[derive(Deserialize, Serialize)]
struct Request {
    key: TCValue,
    value: TCValue,
}

#[async_trait]
impl TCContext for ChainContext {
    async fn post(self: Arc<Self>, txn: Arc<Transaction>, method: Link) -> TCResult<Arc<TCState>> {
        if method.as_str() != "/new" {
            return Err(error::bad_request(
                "ChainContext has no such method",
                method,
            ));
        }

        let new_block = Link::to("/sbin/block/new")?;
        let block = txn.post(new_block).await?.to_block()?;

        Ok(TCState::from_chain(Chain::new(block)))
    }
}
