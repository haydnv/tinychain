use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;

use crate::context::*;
use crate::drive::Drive;
use crate::error;
use crate::transaction::Transaction;

#[derive(Hash)]
pub struct Block {
    path: PathBuf,
}

impl Block {
    fn new(path: PathBuf) -> Arc<Block> {
        Arc::new(Block { path })
    }
}

impl TCContext for Block {}

pub struct BlockContext {
    drive: Arc<Drive>,
}

impl BlockContext {
    pub fn new(drive: Arc<Drive>) -> Arc<BlockContext> {
        Arc::new(BlockContext { drive })
    }
}

#[async_trait]
impl TCContext for BlockContext {
    async fn post(self: Arc<Self>, txn: Arc<Transaction>, method: Link) -> TCResult<Arc<TCState>> {
        if method.as_str() != "/new" {
            return Err(error::bad_request(
                "BlockContext has no such method",
                method,
            ));
        }

        let name = txn.clone().require("name")?.to_value()?.to_string()?;
        let path = self.drive.clone().fs_path(&txn.context(), &name);
        Ok(TCState::from_block(Block::new(path)))
    }
}
