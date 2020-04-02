use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::fs;

use crate::context::*;
use crate::drive::Drive;
use crate::error;
use crate::transaction::Transaction;

const EOT_CHAR: char = 4 as char;

#[derive(Hash)]
pub struct Block {
    path: PathBuf,
}

impl Block {
    fn new(path: PathBuf) -> Arc<Block> {
        Arc::new(Block { path })
    }
}

#[async_trait]
impl TCContext for Block {
    async fn put(self: Arc<Self>, txn: Arc<Transaction>, value: TCValue) -> TCResult<()> {
        let value = value.to_bytes()?;

        if value.contains(&(EOT_CHAR as u8)) {
            let msg = "Attempted to write a block containing the ASCII EOT control character";
            return Err(error::internal(msg));
        }

        let content = [
            &txn.id().to_bytes()[..],
            &[EOT_CHAR as u8],
            &value[..],
            &[EOT_CHAR as u8],
        ]
        .concat();

        match fs::write(&self.path, content).await {
            Ok(()) => Ok(()),
            Err(cause) => {
                eprintln!("Error writing block: {}", cause);
                Err(error::internal("The host encountered a filesystem error"))
            }
        }
    }
}

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
