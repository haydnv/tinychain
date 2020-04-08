use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::fs;

use crate::context::*;
use crate::drive::Drive;
use crate::error;
use crate::state::TCState;
use crate::transaction::Transaction;
use crate::value::{Link, TCValue};

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
    async fn get(self: Arc<Self>, _txn: Arc<Transaction>, path: Link) -> TCResult<TCState> {
        if path.as_str() == "/" {
            match fs::read(&self.path).await {
                Ok(content) => Ok(TCState::Value(TCValue::Bytes(content))),
                Err(cause) => {
                    eprintln!("Error reading block: {}", cause);
                    Err(error::internal("The host encountered a filesystem error"))
                }
            }
        } else {
            Err(error::bad_request(
                "A block itself has no inner directory structure",
                path,
            ))
        }
    }

    async fn put(self: Arc<Self>, txn: Arc<Transaction>, state: TCState) -> TCResult<()> {
        let value = state.to_value()?.to_bytes()?;

        if value.contains(&(EOT_CHAR as u8)) {
            let msg = "Attempted to write a block containing the ASCII EOT control character";
            return Err(error::internal(msg));
        }

        let transaction_id: Vec<u8> = txn.id().into();
        let content = [
            &transaction_id[..],
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
    async fn get(self: Arc<Self>, txn: Arc<Transaction>, path: Link) -> TCResult<TCState> {
        let path = self.drive.clone().fs_path(txn.context(), path)?;
        // TODO: check if the file actually exists before returning success
        Ok(TCState::Block(Block::new(path)))
    }

    async fn put(self: Arc<Self>, txn: Arc<Transaction>, state: TCState) -> TCResult<()> {
        let _path = self.drive.clone().fs_path(txn.context(), state.to_value()?.to_link()?);
        // TODO: touch the file at `path`
        Err(error::not_implemented())
    }
}
