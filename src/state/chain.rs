use std::sync::Arc;

use async_trait::async_trait;

use crate::context::*;
use crate::error;
use crate::fs;
use crate::state::TCState;
use crate::transaction::Transaction;
use crate::value::{Link, TCValue};

#[derive(Hash)]
pub struct Chain {
    fs_dir: Arc<fs::Dir>,
    latest_block: u64,
}

#[async_trait]
impl TCContext for Chain {
    async fn get(self: Arc<Self>, _txn: Arc<Transaction>, _key: TCValue) -> TCResult<TCState> {
        Err(error::not_implemented())
    }

    async fn put(self: Arc<Self>, _txn: Arc<Transaction>, _key: TCValue, _state: TCState) -> TCResult<()> {
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
    fs_dir: Arc<fs::Dir>,
}

impl ChainContext {
    pub fn new(fs_dir: Arc<fs::Dir>) -> Arc<ChainContext> {
        Arc::new(ChainContext { fs_dir })
    }
}

#[async_trait]
impl TCContext for ChainContext {
    async fn get(self: Arc<Self>, _txn: Arc<Transaction>, path: TCValue) -> TCResult<TCState> {
        // TODO: read the contents of each block and provide them to the caller
        let path = path.as_link()?;

        if !self.fs_dir.clone().exists(path.clone()).await? {
            return Err(error::not_found(path));
        }

        let chain_dir = self.fs_dir.clone().reserve(path)?;
        let mut i = 0;
        while self.fs_dir.clone().exists(i.into()).await? {
            i += 1;
        }

        Ok(Arc::new(Chain {
            fs_dir: chain_dir,
            latest_block: i,
        })
        .into())
    }

    async fn put(self: Arc<Self>, _txn: Arc<Transaction>, _key: TCValue, state: TCState) -> TCResult<()> {
        // TODO: support the case where state == TCState::Chain(_) by copying the given chain

        let path = state.as_value()?.as_link()?;
        if self.fs_dir.clone().exists(path.clone()).await? {
            return Err(error::bad_request("There is already an entry at", path));
        }

        self.fs_dir.clone().reserve(path)?;

        Ok(())
    }
}
