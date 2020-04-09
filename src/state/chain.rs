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
        let mut i = self.latest_block;
        loop {
            // TODO: read each entry in each block until the key is found, then return the value
            i -= 1;
            if i == 0 {
                break;
            }
        }

        Err(error::not_implemented())
    }

    async fn put(
        self: Arc<Self>,
        _txn: Arc<Transaction>,
        key: TCValue,
        value: TCState,
    ) -> TCResult<TCState> {
        let value = value.as_value()?;
        let delta = serde_json::to_string_pretty(&(key, value))?
            .as_bytes()
            .to_vec();
        self.fs_dir
            .clone()
            .append(self.latest_block.into(), delta)
            .await?;
        Ok(().into())
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

    async fn put(
        self: Arc<Self>,
        _txn: Arc<Transaction>,
        _key: TCValue,
        state: TCState,
    ) -> TCResult<TCState> {
        // TODO: support the case where state == TCState::Chain(_) by copying the given chain

        let path = state.as_value()?.as_link()?;
        if self.fs_dir.clone().exists(path.clone()).await? {
            return Err(error::bad_request("There is already an entry at", path));
        }

        let chain_dir = self.fs_dir.clone().reserve(path)?;

        Ok(Arc::new(Chain {
            fs_dir: chain_dir,
            latest_block: 0,
        })
        .into())
    }
}
