use std::convert::TryInto;
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
    async fn get(self: &Arc<Self>, _txn: Arc<Transaction>, key: TCValue) -> TCResult<TCState> {
        let mut i = self.latest_block;
        let mut matched: Vec<TCValue> = vec![];
        loop {
            let contents = self.fs_dir.clone().get(i.into()).await?;
            for entry in contents {
                let (k, value) = serde_json::from_slice(&entry).map_err(error::internal)?;
                if key == k {
                    matched.push(value);
                }
            }

            if i == 0 {
                break;
            } else {
                i -= 1;
            }
        }

        Ok(matched.into())
    }

    async fn put(
        self: &Arc<Self>,
        _txn: Arc<Transaction>,
        key: TCValue,
        value: TCState,
    ) -> TCResult<TCState> {
        let value: TCValue = value.try_into()?;
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

#[derive(Debug)]
pub struct ChainContext {
    fs_dir: Arc<fs::Dir>,
}

impl ChainContext {
    pub fn new(fs_dir: Arc<fs::Dir>) -> Arc<ChainContext> {
        Arc::new(ChainContext { fs_dir })
    }
}

#[async_trait]
impl TCExecutable for ChainContext {
    async fn post(self: &Arc<Self>, txn: Arc<Transaction>, method: &Link) -> TCResult<TCState> {
        let path: Link = txn.require("path")?.try_into()?;

        if method == "/new" {
            if self.fs_dir.clone().exists(&path).await? {
                return Err(error::bad_request("There is already an entry at", path));
            }

            let chain_dir = self.fs_dir.clone().reserve(&path)?;

            Ok(Arc::new(Chain {
                fs_dir: chain_dir,
                latest_block: 0,
            })
            .into())
        } else if method == "/load" {
            // TODO: read the contents of each block and provide them to the caller
            if !self.fs_dir.clone().exists(&path).await? {
                return Err(error::not_found(path));
            }

            let chain_dir = self.fs_dir.clone().reserve(&path)?;
            let mut i = 0;
            while self.fs_dir.clone().exists(&i.into()).await? {
                i += 1;
            }

            Ok(Arc::new(Chain {
                fs_dir: chain_dir,
                latest_block: i,
            })
            .into())
        } else {
            Err(error::bad_request(
                "ChainContext has no such method",
                method,
            ))
        }
    }
}
