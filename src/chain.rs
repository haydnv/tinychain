use std::convert::TryInto;
use std::sync::Arc;

use async_trait::async_trait;

use crate::context::*;
use crate::error;
use crate::fs;
use crate::state::TCState;
use crate::transaction::{Transaction, TransactionId};
use crate::value::TCValue;

#[derive(Hash)]
pub struct Chain {
    fs_dir: Arc<fs::Dir>,
    latest_block: u64,
}

impl Chain {
    pub fn new(fs_dir: Arc<fs::Dir>) -> Arc<Chain> {
        Arc::new(Chain { fs_dir, latest_block: 0 })
    }
}

#[async_trait]
impl TCContext for Chain {
    async fn commit(self: &Arc<Self>, _txn_id: TransactionId) {
        // TODO
    }

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
