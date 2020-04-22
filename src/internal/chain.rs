use std::sync::Arc;

use futures::stream::{FuturesOrdered, Stream};
use futures::Future;
use futures_util::{FutureExt, TryFutureExt};

use crate::context::*;
use crate::error;
use crate::internal::FsDir;
use crate::state::TCState;
use crate::transaction::TransactionId;
use crate::value::TCValue;

#[derive(Debug, Hash)]
pub struct Chain {
    fs_dir: Arc<FsDir>,
    latest_block: u64,
}

impl Chain {
    pub fn new(fs_dir: Arc<FsDir>) -> Arc<Chain> {
        Arc::new(Chain {
            fs_dir,
            latest_block: 0,
        })
    }

    pub async fn get(self: &Arc<Self>, _txn_id: TransactionId, key: &TCValue) -> TCResult<TCState> {
        // TODO: use txn_id to return the state of the chain at a specific point in time

        let mut i = self.latest_block;
        let mut matched: Vec<TCValue> = vec![];
        loop {
            let contents = self.fs_dir.clone().get(i.into()).await?;
            for entry in contents {
                let (k, value): (TCValue, TCValue) =
                    serde_json::from_slice(&entry).map_err(error::internal)?;
                if key == &k {
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

    pub async fn put(self: &Arc<Self>, txn_id: TransactionId, mutations: Vec<(TCValue, TCValue)>) {
        let delta: Vec<Vec<u8>> = mutations
            .iter()
            .map(|e| serde_json::to_string_pretty(e).unwrap().as_bytes().to_vec())
            .collect::<Vec<Vec<u8>>>();
        self.fs_dir
            .flush(self.latest_block.into(), txn_id.into(), delta)
            .await;
    }
}

impl From<Chain> for Box<dyn Stream<Item = TCResult<Vec<(TCValue, TCValue)>>>> {
    fn from(chain: Chain) -> Box<dyn Stream<Item = TCResult<Vec<(TCValue, TCValue)>>>> {
        let mut stream: FuturesOrdered<
            Box<dyn Future<Output = TCResult<Vec<(TCValue, TCValue)>>> + Unpin>,
        > = FuturesOrdered::new();

        for i in 0..chain.latest_block {
            let fut = chain
                .fs_dir
                .clone()
                .get(i.into())
                .and_then(|contents| async {
                    let mut results: Vec<(TCValue, TCValue)> = Vec::with_capacity(contents.len());
                    for entry in contents {
                        let result = serde_json::from_slice(&entry).map_err(error::internal)?;
                        results.push(result);
                    }
                    Ok(results)
                });
            stream.push(Box::new(fut.boxed()));
        }

        Box::new(stream)
    }
}
