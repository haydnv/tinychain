use std::sync::Arc;

use bytes::Bytes;
use futures::stream::{FuturesOrdered, Stream};
use futures::Future;
use futures_util::{FutureExt, StreamExt};
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::error;
use crate::internal::FsDir;
use crate::transaction::TransactionId;
use crate::value::{TCResult, TCValue};

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

    pub async fn from(
        mut stream: Box<dyn Stream<Item = Vec<Bytes>> + Unpin>,
        dest: Arc<FsDir>,
    ) -> Arc<Chain> {
        let mut latest_block: u64 = 0;
        while let Some(block) = stream.next().await {
            dest.flush(latest_block.into(), &block[0], &block[1..])
                .await;
            latest_block += 1;
        }

        Arc::new(Chain {
            fs_dir: dest,
            latest_block,
        })
    }

    pub async fn get<T: DeserializeOwned>(
        self: &Arc<Self>,
        _txn_id: TransactionId,
        key: &TCValue,
    ) -> TCResult<Vec<T>> {
        // TODO: use txn_id to return the state of the chain at a specific point in time

        let mut i = self.latest_block;
        let mut matched: Vec<T> = vec![];
        loop {
            let contents = self.fs_dir.clone().get(i.into()).await;
            for entry in contents {
                let (k, value): (TCValue, T) =
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

        Ok(matched)
    }

    pub async fn put<T: Serialize>(self: &Arc<Self>, txn_id: TransactionId, mutations: &[T]) {
        let delta: Vec<Bytes> = mutations
            .iter()
            .map(|e| Bytes::from(serde_json::to_string_pretty(e).unwrap()))
            .collect();
        self.fs_dir
            .flush(self.latest_block.into(), &txn_id.into(), &delta)
            .await;
    }
}

impl From<&Chain> for Box<dyn Stream<Item = Vec<Bytes>> + Send> {
    fn from(chain: &Chain) -> Box<dyn Stream<Item = Vec<Bytes>> + Send> {
        let mut stream: FuturesOrdered<Box<dyn Future<Output = Vec<Bytes>> + Unpin + Send>> =
            FuturesOrdered::new();

        for i in 0..chain.latest_block {
            let fut = chain.fs_dir.clone().get(i.into());
            stream.push(Box::new(fut.boxed()));
        }

        Box::new(stream)
    }
}
