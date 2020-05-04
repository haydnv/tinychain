use std::collections::VecDeque;
use std::sync::{Arc, RwLock};

use bytes::Bytes;
use futures::future;
use futures::stream::{FuturesOrdered, Stream};
use futures::{Future, FutureExt, StreamExt};
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::error;
use crate::internal::block::Store;
use crate::internal::{GROUP_DELIMITER, RECORD_DELIMITER};
use crate::transaction::TransactionId;
use crate::value::{PathSegment, TCResult};

type ChainStream<T> =
    FuturesOrdered<Box<dyn Future<Output = Vec<(TransactionId, Vec<T>)>> + Unpin + Send>>;

pub struct Chain {
    store: Arc<Store>,
    latest_block: RwLock<u64>,
}

impl Chain {
    pub fn new(store: Arc<Store>) -> Arc<Chain> {
        store.new_block(0.into(), Bytes::from(&[0; 32][..]));

        Arc::new(Chain {
            store,
            latest_block: RwLock::new(0),
        })
    }

    pub fn copy_from(
        stream: impl Stream<Item = Vec<(TransactionId, Vec<Bytes>)>> + Unpin,
        dest: Arc<Store>,
    ) -> impl Future<Output = Arc<Chain>> {
        stream
            .fold((0u64, dest), |acc, block| async move {
                let (i, dest) = acc;
                let block_id: PathSegment = i.into();
                dest.new_block(block_id.clone(), Bytes::from(&[0; 32][..]));

                for (txn_id, data) in block {
                    dest.append(&block_id, txn_id.into(), data);
                }

                dest.clone().flush(block_id).await;
                (i, dest)
            })
            .then(|(i, dest)| async move {
                Arc::new(Chain {
                    store: dest,
                    latest_block: RwLock::new(i),
                })
            })
    }

    pub async fn from_store(store: Arc<Store>) -> TCResult<Arc<Chain>> {
        let mut latest_block = 0;
        if !store.exists(&latest_block.into()).await? {
            return Err(error::bad_request(
                "This store does not contain a Chain",
                format!("{:?}", store),
            ));
        }

        while store.exists(&(latest_block + 1).into()).await? {
            latest_block += 1;
        }

        Ok(Arc::new(Chain {
            store,
            latest_block: RwLock::new(latest_block),
        }))
    }

    pub async fn put<T: Serialize>(self: &Arc<Self>, txn_id: &TransactionId, mutations: &[T]) {
        let mut latest_block: u64 = *self.latest_block.read().unwrap();
        let delta: Vec<Bytes> = mutations
            .iter()
            .map(|e| Bytes::from(serde_json::to_string_pretty(e).unwrap()))
            .collect();

        let txn_id: Bytes = txn_id.into();
        if !self
            .store
            .will_fit(&latest_block.into(), &txn_id, &delta)
            .await
        {
            latest_block += 1;
            *self.latest_block.write().unwrap() = latest_block;
        }

        self.store.append(&latest_block.into(), txn_id, delta);
        self.store.clone().flush(latest_block.into()).await;
    }

    fn stream(self: &Arc<Self>) -> impl Stream<Item = Vec<(TransactionId, Vec<Bytes>)>> {
        let mut stream: ChainStream<Bytes> = FuturesOrdered::new();

        for i in 0..*self.latest_block.read().unwrap() + 1 {
            let fut = self
                .store
                .clone()
                .get_block(i.into())
                .then(|block| async move {
                    let mut block: VecDeque<&[u8]> =
                        block.split(|b| *b == GROUP_DELIMITER as u8).collect();
                    block.pop_back();

                    let _header = block.pop_front();

                    let mut records: Vec<(TransactionId, Vec<Bytes>)> =
                        Vec::with_capacity(block.len());
                    while let Some(txn) = block.pop_front() {
                        let mut txn: Vec<&[u8]> =
                            txn.split(|b| *b == RECORD_DELIMITER as u8).collect();
                        txn.pop();

                        let txn_id = TransactionId::from(Bytes::copy_from_slice(txn[0]));
                        let txn: Vec<Bytes> =
                            txn[1..].iter().map(|e| Bytes::copy_from_slice(e)).collect();
                        records.push((txn_id, txn));
                    }

                    records
                });

            stream.push(Box::new(fut.boxed()));
        }

        stream
    }

    pub fn stream_into<T: 'static + Clone + DeserializeOwned>(
        self: &Arc<Self>,
    ) -> impl Stream<Item = Vec<T>> {
        self.stream()
            .map(move |block: Vec<(TransactionId, Vec<Bytes>)>| {
                block
                    .iter()
                    .map(|(_, data)| {
                        data.iter()
                            .map(|e| serde_json::from_slice::<T>(e).unwrap())
                            .collect::<Vec<T>>()
                    })
                    .collect()
            })
            .map(|block: Vec<Vec<T>>| block.iter().flatten().cloned().collect())
    }

    pub fn stream_until(
        self: &Arc<Self>,
        txn_id: TransactionId,
    ) -> impl Stream<Item = Vec<(TransactionId, Vec<Bytes>)>> {
        let txn_id_clone = txn_id.clone();
        self.stream()
            .take_while(move |b| {
                if let Some((time, _)) = b.last() {
                    future::ready(time <= &txn_id_clone)
                } else {
                    future::ready(false)
                }
            })
            .map(move |block| {
                block
                    .iter()
                    .filter(|(time, _)| time <= &txn_id)
                    .cloned()
                    .collect()
            })
    }

    pub fn stream_into_until<T: 'static + Clone + DeserializeOwned>(
        self: &Arc<Self>,
        txn_id: TransactionId,
    ) -> impl Stream<Item = Vec<T>> {
        self.stream_until(txn_id.clone())
            .map(move |block: Vec<(TransactionId, Vec<Bytes>)>| {
                block
                    .iter()
                    .map(|(time, data)| {
                        (
                            time.clone(),
                            data.iter()
                                .map(|e| serde_json::from_slice::<T>(e).unwrap())
                                .collect(),
                        )
                    })
                    .filter(|(time, _)| time <= &txn_id)
                    .collect()
            })
            .map(|block: Vec<(TransactionId, Vec<T>)>| {
                let mut result = vec![];
                for (_time, mutations) in block {
                    result.extend(mutations.to_vec())
                }
                result.to_vec()
            })
            .take_while(|block| future::ready(!block.is_empty()))
    }
}
