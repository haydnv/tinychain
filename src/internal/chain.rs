use std::sync::Arc;

use bytes::Bytes;
use futures::future;
use futures::stream::{FuturesOrdered, Stream};
use futures::{Future, FutureExt, StreamExt};
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::internal::block::Store;
use crate::internal::{GROUP_DELIMITER, RECORD_DELIMITER};
use crate::transaction::TransactionId;

pub struct Chain {
    store: Arc<Store>,
    latest_block: u64,
}

impl Chain {
    pub fn new(store: Arc<Store>) -> Arc<Chain> {
        Arc::new(Chain {
            store,
            latest_block: 0,
        })
    }

    pub fn from(
        stream: impl Stream<Item = Vec<(TransactionId, Vec<Bytes>)>> + Unpin,
        dest: Arc<Store>,
    ) -> impl Future<Output = Arc<Chain>> {
        stream
            .fold((0u64, dest), |acc, block| async move {
                let (i, dest) = acc;
                for (txn_id, data) in block {
                    dest.clone().flush(i.into(), &txn_id.into(), &data).await;
                }
                (i, dest)
            })
            .then(|(i, dest)| async move {
                Arc::new(Chain {
                    store: dest,
                    latest_block: i,
                })
            })
    }

    pub fn until<T: 'static + Clone + DeserializeOwned>(
        self: &Arc<Self>,
        txn_id: TransactionId,
    ) -> impl Stream<Item = Vec<T>> {
        self.into_stream(txn_id.clone())
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

    pub fn put<T: Serialize>(
        self: Arc<Self>,
        txn_id: &TransactionId,
        mutations: &[T],
    ) -> impl Future<Output = ()> {
        let delta: Vec<Bytes> = mutations
            .iter()
            .map(|e| Bytes::from(serde_json::to_string_pretty(e).unwrap()))
            .collect();
        self.store
            .clone()
            .flush(self.latest_block.into(), &txn_id.into(), &delta)
    }

    pub fn into_stream(
        self: &Arc<Self>,
        txn_id: TransactionId,
    ) -> impl Stream<Item = Vec<(TransactionId, Vec<Bytes>)>> {
        let mut stream: FuturesOrdered<
            Box<dyn Future<Output = Vec<(TransactionId, Vec<Bytes>)>> + Unpin + Send>,
        > = FuturesOrdered::new();

        for i in 0..self.latest_block {
            let txn_id = txn_id.clone();
            let fut = self.store.clone().get(i.into()).then(|block| async move {
                let mut block: Vec<&[u8]> = block.split(|b| *b == GROUP_DELIMITER as u8).collect();
                block.pop();

                let mut data: Vec<(TransactionId, Vec<Bytes>)> = vec![];
                for txn in block {
                    let mut txn: Vec<&[u8]> = txn.split(|b| *b == RECORD_DELIMITER as u8).collect();
                    txn.pop();

                    let time = TransactionId::from(Bytes::copy_from_slice(txn[0]));
                    if time <= txn_id {
                        let txn: Vec<Bytes> =
                            txn[1..].iter().map(|e| Bytes::copy_from_slice(e)).collect();
                        data.push((time, txn));
                    }
                }
                data
            });
            stream.push(Box::new(fut.boxed()));
        }

        stream.take_while(|b| future::ready(!b.is_empty()))
    }
}
