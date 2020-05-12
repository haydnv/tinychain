use std::collections::VecDeque;
use std::convert::{TryFrom, TryInto};
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use bytes::{BufMut, Bytes, BytesMut};
use futures::future;
use futures::stream::{self, FuturesOrdered, Stream};
use futures::{Future, FutureExt, StreamExt};
use serde::de::DeserializeOwned;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::error;
use crate::internal::block::{Block, Checksum, Store};
use crate::internal::{GROUP_DELIMITER, RECORD_DELIMITER};
use crate::transaction::{Transact, TransactionId};
use crate::value::{PathSegment, TCResult};

pub trait Mutation: Clone + DeserializeOwned + Serialize + Send + Sync {}

#[derive(Clone)]
pub struct ChainBlock<M: Mutation> {
    checksum: Checksum,
    mutations: Vec<(TransactionId, Vec<M>)>,
}

impl<M: Mutation> ChainBlock<M> {
    fn encode_transaction<I: Iterator<Item = M>>(txn_id: &TransactionId, mutations: I) -> Bytes {
        let mut buf = BytesMut::new();
        let txn_id: Bytes = txn_id.into();
        buf.extend(txn_id);
        buf.put_u8(RECORD_DELIMITER as u8);
        for mutation in mutations {
            buf.extend(
                serde_json::to_string_pretty(&mutation)
                    .unwrap()
                    .into_bytes(),
            );
            buf.put_u8(RECORD_DELIMITER as u8);
        }
        buf.put_u8(GROUP_DELIMITER as u8);

        buf.into()
    }

    fn is_empty(&self) -> bool {
        self.mutations.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<(TransactionId, Vec<M>)> {
        self.mutations.iter()
    }
}

impl<M: Mutation> Block for ChainBlock<M> {}

impl<M: Mutation> From<ChainBlock<M>> for Bytes {
    fn from(mut block: ChainBlock<M>) -> Bytes {
        let mut buf = BytesMut::new();
        buf.put(&block.checksum[..]);
        buf.put_u8(GROUP_DELIMITER as u8);

        for (txn_id, mut mutations) in block.mutations.drain(..) {
            buf.extend(ChainBlock::encode_transaction(&txn_id, mutations.drain(..)));
        }

        Bytes::from(buf)
    }
}

impl<M: Mutation> From<ChainBlock<M>> for Checksum {
    fn from(block: ChainBlock<M>) -> Checksum {
        if block.is_empty() {
            return [0; 32];
        }

        let mut hasher = Sha256::new();
        let data: Bytes = block.into();
        hasher.input(data);
        let mut checksum = [0; 32];
        checksum.copy_from_slice(&hasher.result()[..]);
        checksum
    }
}

impl<M: Mutation> TryFrom<Bytes> for ChainBlock<M> {
    type Error = error::TCError;

    fn try_from(buf: Bytes) -> TCResult<ChainBlock<M>> {
        let mut buf: VecDeque<&[u8]> = buf.split(|b| *b == GROUP_DELIMITER as u8).collect();
        buf.pop_back();

        let mut checksum: Checksum = [0; 32];
        checksum.copy_from_slice(&buf.pop_front().unwrap()[0..32]);

        let mut mutations: Vec<(TransactionId, Vec<M>)> = Vec::with_capacity(buf.len());
        while let Some(record) = buf.pop_front() {
            let record: Vec<&[u8]> = record.split(|b| *b == RECORD_DELIMITER as u8).collect();

            let txn_id: TransactionId = Bytes::copy_from_slice(record[0]).into();
            mutations.push((
                txn_id,
                record[1..record.len() - 1]
                    .iter()
                    .map(|m| serde_json::from_slice::<M>(&m).unwrap())
                    .collect(),
            ));
        }

        Ok(ChainBlock {
            checksum,
            mutations,
        })
    }
}

type BlockStream = FuturesOrdered<
    Box<dyn Future<Output = (Checksum, Vec<(TransactionId, Vec<Bytes>)>)> + Unpin + Send>,
>;

pub struct Chain<M: Mutation> {
    store: Arc<Store>,
    latest_block: RwLock<u64>,
    phantom: PhantomData<M>,
}

impl<M: Mutation> Chain<M> {
    pub async fn new(txn_id: &TransactionId, store: Arc<Store>) -> Arc<Chain<M>> {
        let checksum = Bytes::from(&[0; 32][..]);
        store
            .new_block(&txn_id, 0.into(), delimit_groups(&[checksum]))
            .await
            .unwrap();
        println!("Chain::new created block 0");

        Arc::new(Chain {
            store,
            latest_block: RwLock::new(0),
            phantom: PhantomData,
        })
    }

    pub async fn copy_from(
        mut source: impl Stream<Item = Bytes> + Unpin,
        txn_id: &TransactionId,
        dest: Arc<Store>,
    ) -> Arc<Chain<M>> {
        let mut latest_block: u64 = 0;
        let mut checksum = [0; 32];
        while let Some(block) = source.next().await {
            let block_id: PathSegment = latest_block.into();
            println!("copying Chain block {}", block_id);

            if checksum[..] != block[0..32] {
                panic!(
                    "Checksum failed for block {}, {} != {}",
                    latest_block,
                    hex::encode(checksum),
                    hex::encode(&block[0..32])
                );
            }

            let block: ChainBlock<M> = block.try_into().unwrap();
            checksum = block.clone().into();
            dest.new_block(&txn_id, block_id.clone(), block.into())
                .await
                .unwrap();
            latest_block += 1;
        }

        if latest_block > 0 {
            latest_block -= 1;
        }

        Arc::new(Chain {
            store: dest,
            latest_block: RwLock::new(latest_block),
            phantom: PhantomData,
        })
    }

    pub async fn from_store(txn_id: &TransactionId, store: Arc<Store>) -> TCResult<Arc<Chain<M>>> {
        let mut latest_block = 0;
        if !store.contains_block(txn_id, &latest_block.into()).await {
            return Err(error::bad_request(
                "This store does not contain a Chain",
                "",
            ));
        }

        while store
            .contains_block(txn_id, &(latest_block + 1).into())
            .await
        {
            latest_block += 1;
        }

        Ok(Arc::new(Chain {
            store,
            latest_block: RwLock::new(latest_block),
            phantom: PhantomData,
        }))
    }

    pub async fn put<I: Iterator<Item = M>>(self: &Arc<Self>, txn_id: TransactionId, mutations: I) {
        let block_id: PathSegment = (*self.latest_block.read().unwrap()).into();
        self.store
            .append(
                &txn_id,
                &block_id,
                ChainBlock::encode_transaction(&txn_id, mutations),
            )
            .await
            .unwrap();
    }

    fn stream(
        self: &Arc<Self>,
        txn_id: TransactionId,
    ) -> impl Stream<Item = (Checksum, Vec<(TransactionId, Vec<Bytes>)>)> {
        let mut stream: BlockStream = FuturesOrdered::new();

        let mut i = 0;
        loop {
            stream.push(Box::new(
                self.store
                    .clone()
                    .get_bytes(txn_id.clone(), i.into())
                    .map(|block| {
                        let block = block.unwrap();
                        let mut block: VecDeque<&[u8]> =
                            block.split(|b| *b == GROUP_DELIMITER as u8).collect();
                        block.pop_back();
                        println!("block has {} groups", block.len());

                        let mut checksum = [0u8; 32];
                        checksum.copy_from_slice(&block.pop_front().unwrap()[0..32]);

                        let mut mutations = Vec::with_capacity(block.len());
                        while let Some(record) = block.pop_front() {
                            let mut record: VecDeque<Bytes> = record
                                .split(|b| *b == RECORD_DELIMITER as u8)
                                .map(|m| Bytes::copy_from_slice(m))
                                .collect();
                            record.pop_back();
                            println!("record size {}: {}", record.len(), record[0].len());

                            let txn_id: TransactionId = record.pop_front().unwrap().into();
                            mutations.push((txn_id, record.into_iter().collect()))
                        }

                        (checksum, mutations)
                    })
                    .boxed(),
            ));
            println!("into_stream got block {}", i);

            if i == *self.latest_block.read().unwrap() {
                println!("into_stream completed queue at {}", i);
                break;
            } else {
                i += 1;
            }
        }

        stream
    }

    pub fn stream_blocks(
        self: Arc<Self>,
        txn_id: TransactionId,
    ) -> impl Stream<Item = ChainBlock<M>> {
        let txn_id_clone = txn_id.clone();

        self.stream(txn_id.clone())
            .take_while(move |(_, records)| {
                if let Some((time, _)) = records.last() {
                    future::ready(time <= &txn_id)
                } else {
                    future::ready(false)
                }
            })
            .map(move |(checksum, mutations)| {
                let mutations: Vec<(TransactionId, Vec<Bytes>)> = mutations
                    .into_iter()
                    .filter(|(time, _)| time <= &txn_id_clone)
                    .collect();

                ChainBlock {
                    checksum,
                    mutations: mutations
                        .iter()
                        .map(|(txn_id, list)| {
                            (
                                txn_id.clone(),
                                list.iter()
                                    .map(|m| serde_json::from_slice::<M>(m).unwrap())
                                    .collect(),
                            )
                        })
                        .collect(),
                }
            })
    }

    pub fn stream_bytes(self: Arc<Self>, txn_id: TransactionId) -> impl Stream<Item = Bytes> {
        self.stream_blocks(txn_id)
            .map(|block: ChainBlock<M>| block.into())
    }

    pub fn stream_into(self: Arc<Self>, txn_id: TransactionId) -> impl Stream<Item = M> {
        self.stream_blocks(txn_id)
            .filter_map(|block| {
                println!("stream_into read block");
                let mutations: Vec<M> = block
                    .mutations
                    .into_iter()
                    .map(|(_, mutations)| mutations)
                    .flatten()
                    .collect();

                if mutations.is_empty() {
                    future::ready(None)
                } else {
                    future::ready(Some(stream::iter(mutations)))
                }
            })
            .flatten()
    }
}

#[async_trait]
impl<M: Mutation> Transact for Chain<M> {
    async fn commit(&self, txn_id: &TransactionId) {
        println!("Chain::commit");
        self.store.commit(txn_id).await;
    }
}

fn delimit_groups(groups: &[Bytes]) -> Bytes {
    let mut buf = BytesMut::new();
    for group in groups {
        buf.extend(group);
        buf.put_u8(GROUP_DELIMITER as u8);
    }
    buf.into()
}
