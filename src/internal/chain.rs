use std::collections::{HashMap, VecDeque};
use std::convert::{TryFrom, TryInto};
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

struct TransactionCache<M: Mutation> {
    cache: RwLock<HashMap<TransactionId, Vec<M>>>,
}

impl<M: Mutation> TransactionCache<M> {
    fn new() -> TransactionCache<M> {
        TransactionCache {
            cache: RwLock::new(HashMap::new()),
        }
    }

    fn close(&self, txn_id: &TransactionId) -> Vec<M> {
        self.cache
            .write()
            .unwrap()
            .remove(txn_id)
            .unwrap_or_else(Vec::new)
    }

    fn get(&self, txn_id: &TransactionId) -> Vec<M> {
        self.cache
            .read()
            .unwrap()
            .get(txn_id)
            .map(|v| v.to_vec())
            .unwrap_or_else(Vec::new)
    }

    fn extend<I: Iterator<Item = M>>(&self, txn_id: TransactionId, iter: I) {
        let mut cache = self.cache.write().unwrap();
        if let Some(list) = cache.get_mut(&txn_id) {
            list.extend(iter);
        } else {
            cache.insert(txn_id, iter.collect());
        }
    }
}

#[derive(Clone)]
pub struct ChainBlock<M: Mutation> {
    checksum: Checksum,
    mutations: Vec<(TransactionId, Vec<M>)>,
}

impl<M: Mutation> ChainBlock<M> {
    fn empty(checksum: Checksum) -> ChainBlock<M> {
        ChainBlock {
            checksum,
            mutations: vec![],
        }
    }

    fn new(checksum: Checksum, txn_id: TransactionId, records: Vec<M>) -> ChainBlock<M> {
        ChainBlock {
            checksum,
            mutations: vec![(txn_id, records)],
        }
    }

    fn encode_transaction(txn_id: &TransactionId, mutations: &[M]) -> Bytes {
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
    fn from(block: ChainBlock<M>) -> Bytes {
        let mut buf = BytesMut::new();
        buf.put(&block.checksum[..]);
        buf.put_u8(GROUP_DELIMITER as u8);

        for (txn_id, mutations) in block.mutations {
            buf.extend(ChainBlock::encode_transaction(&txn_id, &mutations));
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
    cache: TransactionCache<M>,
    store: Arc<Store>,
    latest_block: RwLock<u64>,
}

impl<M: Mutation> Chain<M> {
    pub fn new(store: Arc<Store>) -> Arc<Chain<M>> {
        let checksum = Bytes::from(&[0; 32][..]);
        store.new_block(0.into(), delimit_groups(&[checksum]));

        Arc::new(Chain {
            cache: TransactionCache::new(),
            store,
            latest_block: RwLock::new(0),
        })
    }

    pub async fn copy_from(
        mut source: impl Stream<Item = Bytes> + Unpin,
        dest: Arc<Store>,
    ) -> Arc<Chain<M>> {
        let mut latest_block: u64 = 0;
        let mut checksum = [0; 32];
        while let Some(block) = source.next().await {
            let block_id: PathSegment = latest_block.into();
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
            dest.put_block(block_id.clone(), block.into());
            latest_block += 1;
        }

        if latest_block > 0 {
            latest_block -= 1;
        }

        Arc::new(Chain {
            cache: TransactionCache::new(),
            store: dest,
            latest_block: RwLock::new(latest_block),
        })
    }

    pub async fn from_store(store: Arc<Store>) -> TCResult<Arc<Chain<M>>> {
        let mut latest_block = 0;
        if !store.exists(&latest_block.into()).await? {
            return Err(error::bad_request(
                "This store does not contain a Chain",
                "",
            ));
        }

        while store.exists(&(latest_block + 1).into()).await? {
            latest_block += 1;
        }

        Ok(Arc::new(Chain {
            cache: TransactionCache::new(),
            store,
            latest_block: RwLock::new(latest_block),
        }))
    }

    pub fn get_pending(self: &Arc<Self>, txn_id: &TransactionId) -> Vec<M> {
        self.cache.get(txn_id)
    }

    pub fn put<I: Iterator<Item = M>>(self: &Arc<Self>, txn_id: TransactionId, mutations: I) {
        self.cache.extend(txn_id, mutations)
    }

    fn stream(
        self: &Arc<Self>,
    ) -> impl Stream<Item = (Checksum, Vec<(TransactionId, Vec<Bytes>)>)> {
        let mut stream: BlockStream = FuturesOrdered::new();

        let mut i = 0;
        loop {
            stream.push(Box::new(
                self.store
                    .clone()
                    .get_bytes(i.into())
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
        txn_id: Option<TransactionId>,
    ) -> impl Stream<Item = ChainBlock<M>> {
        let txn_id_clone1 = txn_id.clone();
        let txn_id_clone2 = txn_id.clone();
        let blocks = self
            .stream()
            .take_while(move |(_, records)| {
                if let Some(txn_id) = &txn_id {
                    if let Some((time, _)) = records.last() {
                        future::ready(time <= txn_id)
                    } else {
                        future::ready(false)
                    }
                } else {
                    future::ready(true)
                }
            })
            .map(move |(checksum, mutations)| {
                let mutations = if let Some(txn_id) = &txn_id_clone1 {
                    mutations
                        .into_iter()
                        .filter(|(time, _)| time <= txn_id)
                        .collect()
                } else {
                    mutations
                };

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
            });

        let latest_block: u64 = *self.latest_block.read().unwrap();
        blocks.chain(stream::once(async move {
            let latest_block: ChainBlock<M> = self
                .store
                .clone()
                .get_block(latest_block.into())
                .await
                .unwrap();
            let checksum: Checksum = latest_block.into();
            if let Some(txn_id) = txn_id_clone2 {
                let mutations: Vec<M> = self.cache.get(&txn_id);
                ChainBlock::new(checksum, txn_id, mutations)
            } else {
                ChainBlock::empty(checksum)
            }
        }))
    }

    pub fn stream_bytes(
        self: Arc<Self>,
        txn_id: Option<TransactionId>,
    ) -> impl Stream<Item = Bytes> {
        self.stream_blocks(txn_id)
            .map(|block: ChainBlock<M>| block.into())
    }

    pub fn stream_into(self: Arc<Self>, txn_id: Option<TransactionId>) -> impl Stream<Item = M> {
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
        let mutations: Vec<M> = self.cache.close(txn_id);
        let latest_block: u64 = *self.latest_block.read().unwrap();
        let encoded = ChainBlock::encode_transaction(txn_id, &mutations);

        println!(
            "Chain::commit {} mutations in block {}",
            mutations.len(),
            latest_block
        );

        self.store.append(&latest_block.into(), encoded);
        self.store.clone().flush(latest_block.into()).await;
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
