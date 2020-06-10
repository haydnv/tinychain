use std::collections::VecDeque;
use std::convert::{TryFrom, TryInto};
use std::iter;
use std::sync::Arc;

use bytes::{BufMut, Bytes, BytesMut};
use futures::future;
use futures::stream::{self, FuturesOrdered, Stream};
use futures::{Future, FutureExt, StreamExt};
use serde::de::DeserializeOwned;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::error;
use crate::state::Collect;
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::link::PathSegment;
use crate::value::TCResult;

use super::file::File;
use super::{GROUP_DELIMITER, RECORD_DELIMITER};

const BLOCK_SIZE: usize = 1_000_000;

type Checksum = [u8; 32];

// TODO: remove Mutation
// give Chain an object instead with either a value type (: Object) or (Selector, Data) (: Collection)
// use this as the type param for ChainBlock

#[derive(Clone)]
pub struct ChainBlock<M: Clone + DeserializeOwned + Serialize> {
    checksum: Checksum,
    mutations: Vec<(TxnId, Vec<M>)>,
}

impl<M: Clone + DeserializeOwned + Serialize> ChainBlock<M> {
    fn checksum(self) -> Checksum {
        if self.is_empty() {
            return [0; 32];
        }

        let mut hasher = Sha256::new();
        let data: Bytes = self.into();
        hasher.input(data);
        let mut checksum = [0; 32];
        checksum.copy_from_slice(&hasher.result()[..]);
        checksum
    }

    fn iter(&self) -> std::slice::Iter<(TxnId, Vec<M>)> {
        self.mutations.iter()
    }

    fn len(&self) -> usize {
        self.mutations.len() + 1
    }
}

fn valid_encoding(val: &[u8]) -> bool {
    for b in val {
        if *b == GROUP_DELIMITER as u8 || *b == RECORD_DELIMITER as u8 {
            return false;
        }
    }

    true
}

impl<M: Clone + DeserializeOwned + Serialize> ChainBlock<M> {
    fn encode_transaction<I: Iterator<Item = M>>(txn_id: &TxnId, mutations: I) -> TCResult<Bytes> {
        let mut buf = BytesMut::new();
        buf.extend(serde_json::to_string_pretty(txn_id).unwrap().as_bytes());
        buf.put_u8(RECORD_DELIMITER as u8);
        for mutation in mutations {
            let mutation = serde_json::to_string_pretty(&mutation).unwrap();
            let encoded = mutation.as_bytes();

            if !valid_encoding(&encoded) {
                return Err(error::bad_request(
                    "Attempt to encode a value containing an ASCII control character",
                    mutation,
                ));
            }

            buf.extend(encoded);
            buf.put_u8(RECORD_DELIMITER as u8);
        }
        buf.put_u8(GROUP_DELIMITER as u8);

        Ok(buf.into())
    }

    fn is_empty(&self) -> bool {
        self.mutations.is_empty()
    }
}

impl<M: Clone + DeserializeOwned + Serialize> From<ChainBlock<M>> for Bytes {
    fn from(mut block: ChainBlock<M>) -> Bytes {
        let mut buf = BytesMut::new();
        buf.put(&block.checksum[..]);
        buf.put_u8(GROUP_DELIMITER as u8);

        for (txn_id, mut mutations) in block.mutations.drain(..) {
            buf.extend(ChainBlock::encode_transaction(&txn_id, mutations.drain(..)).unwrap());
        }

        Bytes::from(buf)
    }
}

impl<M: Clone + DeserializeOwned + Serialize> TryFrom<Bytes> for ChainBlock<M> {
    type Error = error::TCError;

    fn try_from(buf: Bytes) -> TCResult<ChainBlock<M>> {
        let mut buf: VecDeque<&[u8]> = buf.split(|b| *b == GROUP_DELIMITER as u8).collect();
        buf.pop_back();

        let mut checksum: Checksum = [0; 32];
        checksum.copy_from_slice(&buf.pop_front().unwrap()[0..32]);

        let mut mutations: Vec<(TxnId, Vec<M>)> = Vec::with_capacity(buf.len());
        while let Some(record) = buf.pop_front() {
            let record: Vec<&[u8]> = record.split(|b| *b == RECORD_DELIMITER as u8).collect();

            let txn_id: TxnId = serde_json::from_slice(&record[0]).unwrap();
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

type BlockStream =
    FuturesOrdered<Box<dyn Future<Output = (Checksum, Vec<(TxnId, Vec<Bytes>)>)> + Unpin + Send>>;

pub struct Chain<T: Collect> {
    object: T,
    file: Arc<File>,
    latest_block: u64,
}

impl<T: Collect> Chain<T> {
    pub async fn new(txn_id: TxnId, file: Arc<File>, object: T) -> Chain<T> {
        let checksum = Bytes::from(&[0; 32][..]);
        file
            .new_block(txn_id.clone(), 0u8.into(), delimit_groups(&[checksum]))
            .await
            .unwrap();
        println!("Chain::new created block 0");

        let chain = Chain {
            object,
            file,
            latest_block: 0,
        };

        chain.put(txn_id, iter::empty()).await.unwrap();

        chain
    }

    pub async fn copy_from(
        mut source: impl Stream<Item = Bytes> + Unpin,
        txn: &Arc<Txn>,
        dest: Arc<File>,
        object: T,
    ) -> Chain<T> {
        let mut latest_block: u64 = 0;
        let mut checksum: Checksum = [0; 32];
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

            let block: ChainBlock<(T::Selector, T::Item)> = block.try_into().unwrap();
            checksum = block.clone().checksum();
            Chain::populate(txn, block.clone(), &object).await;

            dest.new_block(txn.id().clone(), block_id.clone(), block.into())
                .await
                .unwrap();

            println!("copied Chain block {}", latest_block);
            latest_block += 1;
        }

        if latest_block > 0 {
            latest_block -= 1;
        } else {
            panic!("Chain::copy_from called on a stream with no blocks!")
        }

        Chain {
            file: dest,
            object,
            latest_block,
        }
    }

    pub async fn from_store(txn: &Arc<Txn>, file: Arc<File>, object: T) -> TCResult<Chain<T>> {
        let mut latest_block = 0;
        if !file.contains_block(txn.id(), &latest_block.into()).await {
            return Err(error::bad_request(
                "This store does not contain a Chain",
                "",
            ));
        } else {
            println!("Chain::from_store");
        }

        while let Some(block) = file.get_block(txn.id(), &(latest_block + 1).into()).await {
            let block: ChainBlock<(T::Selector, T::Item)> = block.try_into()?;
            Chain::populate(txn, block.clone(), &object).await;
            latest_block += 1;
        }

        Ok(Chain {
            file,
            object,
            latest_block,
        })
    }

    pub async fn put<I: Iterator<Item = (T::Selector, T::Item)>>(
        &self,
        txn_id: TxnId,
        mutations: I,
    ) -> TCResult<()> {
        let block_id: PathSegment = self.latest_block.into();

        self.file
            .append(
                &txn_id,
                &block_id,
                ChainBlock::encode_transaction(&txn_id, mutations)?,
            )
            .await
            .unwrap();

        Ok(())
    }

    fn stream(&self, txn_id: TxnId) -> impl Stream<Item = (Checksum, Vec<(TxnId, Vec<Bytes>)>)> {
        let mut stream: BlockStream = FuturesOrdered::new();

        let mut i = 0;
        loop {
            let block_id = i;
            stream.push(Box::new(
                self.file
                    .clone()
                    .get_block_owned(txn_id.clone(), block_id.into())
                    .map(move |block| {
                        let block = block.unwrap_or_else(|| {
                            panic!("This chain has a nonexistent block at {}!", block_id)
                        });
                        let mut block: VecDeque<&[u8]> =
                            block.split(|b| *b == GROUP_DELIMITER as u8).collect();
                        block.pop_back();
                        println!("block has {} groups", block.len());

                        let mut checksum = [0u8; 32];
                        checksum.copy_from_slice(&block.pop_front().unwrap()[0..32]);

                        let mut transactions = Vec::with_capacity(block.len());
                        while let Some(mutations) = block.pop_front() {
                            let mut mutations: VecDeque<Bytes> = mutations
                                .split(|b| *b == RECORD_DELIMITER as u8)
                                .map(|m| Bytes::copy_from_slice(m))
                                .collect();
                            mutations.pop_back();
                            println!("record size {}: {}", mutations.len(), mutations[0].len());

                            let txn_id: TxnId =
                                serde_json::from_slice(&mutations.pop_front().unwrap()).unwrap();
                            transactions.push((txn_id, mutations.into_iter().collect()))
                        }
                        println!("block records {} transactions", transactions.len());

                        (checksum, transactions)
                    })
                    .boxed(),
            ));

            if i == self.latest_block {
                println!("Chain::stream completed queue at {}", i);
                break;
            } else {
                i += 1;
            }
        }

        stream
    }

    pub fn stream_blocks(
        &self,
        txn_id: TxnId,
    ) -> impl Stream<Item = ChainBlock<(T::Selector, T::Item)>> {
        let txn_id_clone = txn_id.clone();

        self.stream(txn_id.clone())
            .take_while(move |(_, records)| {
                if let Some((time, _)) = records.last() {
                    if time <= &txn_id {
                        println!("Passing block through");
                        future::ready(true)
                    } else {
                        println!("Dropping block since its transactions are in the future");
                        future::ready(false)
                    }
                } else {
                    println!("Dropping block since it is empty");
                    future::ready(false)
                }
            })
            .map(move |(checksum, mutations)| {
                let mutations: Vec<(TxnId, Vec<Bytes>)> = mutations
                    .into_iter()
                    .filter(|(time, _)| time <= &txn_id_clone)
                    .collect();

                let block = ChainBlock {
                    checksum,
                    mutations: mutations
                        .iter()
                        .map(|(txn_id, list)| {
                            (
                                txn_id.clone(),
                                list.iter()
                                    .map(|m| {
                                        serde_json::from_slice::<(T::Selector, T::Item)>(m).unwrap()
                                    })
                                    .collect(),
                            )
                        })
                        .collect(),
                };
                println!("ChainBlock size {}", block.len());
                block
            })
    }

    pub fn stream_bytes(&self, txn_id: TxnId) -> impl Stream<Item = Bytes> {
        self.stream_blocks(txn_id)
            .map(|block: ChainBlock<(T::Selector, T::Item)>| block.into())
    }

    pub fn stream_into(&self, txn_id: TxnId) -> impl Stream<Item = (T::Selector, T::Item)> {
        self.stream_blocks(txn_id)
            .filter_map(|block| {
                println!("stream_into read block");
                let mutations: Vec<(T::Selector, T::Item)> = block
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

    pub async fn commit(&mut self, txn_id: &TxnId) {
        println!("Chain::commit");
        let block = self
            .file
            .clone()
            .get_block(txn_id, &self.latest_block.into())
            .await
            .unwrap();

        if block.len() > BLOCK_SIZE {
            self.latest_block += 1;
            let block: ChainBlock<(T::Item, T::Selector)> = block.try_into().unwrap();
            let checksum: Checksum = block.checksum();
            self.file
                .new_block(
                    txn_id.clone(),
                    self.latest_block.into(),
                    delimit_groups(&[Bytes::copy_from_slice(&checksum[..])]),
                )
                .await
                .unwrap();
        }

        self.file.commit(txn_id).await;
    }

    pub async fn rollback(&self, txn_id: &TxnId) {
        println!("Chain::rollback");
        self.file.rollback(txn_id).await
    }

    async fn populate(txn: &Arc<Txn>, block: ChainBlock<(T::Selector, T::Item)>, object: &T) {
        let mut put_ops = Vec::with_capacity(block.len());
        for (_, mutations) in block.iter() {
            for (key, val) in mutations.iter() {
                put_ops.push(object.put(txn, key.clone(), val.clone()));
            }
        }

        future::try_join_all(put_ops).await.unwrap();
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
