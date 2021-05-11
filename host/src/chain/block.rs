//! A [`Chain`] which stores every mutation of its [`Subject`] in a series of [`ChainBlock`]s.
//!
//! Each block in the chain begins with the hash of the previous block.

use std::convert::{TryFrom, TryInto};
use std::num::ParseIntError;
use std::pin::Pin;

use async_trait::async_trait;
use bytes::Bytes;
use destream::{de, en};
use futures::future::TryFutureExt;
use futures::join;
use futures::stream::{self, Stream, StreamExt, TryStreamExt};
use log::debug;

use tc_error::*;
use tc_transact::fs::{BlockData, Dir, File, Persist, Store};
use tc_transact::lock::{Mutable, TxnLock};
use tc_transact::{IntoView, Transact};
use tcgeneric::{Instance, TCPathBuf};

use crate::fs;
use crate::route::Public;
use crate::scalar::{Link, Value};
use crate::state::State;
use crate::transact::Transaction;
use crate::txn::{Txn, TxnId};

use super::data::Mutation;
use super::{Chain, ChainBlock, ChainInstance, ChainType, Schema, Subject, CHAIN, NULL_HASH};
use safecast::TryCastInto;

/// A [`Chain`] which stores every mutation of its [`Subject`] in a series of [`ChainBlock`]s
#[derive(Clone)]
pub struct BlockChain {
    schema: Schema,
    subject: Subject,
    latest: TxnLock<Mutable<u64>>,
    file: fs::File<ChainBlock>,
}

impl BlockChain {
    fn new(schema: Schema, subject: Subject, latest: u64, file: fs::File<ChainBlock>) -> Self {
        Self {
            schema,
            subject,
            latest: TxnLock::new("latest BlockChain block ordinal", latest.into()),
            file,
        }
    }
}

#[async_trait]
impl ChainInstance for BlockChain {
    async fn append_delete(&self, txn_id: TxnId, path: TCPathBuf, key: Value) -> TCResult<()> {
        let latest = self.latest.read(&txn_id).await?;
        let mut block = self.file.write_block(txn_id, (*latest).into()).await?;
        block.append_delete(txn_id, path, key);
        Ok(())
    }

    async fn append_put(
        &self,
        txn_id: TxnId,
        path: TCPathBuf,
        key: Value,
        value: State,
    ) -> TCResult<()> {
        if value.is_ref() {
            return Err(TCError::bad_request(
                "cannot update Chain subject with reference: {}",
                value,
            ));
        }

        let latest = self.latest.read(&txn_id).await?;
        let mut block = self.file.write_block(txn_id, (*latest).into()).await?;
        block.append_put(
            txn_id,
            path,
            key,
            value.try_cast_into(|s| TCError::not_implemented(format!("Chain <- {}", s.class())))?,
        );
        Ok(())
    }

    async fn last_commit(&self, txn_id: TxnId) -> TCResult<Option<TxnId>> {
        let latest = self.latest.read(&txn_id).await?;
        let block = self.file.read_block(txn_id, (*latest).into()).await?;
        Ok(block.mutations().keys().last().cloned())
    }

    fn subject(&self) -> &Subject {
        &self.subject
    }

    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()> {
        let chain = match txn.get(source.append(CHAIN.into()), Value::None).await? {
            State::Chain(Chain::Block(chain)) => chain,
            other => {
                return Err(TCError::bad_request(
                    "cannot replicate with a blockchain",
                    other,
                ))
            }
        };

        let latest = self.latest.read(txn.id()).await?;
        if !chain
            .file
            .contains_block(txn.id(), &(*latest).into())
            .await?
        {
            return Err(TCError::bad_request(
                "cannot replicate from blockchain with fewer blocks",
                *latest,
            ));
        }

        const ERR_DIVERGENT: &str = "blockchain to replicate diverges at block";
        for i in 0..(*latest) {
            let block = self.file.read_block(*txn.id(), i.into()).await?;
            let other = chain.file.read_block(*txn.id(), i.into()).await?;
            if &*block != &*other {
                return Err(TCError::bad_request(ERR_DIVERGENT, i));
            }
        }

        let block = chain.file.read_block(*txn.id(), (*latest).into()).await?;
        let mutations = if let Some(last_commit) = self.last_commit(*txn.id()).await? {
            block.mutations().range(last_commit.next()..)
        } else {
            block.mutations().range(..)
        };

        for (_, ops) in mutations {
            for mutation in ops.iter() {
                match mutation {
                    Mutation::Delete(path, key) => {
                        self.subject.delete(txn, &path, key.clone()).await?
                    }
                    Mutation::Put(path, key, value) => {
                        self.subject
                            .put(txn, &path, key.clone(), value.clone().into())
                            .await?
                    }
                }
            }
        }

        (*self.file.write_block(*txn.id(), (*latest).into()).await?) = (*block).clone();

        let mut new_blocks = false;
        let mut i = (*latest) + 1;
        while chain.file.contains_block(txn.id(), &i.into()).await? {
            new_blocks = true;

            let block = chain.file.read_block(*txn.id(), i.into()).await?;

            for (_, ops) in block.mutations() {
                for mutation in ops.iter() {
                    match mutation {
                        Mutation::Delete(path, key) => {
                            self.subject.delete(txn, path, key.clone()).await?
                        }
                        Mutation::Put(path, key, value) => {
                            self.subject
                                .put(txn, path, key.clone(), value.clone().into())
                                .await?
                        }
                    }
                }
            }

            self.file
                .create_block(*txn.id(), i.into(), (*block).clone())
                .await?;

            i += 1;
        }

        if new_blocks {
            *latest.upgrade().await? = i;
        }

        Ok(())
    }

    async fn prepare_commit(&self, txn_id: &TxnId) {
        let latest = self.latest.read(txn_id).await.expect("latest block");

        self.file
            .sync_block(*txn_id, (*latest).into())
            .await
            .expect("prepare BlockChain commit");
    }
}

#[async_trait]
impl Persist<fs::Dir, Txn> for BlockChain {
    type Schema = Schema;
    type Store = fs::Dir;

    fn schema(&self) -> &Schema {
        &self.schema
    }

    async fn load(txn: &Txn, schema: Schema, dir: fs::Dir) -> TCResult<Self> {
        let subject = Subject::load(txn, schema.clone(), &dir).await?;

        let txn_id = *txn.id();
        if let Some(file) = dir.get_file(&txn_id, &CHAIN.into()).await? {
            let file = fs::File::<ChainBlock>::try_from(file)?;

            let block_ids = file.block_ids(&txn_id).await?;
            let block_ids = block_ids
                .into_iter()
                .map(|id| id.as_str().parse())
                .collect::<Result<Vec<u64>, ParseIntError>>()
                .map_err(TCError::internal)?;

            let latest = block_ids.into_iter().fold(0, Ord::max);
            let block = file.read_block(txn_id, latest.into()).await?;
            if let Some((last_txn_id, ops)) = block.mutations().iter().last() {
                for op in ops {
                    subject
                        .apply(txn, op)
                        .map_err(|e| {
                            TCError::internal(format!(
                                "error replaying last transaction {}: {}",
                                last_txn_id, e
                            ))
                        })
                        .await?;
                }
            }

            Ok(BlockChain::new(schema, subject, latest, file))
        } else {
            let latest = 0u64;
            let file = dir
                .create_file(txn_id, CHAIN.into(), ChainType::Sync)
                .await?;

            let file = fs::File::<ChainBlock>::try_from(file)?;
            if !file.contains_block(&txn_id, &latest.into()).await? {
                file.create_block(txn_id, latest.into(), ChainBlock::new(NULL_HASH))
                    .await?;
            }

            Ok(BlockChain::new(schema, subject, 0, file))
        }
    }
}

#[async_trait]
impl Transact for BlockChain {
    async fn commit(&self, txn_id: &TxnId) {
        {
            let latest = self.latest.read(txn_id).await.expect("latest block number");

            let block = self
                .file
                .read_block(*txn_id, (*latest).into())
                .await
                .expect("read latest chain block");

            if block.size().await.expect("block size") >= super::BLOCK_SIZE {
                let mut latest = latest.upgrade().await.expect("latest block number");
                (*latest) += 1;

                let hash = block.hash().await.expect("block hash");

                self.file
                    .create_block(*txn_id, (*latest).into(), ChainBlock::new(hash))
                    .await
                    .expect("bump chain block number");
            }
        }

        join!(
            self.latest.commit(txn_id),
            self.subject.commit(txn_id),
            self.file.commit(txn_id)
        );
    }

    async fn finalize(&self, txn_id: &TxnId) {
        join!(
            self.latest.finalize(txn_id),
            self.subject.finalize(txn_id),
            self.file.finalize(txn_id)
        );
    }
}

struct ChainVisitor {
    txn: Txn,
}

#[async_trait]
impl de::Visitor for ChainVisitor {
    type Value = BlockChain;

    fn expecting() -> &'static str {
        "a BlockChain"
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let txn_id = *self.txn.id();
        let schema = seq
            .next_element(())
            .await?
            .ok_or_else(|| de::Error::invalid_length(0, "a BlockChain schema"))?;

        let file = self
            .txn
            .context()
            .create_file(txn_id, CHAIN.into(), ChainType::Block)
            .map_err(de::Error::custom)
            .await?;

        let file = seq
            .next_element((txn_id, file.try_into().map_err(de::Error::custom)?))
            .await?
            .ok_or_else(|| de::Error::invalid_length(1, "a BlockChain file"))?;

        validate(self.txn, schema, file)
            .map_err(de::Error::custom)
            .await
    }
}

#[async_trait]
impl de::FromStream for BlockChain {
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        let visitor = ChainVisitor { txn };
        decoder.decode_seq(visitor).await
    }
}

pub type BlockStream = Pin<Box<dyn Stream<Item = TCResult<ChainBlock>> + Send>>;
pub type BlockSeq = en::SeqStream<TCError, ChainBlock, BlockStream>;

#[async_trait]
impl<'en> IntoView<'en, fs::Dir> for BlockChain {
    type Txn = Txn;
    type View = (Schema, BlockSeq);

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        let txn_id = *txn.id();
        let file = self.file;
        let latest = self.latest.read(txn.id()).await?;
        let blocks = stream::iter(0..(*latest + 1))
            .then(move |i| file.clone().read_block_owned(txn_id, i.into()))
            .map_ok(|block| (*block).clone());

        let blocks: BlockStream = Box::pin(blocks);
        let blocks: BlockSeq = en::SeqStream::from(blocks);
        Ok((self.schema, blocks))
    }
}

async fn validate(txn: Txn, schema: Schema, file: fs::File<ChainBlock>) -> TCResult<BlockChain> {
    let txn_id = txn.id();

    if file.is_empty(txn_id).await? {
        let subject = Subject::create(schema.clone(), txn.context(), *txn_id).await?;
        return Ok(BlockChain::new(schema, subject, 0, file));
    }

    let subject = Subject::create(schema.clone(), txn.context(), *txn.id()).await?;

    let on_err =
        |latest| move |e| TCError::bad_request(format!("error replaying block {}", latest), e);

    let mut latest = 0u64;
    let mut hash = Bytes::from(NULL_HASH);
    while file.contains_block(&txn_id, &latest.into()).await? {
        let block = file.read_block(*txn_id, latest.into()).await?;
        if block.last_hash() != &hash {
            let last_hash = base64::encode(block.last_hash());
            let hash = base64::encode(hash);
            return Err(TCError::bad_request(
                format!("block {} has invalid hash {}, expected", latest, last_hash),
                hash,
            ));
        }

        for (_, ops) in block.mutations() {
            for mutation in ops.iter().cloned() {
                match mutation {
                    Mutation::Delete(path, key) => {
                        debug!("replay DELETE op: {}: {}", path, key);
                        subject
                            .delete(&txn, &path, key)
                            .map_err(on_err(latest))
                            .await?
                    }
                    Mutation::Put(path, key, value) => {
                        debug!("replay PUT op: {}: {} <- {}", path, key, value);
                        subject
                            .put(&txn, &path, key, value.into())
                            .map_err(on_err(latest))
                            .await?
                    }
                }
            }
        }

        hash = block.last_hash().clone();
        latest += 1;
    }

    Ok(BlockChain::new(schema, subject, latest, file))
}
