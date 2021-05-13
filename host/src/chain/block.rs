//! A [`Chain`] which stores every mutation of its [`Subject`] in a series of [`ChainBlock`]s.
//!
//! Each block in the chain begins with the hash of the previous block.

use async_trait::async_trait;
use destream::de;
use futures::future::TryFutureExt;
use futures::{join, try_join};
use log::debug;

use tc_error::*;
use tc_transact::fs::{Block, BlockData, Persist, Store};
use tc_transact::{IntoView, Transact};
use tcgeneric::TCPathBuf;

use crate::fs;
use crate::route::Public;
use crate::scalar::{Link, Value};
use crate::state::State;
use crate::transact::Transaction;
use crate::txn::{Txn, TxnId};

use super::data::Mutation;
use super::internal::{ChainData, ChainDataView};
use super::{Chain, ChainInstance, ChainType, Schema, Subject, CHAIN};

/// A [`Chain`] which stores every mutation of its [`Subject`] in a series of [`ChainBlock`]s
#[derive(Clone)]
pub struct BlockChain {
    schema: Schema,
    subject: Subject,
    history: ChainData,
}

impl BlockChain {
    fn new(schema: Schema, subject: Subject, history: ChainData) -> Self {
        Self {
            schema,
            subject,
            history,
        }
    }
}

#[async_trait]
impl ChainInstance for BlockChain {
    async fn append_delete(&self, txn_id: TxnId, path: TCPathBuf, key: Value) -> TCResult<()> {
        self.history.append_delete(txn_id, path, key).await
    }

    async fn append_put(
        &self,
        txn_id: TxnId,
        path: TCPathBuf,
        key: Value,
        value: State,
    ) -> TCResult<()> {
        self.history.append_put(txn_id, path, key, value).await
    }

    async fn last_commit(&self, txn_id: TxnId) -> TCResult<Option<TxnId>> {
        self.history.last_commit(txn_id).await
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

        let (latest, other_latest) = try_join!(
            self.history.latest_block_id(txn.id()),
            chain.history.latest_block_id(txn.id())
        )?;
        if latest > other_latest {
            return Err(TCError::bad_request(
                "cannot replicate from blockchain with fewer blocks",
                latest,
            ));
        }

        const ERR_DIVERGENT: &str = "blockchain to replicate diverges at block";
        for i in 0u64..latest {
            let block = self.history.read_block(*txn.id(), i.into()).await?;
            let other = chain.history.read_block(*txn.id(), i.into()).await?;
            if &*block != &*other {
                return Err(TCError::bad_request(ERR_DIVERGENT, i));
            }
        }

        let block = chain.history.read_block(*txn.id(), latest.into()).await?;
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

        (*self.history.write_block(*txn.id(), latest.into()).await?) = (*block).clone();

        let mut i = latest + 1;
        while chain.history.contains_block(txn.id(), i).await? {
            let block = chain.history.read_block(*txn.id(), i.into()).await?;

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

            let next_block = self.history.create_next_block(*txn.id()).await?;
            (*next_block.write().await) = block.clone();

            i += 1;
        }

        Ok(())
    }

    async fn prepare_commit(&self, txn_id: &TxnId) {
        self.history.prepare_commit(txn_id).await
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
        let is_new = dir.is_empty(txn.id()).await?;
        let subject = Subject::load(txn, schema.clone(), &dir).await?;

        let history = if is_new {
            ChainData::create(*txn.id(), dir, ChainType::Block).await?
        } else {
            ChainData::load(txn, (), dir).await?
        };

        let block = history.read_latest(*txn.id()).await?;
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

        Ok(BlockChain::new(schema, subject, history))
    }
}

#[async_trait]
impl Transact for BlockChain {
    async fn commit(&self, txn_id: &TxnId) {
        {
            let block = self
                .history
                .read_latest(*txn_id)
                .await
                .expect("read latest chain block");

            if block.size().await.expect("block size") >= super::BLOCK_SIZE {
                self.history
                    .create_next_block(*txn_id)
                    .await
                    .expect("bump chain block number");
            }
        }

        join!(self.subject.commit(txn_id), self.history.commit(txn_id));
    }

    async fn finalize(&self, txn_id: &TxnId) {
        join!(self.subject.finalize(txn_id), self.history.finalize(txn_id));
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
        let schema = seq
            .next_element(())
            .await?
            .ok_or_else(|| de::Error::invalid_length(0, "a BlockChain schema"))?;

        let history = seq
            .next_element(self.txn.clone())
            .await?
            .ok_or_else(|| de::Error::invalid_length(1, "a BlockChain history"))?;

        validate(self.txn, schema, history)
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

#[async_trait]
impl<'en> IntoView<'en, fs::Dir> for BlockChain {
    type Txn = Txn;
    type View = (Schema, ChainDataView<'en>);

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        let history = self.history.into_view(txn).await?;
        Ok((self.schema, history))
    }
}

async fn validate(txn: Txn, schema: Schema, history: ChainData) -> TCResult<BlockChain> {
    let txn_id = txn.id();

    let subject = Subject::create(schema.clone(), txn.context(), *txn.id()).await?;

    let on_err =
        |latest| move |e| TCError::bad_request(format!("error replaying block {}", latest), e);

    let mut i = 0u64;
    while history.contains_block(&txn_id, i).await? {
        let block = history.read_block(*txn_id, i.into()).await?;

        for (_, ops) in block.mutations() {
            for mutation in ops.iter().cloned() {
                match mutation {
                    Mutation::Delete(path, key) => {
                        debug!("replay DELETE op: {}: {}", path, key);
                        subject.delete(&txn, &path, key).map_err(on_err(i)).await?
                    }
                    Mutation::Put(path, key, value) => {
                        debug!("replay PUT op: {}: {} <- {}", path, key, value);
                        subject
                            .put(&txn, &path, key, value.into())
                            .map_err(on_err(i))
                            .await?
                    }
                }
            }
        }

        i += 1;
    }

    Ok(BlockChain::new(schema, subject, history))
}
