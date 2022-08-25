//! A [`Chain`] which stores every mutation of its [`Subject`] in a series of `ChainBlock`s.
//!
//! Each block in the chain begins with the hash of the previous block.

use async_trait::async_trait;
use destream::de;
use futures::future::TryFutureExt;
use futures::join;
use log::debug;
use sha2::digest::Output;
use sha2::Sha256;

use tc_error::*;
use tc_transact::fs::{Dir, Persist};
use tc_transact::{IntoView, Transact};
use tc_value::{Link, Value};
use tcgeneric::TCPathBuf;

use crate::fs;
use crate::state::State;
use crate::transact::Transaction;
use crate::txn::{Txn, TxnId};

use super::data::History;
use super::{Chain, ChainInstance, Schema, Subject, CHAIN};

/// A [`Chain`] which stores every mutation of its [`Subject`] in a series of `ChainBlock`s
#[derive(Clone)]
pub struct BlockChain {
    schema: Schema,
    subject: Subject,
    history: History,
}

impl BlockChain {
    fn new(schema: Schema, subject: Subject, history: History) -> Self {
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
        txn: &Txn,
        path: TCPathBuf,
        key: Value,
        value: State,
    ) -> TCResult<()> {
        self.history.append_put(txn, path, key, value).await
    }

    async fn hash(self, txn: Txn) -> TCResult<Output<Sha256>> {
        self.history
            .read_latest(*txn.id())
            .map_ok(|block| block.hash())
            .await
    }

    fn subject(&self) -> &Subject {
        &self.subject
    }

    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()> {
        let chain = match txn.get(source.append(CHAIN.into()), Value::None).await? {
            State::Chain(Chain::Block(chain)) => chain,
            other => {
                return Err(TCError::bad_request(
                    "blockchain expected to replicate a chain of blocks, but found",
                    other,
                ))
            }
        };

        self.history
            .replicate(txn, &self.subject, chain.history)
            .await
    }

    async fn write_ahead(&self, txn_id: &TxnId) {
        self.history.write_ahead(txn_id).await
    }
}

#[async_trait]
impl Persist<fs::Dir> for BlockChain {
    type Schema = Schema;
    type Store = fs::Dir;
    type Txn = Txn;

    fn schema(&self) -> &Schema {
        &self.schema
    }

    async fn load(txn: &Txn, schema: Schema, dir: fs::Dir) -> TCResult<Self> {
        let subject = Subject::load(txn, schema.clone(), &dir).await?;
        let history = History::load(txn, (), dir).await?;

        let write_ahead_log = history.read_log().await?;
        for (past_txn_id, mutations) in &write_ahead_log.mutations {
            super::data::replay_all(&subject, past_txn_id, mutations, txn, history.store()).await?;
        }

        Ok(BlockChain::new(schema, subject, history))
    }
}

#[async_trait]
impl Transact for BlockChain {
    async fn commit(&self, txn_id: &TxnId) {
        debug!("BlockChain::commit");
        self.subject.commit(txn_id).await;
        // make sure `self.subject` is committed before moving mutations out of the write-ahead log
        self.history.commit(txn_id).await;
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
        let schema: Schema = seq
            .next_element(())
            .await?
            .ok_or_else(|| de::Error::invalid_length(0, "a BlockChain schema"))?;

        let history: History = seq
            .next_element(self.txn.clone())
            .await?
            .ok_or_else(|| de::Error::invalid_length(1, "a BlockChain history"))?;

        let dir = self
            .txn
            .context()
            .create_dir_unique(*self.txn.id())
            .map_err(de::Error::custom)
            .await?;

        let subject = Subject::create(schema.clone(), &dir, *self.txn.id())
            .map_err(de::Error::custom)
            .await?;

        // TODO: validate that `subject` is the end-state of `history` as applied to `schema`

        Ok(BlockChain::new(schema, subject, history))
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
    type View = (Schema, super::data::HistoryView<'en>);

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        let history = self.history.into_view(txn).await?;
        Ok((self.schema, history))
    }
}
