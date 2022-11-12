//! A [`Chain`] which stores every mutation of its [`Subject`] in a series of `ChainBlock`s.
//!
//! Each block in the chain begins with the hash of the previous block.

use std::convert::{TryFrom, TryInto};
use std::marker::PhantomData;

use async_trait::async_trait;
use destream::de;
use futures::future::TryFutureExt;
use futures::join;
use log::debug;

use tc_error::*;
use tc_transact::fs::{Dir, DirCreate, Persist};
use tc_transact::{IntoView, Transact};
use tc_value::{Link, Value};
use tcgeneric::{label, Label, TCPathBuf};

use crate::fs;
use crate::route::{Public, Route};
use crate::state::State;
use crate::transact::Transaction;
use crate::txn::{Txn, TxnId};

use super::data::History;
use super::{Chain, ChainInstance, CHAIN, SUBJECT};

const HISTORY: Label = label("history");

/// A [`Chain`] which stores every mutation of its [`Subject`] in a series of `ChainBlock`s
#[derive(Clone)]
pub struct BlockChain<T> {
    subject: T,
    history: History,
}

impl<T> BlockChain<T> {
    fn new(subject: T, history: History) -> Self {
        Self { subject, history }
    }
}

#[async_trait]
impl<T> ChainInstance<T> for BlockChain<T>
where
    T: Route + Public,
{
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

    fn subject(&self) -> &T {
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
impl<T> Persist<fs::Dir> for BlockChain<T>
where
    T: Route + Public + Persist<fs::Dir, Txn = Txn>,
    <T as Persist<fs::Dir>>::Store: TryFrom<fs::Store>,
    TCError: From<<<T as Persist<fs::Dir>>::Store as TryFrom<fs::Store>>::Error>,
{
    type Schema = T::Schema;
    type Store = fs::Dir;
    type Txn = Txn;

    async fn create(txn: &Self::Txn, schema: Self::Schema, store: Self::Store) -> TCResult<Self> {
        let mut dir = store.write(*txn.id()).await?;

        let history = dir.create_dir(HISTORY.into())?;
        let history = History::create(txn, (), history).await?;

        let store = dir
            .get_or_create_store(SUBJECT.into())
            .try_into()
            .map_err(TCError::from)?;

        let subject = T::create(txn, schema.clone(), store).await?;

        Ok(BlockChain::new(subject, history))
    }

    async fn load(txn: &Txn, schema: Self::Schema, store: Self::Store) -> TCResult<Self> {
        let mut dir = store.write(*txn.id()).await?;

        let history = dir.get_or_create_dir(HISTORY.into())?;
        let history = History::load(txn, (), history).await?;

        let store = dir
            .get_or_create_store(SUBJECT.into())
            .try_into()
            .map_err(TCError::from)?;

        let subject = T::load(txn, schema.clone(), store).await?;

        let write_ahead_log = history.read_log().await?;
        for (past_txn_id, mutations) in &write_ahead_log.mutations {
            super::data::replay_all(&subject, past_txn_id, mutations, txn, history.store()).await?;
        }

        Ok(BlockChain::new(subject, history))
    }

    async fn schema(&self, txn_id: TxnId) -> TCResult<Self::Schema> {
        self.subject.schema(txn_id).await
    }
}

#[async_trait]
impl<T> Transact for BlockChain<T>
where
    T: Transact + Send + Sync,
{
    type Commit = T::Commit;

    async fn commit(&self, txn_id: &TxnId) -> Self::Commit {
        debug!("BlockChain::commit");
        let guard = self.subject.commit(txn_id).await;
        // make sure `self.subject` is committed before moving mutations out of the write-ahead log
        self.history.commit(txn_id).await;
        guard
    }

    async fn finalize(&self, txn_id: &TxnId) {
        join!(self.subject.finalize(txn_id), self.history.finalize(txn_id));
    }
}

struct ChainVisitor<T> {
    txn: Txn,
    phantom: PhantomData<T>,
}

impl<T> ChainVisitor<T> {
    fn new(txn: Txn) -> Self {
        Self {
            txn,
            phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<T> de::Visitor for ChainVisitor<T>
where
    T: Route + Public + de::FromStream<Context = Txn>,
{
    type Value = BlockChain<T>;

    fn expecting() -> &'static str {
        "a BlockChain"
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let txn = self
            .txn
            .subcontext(SUBJECT.into())
            .map_err(de::Error::custom)
            .await?;

        let subject = seq.next_element::<T>(txn).await?;
        let subject = subject.ok_or_else(|| de::Error::invalid_length(0, "a BlockChain schema"))?;

        let txn = self
            .txn
            .subcontext(HISTORY.into())
            .map_err(de::Error::custom)
            .await?;

        let history = seq.next_element::<History>(txn).await?;
        let history =
            history.ok_or_else(|| de::Error::invalid_length(1, "a BlockChain history"))?;

        let write_ahead_log = history.read_log().map_err(de::Error::custom).await?;
        for (past_txn_id, mutations) in &write_ahead_log.mutations {
            super::data::replay_all(&subject, past_txn_id, mutations, &self.txn, history.store())
                .map_err(de::Error::custom)
                .await?;
        }

        Ok(BlockChain::new(subject, history))
    }
}

#[async_trait]
impl<T> de::FromStream for BlockChain<T>
where
    T: Route + Public + de::FromStream<Context = Txn>,
{
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        decoder.decode_seq(ChainVisitor::new(txn)).await
    }
}

#[async_trait]
impl<'en, T> IntoView<'en, fs::Dir> for BlockChain<T>
where
    T: IntoView<'en, fs::Dir, Txn = Txn> + Send + Sync,
{
    type Txn = Txn;
    type View = (T::View, super::data::HistoryView<'en>);

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        let history = self.history.into_view(txn.clone()).await?;
        let subject = self.subject.into_view(txn).await?;
        Ok((subject, history))
    }
}
