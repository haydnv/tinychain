//! A [`Chain`] which stores every mutation of its [`Subject`] in a series of `ChainBlock`s.
//!
//! Each block in the chain begins with the hash of the previous block.

use std::fmt;
use std::marker::PhantomData;

use async_trait::async_trait;
use destream::de;
use futures::future::TryFutureExt;
use futures::join;
use log::debug;
use safecast::TryCastInto;

use tc_error::*;
use tc_transact::fs::{Dir, Persist};
use tc_transact::{IntoView, Transact};
use tc_value::{Link, Value, Version as VersionNumber};
use tcgeneric::{label, Label, Map};

use crate::cluster::Replica;
use crate::collection::CollectionBase;
use crate::fs;
use crate::route::{Public, Route};
use crate::scalar::Scalar;
use crate::state::{State, ToStateAsync};
use crate::transact::Transaction;
use crate::txn::{Txn, TxnId};

use super::data::History;
use super::{ChainInstance, CHAIN};

const HISTORY: Label = label(".history");

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
    async fn append_delete(&self, txn_id: TxnId, key: Value) -> TCResult<()> {
        self.history.append_delete(txn_id, key).await
    }

    async fn append_put(&self, txn: &Txn, key: Value, value: State) -> TCResult<()> {
        self.history.append_put(txn, key, value).await
    }

    fn subject(&self) -> &T {
        &self.subject
    }
}

#[async_trait]
impl Replica for BlockChain<crate::cluster::Library> {
    async fn state(&self, txn_id: TxnId) -> TCResult<State> {
        self.subject.to_state(txn_id).await
    }

    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()> {
        let state = txn.get(source.append(CHAIN.into()), Value::None).await?;
        let library: Map<Map<Scalar>> =
            state.try_cast_into(|s| TCError::bad_request("invalid library version history", s))?;

        let latest_version = self.subject.latest(*txn.id()).await?;
        for (number, version) in library {
            let number: VersionNumber = number.as_str().parse()?;
            if let Some(latest) = latest_version {
                if number > latest {
                    self.put(txn, &[], number.into(), version.into()).await?;
                }
            } else {
                self.put(txn, &[], number.into(), version.into()).await?;
            }
        }

        Ok(())
    }
}

#[async_trait]
impl Replica for BlockChain<crate::cluster::Dir<crate::cluster::Library>> {
    async fn state(&self, txn_id: TxnId) -> TCResult<State> {
        self.subject.state(txn_id).await
    }

    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()> {
        self.subject.replicate(txn, source).await
    }
}

#[async_trait]
impl Replica for BlockChain<CollectionBase> {
    async fn state(&self, _txn_id: TxnId) -> TCResult<State> {
        Ok(State::Chain(self.clone().into()))
    }

    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()> {
        let chain = txn.get(source.append(CHAIN.into()), Value::None).await?;
        let chain: Self = chain.try_cast_into(|s| {
            TCError::bad_request(
                "blockchain expected to replicate a chain of blocks but found",
                s,
            )
        })?;

        self.history
            .replicate(txn, &self.subject, chain.history)
            .await
    }
}

#[async_trait]
impl<T> Persist<fs::Dir> for BlockChain<T>
where
    T: Route + Public + Persist<fs::Dir, Txn = Txn>,
{
    type Txn = Txn;
    type Schema = T::Schema;

    async fn create(txn: &Self::Txn, schema: Self::Schema, store: fs::Store) -> TCResult<Self> {
        let subject = T::create(txn, schema.clone(), store).await?;
        let mut dir = subject.dir().write().await;

        let history = dir
            .create_dir(HISTORY.to_string())
            .map(fs::Dir::new)
            .map_err(fs::io_err)?;

        let history = History::create(txn, (), history.into()).await?;

        Ok(BlockChain::new(subject, history))
    }

    async fn load(txn: &Txn, schema: Self::Schema, store: fs::Store) -> TCResult<Self> {
        let subject = T::load(txn, schema.clone(), store).await?;

        let mut dir = subject.dir().write().await;

        let history = dir
            .get_or_create_dir(HISTORY.to_string())
            .map(fs::Dir::new)
            .map_err(fs::io_err)?;

        let history = History::load(txn, (), history.into()).await?;

        let write_ahead_log = history.read_log().await?;
        for (past_txn_id, mutations) in &write_ahead_log.mutations {
            super::data::replay_all(&subject, past_txn_id, mutations, txn, history.store()).await?;
        }

        Ok(BlockChain::new(subject, history))
    }

    fn dir(&self) -> <fs::Dir as Dir>::Inner {
        self.subject.dir()
    }
}

#[async_trait]
impl<T: Transact + Send + Sync> Transact for BlockChain<T> {
    type Commit = T::Commit;

    async fn commit(&self, txn_id: &TxnId) -> Self::Commit {
        debug!("BlockChain::commit");

        self.history.write_ahead(txn_id).await;

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
        let subject = seq.next_element::<T>(self.txn.clone()).await?;
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

impl<T> fmt::Display for BlockChain<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "BlockChain<{}>", std::any::type_name::<T>())
    }
}
