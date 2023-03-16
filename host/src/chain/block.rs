//! A `Chain` which stores every mutation of its subject in a series of `ChainBlock`s.
//!
//! Each block in the chain begins with the hash of the previous block.

use std::fmt;
use std::marker::PhantomData;

use async_hash::{Digest, Hash, Output, Sha256};
use async_trait::async_trait;
use destream::de::{self, Error};
use futures::future::TryFutureExt;
use futures::join;
use log::debug;
use safecast::TryCastInto;

use tc_error::*;
use tc_transact::fs::{CopyFrom, Dir, Persist};
use tc_transact::{AsyncHash, IntoView, Transact, Transaction};
use tc_value::{Link, Value, Version as VersionNumber};
use tcgeneric::{label, Label, Map};

use crate::cluster::{Replica, REPLICAS};
use crate::collection::CollectionBase;
use crate::fs;
use crate::object::InstanceClass;
use crate::route::{Public, Route};
use crate::scalar::Scalar;
use crate::state::State;
use crate::txn::{Txn, TxnId};

use super::data::History;
use super::{ChainInstance, Recover, CHAIN};

pub(crate) const HISTORY: Label = label(".history");

/// A `Chain` which stores every mutation of its subject in a series of `ChainBlock`s
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
    T: Route + fmt::Debug,
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
impl Replica for BlockChain<crate::cluster::Class> {
    async fn state(&self, txn_id: TxnId) -> TCResult<State> {
        self.subject.to_state(txn_id).await
    }

    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()> {
        let params = Map::one(label("add"), txn.host().clone().into());

        let state = txn
            .post(source.append(REPLICAS), State::Map(params))
            .await?;

        let classes: Map<Map<InstanceClass>> =
            state.try_cast_into(|s| TCError::unexpected(s, "Class version history"))?;

        // TODO: verify equality of existing versions
        let latest_version = self.subject.latest(*txn.id()).await?;
        for (number, version) in classes {
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
impl Replica for BlockChain<crate::cluster::Library> {
    async fn state(&self, txn_id: TxnId) -> TCResult<State> {
        self.subject.to_state(txn_id).await
    }

    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()> {
        let params = Map::one(label("add"), txn.host().clone().into());

        let state = txn
            .post(source.clone().append(REPLICAS), State::Map(params))
            .await?;

        let library: Map<Map<Scalar>> =
            state.try_cast_into(|s| TCError::unexpected(s, "Library version history"))?;

        // TODO: verify equality of existing versions
        let latest_version = self.subject.latest(*txn.id()).await?;
        for (number, version) in library {
            let number: VersionNumber = number.as_str().parse()?;
            let class = InstanceClass::extend(source.clone(), version);
            if let Some(latest) = latest_version {
                if number > latest {
                    self.put(txn, &[], number.into(), class.into()).await?;
                }
            } else {
                self.put(txn, &[], number.into(), class.into()).await?;
            }
        }

        Ok(())
    }
}

#[async_trait]
impl Replica for BlockChain<crate::cluster::Service> {
    async fn state(&self, txn_id: TxnId) -> TCResult<State> {
        self.subject.state(txn_id).await
    }

    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()> {
        let params = Map::one(label("add"), txn.host().clone().into());

        let state = txn
            .post(source.clone().append(REPLICAS), State::Map(params))
            .await?;

        let library: Map<InstanceClass> =
            state.try_cast_into(|s| TCError::unexpected(s, "Service version history"))?;

        // TODO: verify equality of existing versions
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

        self.subject.replicate(txn, source).await
    }
}

#[async_trait]
impl Replica for BlockChain<CollectionBase> {
    async fn state(&self, _txn_id: TxnId) -> TCResult<State> {
        Ok(State::Chain(self.clone().into()))
    }

    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()> {
        let chain = txn.get(source.append(CHAIN), Value::default()).await?;
        let chain: Self = chain.try_cast_into(|s| {
            bad_request!(
                "blockchain expected to replicate a chain of blocks but found {:?}",
                s,
            )
        })?;

        self.history
            .replicate(txn, &self.subject, chain.history)
            .await
    }
}

#[async_trait]
impl<T> Persist<fs::CacheBlock> for BlockChain<T>
where
    T: Route + Persist<fs::CacheBlock, Txn = Txn> + fmt::Debug,
{
    type Txn = Txn;
    type Schema = T::Schema;

    async fn create(txn_id: TxnId, schema: Self::Schema, store: fs::Dir) -> TCResult<Self> {
        let subject = T::create(txn_id, schema.clone(), store).await?;
        let mut dir = subject.dir().try_write_owned()?;

        let history = {
            let dir = dir.get_or_create_dir(HISTORY.to_string())?;
            fs::Dir::load(txn_id, dir).await?
        };

        let history = History::create(txn_id, (), history.into()).await?;

        Ok(BlockChain::new(subject, history))
    }

    async fn load(txn_id: TxnId, schema: Self::Schema, store: fs::Dir) -> TCResult<Self> {
        let subject = T::load(txn_id, schema.clone(), store).await?;

        let mut dir = subject.dir().try_write_owned()?;

        let history = {
            let dir = dir.get_or_create_dir(HISTORY.to_string())?;
            fs::Dir::load(txn_id, dir).await?
        };

        let history = History::load(txn_id, (), history.into()).await?;

        Ok(BlockChain::new(subject, history))
    }

    fn dir(&self) -> tc_transact::fs::Inner<fs::CacheBlock> {
        self.subject.dir()
    }
}

#[async_trait]
impl<T: Route + fmt::Debug> Recover for BlockChain<T> {
    async fn recover(&self, txn: &Txn) -> TCResult<()> {
        let write_ahead_log = self.history.read_log().await?;

        for (past_txn_id, mutations) in &write_ahead_log.mutations {
            super::data::replay_all(
                &self.subject,
                past_txn_id,
                mutations,
                txn,
                self.history.store(),
            )
            .await?;
        }

        Ok(())
    }
}

#[async_trait]
impl<T> CopyFrom<fs::CacheBlock, BlockChain<T>> for BlockChain<T>
where
    T: Route + Persist<fs::CacheBlock, Txn = Txn> + fmt::Debug,
{
    async fn copy_from(
        _txn: &<Self as Persist<fs::CacheBlock>>::Txn,
        _store: fs::Dir,
        _instance: BlockChain<T>,
    ) -> TCResult<Self> {
        Err(not_implemented!("BlockChain::copy_from"))
    }
}

#[async_trait]
impl<T: Send + Sync> AsyncHash<fs::CacheBlock> for BlockChain<T> {
    type Txn = Txn;

    async fn hash(self, txn: &Self::Txn) -> TCResult<Output<Sha256>> {
        self.history.hash(txn).await
    }
}

#[async_trait]
impl<T: Transact + Send + Sync> Transact for BlockChain<T> {
    type Commit = T::Commit;

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        debug!("BlockChain::commit");

        self.history.write_ahead(txn_id).await;

        let guard = self.subject.commit(txn_id).await;
        // make sure `self.subject` is committed before moving mutations out of the write-ahead log
        self.history.commit(txn_id).await;
        guard
    }

    async fn rollback(&self, txn_id: &TxnId) {
        join!(self.subject.rollback(txn_id), self.history.rollback(txn_id));
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
    T: Route + de::FromStream<Context = Txn> + fmt::Debug,
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
    T: Route + de::FromStream<Context = Txn> + fmt::Debug,
{
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        decoder.decode_seq(ChainVisitor::new(txn)).await
    }
}

#[async_trait]
impl<'en, T> IntoView<'en, fs::CacheBlock> for BlockChain<T>
where
    T: IntoView<'en, fs::CacheBlock, Txn = Txn> + Send + Sync,
{
    type Txn = Txn;
    type View = (T::View, super::data::HistoryView<'en>);

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        let history = self.history.into_view(txn.clone()).await?;
        let subject = self.subject.into_view(txn).await?;
        Ok((subject, history))
    }
}

impl<T> fmt::Debug for BlockChain<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "BlockChain<{}>", std::any::type_name::<T>())
    }
}
