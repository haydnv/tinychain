//! A `Chain` which stores every mutation of its subject in a series of `ChainBlock`s.
//!
//! Each block in the chain begins with the hash of the previous block.

use std::fmt;
use std::marker::PhantomData;

use async_hash::{Digest, Hash, Output, Sha256};
use async_trait::async_trait;
// use destream::de;
use futures::future::TryFutureExt;
use futures::join;
use log::debug;
use safecast::TryCastInto;

use tc_error::*;
use tc_transact::fs::{CopyFrom, Persist};
use tc_transact::{AsyncHash, IntoView, Transact, Transaction};
use tc_value::{Link, Value, Version as VersionNumber};
use tcgeneric::{label, Label, Map};

use crate::collection::CollectionBase;
use crate::fs;
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

    pub(crate) fn history(&self) -> &History {
        &self.history
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
impl<T> Persist<fs::CacheBlock> for BlockChain<T>
where
    T: Route + Persist<fs::CacheBlock, Txn = Txn> + fmt::Debug,
{
    type Txn = Txn;
    type Schema = T::Schema;

    async fn create(txn_id: TxnId, schema: Self::Schema, store: fs::Dir) -> TCResult<Self> {
        debug!("BlockChain::create");

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
        debug!("BlockChain::load");

        let subject = T::load(txn_id, schema.clone(), store).await?;

        let mut dir = subject.dir().write_owned().await;

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

// #[async_trait]
// impl<T> CopyFrom<fs::CacheBlock, BlockChain<T>> for BlockChain<T>
// where
//     T: Route + Persist<fs::CacheBlock, Txn = Txn> + fmt::Debug,
// {
//     async fn copy_from(
//         _txn: &<Self as Persist<fs::CacheBlock>>::Txn,
//         _store: fs::Dir,
//         _instance: BlockChain<T>,
//     ) -> TCResult<Self> {
//         Err(not_implemented!("BlockChain::copy_from"))
//     }
// }

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

// struct ChainVisitor<T> {
//     txn: Txn,
//     phantom: PhantomData<T>,
// }
//
// impl<T> ChainVisitor<T> {
//     fn new(txn: Txn) -> Self {
//         Self {
//             txn,
//             phantom: PhantomData,
//         }
//     }
// }

// #[async_trait]
// impl<T> de::Visitor for ChainVisitor<T>
// where
//     T: Route + de::FromStream<Context = Txn> + fmt::Debug,
// {
//     type Value = BlockChain<T>;
//
//     fn expecting() -> &'static str {
//         "a BlockChain"
//     }
//
//     async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
//         let subject = seq.next_element::<T>(self.txn.clone()).await?;
//         let subject = subject.ok_or_else(|| de::Error::invalid_length(0, "a BlockChain schema"))?;
//
//         let txn = self
//             .txn
//             .subcontext(HISTORY.into())
//             .map_err(de::Error::custom)
//             .await?;
//
//         let history = seq.next_element::<History>(txn).await?;
//         let history =
//             history.ok_or_else(|| de::Error::invalid_length(1, "a BlockChain history"))?;
//
//         let write_ahead_log = history.read_log().map_err(de::Error::custom).await?;
//         for (past_txn_id, mutations) in &write_ahead_log.mutations {
//             super::data::replay_all(&subject, past_txn_id, mutations, &self.txn, history.store())
//                 .map_err(de::Error::custom)
//                 .await?;
//         }
//
//         Ok(BlockChain::new(subject, history))
//     }
// }

// #[async_trait]
// impl<T> de::FromStream for BlockChain<T>
// where
//     T: Route + de::FromStream<Context = Txn> + fmt::Debug,
// {
//     type Context = Txn;
//
//     async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
//         decoder.decode_seq(ChainVisitor::new(txn)).await
//     }
// }

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
