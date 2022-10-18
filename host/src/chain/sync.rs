//! A [`super::Chain`] which keeps only the data needed to recover the state of its subject in the
//! event of a transaction failure.

use std::convert::{TryFrom, TryInto};
use std::fmt;

use async_trait::async_trait;
use destream::{de, FromStream};
use freqfs::{FileLock, FileWriteGuard};
use futures::future::TryFutureExt;
use futures::try_join;
use log::debug;
use safecast::{TryCastFrom, TryCastInto};

use tc_error::*;
use tc_transact::fs::{Dir, DirCreate, DirCreateFile, Persist, Restore};
use tc_transact::{IntoView, Transact, Transaction, TxnId};
use tc_value::{Link, Value};
use tcgeneric::{label, Id, Label, TCPathBuf};

use crate::fs;
use crate::route::{Public, Route};
use crate::state::State;
use crate::txn::Txn;

use super::{null_hash, ChainBlock, ChainInstance};

const BLOCKS: Label = label("blocks");
const COMMITTED: &str = "committed.chain_block";
const PENDING: &str = "pending.chain_block";
const STORE: Label = label("store");

/// A [`super::Chain`] which keeps only the data needed to recover the state of its subject in the
/// event of a transaction failure.
#[derive(Clone)]
pub struct SyncChain<T> {
    subject: T,
    pending: FileLock<fs::CacheBlock>,
    committed: FileLock<fs::CacheBlock>,
    store: super::data::Store,
}

#[async_trait]
impl<T> ChainInstance<T> for SyncChain<T>
where
    T: Persist<fs::Dir, Txn = Txn>
        + Restore<fs::Dir>
        + TryCastFrom<State>
        + Route
        + Public
        + fmt::Display,
{
    async fn append_delete(&self, txn_id: TxnId, path: TCPathBuf, key: Value) -> TCResult<()> {
        let mut block: FileWriteGuard<_, ChainBlock> =
            self.pending.write().map_err(fs::io_err).await?;

        block.append_delete(txn_id, path, key);
        Ok(())
    }

    async fn append_put(
        &self,
        txn: &Txn,
        path: TCPathBuf,
        key: Value,
        value: State,
    ) -> TCResult<()> {
        debug!("SyncChain::append_put {}: {} <- {}", path, key, value);

        let value = self.store.save_state(txn, value).await?;

        debug!("locking pending transaction log block...");
        let mut block: FileWriteGuard<_, ChainBlock> =
            self.pending.write().map_err(fs::io_err).await?;

        block.append_put(*txn.id(), path, key, value);

        debug!("locked pending transaction log block");
        Ok(())
    }

    fn subject(&self) -> &T {
        &self.subject
    }

    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()> {
        let backup = txn.get(source, Value::None).await?;
        let backup = backup.try_cast_into(|backup| {
            TCError::unsupported(format!(
                "{} is not a valid backup of {}",
                backup, &self.subject
            ))
        })?;
        self.subject.restore(&backup, *txn.id()).await?;

        let (mut pending, mut committed): (
            FileWriteGuard<_, ChainBlock>,
            FileWriteGuard<_, ChainBlock>,
        ) = try_join!(
            self.pending.write().map_err(fs::io_err),
            self.committed.write().map_err(fs::io_err)
        )?;

        *pending = ChainBlock::with_txn(null_hash().to_vec(), *txn.id());
        *committed = ChainBlock::new(null_hash().to_vec());

        Ok(())
    }

    async fn write_ahead(&self, txn_id: &TxnId) {
        self.store.commit(txn_id).await;

        {
            let (mut pending, mut committed): (
                FileWriteGuard<_, ChainBlock>,
                FileWriteGuard<_, ChainBlock>,
            ) = try_join!(
                self.pending.write().map_err(fs::io_err),
                self.committed.write().map_err(fs::io_err)
            )
            .expect("SyncChain blocks");

            if let Some(mutations) = pending.mutations.remove(txn_id) {
                committed.mutations.insert(*txn_id, mutations);
            }
        }

        try_join!(self.pending.sync(false), self.committed.sync(false))
            .expect("sync SyncChain blocks");
    }
}

#[async_trait]
impl<T> Transact for SyncChain<T>
where
    T: Transact + Send + Sync,
{
    type Commit = T::Commit;

    async fn commit(&self, txn_id: &TxnId) -> Self::Commit {
        debug!("SyncChain::commit");

        let guard = self.subject.commit(txn_id).await;

        // assume the mutations for the transaction have already been moved and sync'd
        // from `self.pending` to `self.committed` by calling the `write_ahead` method
        {
            let mut committed: FileWriteGuard<_, ChainBlock> =
                self.committed.write().await.expect("committed");

            committed.mutations.remove(txn_id);
        }

        self.committed.sync(false).await.expect("sync commit block");

        guard
    }

    async fn finalize(&self, txn_id: &TxnId) {
        {
            let mut pending: FileWriteGuard<_, ChainBlock> =
                self.pending.write().await.expect("pending");

            pending.mutations.remove(txn_id);
        }

        self.pending.sync(false).await.expect("sync pending block");
        self.subject.finalize(txn_id).await
    }
}

#[async_trait]
impl<T> Persist<fs::Dir> for SyncChain<T>
where
    T: Persist<fs::Dir, Txn = Txn> + Send + Sync,
    <T as Persist<fs::Dir>>::Store: TryFrom<fs::Store>,
    TCError: From<<<T as Persist<fs::Dir>>::Store as TryFrom<fs::Store>>::Error>,
{
    type Schema = T::Schema;
    type Store = fs::Dir;
    type Txn = Txn;

    async fn create(txn: &Self::Txn, schema: Self::Schema, dir: Self::Store) -> TCResult<Self> {
        debug!("SyncChain::create");

        let mut dir = dir.write(*txn.id()).await?;

        let store = dir
            .get_or_create_dir(STORE.into())
            .map(super::data::Store::new)?;

        let mut blocks_dir = {
            let file: fs::File<Id, ChainBlock> = dir.get_or_create_file(BLOCKS.into())?;
            file.into_inner().write().await
        };

        let block = ChainBlock::with_txn(null_hash().to_vec(), *txn.id());
        let pending = blocks_dir
            .create_file(PENDING.to_string(), block, Some(0))
            .map_err(fs::io_err)?;

        let block = ChainBlock::new(null_hash().to_vec());
        let committed = blocks_dir
            .create_file(COMMITTED.to_string(), block, Some(0))
            .map_err(fs::io_err)?;

        let subject_store = dir
            .get_or_create_store(super::SUBJECT.into())
            .try_into()
            .map_err(TCError::from)?;

        let subject = T::create(txn, schema, subject_store).await?;

        Ok(Self {
            subject,
            pending,
            committed,
            store,
        })
    }

    async fn load(txn: &Txn, schema: Self::Schema, dir: fs::Dir) -> TCResult<Self> {
        debug!("SyncChain::load");

        let mut dir = dir.write(*txn.id()).await?;

        let store = dir
            .get_or_create_dir(STORE.into())
            .map(super::data::Store::new)?;

        let mut blocks_dir = {
            let file: fs::File<Id, ChainBlock> = dir.get_or_create_file(BLOCKS.into())?;
            file.into_inner().write().await
        };

        let pending = if let Some(file) = blocks_dir.get_file(&PENDING.to_string()) {
            file
        } else {
            let block = ChainBlock::with_txn(null_hash().to_vec(), *txn.id());
            blocks_dir
                .create_file(PENDING.to_string(), block, Some(0))
                .map_err(fs::io_err)?
        };

        let committed = if let Some(file) = blocks_dir.get_file(&COMMITTED.to_string()) {
            file
        } else {
            let block = ChainBlock::new(null_hash().to_vec());
            blocks_dir
                .create_file(COMMITTED.to_string(), block, Some(0))
                .map_err(fs::io_err)?
        };

        let subject_store = dir
            .get_or_create_store(super::SUBJECT.into())
            .try_into()
            .map_err(TCError::from)?;

        let subject = T::load_or_create(txn, schema, subject_store).await?;

        Ok(Self {
            subject,
            pending,
            committed,
            store,
        })
    }
}

#[async_trait]
impl<T> de::FromStream for SyncChain<T>
where
    T: FromStream<Context = Txn>,
{
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        let subject = txn
            .subcontext(super::SUBJECT.into())
            .map_err(de::Error::custom)
            .and_then(|txn| T::from_stream(txn, decoder))
            .await?;

        let mut dir = txn
            .context()
            .write(*txn.id())
            .map_err(de::Error::custom)
            .await?;

        let store = dir
            .create_dir(STORE.into())
            .map(super::data::Store::new)
            .map_err(de::Error::custom)?;

        let mut blocks_dir = {
            let file: fs::File<Id, ChainBlock> =
                dir.create_file(BLOCKS.into()).map_err(de::Error::custom)?;

            file.into_inner().write().await
        };

        let null_hash = null_hash();
        let committed = blocks_dir
            .create_file(
                COMMITTED.to_string(),
                ChainBlock::new(null_hash.to_vec()),
                Some(null_hash.len()),
            )
            .map_err(de::Error::custom)?;

        let pending = blocks_dir
            .create_file(
                PENDING.to_string(),
                ChainBlock::with_txn(null_hash.to_vec(), *txn.id()),
                Some(null_hash.len()),
            )
            .map_err(de::Error::custom)?;

        Ok(Self {
            subject,
            pending,
            committed,
            store,
        })
    }
}

#[async_trait]
impl<'en, T> IntoView<'en, fs::Dir> for SyncChain<T>
where
    T: IntoView<'en, fs::Dir, Txn = Txn> + Send + Sync,
    Self: Send + Sync,
{
    type Txn = Txn;
    type View = T::View;

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        self.subject.into_view(txn).await
    }
}
