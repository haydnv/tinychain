//! A [`super::Chain`] which keeps only the data needed to recover the state of its subject in the
//! event of a transaction failure.

use std::fmt;

use async_trait::async_trait;
use destream::{de, FromStream};
use freqfs::{FileLock, FileWriteGuard};
use futures::future::TryFutureExt;
use futures::try_join;
use log::{debug, trace};
use safecast::{TryCastFrom, TryCastInto};
use sha2::digest::Output;
use sha2::Sha256;

use tc_error::*;
use tc_transact::fs::{CopyFrom, Dir, DirCreate, DirCreateFile, File, Persist, Restore};
use tc_transact::{AsyncHash, IntoView, Transact, Transaction, TxnId};
use tc_value::{Link, Value};
use tcgeneric::{label, Id, Label};

use crate::cluster::Replica;
use crate::fs;
use crate::fs::CacheBlock;
use crate::route::Route;
use crate::state::State;
use crate::txn::Txn;

use super::{null_hash, ChainBlock, ChainInstance, Recover};

const BLOCKS: Label = label(".blocks");
const COMMITTED: &str = "committed.chain_block";
const PENDING: &str = "pending.chain_block";
const STORE: Label = label(".store");

/// A [`super::Chain`] which keeps only the data needed to recover the state of its subject in the
/// event of a transaction failure.
#[derive(Clone)]
pub struct SyncChain<T> {
    subject: T,
    pending: FileLock<fs::CacheBlock>,
    committed: FileLock<fs::CacheBlock>,
    store: super::data::Store,
}

impl<T> SyncChain<T> {
    async fn write_ahead(&self, txn_id: TxnId) {
        trace!("SyncChain::write_ahead {}", txn_id);

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

            if let Some(mutations) = pending.mutations.remove(&txn_id) {
                committed.mutations.insert(txn_id, mutations);
            }
        }

        try_join!(self.pending.sync(false), self.committed.sync(false))
            .expect("sync SyncChain blocks");
    }
}

#[async_trait]
impl<T> ChainInstance<T> for SyncChain<T>
where
    T: Persist<fs::Dir, Txn = Txn> + Route + fmt::Display,
{
    async fn append_delete(&self, txn_id: TxnId, key: Value) -> TCResult<()> {
        let mut block: FileWriteGuard<_, ChainBlock> =
            self.pending.write().map_err(fs::io_err).await?;

        block.append_delete(txn_id, key);
        Ok(())
    }

    async fn append_put(&self, txn: &Txn, key: Value, value: State) -> TCResult<()> {
        debug!("SyncChain::append_put {} <- {}", key, value);

        let value = self.store.save_state(txn, value).await?;

        debug!("locking pending transaction log block...");
        let mut block: FileWriteGuard<_, ChainBlock> =
            self.pending.write().map_err(fs::io_err).await?;

        block.append_put(*txn.id(), key, value);

        debug!("locked pending transaction log block");
        Ok(())
    }

    fn subject(&self) -> &T {
        &self.subject
    }
}

#[async_trait]
impl<T> Replica for SyncChain<T>
where
    T: Restore<fs::Dir> + Transact + Clone + Send + Sync,
    T: TryCastFrom<State>,
    State: From<T>,
{
    async fn state(&self, _txn_id: TxnId) -> TCResult<State> {
        Ok(self.subject.clone().into())
    }

    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()> {
        let backup = txn.get(source, Value::default()).await?;
        let backup =
            backup.try_cast_into(|backup| TCError::bad_request("not a valid backup", backup))?;

        self.subject.restore(*txn.id(), &backup).await?;

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
}

#[async_trait]
impl<T: AsyncHash<fs::Dir, Txn = Txn> + Send + Sync> AsyncHash<fs::Dir> for SyncChain<T> {
    type Txn = Txn;

    async fn hash(self, txn: &Self::Txn) -> TCResult<Output<Sha256>> {
        self.subject.hash(txn).await
    }
}

#[async_trait]
impl<T: Transact + Send + Sync> Transact for SyncChain<T>
where
    Self: ChainInstance<T>,
{
    type Commit = T::Commit;

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        debug!("SyncChain::commit");

        self.write_ahead(txn_id).await;
        trace!("SyncChain::commit logged the mutations to be applied");

        let guard = self.subject.commit(txn_id).await;
        trace!("SyncChain committed subject, moving its mutations out of the write-head log...");

        // assume the mutations for the transaction have already been moved and sync'd
        // from `self.pending` to `self.committed` by calling the `write_ahead` method
        {
            let mut committed: FileWriteGuard<_, ChainBlock> =
                self.committed.write().await.expect("committed");

            committed.mutations.remove(&txn_id);
            trace!("mutations are out of the write-ahead log");
        }

        self.committed.sync(false).await.expect("sync commit block");

        guard
    }

    async fn rollback(&self, txn_id: &TxnId) {
        debug!("SyncChain::rollback");

        self.subject.rollback(txn_id).await;

        {
            let mut pending: FileWriteGuard<_, ChainBlock> =
                self.pending.write().await.expect("pending");

            pending.mutations.remove(txn_id);
        }

        self.pending.sync(false).await.expect("sync pending block");
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

impl<T> Persist<fs::Dir> for SyncChain<T>
where
    T: Persist<fs::Dir, Txn = Txn> + Send + Sync,
{
    type Txn = Txn;
    type Schema = T::Schema;

    fn create(txn_id: TxnId, schema: Self::Schema, store: fs::Store) -> TCResult<Self> {
        debug!("SyncChain::create");

        let subject = T::create(txn_id, schema, store)?;

        let mut dir = subject.dir().try_write().map_err(fs::io_err)?;

        let store = dir
            .create_dir(STORE.to_string())
            .map_err(fs::io_err)
            .map(|dir| fs::Dir::new(dir, txn_id))
            .map(super::data::Store::new)?;

        let mut blocks_dir = dir
            .create_dir(BLOCKS.to_string())
            .and_then(|dir| dir.try_write())
            .map_err(fs::io_err)?;

        let block = ChainBlock::with_txn(null_hash().to_vec(), txn_id);
        let pending = blocks_dir
            .create_file(PENDING.to_string(), block, Some(0))
            .map_err(fs::io_err)?;

        let block = ChainBlock::new(null_hash().to_vec());
        let committed = blocks_dir
            .create_file(COMMITTED.to_string(), block, Some(0))
            .map_err(fs::io_err)?;

        Ok(Self {
            subject,
            pending,
            committed,
            store,
        })
    }

    fn load(txn_id: TxnId, schema: Self::Schema, store: fs::Store) -> TCResult<Self> {
        debug!("SyncChain::load");

        let subject = T::load_or_create(txn_id, schema, store)?;

        let mut dir = subject.dir().try_write().map_err(fs::io_err)?;

        let store = dir
            .get_or_create_dir(STORE.to_string())
            .map_err(fs::io_err)
            .map(|dir| fs::Dir::new(dir, txn_id))
            .map(super::data::Store::new)?;

        let mut blocks_dir = dir
            .get_or_create_dir(BLOCKS.to_string())
            .and_then(|dir| dir.try_write())
            .map_err(fs::io_err)?;

        let pending = if let Some(file) = blocks_dir.get_file(&PENDING.to_string()) {
            file
        } else {
            let block = ChainBlock::with_txn(null_hash().to_vec(), txn_id);

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

        Ok(Self {
            subject,
            pending,
            committed,
            store,
        })
    }

    fn dir(&self) -> <fs::Dir as Dir>::Inner {
        self.subject.dir()
    }
}

#[async_trait]
impl<T: Route + fmt::Display + Send + Sync> Recover for SyncChain<T> {
    async fn recover(&self, txn: &Txn) -> TCResult<()> {
        {
            let mut committed: freqfs::FileWriteGuard<CacheBlock, ChainBlock> =
                self.committed.write().map_err(fs::io_err).await?;

            for (txn_id, mutations) in &committed.mutations {
                super::data::replay_all(&self.subject, txn_id, mutations, txn, &self.store).await?;
            }

            committed.mutations.clear()
        }

        self.committed.sync(false).map_err(fs::io_err).await
    }
}

#[async_trait]
impl<T> CopyFrom<fs::Dir, SyncChain<T>> for SyncChain<T>
where
    T: Persist<fs::Dir, Txn = Txn> + Route,
{
    async fn copy_from(
        _txn: &<Self as Persist<fs::Dir>>::Txn,
        _store: fs::Store,
        _instance: SyncChain<T>,
    ) -> TCResult<Self> {
        Err(TCError::not_implemented("SyncChain::copy_from"))
    }
}

#[async_trait]
impl<T> de::FromStream for SyncChain<T>
where
    T: FromStream<Context = Txn>,
{
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        let subject = T::from_stream(txn.clone(), decoder).await?;

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

impl<T> fmt::Display for SyncChain<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SyncChain<{}>", std::any::type_name::<T>())
    }
}
