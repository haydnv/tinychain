//! A [`super::Chain`] which keeps only the data needed to recover the state of its subject in the
//! event of a transaction failure.

use std::fmt;

use async_hash::{Digest, Hash, Output, Sha256};
use async_trait::async_trait;
use destream::{de, FromStream};
use freqfs::{FileLock, FileWriteGuard};
use futures::future::TryFutureExt;
use futures::try_join;
use get_size::GetSize;
use log::{debug, trace};
use safecast::{TryCastFrom, TryCastInto};

use tc_error::*;
use tc_transact::fs::{CopyFrom, Persist, Restore};
use tc_transact::{AsyncHash, IntoView, Transact, Transaction, TxnId};
use tc_value::{Link, Value};
use tcgeneric::{label, Label};

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
                FileWriteGuard<ChainBlock>,
                FileWriteGuard<ChainBlock>,
            ) = try_join!(self.pending.write(), self.committed.write()).expect("SyncChain blocks");

            if let Some(mutations) = pending.mutations.remove(&txn_id) {
                committed.mutations.insert(txn_id, mutations);
            }
        }

        try_join!(self.pending.sync(), self.committed.sync()).expect("sync SyncChain blocks");
    }
}

#[async_trait]
impl<T> ChainInstance<T> for SyncChain<T>
where
    T: Persist<CacheBlock, Txn = Txn> + Route + fmt::Debug,
{
    async fn append_delete(&self, txn_id: TxnId, key: Value) -> TCResult<()> {
        let mut block: FileWriteGuard<ChainBlock> = self.pending.write().await?;
        block.append_delete(txn_id, key);
        Ok(())
    }

    async fn append_put(&self, txn: &Txn, key: Value, value: State) -> TCResult<()> {
        debug!("SyncChain::append_put {} <- {:?}", key, value);

        let value = self.store.save_state(txn, value).await?;

        debug!("locking pending transaction log block...");
        let mut block: FileWriteGuard<ChainBlock> = self.pending.write().await?;

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
    T: Restore<CacheBlock> + Transact + Clone + Send + Sync,
    T: TryCastFrom<State>,
    State: From<T>,
{
    async fn state(&self, _txn_id: TxnId) -> TCResult<State> {
        Ok(self.subject.clone().into())
    }

    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()> {
        let backup = txn.get(source, Value::default()).await?;
        let backup =
            backup.try_cast_into(|backup| bad_request!("{:?} is not a valid backup", backup))?;

        self.subject.restore(*txn.id(), &backup).await?;

        let (mut pending, mut committed) = try_join!(self.pending.write(), self.committed.write())?;

        *pending = ChainBlock::with_txn(null_hash().to_vec(), *txn.id());
        *committed = ChainBlock::new(null_hash().to_vec());

        Ok(())
    }
}

#[async_trait]
impl<T: AsyncHash<CacheBlock, Txn = Txn> + Send + Sync> AsyncHash<CacheBlock> for SyncChain<T> {
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
            let mut committed: FileWriteGuard<ChainBlock> =
                self.committed.write().await.expect("committed");

            committed.mutations.remove(&txn_id);
            trace!("mutations are out of the write-ahead log");
        }

        self.committed.sync().await.expect("sync commit block");

        guard
    }

    async fn rollback(&self, txn_id: &TxnId) {
        debug!("SyncChain::rollback");

        self.subject.rollback(txn_id).await;

        {
            let mut pending: FileWriteGuard<ChainBlock> =
                self.pending.write().await.expect("pending");

            pending.mutations.remove(txn_id);
        }

        self.pending.sync().await.expect("sync pending block");
    }

    async fn finalize(&self, txn_id: &TxnId) {
        {
            let mut pending: FileWriteGuard<ChainBlock> =
                self.pending.write().await.expect("pending");

            pending.mutations.remove(txn_id);
        }

        self.pending.sync().await.expect("sync pending block");
        self.subject.finalize(txn_id).await
    }
}

// #[async_trait]
// impl<T> Persist<CacheBlock> for SyncChain<T>
// where
//     T: Persist<CacheBlock, Txn = Txn> + Send + Sync,
// {
//     type Txn = Txn;
//     type Schema = T::Schema;
//
//     async fn create(txn_id: TxnId, schema: Self::Schema, store: fs::Dir) -> TCResult<Self> {
//         debug!("SyncChain::create");
//
//         let subject = T::create(txn_id, schema, store).await?;
//
//         let mut dir = subject.dir().try_write_owned()?;
//
//         let store = {
//             let dir = dir.create_dir(STORE.to_string())?;
//             fs::Dir::load(txn_id, dir)
//                 .map_ok(super::data::Store::new)
//                 .await?
//         };
//
//         let mut blocks_dir = dir
//             .create_dir(BLOCKS.to_string())
//             .and_then(|dir| dir.try_write_owned())?;
//
//         let block = ChainBlock::with_txn(null_hash().to_vec(), txn_id);
//         let size_hint = block.get_size();
//         let pending = blocks_dir.create_file(PENDING.to_string(), block, size_hint)?;
//
//         let block = ChainBlock::new(null_hash().to_vec());
//         let size_hint = block.get_size();
//         let committed = blocks_dir.create_file(COMMITTED.to_string(), block, size_hint)?;
//
//         Ok(Self {
//             subject,
//             pending,
//             committed,
//             store,
//         })
//     }
//
//     async fn load(txn_id: TxnId, schema: Self::Schema, store: fs::Dir) -> TCResult<Self> {
//         debug!("SyncChain::load");
//
//         let subject = T::load_or_create(txn_id, schema, store).await?;
//
//         let mut dir = subject.dir().write_owned().await;
//
//         let store = {
//             let dir = dir.get_or_create_dir(STORE.to_string())?;
//             fs::Dir::load(txn_id, dir)
//                 .map_ok(super::data::Store::new)
//                 .await?
//         };
//
//         let mut blocks_dir = dir
//             .get_or_create_dir(BLOCKS.to_string())
//             .and_then(|dir| dir.try_write_owned())?;
//
//         let pending = if let Some(file) = blocks_dir.get_file(&PENDING.to_string()) {
//             file.clone()
//         } else {
//             let block = ChainBlock::with_txn(null_hash().to_vec(), txn_id);
//             let size_hint = block.get_size();
//             blocks_dir.create_file(PENDING.to_string(), block, size_hint)?
//         };
//
//         let committed = if let Some(file) = blocks_dir.get_file(&COMMITTED.to_string()) {
//             file.clone()
//         } else {
//             let block = ChainBlock::new(null_hash().to_vec());
//             let size_hint = block.get_size();
//             blocks_dir.create_file(COMMITTED.to_string(), block, size_hint)?
//         };
//
//         Ok(Self {
//             subject,
//             pending,
//             committed,
//             store,
//         })
//     }
//
//     fn dir(&self) -> tc_transact::fs::Inner<CacheBlock> {
//         self.subject.dir()
//     }
// }

#[async_trait]
impl<T: Route + fmt::Debug + Send + Sync> Recover for SyncChain<T> {
    async fn recover(&self, txn: &Txn) -> TCResult<()> {
        {
            let mut committed: FileWriteGuard<ChainBlock> = self.committed.write().await?;

            for (txn_id, mutations) in &committed.mutations {
                super::data::replay_all(&self.subject, txn_id, mutations, txn, &self.store).await?;
            }

            committed.mutations.clear()
        }

        self.committed.sync().map_err(TCError::from).await
    }
}

// #[async_trait]
// impl<T> CopyFrom<CacheBlock, SyncChain<T>> for SyncChain<T>
// where
//     T: Persist<CacheBlock, Txn = Txn> + Route,
// {
//     async fn copy_from(
//         _txn: &<Self as Persist<CacheBlock>>::Txn,
//         _store: fs::Dir,
//         _instance: SyncChain<T>,
//     ) -> TCResult<Self> {
//         Err(not_implemented!("SyncChain::copy_from"))
//     }
// }

// #[async_trait]
// impl<T> de::FromStream for SyncChain<T>
// where
//     T: FromStream<Context = Txn>,
// {
//     type Context = Txn;
//
//     async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
//         let subject = T::from_stream(txn.clone(), decoder).await?;
//
//         let mut context = txn.context().write().await;
//
//         let store = {
//             let dir = context
//                 .create_dir(STORE.to_string())
//                 .map_err(de::Error::custom)?;
//
//             fs::Dir::load(*txn.id(), dir)
//                 .map_ok(super::data::Store::new)
//                 .map_err(de::Error::custom)
//                 .await?
//         };
//
//         let mut blocks_dir = {
//             let file = context
//                 .create_dir(BLOCKS.to_string())
//                 .map_err(de::Error::custom)?;
//
//             file.write_owned().await
//         };
//
//         let null_hash = null_hash();
//         let block = ChainBlock::new(null_hash.to_vec());
//         let size_hint = block.get_size();
//         let committed = blocks_dir
//             .create_file(COMMITTED.to_string(), block, size_hint)
//             .map_err(de::Error::custom)?;
//
//         let block = ChainBlock::with_txn(null_hash.to_vec(), *txn.id());
//         let size_hint = block.get_size();
//         let pending = blocks_dir
//             .create_file(PENDING.to_string(), block, size_hint)
//             .map_err(de::Error::custom)?;
//
//         Ok(Self {
//             subject,
//             pending,
//             committed,
//             store,
//         })
//     }
// }

#[async_trait]
impl<'en, T> IntoView<'en, CacheBlock> for SyncChain<T>
where
    T: IntoView<'en, CacheBlock, Txn = Txn> + Send + Sync,
    Self: Send + Sync,
{
    type Txn = Txn;
    type View = T::View;

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        self.subject.into_view(txn).await
    }
}

impl<T> fmt::Debug for SyncChain<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SyncChain<{}>", std::any::type_name::<T>())
    }
}
