//! A [`super::Chain`] which keeps only the data needed to recover the state of its subject in the
//! event of a transaction failure.

use std::fmt;

use async_hash::{Output, Sha256};
use async_trait::async_trait;
use destream::{de, FromStream};
use freqfs::{FileLock, FileWriteGuard};
use futures::future::TryFutureExt;
use futures::try_join;
use get_size::GetSize;
use log::{debug, trace};
use safecast::{AsType, TryCastFrom, TryCastInto};

use tc_collection::btree::Node as BTreeNode;
use tc_collection::tensor::{DenseCacheFile, Node as TensorNode};
use tc_collection::Collection;
use tc_error::*;
use tc_scalar::Scalar;
use tc_transact::fs;
use tc_transact::public::{Route, StateInstance};
use tc_transact::{AsyncHash, IntoView, RPCClient, Transact, Transaction, TxnId};
use tc_value::{Link, Value};
use tcgeneric::{label, Label};

use crate::data::StoreEntry;

use super::{null_hash, ChainBlock, ChainInstance, Recover};

const BLOCKS: Label = label(".blocks");
const COMMITTED: &str = "committed.chain_block";
const PENDING: &str = "pending.chain_block";
const STORE: Label = label(".store");

/// A [`super::Chain`] which keeps only the data needed to recover the state of its subject in the
/// event of a transaction failure.
#[derive(Clone)]
pub struct SyncChain<State: StateInstance, T> {
    committed: FileLock<State::FE>,
    pending: FileLock<State::FE>,
    store: super::data::Store<State::Txn, State::FE>,
    subject: T,
}

impl<State, T> SyncChain<State, T>
where
    State: StateInstance,
    State::FE: AsType<ChainBlock> + for<'a> fs::FileSave<'a>,
{
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

impl<State, T> SyncChain<State, T>
where
    State: StateInstance,
    State::FE: AsType<ChainBlock>,
    T: fs::Persist<State::FE, Txn = State::Txn> + fs::Restore<State::FE> + TryCastFrom<State>,
{
    pub async fn restore_from(&self, txn: &State::Txn, source: Link) -> TCResult<()> {
        debug!("restore {self:?} from {source}");

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
impl<State, T> ChainInstance<State, T> for SyncChain<State, T>
where
    State: StateInstance,
    State::FE: DenseCacheFile + AsType<ChainBlock> + AsType<BTreeNode> + AsType<TensorNode>,
    T: fs::Persist<State::FE, Txn = State::Txn> + Route<State> + fmt::Debug,
    Collection<State::Txn, State::FE>: TryCastFrom<State>,
    Scalar: TryCastFrom<State>,
{
    async fn append_delete(&self, txn_id: TxnId, key: Value) -> TCResult<()> {
        let mut block: FileWriteGuard<ChainBlock> = self.pending.write().await?;
        block.append_delete(txn_id, key);
        Ok(())
    }

    async fn append_put(&self, txn: &State::Txn, key: Value, value: State) -> TCResult<()> {
        debug!("SyncChain::append_put {} <- {:?}", key, value);

        let value = StoreEntry::try_from_state(value)?;
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
impl<State, T> AsyncHash for SyncChain<State, T>
where
    State: StateInstance,
    T: AsyncHash + Send + Sync,
{
    async fn hash(self, txn_id: TxnId) -> TCResult<Output<Sha256>> {
        self.subject.hash(txn_id).await
    }
}

#[async_trait]
impl<State, T> Transact for SyncChain<State, T>
where
    State: StateInstance,
    State::FE: AsType<ChainBlock> + for<'a> fs::FileSave<'a>,
    T: Transact + Send + Sync,
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

#[async_trait]
impl<State, T> fs::Persist<State::FE> for SyncChain<State, T>
where
    State: StateInstance,
    State::FE: AsType<ChainBlock> + for<'a> fs::FileSave<'a>,
    T: fs::Persist<State::FE, Txn = State::Txn> + Send + Sync,
{
    type Txn = State::Txn;
    type Schema = T::Schema;

    async fn create(
        txn_id: TxnId,
        schema: Self::Schema,
        store: fs::Dir<State::FE>,
    ) -> TCResult<Self> {
        debug!("SyncChain::create");

        let subject = T::create(txn_id, schema, store).await?;

        let mut dir = subject.dir().try_write_owned()?;

        let store = {
            let dir = dir.create_dir(STORE.to_string())?;
            fs::Dir::load(txn_id, dir)
                .map_ok(super::data::Store::new)
                .await?
        };

        let mut blocks_dir = dir
            .create_dir(BLOCKS.to_string())
            .and_then(|dir| dir.try_write_owned())?;

        let block = ChainBlock::with_txn(null_hash().to_vec(), txn_id);
        let size_hint = block.get_size();
        let pending = blocks_dir.create_file(PENDING.to_string(), block, size_hint)?;

        let block = ChainBlock::new(null_hash().to_vec());
        let size_hint = block.get_size();
        let committed = blocks_dir.create_file(COMMITTED.to_string(), block, size_hint)?;

        Ok(Self {
            subject,
            pending,
            committed,
            store,
        })
    }

    async fn load(
        txn_id: TxnId,
        schema: Self::Schema,
        store: fs::Dir<State::FE>,
    ) -> TCResult<Self> {
        debug!("SyncChain::load");

        let subject = T::load_or_create(txn_id, schema, store).await?;

        let mut dir = subject.dir().write_owned().await;

        let store = {
            let dir = dir.get_or_create_dir(STORE.to_string())?;
            fs::Dir::load(txn_id, dir)
                .map_ok(super::data::Store::new)
                .await?
        };

        let mut blocks_dir = dir
            .get_or_create_dir(BLOCKS.to_string())
            .and_then(|dir| dir.try_write_owned())?;

        let pending = if let Some(file) = blocks_dir.get_file(&PENDING.to_string()) {
            file.clone()
        } else {
            let block = ChainBlock::with_txn(null_hash().to_vec(), txn_id);
            let size_hint = block.get_size();
            blocks_dir.create_file(PENDING.to_string(), block, size_hint)?
        };

        let committed = if let Some(file) = blocks_dir.get_file(&COMMITTED.to_string()) {
            file.clone()
        } else {
            let block = ChainBlock::new(null_hash().to_vec());
            let size_hint = block.get_size();
            blocks_dir.create_file(COMMITTED.to_string(), block, size_hint)?
        };

        Ok(Self {
            subject,
            pending,
            committed,
            store,
        })
    }

    fn dir(&self) -> fs::Inner<State::FE> {
        self.subject.dir()
    }
}

#[async_trait]
impl<State, T> Recover<State::FE> for SyncChain<State, T>
where
    State: StateInstance + From<Collection<State::Txn, State::FE>> + From<Scalar>,
    State::FE: DenseCacheFile
        + AsType<BTreeNode>
        + AsType<TensorNode>
        + AsType<ChainBlock>
        + for<'a> fs::FileSave<'a>,
    T: Route<State> + fmt::Debug + Send + Sync,
    Collection<State::Txn, State::FE>: TryCastFrom<State>,
    Scalar: TryCastFrom<State>,
    BTreeNode: freqfs::FileLoad,
{
    type Txn = State::Txn;

    async fn recover(&self, txn: &State::Txn) -> TCResult<()> {
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

#[async_trait]
impl<State, T> fs::CopyFrom<State::FE, Self> for SyncChain<State, T>
where
    State: StateInstance,
    State::FE: AsType<ChainBlock> + for<'a> fs::FileSave<'a>,
    T: fs::Persist<State::FE, Txn = State::Txn> + Route<State> + fmt::Debug,
{
    async fn copy_from(
        _txn: &State::Txn,
        _store: fs::Dir<State::FE>,
        _instance: Self,
    ) -> TCResult<Self> {
        Err(not_implemented!("SyncChain::copy_from"))
    }
}

#[async_trait]
impl<State, T> de::FromStream for SyncChain<State, T>
where
    State: StateInstance,
    State::FE: AsType<ChainBlock>,
    T: FromStream<Context = State::Txn>,
{
    type Context = State::Txn;

    async fn from_stream<D: de::Decoder>(
        txn: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        let subject = T::from_stream(txn.clone(), decoder).await?;

        let mut context = txn.context().write().await;

        let store = {
            let dir = context
                .create_dir(STORE.to_string())
                .map_err(de::Error::custom)?;

            fs::Dir::load(*txn.id(), dir)
                .map_ok(super::data::Store::new)
                .map_err(de::Error::custom)
                .await?
        };

        let mut blocks_dir = {
            let file = context
                .create_dir(BLOCKS.to_string())
                .map_err(de::Error::custom)?;

            file.write_owned().await
        };

        let null_hash = null_hash();
        let block = ChainBlock::new(null_hash.to_vec());
        let size_hint = block.get_size();
        let committed = blocks_dir
            .create_file(COMMITTED.to_string(), block, size_hint)
            .map_err(de::Error::custom)?;

        let block = ChainBlock::with_txn(null_hash.to_vec(), *txn.id());
        let size_hint = block.get_size();
        let pending = blocks_dir
            .create_file(PENDING.to_string(), block, size_hint)
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
impl<'en, State, T> IntoView<'en, State::FE> for SyncChain<State, T>
where
    State: StateInstance,
    T: IntoView<'en, State::FE, Txn = State::Txn> + Send + Sync,
{
    type Txn = State::Txn;
    type View = T::View;

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        self.subject.into_view(txn).await
    }
}

impl<State, T> fmt::Debug for SyncChain<State, T>
where
    State: StateInstance,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SyncChain<{}>", std::any::type_name::<T>())
    }
}
