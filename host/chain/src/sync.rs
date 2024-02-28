//! A [`super::Chain`] which keeps only the data needed to recover the state of its subject in the
//! event of a transaction failure.

use std::fmt;
use std::marker::PhantomData;

use async_trait::async_trait;
use destream::{de, FromStream};
use freqfs::{FileLock, FileWriteGuard};
use futures::TryFutureExt;
use get_size::GetSize;
use log::{debug, trace};
use safecast::{AsType, TryCastFrom, TryCastInto};

use tc_collection::btree::Node as BTreeNode;
use tc_collection::tensor::{DenseCacheFile, Node as TensorNode};
use tc_collection::Collection;
use tc_error::*;
use tc_scalar::Scalar;
use tc_transact::fs;
use tc_transact::hash::{AsyncHash, Output, Sha256};
use tc_transact::lock::TxnTaskQueue;
use tc_transact::public::{Route, StateInstance};
use tc_transact::{Gateway, IntoView, Transact, Transaction, TxnId};
use tc_value::{Link, Value};
use tcgeneric::{label, Label};

use crate::data::{MutationPending, MutationRecord, StoreEntry};

use super::{new_queue, null_hash, ChainBlock, ChainInstance, Recover};

const BLOCKS: Label = label(".blocks");
const COMMITTED: &str = "committed.chain_block";
const STORE: Label = label(".store");

/// A [`super::Chain`] which keeps only the data needed to recover the state of its subject in the
/// event of a transaction failure.
pub struct SyncChain<State, Txn, FE, T> {
    committed: FileLock<FE>,
    queue: TxnTaskQueue<MutationPending<Txn, FE>, TCResult<MutationRecord>>,
    store: super::data::Store<Txn, FE>,
    subject: T,
    state: PhantomData<State>,
}

impl<State, Txn, FE, T> Clone for SyncChain<State, Txn, FE, T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self {
            committed: self.committed.clone(),
            queue: self.queue.clone(),
            store: self.store.clone(),
            subject: self.subject.clone(),
            state: self.state,
        }
    }
}

impl<State, T> SyncChain<State, State::Txn, State::FE, T>
where
    State: StateInstance,
    State::FE: AsType<ChainBlock> + for<'a> fs::FileSave<'a>,
{
    async fn write_ahead(&self, txn_id: TxnId) {
        trace!("SyncChain::write_ahead {}", txn_id);

        let handles = self.queue.commit(txn_id).await;

        let mutations = handles
            .into_iter()
            .collect::<TCResult<Vec<_>>>()
            .expect("mutations");

        if mutations.is_empty() {
            return;
        }

        self.store.commit(txn_id).await;

        {
            let mut committed: FileWriteGuard<ChainBlock> =
                self.committed.write().await.expect("SyncChain block");

            committed.mutations.insert(txn_id, mutations);
        }

        self.committed.sync().await.expect("sync SyncChain block")
    }
}

impl<State, T> SyncChain<State, State::Txn, State::FE, T>
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

        let mut committed = self.committed.write().await?;

        *committed = ChainBlock::new(null_hash().to_vec());

        Ok(())
    }
}

impl<State, T> ChainInstance<State, T> for SyncChain<State, State::Txn, State::FE, T>
where
    State: StateInstance,
    State::FE: DenseCacheFile + AsType<ChainBlock> + AsType<BTreeNode> + AsType<TensorNode>,
    T: fs::Persist<State::FE, Txn = State::Txn> + Route<State> + fmt::Debug,
    Collection<State::Txn, State::FE>: TryCastFrom<State>,
    Scalar: TryCastFrom<State>,
{
    fn append_delete(&self, txn_id: TxnId, key: Value) -> TCResult<()> {
        self.queue
            .push(txn_id, MutationPending::Delete(key))
            .map_err(TCError::from)
    }

    fn append_put(&self, txn: State::Txn, key: Value, value: State) -> TCResult<()> {
        let txn_id = *txn.id();
        let value = StoreEntry::try_from_state(value)?;
        let mutation = MutationPending::Put(txn, key, value);
        self.queue.push(txn_id, mutation).map_err(TCError::from)
    }

    fn subject(&self) -> &T {
        &self.subject
    }
}

#[async_trait]
impl<State, T> AsyncHash for SyncChain<State, State::Txn, State::FE, T>
where
    State: StateInstance,
    T: AsyncHash + Send + Sync,
{
    async fn hash(self, txn_id: TxnId) -> TCResult<Output<Sha256>> {
        self.subject.hash(txn_id).await
    }
}

#[async_trait]
impl<State, T> Transact for SyncChain<State, State::Txn, State::FE, T>
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

        self.queue.rollback(txn_id);
        self.subject.rollback(txn_id).await;
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.queue.finalize(*txn_id);
        self.subject.finalize(txn_id).await
    }
}

#[async_trait]
impl<State, T> fs::Persist<State::FE> for SyncChain<State, State::Txn, State::FE, T>
where
    State: StateInstance,
    State::FE: DenseCacheFile
        + AsType<BTreeNode>
        + AsType<ChainBlock>
        + AsType<TensorNode>
        + for<'a> fs::FileSave<'a>,
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

        let queue = new_queue::<State>(store.clone());

        // TODO: is this necessary?
        let mut blocks_dir = dir
            .create_dir(BLOCKS.to_string())
            .and_then(|dir| dir.try_write_owned())?;

        let block = ChainBlock::new(null_hash().to_vec());
        let size_hint = block.get_size();
        let committed = blocks_dir.create_file(COMMITTED.to_string(), block, size_hint)?;

        Ok(Self {
            subject,
            queue,
            committed,
            store,
            state: PhantomData,
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

        let queue = new_queue::<State>(store.clone());

        let mut blocks_dir = dir
            .get_or_create_dir(BLOCKS.to_string())
            .and_then(|dir| dir.try_write_owned())?;

        let committed = if let Some(file) = blocks_dir.get_file(&*COMMITTED) {
            file.clone()
        } else {
            let block = ChainBlock::new(null_hash().to_vec());
            let size_hint = block.get_size();
            blocks_dir.create_file(COMMITTED.to_string(), block, size_hint)?
        };

        Ok(Self {
            subject,
            queue,
            committed,
            store,
            state: PhantomData,
        })
    }

    fn dir(&self) -> fs::Inner<State::FE> {
        self.subject.dir()
    }
}

#[async_trait]
impl<State, T> Recover<State::FE> for SyncChain<State, State::Txn, State::FE, T>
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
impl<State, T> fs::CopyFrom<State::FE, Self> for SyncChain<State, State::Txn, State::FE, T>
where
    State: StateInstance,
    State::FE: DenseCacheFile
        + AsType<BTreeNode>
        + AsType<ChainBlock>
        + AsType<TensorNode>
        + for<'a> fs::FileSave<'a>,
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
impl<State, T> de::FromStream for SyncChain<State, State::Txn, State::FE, T>
where
    State: StateInstance,
    State::FE: DenseCacheFile + AsType<BTreeNode> + AsType<ChainBlock> + AsType<TensorNode>,
    T: FromStream<Context = State::Txn>,
{
    type Context = State::Txn;

    async fn from_stream<D: de::Decoder>(
        txn: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        let subject = T::from_stream(txn.clone(), decoder).await?;

        let cxt = txn.context().map_err(de::Error::custom).await?;

        let store = {
            let dir = {
                let mut cxt = cxt.write().await;

                cxt.create_dir(STORE.to_string())
                    .map_err(de::Error::custom)?
            };

            fs::Dir::load(*txn.id(), dir)
                .map_ok(super::data::Store::new)
                .map_err(de::Error::custom)
                .await?
        };

        let queue = new_queue::<State>(store.clone());

        let mut blocks_dir = {
            let file = {
                let mut cxt = cxt.write().await;

                cxt.create_dir(BLOCKS.to_string())
                    .map_err(de::Error::custom)?
            };

            file.write_owned().await
        };

        let null_hash = null_hash();
        let block = ChainBlock::new(null_hash.to_vec());
        let size_hint = block.get_size();
        let committed = blocks_dir
            .create_file(COMMITTED.to_string(), block, size_hint)
            .map_err(de::Error::custom)?;

        Ok(Self {
            subject,
            queue,
            committed,
            store,
            state: PhantomData,
        })
    }
}

#[async_trait]
impl<'en, State, T> IntoView<'en, State::FE> for SyncChain<State, State::Txn, State::FE, T>
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

impl<State, Txn, FE, T> fmt::Debug for SyncChain<State, Txn, FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SyncChain<{}>", std::any::type_name::<T>())
    }
}
