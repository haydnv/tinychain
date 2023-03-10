use async_trait::async_trait;
use freqfs::{FileLoad, FileSave};
use futures::TryFutureExt;
use get_size::GetSize;
use safecast::AsType;
use std::marker::PhantomData;

use tc_error::*;
use tcgeneric::{Id, ThreadSafe};

use super::{TCResult, Transaction, TxnId};

/// The underlying filesystem directory type backing a [`Dir`]
pub type Inner<FE> = freqfs::DirLock<FE>;

/// A read lock on a block in a [`File`]
pub type BlockRead<FE, B> = txfs::FileVersionRead<TxnId, FE, B>;

/// A write lock on a block in a [`File`]
pub type BlockWrite<FE, B> = txfs::FileVersionWrite<TxnId, FE, B>;

/// A transactional directory
pub struct Dir<FE> {
    inner: txfs::Dir<TxnId, FE>,
}

impl<FE: ThreadSafe> Dir<FE> {
    /// Check whether this [`Dir`] has an entry with the given `name` at `txn_id`.
    pub async fn contains(&self, txn_id: TxnId, name: &Id) -> TCResult<bool> {
        self.inner
            .contains(txn_id, name.as_str())
            .map_err(TCError::from)
            .await
    }

    /// Create a new sub-directory with the given `name` at `txn_id`.
    pub async fn create_dir(&self, txn_id: TxnId, name: Id) -> TCResult<Self> {
        self.inner
            .create_dir(txn_id, name.into())
            .map_ok(|inner| Self { inner })
            .map_err(TCError::from)
            .await
    }

    /// Create a new [`File`] with the given `name` at `txn_id`.
    pub async fn create_file<B>(&self, txn_id: TxnId, name: Id) -> TCResult<File<FE, B>>
    where
        B: GetSize + Clone,
        FE: AsType<B>,
    {
        self.inner
            .create_dir(txn_id, name.into())
            .map_ok(File::new)
            .map_err(TCError::from)
            .await
    }

    /// Get the sub-[`Dir`] with the given `name` at `txn_id`, or return a "not found" error.
    pub async fn get_dir(&self, txn_id: TxnId, name: &Id) -> TCResult<Self> {
        if let Some(dir) = self.inner.get_dir(txn_id, name.as_str()).await? {
            Ok(Self { inner: dir.clone() })
        } else {
            Err(TCError::not_found(name))
        }
    }

    /// Get the sub-[`Dir`] with the given `name` at `txn_id`, or create a new one.
    pub async fn get_or_create_dir(&self, txn_id: TxnId, name: Id) -> TCResult<Self> {
        if let Some(dir) = self.inner.get_dir(txn_id, name.as_str()).await? {
            Ok(Self { inner: dir.clone() })
        } else {
            self.create_dir(txn_id, name).await
        }
    }

    /// Get the [`File`] with the given `name` at `txn_id`, or return a "not found" error.
    pub async fn get_file<B>(&self, txn_id: TxnId, name: &Id) -> TCResult<File<FE, B>>
    where
        B: GetSize + Clone,
        FE: AsType<B>,
    {
        if let Some(blocks) = self.inner.get_dir(txn_id, name.as_str()).await? {
            Ok(File::new(blocks.clone()))
        } else {
            Err(TCError::not_found(name))
        }
    }

    /// Return `true` if this [`Dir`] is empty at `txn_id`.
    pub async fn is_empty(&self, txn_id: TxnId) -> TCResult<bool> {
        self.inner.is_empty(txn_id).map_err(TCError::from).await
    }
}

/// A transactional file
pub struct File<FE, B> {
    inner: txfs::Dir<TxnId, FE>,
    phantom: PhantomData<B>,
}

impl<FE, B> File<FE, B> {
    fn new(inner: txfs::Dir<TxnId, FE>) -> Self {
        Self {
            inner,
            phantom: PhantomData,
        }
    }
}

impl<FE, B> File<FE, B>
where
    FE: for<'a> FileSave<'a> + AsType<B> + Clone + Send + Sync,
    B: FileLoad + GetSize + Clone,
{
    /// Create a new block at `txn_id` with the given `name` and `contents`.
    pub async fn create_block(
        &self,
        txn_id: TxnId,
        name: Id,
        contents: B,
    ) -> TCResult<BlockWrite<FE, B>> {
        let block = self
            .inner
            .create_file(txn_id, name.into(), contents)
            .await?;

        block.into_write(txn_id).map_err(TCError::from).await
    }

    /// Delete the block with the given `name` at `txn_id` and return `true` if it was present.
    pub async fn delete_block(&self, txn_id: TxnId, name: Id) -> TCResult<bool> {
        self.inner
            .delete(txn_id, name.into())
            .map_err(TCError::from)
            .await
    }

    /// Lock the block at `name` for reading at `txn_id`.
    pub async fn read_block(&self, txn_id: TxnId, name: &Id) -> TCResult<BlockRead<FE, B>> {
        self.inner
            .read_file(txn_id, name.as_str())
            .map_err(TCError::from)
            .await
    }

    /// Lock the block at `name` for writing at `txn_id`.
    pub async fn write_block(&self, txn_id: TxnId, name: Id) -> TCResult<BlockWrite<FE, B>> {
        self.inner
            .write_file(txn_id, name.as_str())
            .map_err(TCError::from)
            .await
    }
}

/// Defines how to load a persistent data structure from the filesystem.
#[async_trait]
pub trait Persist<FE: ThreadSafe>: Sized {
    type Txn: Transaction<FE>;
    type Schema: Clone + Send + Sync;

    /// Create a new instance of [`Self`] from an empty `Store`.
    async fn create(txn_id: TxnId, schema: Self::Schema, store: Dir<FE>) -> TCResult<Self>;

    /// Load a saved instance of [`Self`] from persistent storage.
    /// Should only be invoked at startup time.
    async fn load(txn_id: TxnId, schema: Self::Schema, store: Dir<FE>) -> TCResult<Self>;

    /// Load a saved instance of [`Self`] from persistent storage if present, or create a new one.
    async fn load_or_create(txn_id: TxnId, schema: Self::Schema, store: Dir<FE>) -> TCResult<Self> {
        if store.is_empty(txn_id).await? {
            Self::create(txn_id, schema, store).await
        } else {
            Self::load(txn_id, schema, store).await
        }
    }

    /// Access the filesystem directory backing this persistent data structure.
    fn dir(&self) -> &Inner<FE>;
}

/// Copy a base state from another instance, possibly a view.
#[async_trait]
pub trait CopyFrom<FE: ThreadSafe, I>: Persist<FE> {
    /// Copy a new instance of `Self` from an existing instance.
    async fn copy_from(
        txn: &<Self as Persist<FE>>::Txn,
        store: Dir<FE>,
        instance: I,
    ) -> TCResult<Self>;
}

/// Restore a persistent state from a backup.
#[async_trait]
pub trait Restore<FE: ThreadSafe>: Persist<FE> {
    /// Restore this persistent state from a backup.
    async fn restore(&self, txn_id: TxnId, backup: &Self) -> TCResult<()>;
}
