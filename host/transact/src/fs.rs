//! Transactional filesystem traits and data structures.

use std::borrow::Borrow;
use std::ops::{Deref, DerefMut};

use async_trait::async_trait;

use tc_error::*;
use tcgeneric::PathSegment;

use super::{Transaction, TxnId};

/// The data contained by a single block on the filesystem
pub trait BlockData: Clone + Send + Sync + 'static {
    fn ext() -> &'static str;
}

#[cfg(feature = "tensor")]
impl BlockData for afarray::Array {
    fn ext() -> &'static str {
        "array"
    }
}

/// A read lock on a block
pub trait BlockRead<B: BlockData>: Deref<Target = B> + Send {}

/// An exclusive read lock on a block
pub trait BlockReadExclusive: Deref<Target = <Self::File as File>::Block> + Send {
    /// The type of [`File`] that this block is part of
    type File: File;

    fn upgrade(self) -> <Self::File as File>::BlockWrite;
}

/// A write lock on a block
pub trait BlockWrite: DerefMut<Target = <Self::File as File>::Block> + Send {
    /// The type of [`File`] that this block is part of
    type File: File;

    fn downgrade(self) -> <Self::File as File>::BlockReadExclusive;
}

/// A read lock on a [`File`]
#[async_trait]
pub trait FileRead: Sized + Send + Sync {
    /// The type of this [`File`]
    type File: File;

    /// Iterate over the names of each block in this [`File`].
    fn block_ids(&self) -> crate::lock::Keys<<Self::File as File>::Key>;

    /// Return `true` if this [`File`] contains a block with the given `name`.
    fn contains<Q>(&self, name: Q) -> bool
    where
        Q: Borrow<<Self::File as File>::Key>;

    /// Return `true` if there are no blocks in this [`File`].
    fn is_empty(&self) -> bool;

    /// Lock the block at `name` for reading.
    async fn read_block<Q>(&self, name: Q) -> TCResult<<Self::File as File>::BlockRead>
    where
        Q: Borrow<<Self::File as File>::Key> + Send + Sync;

    /// Lock the block at `name` for reading exclusively,
    /// i.e. prevent any more read locks being acquired while this one is active.
    async fn read_block_exclusive<Q>(
        &self,
        name: Q,
    ) -> TCResult<<Self::File as File>::BlockReadExclusive>
    where
        Q: Borrow<<Self::File as File>::Key> + Send + Sync;

    /// Lock the block at `name` for reading, without borrowing.
    async fn read_block_owned<Q>(self, name: Q) -> TCResult<<Self::File as File>::BlockRead>
    where
        Q: Borrow<<Self::File as File>::Key> + Send + Sync,
    {
        self.read_block(name).await
    }

    /// Convenience method to lock the block at `name` for writing.
    async fn write_block<Q>(&self, name: Q) -> TCResult<<Self::File as File>::BlockWrite>
    where
        Q: Borrow<<Self::File as File>::Key> + Send + Sync;
}

/// An exclusive read lock on a [`File`]
pub trait FileReadExclusive: FileRead {
    /// Upgrade this read lock to a write lock
    fn upgrade(self) -> <Self::File as File>::Write;
}

/// A write lock on a [`File`]
#[async_trait]
pub trait FileWrite: FileRead {
    /// Downgrade this write lock to an exclusive read lock.
    fn downgrade(self) -> <Self::File as File>::ReadExclusive;

    /// Create a new block.
    async fn create_block(
        &mut self,
        name: <Self::File as File>::Key,
        initial_value: <Self::File as File>::Block,
        size_hint: usize,
    ) -> TCResult<<Self::File as File>::BlockWrite>;

    /// Create a new block.
    async fn create_block_unique(
        &mut self,
        initial_value: <Self::File as File>::Block,
        size_hint: usize,
    ) -> TCResult<(<Self::File as File>::Key, <Self::File as File>::BlockWrite)>;

    /// Delete the block with the given `name`.
    async fn delete_block<Q>(&mut self, name: Q) -> TCResult<()>
    where
        Q: Borrow<<Self::File as File>::Key> + Send + Sync;

    /// Delete all of this `File`'s blocks.
    async fn copy_from<O>(&mut self, other: &O, truncate: bool) -> TCResult<()>
    where
        O: FileRead,
        O::File: File<Key = <Self::File as File>::Key, Block = <Self::File as File>::Block>;

    /// Delete all of this `File`'s blocks.
    async fn truncate(&mut self) -> TCResult<()>;
}

/// A transactional file
#[async_trait]
pub trait File: Store + Clone + 'static {
    /// The type used to identify blocks in this [`File`]
    type Key;

    /// The type used to identify blocks in this [`File`]
    type Block: BlockData;

    /// The type of read guard used by this `File`
    type Read: FileRead<File = Self> + Clone;

    /// The type of exclusive read guard used by this `File`
    type ReadExclusive: FileReadExclusive<File = Self>;

    /// The type of write guard used by this `File`
    type Write: FileWrite<File = Self>;

    /// A read lock on a block in this file.
    type BlockRead: BlockRead<Self::Block>;

    /// An exclusive read lock on a block in this file.
    type BlockReadExclusive: BlockReadExclusive<File = Self>;

    /// A write lock on a block in this file.
    type BlockWrite: BlockWrite<File = Self>;

    /// Lock the contents of this file for reading at the given `txn_id`.
    async fn read(&self, txn_id: TxnId) -> TCResult<Self::Read>;

    /// Lock the contents of this file for reading at the given `txn_id`, exclusively,
    /// i.e. don't allow any more read locks while this one is active.
    async fn read_exclusive(&self, txn_id: TxnId) -> TCResult<Self::ReadExclusive>;

    /// Lock the contents of this file for writing.
    async fn write(&self, txn_id: TxnId) -> TCResult<Self::Write>;

    /// Convenience method to lock the block at `name` for reading.
    async fn read_block<Q>(&self, txn_id: TxnId, name: Q) -> TCResult<Self::BlockRead>
    where
        Q: Borrow<Self::Key> + Send + Sync,
    {
        let file = self.read(txn_id).await?;
        file.read_block(name).await
    }

    /// Convenience method to lock the block at `name` for writing.
    async fn write_block<I>(&self, txn_id: TxnId, name: I) -> TCResult<Self::BlockWrite>
    where
        I: Borrow<Self::Key> + Send + Sync,
    {
        let file = self.read(txn_id).await?;
        file.write_block(name).await
    }
}

/// A read lock on a [`Dir`]
pub trait DirRead: Send {
    /// The type of lock used to guard subdirectories in this [`Dir`]
    type Lock: Dir;

    /// Return `true` if this directory has an entry at the given [`PathSegment`].
    fn contains(&self, name: &PathSegment) -> bool;

    /// Look up a subdirectory of this `Dir`.
    fn get_dir(&self, name: &PathSegment) -> TCResult<Option<Self::Lock>>;

    /// Return `true` if there are no files or subdirectories in this [`Dir`].
    fn is_empty(&self) -> bool;
}

/// A read lock on a [`Dir`], used to read the files it stores
pub trait DirReadFile<F: File>: DirRead {
    /// Get a [`File`] in this `Dir`.
    fn get_file(&self, name: &PathSegment) -> TCResult<Option<F>>;
}

/// A write lock on a [`Dir`] used to create a subdirectory
pub trait DirCreate: DirRead {
    /// Create a new `Dir`.
    fn create_dir(&mut self, name: PathSegment) -> TCResult<Self::Lock>;

    /// Create a new `Dir` with a new unique ID.
    fn create_dir_unique(&mut self) -> TCResult<Self::Lock>;

    /// Get the [`Dir`] with the given `name` and create a new one if none exists.
    fn get_or_create_dir(&mut self, name: PathSegment) -> TCResult<Self::Lock> {
        if let Some(dir) = self.get_dir(&name)? {
            Ok(dir)
        } else {
            self.create_dir(name)
        }
    }
}

/// A write lock on a [`Dir`] used to create a file
pub trait DirCreateFile<F: File>: DirRead {
    /// Create a new [`File`].
    fn create_file(&mut self, name: PathSegment) -> TCResult<F>;

    /// Create a new [`File`] with a new unique ID.
    fn create_file_unique(&mut self) -> TCResult<F>;

    /// Get the [`File`] with the given `name` and create a new one if none exists.
    fn get_or_create_file(&mut self, name: PathSegment) -> TCResult<F>;
}

/// A transactional directory
#[async_trait]
pub trait Dir: Store + Clone + Send + Sized + 'static {
    /// The type of read guard used by this `Dir`
    type Read: DirRead<Lock = Self>;

    /// The type of write guard used by this `Dir`
    type Write: DirCreate<Lock = Self>;

    /// Lock this [`Dir`] for reading.
    async fn read(&self, txn_id: TxnId) -> TCResult<Self::Read>;

    /// Lock this [`Dir`] for writing.
    async fn write(&self, txn_id: TxnId) -> TCResult<Self::Write>;

    /// Convenience method to create a temporary working directory
    async fn create_dir_unique(&self, txn_id: TxnId) -> TCResult<Self> {
        let mut dir = self.write(txn_id).await?;
        dir.create_dir_unique()
    }

    /// Convenience method to create a temporary file
    async fn create_file_unique<F>(&self, txn_id: TxnId) -> TCResult<F>
    where
        F: File,
        Self::Write: DirCreateFile<F>,
    {
        let mut dir = self.write(txn_id).await?;
        dir.create_file_unique()
    }
}

/// A transactional persistent data store, i.e. a [`File`] or [`Dir`].
// TODO: add `is_empty` method
pub trait Store: Send + Sync {}

/// Defines how to load a persistent data structure from the filesystem.
// TODO: add `create` and `load_or_create` methods
#[async_trait]
pub trait Persist<D: Dir>: Sized {
    type Schema: Clone + Send + Sync;
    type Store: Store;
    type Txn: Transaction<D>;

    /// Load a saved state from persistent storage.
    async fn load(txn: &Self::Txn, schema: Self::Schema, store: Self::Store) -> TCResult<Self>;
}

/// Defines how to copy a base state from another instance, possibly a view.
#[async_trait]
pub trait CopyFrom<D: Dir, I>: Persist<D> {
    /// Copy a new instance of `Self` from an existing instance.
    async fn copy_from(
        instance: I,
        store: <Self as Persist<D>>::Store,
        txn: &<Self as Persist<D>>::Txn,
    ) -> TCResult<Self>;
}

/// Defines how to restore persistent state from backup.
#[async_trait]
pub trait Restore<D: Dir>: Persist<D> {
    /// Restore this persistent state from a backup.
    async fn restore(&self, backup: &Self, txn_id: TxnId) -> TCResult<()>;
}
