//! Transactional filesystem traits and data structures.

use std::borrow::Borrow;
use std::collections::HashSet;
use std::fmt;
use std::ops::{Deref, DerefMut};

use async_trait::async_trait;
use safecast::AsType;

use tc_error::*;
use tcgeneric::{Id, PathSegment};

use super::{Transaction, TxnId};

/// An alias for [`Id`] used for code clarity.
pub type BlockId = PathSegment;

/// The data contained by a single block on the filesystem.
pub trait BlockData: Clone + Send + Sync + 'static {
    fn ext() -> &'static str;
}

#[cfg(feature = "tensor")]
impl BlockData for afarray::Array {
    fn ext() -> &'static str {
        "array"
    }
}

/// A block in a [`File`]
#[async_trait]
pub trait Block<B: Clone>: Send + Sync {
    /// A read lock on a block in this file.
    type Read: Deref<Target = B> + Send;

    /// A write lock on a block in this file.
    type Write: DerefMut<Target = B> + Send;

    /// Lock this [`Block`] for reading.
    async fn read(&self) -> TCResult<Self::Read>;

    /// Lock this [`Block`] for writing.
    async fn write(&self) -> TCResult<Self::Write>;
}

/// A file in a [`Dir`]
#[async_trait]
pub trait File<B: Clone>: Send + Sync {
    type Block: Block<B>;

    /// Get the set of all [`BlockId`]s in this [`File`].
    async fn block_ids(&self) -> TCResult<HashSet<BlockId>>;

    /// Copy all blocks from the source [`FileLock`] into this [`FileLock`].
    async fn copy_from(&mut self, other: &Self) -> TCResult<()>;

    /// Create a new block.
    async fn create_block(
        &mut self,
        name: BlockId,
        initial_value: B,
        size_hint: usize,
    ) -> TCResult<<Self::Block as Block<B>>::Write>;

    /// Create a new block.
    async fn create_block_unique(
        &mut self,
        initial_value: B,
        size_hint: usize,
    ) -> TCResult<(BlockId, <Self::Block as Block<B>>::Write)>;

    /// Delete the block with the given ID.
    async fn delete_block(&mut self, name: BlockId) -> TCResult<()>;

    /// Return the specified [`Block`] in this [`File`], if present.
    async fn get_block(&self, block_id: &BlockId) -> TCResult<Option<Self::Block>>;

    /// Return `true` if there are no blocks in this [`File`].
    async fn is_empty(&self) -> bool;

    /// Delete all of this `File`'s blocks.
    async fn truncate(&mut self) -> TCResult<()>;

    /// Convenience method to lock the block at `name` for reading.
    async fn read_block<I>(&self, name: I) -> TCResult<<Self::Block as Block<B>>::Read>
    where
        I: Borrow<BlockId> + Send + Sync,
    {
        let block = {
            let block = self.get_block(name.borrow()).await?;
            block.ok_or_else(|| TCError::not_found(name.borrow()))?
        };

        block.read().await
    }

    /// Convenience method to lock the block at `name` for writing.
    async fn write_block<I>(&self, name: I) -> TCResult<<Self::Block as Block<B>>::Write>
    where
        I: Borrow<BlockId> + Send + Sync,
    {
        let block = {
            let block = self.get_block(name.borrow()).await?;
            block.ok_or_else(|| TCError::not_found(name.borrow()))?
        };

        block.write().await
    }
}

/// A lock on the contents of a transactional [`File`].
#[async_trait]
pub trait FileLock<B: BlockData>: Store + 'static {
    /// The type of this [`File`]
    type File: File<B>;

    /// A read lock on this [`File`]
    type Read: Deref<Target = Self::File> + Send + Sync + 'static;

    /// A write lock on this [`File`]
    type Write: DerefMut<Target = Self::File> + Send + Sync + 'static;

    /// Lock the contents of this [`File`] for reading.
    async fn read(&self, txn_id: TxnId) -> TCResult<Self::Read>;

    /// Lock the contents of this [`File`] for writing.
    async fn write(&self, txn_id: TxnId) -> TCResult<Self::Write>;

    /// Convenience method to lock the block at `name` for reading.
    async fn read_block<I>(
        &self,
        txn_id: TxnId,
        name: I,
    ) -> TCResult<<<Self::File as File<B>>::Block as Block<B>>::Read>
        where
            I: Borrow<BlockId> + Send + Sync,
    {
        let block = {
            let contents = self.read(txn_id).await?;
            let block = contents.get_block(name.borrow()).await?;
            block.ok_or_else(|| TCError::not_found(name.borrow()))?
        };

        block.read().await
    }

    /// Convenience method to lock the block at `name` for reading, without borrowing.
    async fn read_block_owned<I>(
        self,
        txn_id: TxnId,
        name: I,
    ) -> TCResult<<<Self::File as File<B>>::Block as Block<B>>::Read>
        where
            I: Borrow<BlockId> + Send + Sync,
    {
        let block = {
            let contents = self.read(txn_id).await?;
            let block = contents.get_block(name.borrow()).await?;
            block.ok_or_else(|| TCError::not_found(name.borrow()))?
        };

        block.read().await
    }

    /// Convenience method to lock the block at `name` for writing.
    async fn write_block<I>(
        &self,
        txn_id: TxnId,
        name: I,
    ) -> TCResult<<<Self::File as File<B>>::Block as Block<B>>::Write>
    where
        I: Borrow<BlockId> + Send + Sync,
    {
        let block = {
            let contents = self.read(txn_id).await?;
            let block = contents.get_block(name.borrow()).await?;
            block.ok_or_else(|| TCError::not_found(name.borrow()))?
        };

        block.write().await
    }
}

/// A transactional directory, containing sub-directories and [`File`]s
#[async_trait]
pub trait Dir: Clone + Send + Sync {
    /// The type of a file entry in this [`Dir`]
    type FileEntry;

    /// The `Class` of a file stored in this [`Dir`]
    type FileClass: Send;

    /// The type of lock used to guard subdirectories in this [`Dir`]
    type Lock: DirLock<Dir = Self>;

    /// Return `true` if this directory has an entry at the given [`PathSegment`].
    async fn contains(&self, name: &PathSegment) -> TCResult<bool>;

    /// Create a new `Dir`.
    async fn create_dir(&mut self, name: PathSegment) -> TCResult<Self::Lock>;

    /// Create a new `Dir` with a new unique ID.
    async fn create_dir_unique(&mut self) -> TCResult<Self::Lock>;

    /// Create a new [`Self::File`].
    async fn create_file<C, F, B>(&mut self, name: Id, class: C) -> TCResult<F>
    where
        Self::FileEntry: AsType<F>,
        C: Copy + Send + fmt::Display,
        B: BlockData,
        F: FileLock<B>;

    /// Create a new [`Self::File`] with a new unique ID.
    async fn create_file_unique<C, F, B>(&mut self, class: C) -> TCResult<F>
    where
        Self::FileEntry: AsType<F>,
        C: Copy + Send + fmt::Display,
        B: BlockData,
        F: FileLock<B>;

    /// Look up a subdirectory of this `Dir`.
    async fn get_dir(&self, name: &PathSegment) -> TCResult<Option<Self::Lock>>;

    /// Get a [`Self::File`] in this `Dir`.
    async fn get_file<F, B>(&self, name: &Id) -> TCResult<Option<F>>
    where
        Self::FileEntry: AsType<F>,
        B: BlockData,
        F: FileLock<B>;

    /// Return `true` if there are no files or subdirectories in this [`Dir`].
    async fn is_empty(&self) -> bool;
}

/// A transactional directory
#[async_trait]
pub trait DirLock: Store + Send + Sized + 'static {
    /// The type of a subdirectory in this `Dir`
    type Dir: Dir<Lock = Self>;

    /// A read lock on this [`Dir`].
    type Read: Deref<Target = Self::Dir>;

    /// A write lock on this [`Dir`].
    type Write: DerefMut<Target = Self::Dir>;

    /// Lock this [`Dir`] for reading.
    async fn read(&self, txn_id: TxnId) -> TCResult<Self::Dir>;

    /// Lock this [`Dir`] for writing.
    async fn write(&self, txn_id: TxnId) -> TCResult<Self::Dir>;

    /// Convenience method to create a temporary working directory
    async fn create_dir_unique(&self, txn_id: TxnId) -> TCResult<Self> {
        let mut dir = self.write(txn_id).await?;
        dir.create_dir_unique().await
    }

    /// Convenience method to create a temporary file
    async fn create_file_unique<C, F, B>(&self, txn_id: TxnId, class: C) -> TCResult<F>
    where
        <Self::Dir as Dir>::FileEntry: AsType<F>,
        C: Copy + Send + fmt::Display,
        B: BlockData,
        F: FileLock<B>,
    {
        let mut dir = self.write(txn_id).await?;
        dir.create_file_unique(class).await
    }
}

/// A transactional persistent data store, i.e. a [`File`] or [`Dir`].
#[async_trait]
pub trait Store: Clone + Send + Sync {}

/// Defines how to load a persistent data structure from the filesystem.
#[async_trait]
pub trait Persist<D: DirLock>: Sized {
    type Schema;
    type Store: Store;
    type Txn: Transaction<D>;

    /// Return the schema of this persistent state.
    fn schema(&self) -> &Self::Schema;

    /// Load a saved state from persistent storage.
    async fn load(txn: &Self::Txn, schema: Self::Schema, store: Self::Store) -> TCResult<Self>;
}

/// Defines how to copy a base state from another instance, possibly a view.
#[async_trait]
pub trait CopyFrom<D: DirLock, I>: Persist<D> {
    /// Copy a new instance of `Self` from an existing instance.
    async fn copy_from(
        instance: I,
        store: <Self as Persist<D>>::Store,
        txn: &<Self as Persist<D>>::Txn,
    ) -> TCResult<Self>;
}

/// Defines how to restore persistent state from backup.
#[async_trait]
pub trait Restore<D: DirLock>: Persist<D> {
    /// Restore this persistent state from a backup.
    async fn restore(&self, backup: &Self, txn_id: TxnId) -> TCResult<()>;
}
