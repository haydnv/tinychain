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

/// A read lock on a [`File`]
#[async_trait]
pub trait FileRead<B: Clone>: Send + Sync {
    /// A read lock on a block in this file.
    type Read: Deref<Target = B> + Send;

    /// A write lock on a block in this file.
    type Write: DerefMut<Target = B> + Send;

    /// Get the set of all [`BlockId`]s in this [`File`].
    fn block_ids(&self) -> HashSet<&BlockId>;

    /// Return `true` if there are no blocks in this [`File`].
    fn is_empty(&self) -> bool;

    /// Convenience method to lock the block at `name` for reading.
    async fn read_block<I>(&self, name: I) -> TCResult<Self::Read>
    where
        I: Borrow<BlockId> + Send + Sync;

    /// Convenience method to lock the block at `name` for writing.
    async fn write_block<I>(&self, name: I) -> TCResult<Self::Write>
    where
        I: Borrow<BlockId> + Send + Sync;
}

/// A write lock on a [`File`]
#[async_trait]
pub trait FileWrite<B: Clone>: FileRead<B> {
    /// Create a new block.
    async fn create_block(
        &mut self,
        name: BlockId,
        initial_value: B,
        size_hint: usize,
    ) -> TCResult<Self::Write>;

    /// Create a new block.
    async fn create_block_unique(
        &mut self,
        initial_value: B,
        size_hint: usize,
    ) -> TCResult<(BlockId, Self::Write)>;

    /// Delete the block with the given ID.
    async fn delete_block(&mut self, name: BlockId) -> TCResult<()>;

    /// Delete all of this `File`'s blocks.
    async fn truncate(&mut self) -> TCResult<()>;
}

/// A transactional file
#[async_trait]
pub trait File<B: BlockData>: Store + 'static {
    /// The type of read guard used by this `File`
    type Read: FileRead<B> + Clone;

    /// The type of write guard used by this `File`
    type Write: FileWrite<
        B,
        Read = <Self::Read as FileRead<B>>::Read,
        Write = <Self::Read as FileRead<B>>::Write,
    >;

    /// Copy all blocks from the source [`File`] into this [`File`].
    async fn copy_from(&self, txn_id: TxnId, other: &Self, truncate: bool) -> TCResult<()>;

    /// Lock the contents of this file for reading.
    async fn read(&self, txn_id: TxnId) -> TCResult<Self::Read>;

    /// Lock the contents of this file for writing.
    async fn write(&self, txn_id: TxnId) -> TCResult<Self::Write>;

    /// Convenience method to lock the block at `name` for reading.
    async fn read_block<I>(
        &self,
        txn_id: TxnId,
        name: I,
    ) -> TCResult<<Self::Read as FileRead<B>>::Read>
    where
        I: Borrow<BlockId> + Send + Sync,
    {
        let file = self.read(txn_id).await?;
        file.read_block(name).await
    }

    /// Convenience method to lock the block at `name` for reading, without borrowing.
    async fn read_block_owned<I>(
        self,
        txn_id: TxnId,
        name: I,
    ) -> TCResult<<Self::Read as FileRead<B>>::Read>
    where
        I: Borrow<BlockId> + Send + Sync,
    {
        let file = self.read(txn_id).await?;
        file.read_block(name).await
    }

    /// Convenience method to lock the block at `name` for writing.
    async fn write_block<I>(
        &self,
        txn_id: TxnId,
        name: I,
    ) -> TCResult<<Self::Read as FileRead<B>>::Write>
    where
        I: Borrow<BlockId> + Send + Sync,
    {
        let file = self.read(txn_id).await?;
        file.write_block(name).await
    }
}

/// A read lock on a [`Dir`]
pub trait DirRead: Send + Sync {
    /// The type of a file entry in this [`Dir`]
    type FileEntry;

    /// The type of lock used to guard subdirectories in this [`Dir`]
    type Lock: Dir;

    /// Return `true` if this directory has an entry at the given [`PathSegment`].
    fn contains(&self, name: &PathSegment) -> bool;

    /// Look up a subdirectory of this `Dir`.
    fn get_dir(&self, name: &PathSegment) -> TCResult<Option<Self::Lock>>;

    /// Get a [`Self::File`] in this `Dir`.
    fn get_file<F, B>(&self, name: &Id) -> TCResult<Option<F>>
    where
        Self::FileEntry: AsType<F>,
        B: BlockData,
        F: File<B>;

    /// Return `true` if there are no files or subdirectories in this [`Dir`].
    fn is_empty(&self) -> bool;
}

/// A write lock on a [`Dir`]
pub trait DirWrite: DirRead {
    /// The `Class` of a file stored in this [`Dir`]
    type FileClass: Copy + Send + fmt::Display;

    /// Create a new `Dir`.
    fn create_dir(&mut self, name: PathSegment) -> TCResult<Self::Lock>;

    /// Create a new `Dir` with a new unique ID.
    fn create_dir_unique(&mut self) -> TCResult<Self::Lock>;

    /// Create a new [`Self::File`].
    fn create_file<C, F, B>(&mut self, name: Id, class: C) -> TCResult<F>
    where
        Self::FileClass: From<C>,
        Self::FileEntry: AsType<F>,
        C: Copy + Send + fmt::Display,
        B: BlockData,
        F: File<B>;

    /// Create a new [`Self::File`] with a new unique ID.
    fn create_file_unique<C, F, B>(&mut self, class: C) -> TCResult<F>
    where
        Self::FileClass: From<C>,
        Self::FileEntry: AsType<F>,
        C: Copy + Send + fmt::Display,
        B: BlockData,
        F: File<B>;

    /// Get the [`Dir`] with the given `name` and create a new one if none exists.
    fn get_or_create_dir(&mut self, name: PathSegment) -> TCResult<Self::Lock> {
        if let Some(dir) = self.get_dir(&name)? {
            Ok(dir)
        } else {
            self.create_dir(name)
        }
    }
}

/// A transactional directory
#[async_trait]
pub trait Dir: Store + Send + Sized + 'static {
    /// The type of read guard used by this `Dir`
    type Read: DirRead<Lock = Self>;

    /// The type of write guard used by this `Dir`
    type Write: DirWrite<FileEntry = <Self::Read as DirRead>::FileEntry, Lock = Self>;

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
    async fn create_file_unique<C, F, B>(&self, txn_id: TxnId, class: C) -> TCResult<F>
    where
        <Self::Write as DirWrite>::FileClass: From<C>,
        <Self::Read as DirRead>::FileEntry: AsType<F>,
        C: Copy + Send + fmt::Display,
        B: BlockData,
        F: File<B>,
    {
        let mut dir = self.write(txn_id).await?;
        dir.create_file_unique(class)
    }
}

/// A transactional persistent data store, i.e. a [`File`] or [`Dir`].
pub trait Store: Clone + Send + Sync {}

/// Defines how to load a persistent data structure from the filesystem.
#[async_trait]
pub trait Persist<D: Dir>: Sized {
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
