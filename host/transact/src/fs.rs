//! Transactional filesystem traits and data structures.

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

/// A transactional persistent data store.
#[async_trait]
pub trait Store: Clone + Send + Sync {
    /// Return `true` if this store contains no data as of the given [`TxnId`].
    async fn is_empty(&self, txn_id: TxnId) -> TCResult<bool>;
}

/// A transactional file.
#[async_trait]
pub trait File<B: Clone>: Store + Sized + 'static {
    /// A read Lock on a block in this file.
    type Read: Deref<Target = B> + Send;

    /// A read Lock on a block in this file.
    type Write: DerefMut<Target = B> + Send;

    /// Return the IDs of all this `File`'s blocks.
    async fn block_ids(&self, txn_id: TxnId) -> TCResult<HashSet<BlockId>>;

    /// Return true if this `File` contains the given [`BlockId`] as of the given [`TxnId`].
    async fn contains_block(&self, txn_id: TxnId, name: &BlockId) -> TCResult<bool>;

    /// Copy all blocks from the source `File` into this `File`.
    async fn copy_from(&self, other: &Self, txn_id: TxnId) -> TCResult<()>;

    /// Create a new block.
    ///
    /// `size_hint` should be the maximum allowed size of the block.
    async fn create_block(
        &self,
        txn_id: TxnId,
        name: BlockId,
        initial_value: B,
        size_hint: usize,
    ) -> TCResult<Self::Write>;

    /// Create a new block.
    ///
    /// `size_hint` should be the maximum allowed size of the block.
    async fn create_block_unique(
        &self,
        txn_id: TxnId,
        initial_value: B,
        size_hint: usize,
    ) -> TCResult<(BlockId, Self::Write)>;

    /// Delete the block with the given ID.
    async fn delete_block(&self, txn_id: TxnId, name: BlockId) -> TCResult<()>;

    /// Get a read lock on the block at `name`.
    async fn read_block(&self, txn_id: TxnId, name: BlockId) -> TCResult<Self::Read>;

    /// Get a read lock on the block at `name`, without borrowing.
    async fn read_block_owned(self, txn_id: TxnId, name: BlockId) -> TCResult<Self::Read>;

    /// Get a read lock on the block at `name` as of [`TxnId`].
    async fn write_block(&self, txn_id: TxnId, name: BlockId) -> TCResult<Self::Write>;

    /// Delete all of this `File`'s blocks.
    async fn truncate(&self, txn_id: TxnId) -> TCResult<()>;
}

/// A transactional directory
#[async_trait]
pub trait Dir: Store + Send + Sized + 'static {
    /// The type of a file entry in this `Dir`
    type File: Send;

    /// The `Class` of a file stored in this `Dir`
    type FileClass: Send;

    /// Return `true` if this directory has an entry at the given [`PathSegment`].
    async fn contains(&self, txn_id: TxnId, name: &PathSegment) -> TCResult<bool>;

    /// Create a new `Dir`.
    async fn create_dir(&self, txn_id: TxnId, name: PathSegment) -> TCResult<Self>;

    /// Create a new `Dir` with a new unique ID.
    async fn create_dir_unique(&self, txn_id: TxnId) -> TCResult<Self>;

    /// Create a new [`Self::File`].
    async fn create_file<C, F, B>(&self, txn_id: TxnId, name: Id, class: C) -> TCResult<F>
    where
        C: Copy + Send + fmt::Display,
        F: Clone,
        B: BlockData,
        Self::FileClass: From<C>,
        Self::File: AsType<F>,
        F: File<B>;

    /// Create a new [`Self::File`] with a new unique ID.
    async fn create_file_unique<C, F, B>(&self, txn_id: TxnId, class: C) -> TCResult<F>
    where
        C: Copy + Send + fmt::Display,
        F: Clone,
        B: BlockData,
        Self::FileClass: From<C>,
        Self::File: AsType<F>,
        F: File<B>;

    /// Look up a subdirectory of this `Dir`.
    async fn get_dir(&self, txn_id: TxnId, name: &PathSegment) -> TCResult<Option<Self>>;

    /// Get a [`Self::File`] in this `Dir`.
    async fn get_file<F, B>(&self, txn_id: TxnId, name: &Id) -> TCResult<Option<F>>
    where
        F: Clone,
        B: BlockData,
        Self::File: AsType<F>,
        F: File<B>;
}

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
