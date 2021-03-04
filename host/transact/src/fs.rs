//! Transactional filesystem traits and data structures. Unstable.

use std::convert::TryFrom;
use std::ops::{Deref, DerefMut};

use async_trait::async_trait;
use bytes::Bytes;

use tc_error::*;
use tcgeneric::{Id, PathSegment};

use super::TxnId;

/// An alias for [`Id`] used for code clarity.
pub type BlockId = PathSegment;

/// The contents of a [`Block`].
pub trait BlockData: Clone + TryFrom<Bytes> + Into<Bytes> + Send + Sync {}

impl BlockData for Bytes {}

/// A transactional filesystem block.
#[async_trait]
pub trait Block<B: BlockData>: Send + Sync {
    type ReadLock: Deref<Target = B>;
    type WriteLock: DerefMut<Target = B>;

    /// Get a read lock on this block.
    async fn read(&self, txn_id: &TxnId) -> TCResult<Self::ReadLock>;

    /// Get a write lock on this block.
    async fn write(&self, txn_id: &TxnId) -> TCResult<Self::WriteLock>;
}

/// A transactional persistent data store.
#[async_trait]
pub trait Store: Send + Sync {
    /// Return `true` if this store contains no data as of the given [`TxnId`].
    async fn is_empty(&self, txn_id: &TxnId) -> TCResult<bool>;
}

/// A transactional file.
#[async_trait]
pub trait File<B: BlockData>: Store + Sized {
    /// The type of block which this file is divided into.
    type Block: Block<B>;

    /// Return true if this file contains the given [`BlockId`] as of the given [`TxnId`].
    async fn contains_block(&self, txn_id: &TxnId, name: &BlockId) -> TCResult<bool>;

    /// Create a new [`Self::Block`].
    async fn create_block(
        &self,
        txn_id: TxnId,
        name: BlockId,
        initial_value: B,
    ) -> TCResult<<Self::Block as Block<B>>::ReadLock>;

    /// Delete the block with the given ID.
    async fn delete_block(
        &self,
        txn_id: &TxnId,
        name: BlockId,
    ) -> TCResult<()>;

    /// Obtain a read lock on block `name` as of [`TxnId`].
    async fn get_block(
        &self,
        txn_id: &TxnId,
        name: BlockId,
    ) -> TCResult<<Self::Block as Block<B>>::ReadLock>;

    /// Obtain a write lock on block `name` as of [`TxnId`].
    async fn get_block_mut(
        &self,
        txn_id: &TxnId,
        name: BlockId,
    ) -> TCResult<<Self::Block as Block<B>>::WriteLock>;
}

/// A transactional directory
#[async_trait]
pub trait Dir: Store + Sized {
    /// The type of a file entry in this `Dir`
    type File;

    /// The `Class` of a file stored in this `Dir`
    type FileClass;

    /// Create a new [`Dir`].
    async fn create_dir(&self, txn_id: TxnId, name: PathSegment) -> TCResult<Self>;

    /// Create a new [`Self::File`].
    async fn create_file(
        &self,
        txn_id: TxnId,
        name: Id,
        class: Self::FileClass,
    ) -> TCResult<Self::File>;

    /// Look up a subdirectory of this `Dir`.
    async fn get_dir(&self, txn_id: &TxnId, name: &PathSegment) -> TCResult<Option<Self>>;

    /// Get a [`Self::File`] in this `Dir`.
    async fn get_file(&self, txn_id: &TxnId, name: &Id) -> TCResult<Option<Self::File>>;
}

/// Defines how to load a persistent data structure from the filesystem.
#[async_trait]
pub trait Persist: Sized {
    type Schema;
    type Store: Store;

    /// Return the schema of this persistent state.
    fn schema(&'_ self) -> &'_ Self::Schema;

    /// Load a saved state from persistent storage.
    async fn load(schema: Self::Schema, store: Self::Store, txn_id: TxnId) -> TCResult<Self>;
}
