//! Transactional filesystem traits and data structures. Unstable.

use std::collections::HashSet;
use std::fmt;
use std::ops::{Deref, DerefMut};

use async_trait::async_trait;
use bytes::Bytes;
use destream::en;
use futures::{TryFutureExt, TryStreamExt};
use safecast::AsType;
use sha2::{Digest, Sha256};

use tc_error::*;
use tcgeneric::{Id, PathSegment, TCBoxTryStream};

use super::{Transaction, TxnId};

/// An alias for [`Id`] used for code clarity.
pub type BlockId = PathSegment;

pub trait BlockData: Clone + Send + Sync + 'static {
    fn ext() -> &'static str;
}

#[cfg(feature = "tensor")]
impl BlockData for afarray::Array {
    fn ext() -> &'static str {
        "array"
    }
}

impl BlockData for tc_value::Value {
    fn ext() -> &'static str {
        "value"
    }
}

/// A transactional filesystem block.
#[async_trait]
pub trait Block<B>: Send + Sync {
    type ReadLock: Deref<Target = B> + Send;
    type WriteLock: DerefMut<Target = B> + Send + Sync;

    /// Get a read lock on this block.
    async fn read(self) -> TCResult<Self::ReadLock>;

    /// Get a write lock on this block.
    async fn write(self) -> TCResult<Self::WriteLock>;
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
    /// The type of block which this file is divided into.
    type Block: Block<B>;

    /// Return the IDs of all this `File`'s blocks.
    async fn block_ids(&self, txn_id: TxnId) -> TCResult<HashSet<BlockId>>;

    /// Return true if this `File` contains the given [`BlockId`] as of the given [`TxnId`].
    async fn contains_block(&self, txn_id: TxnId, name: &BlockId) -> TCResult<bool>;

    /// Copy all blocks from the source `File` into this `File`.
    async fn copy_from(&self, other: &Self, txn_id: TxnId) -> TCResult<()>;

    /// Create a new [`Self::Block`].
    ///
    /// `size_hint` should be the maximum allowed size of the block.
    async fn create_block(
        &self,
        txn_id: TxnId,
        name: BlockId,
        initial_value: B,
        size_hint: usize,
    ) -> TCResult<Self::Block>;

    // TODO: rename to create_block_unique
    /// Create a new [`Self::Block`].
    ///
    /// `size_hint` should be the maximum allowed size of the block.
    async fn create_block_tmp(
        &self,
        txn_id: TxnId,
        initial_value: B,
        size_hint: usize,
    ) -> TCResult<(BlockId, Self::Block)>;

    /// Delete the block with the given ID.
    async fn delete_block(&self, txn_id: TxnId, name: BlockId) -> TCResult<()>;

    /// Get a read lock on the block at `name`.
    async fn read_block(
        &self,
        txn_id: TxnId,
        name: BlockId,
    ) -> TCResult<<Self::Block as Block<B>>::ReadLock>;

    /// Get a read lock on the block at `name`, without borrowing.
    async fn read_block_owned(
        self,
        txn_id: TxnId,
        name: BlockId,
    ) -> TCResult<<Self::Block as Block<B>>::ReadLock>;

    /// Get a read lock on the block at `name` as of [`TxnId`].
    async fn write_block(
        &self,
        txn_id: TxnId,
        name: BlockId,
    ) -> TCResult<<Self::Block as Block<B>>::WriteLock>;

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

    // TODO: rename to create_dir_unique
    /// Create a new `Dir` with a new unique ID.
    async fn create_dir_tmp(&self, txn_id: TxnId) -> TCResult<Self>;

    /// Create a new [`Self::File`].
    async fn create_file<C, F, B>(&self, txn_id: TxnId, name: Id, class: C) -> TCResult<F>
    where
        C: Copy + Send + fmt::Display,
        F: Clone,
        B: BlockData,
        Self::FileClass: From<C>,
        Self::File: AsType<F>,
        F: File<B>;

    // TODO: rename to create_file_unique
    /// Create a new [`Self::File`] with a new unique ID.
    async fn create_file_tmp<C, F, B>(&self, txn_id: TxnId, class: C) -> TCResult<F>
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

/// Defines a standard hash for a persistent state.
#[async_trait]
pub trait Hash<'en, D: Dir> {
    type Item: en::IntoStream<'en> + Send + 'en;
    type Txn: Transaction<D>;

    /// Return the SHA256 hash of this state as a hexadecimal string.
    async fn hash_hex(&'en self, txn: &'en Self::Txn) -> TCResult<String> {
        self.hash(txn).map_ok(|hash| hex::encode(hash)).await
    }

    /// Compute the SHA256 hash of this state.
    async fn hash(&'en self, txn: &'en Self::Txn) -> TCResult<Bytes> {
        let mut data = self.hashable(txn).await?;

        let mut hasher = Sha256::default();
        while let Some(item) = data.try_next().await? {
            hash_chunks(&mut hasher, item).await?;
        }

        let digest = hasher.finalize();
        Ok(Bytes::from(digest.to_vec()))
    }

    /// Return a stream of hashable items which this state comprises, in a consistent order.
    async fn hashable(&'en self, txn: &'en Self::Txn) -> TCResult<TCBoxTryStream<'en, Self::Item>>;
}

async fn hash_chunks<'en, T: en::IntoStream<'en> + 'en>(
    hasher: &mut Sha256,
    data: T,
) -> TCResult<()> {
    let mut data = tbon::en::encode(data).map_err(TCError::internal)?;
    while let Some(chunk) = data.try_next().map_err(TCError::internal).await? {
        hasher.update(&chunk);
    }

    Ok(())
}
