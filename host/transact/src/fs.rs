//! Transactional filesystem traits and data structures. Unstable.

use std::io;
use std::ops::{Deref, DerefMut};

use async_trait::async_trait;
use bytes::Bytes;
use futures::{future, TryFutureExt, TryStreamExt};
use tokio::io::{AsyncReadExt, AsyncWrite};
use tokio_util::io::StreamReader;

use tc_error::*;
use tc_value::Value;
use tcgeneric::{Id, PathSegment};

use super::TxnId;

/// An alias for [`Id`] used for code clarity.
pub type BlockId = PathSegment;

/// The contents of a [`Block`].
#[async_trait]
pub trait BlockData: Clone + Send + Sync {
    async fn load<S: AsyncReadExt + Send + Unpin>(source: S) -> TCResult<Self>;

    async fn persist<W: AsyncWrite + Send + Unpin>(&self, sink: &mut W) -> TCResult<u64>;

    async fn size(&self) -> TCResult<u64>;
}

#[async_trait]
impl BlockData for Value {
    async fn load<S: AsyncReadExt + Send + Unpin>(source: S) -> TCResult<Self> {
        destream_json::read_from((), source)
            .map_err(|e| TCError::internal(format!("unable to parse Value: {}", e)))
            .await
    }

    async fn persist<W: AsyncWrite + Send + Unpin>(&self, sink: &mut W) -> TCResult<u64> {
        let encoded = destream_json::encode(self.clone())
            .map_err(|e| TCError::internal(format!("unable to serialize Value: {}", e)))?;

        let mut reader = StreamReader::new(
            encoded
                .map_ok(Bytes::from)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e)),
        );

        tokio::io::copy(&mut reader, sink)
            .map_err(|e| TCError::bad_gateway(e))
            .await
    }

    async fn size(&self) -> TCResult<u64> {
        let encoded = destream_json::encode(self)
            .map_err(|e| TCError::bad_request("serialization error", e))?;

        encoded
            .map_err(|e| TCError::bad_request("serialization error", e))
            .try_fold(0, |size, chunk| {
                future::ready(Ok(size + chunk.len() as u64))
            })
            .await
    }
}

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
    async fn delete_block(&self, txn_id: &TxnId, name: BlockId) -> TCResult<()>;

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

    /// Return `true` if this directory has an entry at the given [`PathSegment`].
    async fn contains(&self, txn_id: &TxnId, name: &PathSegment) -> TCResult<bool>;

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
