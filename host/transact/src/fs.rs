//! Transactional filesystem traits and data structures. Unstable.

use std::collections::HashSet;
use std::io;
use std::ops::{Deref, DerefMut};

use async_trait::async_trait;
use bytes::Bytes;
use destream::{de, en};
use futures::{future, TryFutureExt, TryStreamExt};
use sha2::{Digest, Sha256};
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
pub trait BlockData: de::FromStream<Context = ()> + Clone + Send + Sync {
    fn ext() -> &'static str;

    async fn hash<'en>(&'en self) -> TCResult<Bytes> where Self: en::ToStream<'en> {
        let mut data = tbon::en::encode(self).map_err(TCError::internal)?;
        let mut hasher = Sha256::default();
        while let Some(chunk) = data.try_next().map_err(TCError::internal).await? {
            hasher.update(&chunk);
        }

        let digest = hasher.finalize();
        Ok(Bytes::from(digest.to_vec()))
    }

    async fn load<S: AsyncReadExt + Send + Unpin>(source: S) -> TCResult<Self> {
        tbon::de::read_from((), source)
            .map_err(|e| TCError::internal(format!("unable to parse Value: {}", e)))
            .await
    }

    async fn persist<'en, W: AsyncWrite + Send + Unpin>(&'en self, sink: &mut W) -> TCResult<u64> where Self: en::ToStream<'en> {
        let encoded = tbon::en::encode(self)
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

    async fn size<'en>(&'en self) -> TCResult<u64> where Self: en::ToStream<'en> {
        let encoded = tbon::en::encode(self).map_err(|e| TCError::bad_request("serialization error", e))?;

        encoded
            .map_err(|e| TCError::bad_request("serialization error", e))
            .try_fold(0, |size, chunk| {
                future::ready(Ok(size + chunk.len() as u64))
            })
            .await
    }

    async fn into_size<'en>(self) -> TCResult<u64> where Self: Clone + en::IntoStream<'en> + 'en {
        let encoded = tbon::en::encode(self).map_err(|e| TCError::bad_request("serialization error", e))?;

        encoded
            .map_err(|e| TCError::bad_request("serialization error", e))
            .try_fold(0, |size, chunk| {
                future::ready(Ok(size + chunk.len() as u64))
            })
            .await
    }
}

#[async_trait]
impl BlockData for Value {
    fn ext() -> &'static str {
        "value"
    }
}

/// A transactional filesystem block.
#[async_trait]
pub trait Block<B: BlockData>: Send + Sync {
    type ReadLock: Deref<Target = B>;
    type WriteLock: DerefMut<Target = B>;

    /// Get a read lock on this block.
    async fn read(&self) -> Self::ReadLock;

    /// Get a write lock on this block.
    async fn write(&self) -> Self::WriteLock;
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

    /// Return the IDs of all this file's blocks.
    async fn block_ids(&self, txn_id: &TxnId) -> TCResult<HashSet<BlockId>>;

    /// Return true if this file contains the given [`BlockId`] as of the given [`TxnId`].
    async fn contains_block(&self, txn_id: &TxnId, name: &BlockId) -> TCResult<bool>;

    /// Create a new [`Self::Block`].
    async fn create_block(
        &self,
        txn_id: TxnId,
        name: BlockId,
        initial_value: B,
    ) -> TCResult<Self::Block>;

    /// Delete the block with the given ID.
    async fn delete_block(&self, txn_id: &TxnId, name: &BlockId) -> TCResult<()>;

    /// Get a read lock on the block at `name` as of [`TxnId`].
    async fn read_block(
        &self,
        txn_id: &TxnId,
        name: &BlockId,
    ) -> TCResult<<Self::Block as Block<B>>::ReadLock>;

    /// Get a read lock on the block at `name` as of [`TxnId`], without borrowing.
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
