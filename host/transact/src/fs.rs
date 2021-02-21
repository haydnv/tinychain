//! Transactional filesystem traits and data structures.

use std::convert::TryFrom;
use std::fmt;
use std::ops::{Deref, DerefMut};

use async_trait::async_trait;
use bytes::Bytes;
use destream::en;
use futures::Future;
use uplock::{RwLockReadGuard, RwLockWriteGuard};

use tc_error::*;
use tcgeneric::{Id, PathSegment};

use super::TxnId;

/// An alias for [`Id`] used for code clarity.
pub type BlockId = PathSegment;

/// A transactional persistent data store.
#[async_trait]
pub trait Store: Send + Sync {
    /// Return `true` if this store contains no data as of the given [`TxnId`].
    async fn is_empty(&self, txn_id: &TxnId) -> TCResult<bool>;
}

/// A transactional file.
#[async_trait]
pub trait File: Store + Sized {
    /// The type of block which this file is divided into.
    type Block: BlockData;

    /// Return true if this file contains the given [`BlockId`] as of the given [`TxnId`].
    async fn block_exists(&self, txn_id: &TxnId, name: &BlockId) -> TCResult<bool>;

    /// Create a new [`Self::Block`].
    async fn create_block(
        &self,
        txn_id: TxnId,
        name: BlockId,
        initial_value: Self::Block,
    ) -> TCResult<BlockOwned<Self>>;

    /// Get the data in block `name` as of [`TxnId`].
    async fn get_block<'a>(
        &'a self,
        txn_id: &'a TxnId,
        name: &'a BlockId,
    ) -> TCResult<Block<'a, Self>>;

    /// Get a mutable lock on the data in block `name` as of [`TxnId`].
    async fn get_block_mut<'a>(
        &'a self,
        txn_id: &'a TxnId,
        name: &'a BlockId,
    ) -> TCResult<BlockMut<'a, Self>>;

    /// Get the data in block `name` at [`TxnId`] without borrowing.
    async fn get_block_owned(self, txn_id: TxnId, name: BlockId) -> TCResult<BlockOwned<Self>>;

    /// Get a mutable lock on the data in block `name` at [`TxnId`] without borrowing.
    async fn get_block_owned_mut(
        self,
        txn_id: TxnId,
        name: BlockId,
    ) -> TCResult<BlockOwnedMut<Self>>;
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

    fn schema(&'_ self) -> &'_ Self::Schema;

    async fn load(schema: Self::Schema, store: Self::Store, txn_id: TxnId) -> TCResult<Self>;
}

/// A read lock on one block of a [`File`].
pub struct Block<'a, F: File> {
    file: &'a F,
    txn_id: &'a TxnId,
    block_id: &'a BlockId,
    lock: RwLockReadGuard<F::Block>,
}

impl<'a, F: File> Block<'a, F> {
    /// Construct a new lock.
    pub fn new(
        file: &'a F,
        txn_id: &'a TxnId,
        block_id: &'a BlockId,
        lock: RwLockReadGuard<F::Block>,
    ) -> Block<'a, F> {
        Block {
            file,
            txn_id,
            block_id,
            lock,
        }
    }

    /// Upgrade this read lock to a write lock.
    pub async fn upgrade(self) -> TCResult<BlockMut<'a, F>> {
        let lock = self._upgrade();
        // make sure to drop self (with its read lock) before acquiring the write lock
        lock.await
    }

    fn _upgrade(self) -> impl Future<Output = TCResult<BlockMut<'a, F>>> {
        self.file.get_block_mut(self.txn_id, self.block_id)
        // self dropped here
    }
}

impl<'a, F: File> Deref for Block<'a, F> {
    type Target = F::Block;

    fn deref(&self) -> &F::Block {
        self.lock.deref()
    }
}

impl<'a, F: File> fmt::Display for Block<'a, F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "block {}", &self.block_id)
    }
}

/// An exclusive mutable lock on one block of a [`File`].
pub struct BlockMut<'a, F: File> {
    file: &'a F,
    txn_id: &'a TxnId,
    block_id: &'a BlockId,
    lock: RwLockWriteGuard<F::Block>,
}

impl<'a, F: File> BlockMut<'a, F> {
    /// Construct a new lock.
    pub fn new(
        file: &'a F,
        txn_id: &'a TxnId,
        block_id: &'a BlockId,
        lock: RwLockWriteGuard<F::Block>,
    ) -> Self {
        Self {
            file,
            txn_id,
            block_id,
            lock,
        }
    }

    /// Downgrade this write lock to a read lock.
    pub async fn downgrade(self) -> TCResult<Block<'a, F>> {
        let lock = self._downgrade();
        // make sure to drop self (with its write lock) before acquiring the read lock
        lock.await
    }

    fn _downgrade(self) -> impl Future<Output = TCResult<Block<'a, F>>> {
        self.file.get_block(self.txn_id, self.block_id)
        // self dropped here
    }
}

impl<'a, F: File> Deref for BlockMut<'a, F> {
    type Target = F::Block;

    fn deref(&self) -> &F::Block {
        self.lock.deref()
    }
}

impl<'a, F: File> DerefMut for BlockMut<'a, F> {
    fn deref_mut(&mut self) -> &mut F::Block {
        self.lock.deref_mut()
    }
}

/// An owned read lock on one block of a [`File`].
pub struct BlockOwned<F: File> {
    file: F,
    txn_id: TxnId,
    block_id: BlockId,
    lock: RwLockReadGuard<F::Block>,
}

impl<F: File + 'static> BlockOwned<F> {
    /// Construct a new lock.
    pub fn new(
        file: F,
        txn_id: TxnId,
        block_id: BlockId,
        lock: RwLockReadGuard<F::Block>,
    ) -> BlockOwned<F> {
        BlockOwned {
            file,
            txn_id,
            block_id,
            lock,
        }
    }

    /// Upgrade this read lock to a write lock.
    pub async fn upgrade(self) -> TCResult<BlockOwnedMut<F>> {
        let lock = self._upgrade();
        // make sure to drop self before acquiring the write lock
        lock.await
    }

    fn _upgrade(self) -> impl Future<Output = TCResult<BlockOwnedMut<F>>> {
        self.file.get_block_owned_mut(self.txn_id, self.block_id)
        // self dropped here
    }
}

impl<F: File> Deref for BlockOwned<F> {
    type Target = F::Block;

    fn deref(&'_ self) -> &'_ F::Block {
        self.lock.deref()
    }
}

impl<'en, F: File> en::IntoStream<'en> for BlockOwned<F>
where
    F::Block: en::IntoStream<'en>,
{
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream(self.deref().clone(), encoder)
    }
}

/// An owned exclusive write lock on one block of a [`File`].
pub struct BlockOwnedMut<F: File> {
    file: F,
    txn_id: TxnId,
    block_id: BlockId,
    lock: RwLockWriteGuard<F::Block>,
}

impl<F: File + 'static> BlockOwnedMut<F> {
    /// Construct a new lock.
    pub fn new(
        file: F,
        txn_id: TxnId,
        block_id: BlockId,
        lock: RwLockWriteGuard<F::Block>,
    ) -> Self {
        Self {
            file,
            txn_id,
            block_id,
            lock,
        }
    }

    /// Downgrade this write lock to a read lock.
    pub async fn downgrade(self) -> TCResult<BlockOwned<F>> {
        let lock = self._downgrade();
        // make sure to drop self (with its write lock) before acquiring the read lock
        lock.await
    }

    fn _downgrade(self) -> impl Future<Output = TCResult<BlockOwned<F>>> {
        self.file.get_block_owned(self.txn_id, self.block_id)
        // self dropped here
    }
}

impl<F: File> Deref for BlockOwnedMut<F> {
    type Target = F::Block;

    fn deref(&self) -> &F::Block {
        self.lock.deref()
    }
}

impl<F: File> DerefMut for BlockOwnedMut<F> {
    fn deref_mut(&mut self) -> &mut F::Block {
        self.lock.deref_mut()
    }
}

pub trait BlockData: Clone + TryFrom<Bytes> + Into<Bytes> + Send + Sync {}

impl BlockData for Bytes {}
