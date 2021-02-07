use std::convert::TryFrom;
use std::fmt;
use std::ops::{Deref, DerefMut};

use bytes::Bytes;
use destream::en;
use futures::Future;
use futures_locks::{RwLockReadGuard, RwLockWriteGuard};

use error::*;
use generic::PathSegment;
use hostfs;

use super::TxnId;

mod dir;
mod file;

pub use dir::{Dir, DirEntry, FileEntry};
pub use file::File;
pub use hostfs::{mount, Dir as HostDir};

pub type BlockId = PathSegment;

pub struct Block<'a, B: BlockData> {
    file: &'a File<B>,
    txn_id: &'a TxnId,
    block_id: BlockId,
    lock: RwLockReadGuard<B>,
}

impl<'a, B: BlockData> Block<'a, B> {
    pub fn new(
        file: &'a File<B>,
        txn_id: &'a TxnId,
        block_id: BlockId,
        lock: RwLockReadGuard<B>,
    ) -> Block<'a, B> {
        Block {
            file,
            txn_id,
            block_id,
            lock,
        }
    }

    pub async fn upgrade(self) -> TCResult<BlockMut<'a, B>> {
        let lock = self._upgrade();
        // make sure to drop self (with its read lock) before acquiring the write lock
        lock.await
    }

    fn _upgrade(self) -> impl Future<Output = TCResult<BlockMut<'a, B>>> {
        self.file.get_block_mut(self.txn_id, self.block_id)
        // self dropped here
    }
}

impl<'a, B: BlockData> Deref for Block<'a, B> {
    type Target = B;

    fn deref(&self) -> &B {
        self.lock.deref()
    }
}

impl<'a, B: BlockData> fmt::Display for Block<'a, B> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "block {}", &self.block_id)
    }
}

pub struct BlockMut<'a, B: BlockData> {
    file: &'a File<B>,
    txn_id: &'a TxnId,
    block_id: BlockId,
    lock: RwLockWriteGuard<B>,
}

impl<'a, B: BlockData> BlockMut<'a, B> {
    pub fn new(
        file: &'a File<B>,
        txn_id: &'a TxnId,
        block_id: BlockId,
        lock: RwLockWriteGuard<B>,
    ) -> Self {
        Self {
            file,
            txn_id,
            block_id,
            lock,
        }
    }

    pub async fn downgrade(self) -> TCResult<Block<'a, B>> {
        let lock = self._downgrade();
        // make sure to drop self (with its write lock) before acquiring the read lock
        lock.await
    }

    fn _downgrade(self) -> impl Future<Output = TCResult<Block<'a, B>>> {
        self.file.get_block(self.txn_id, self.block_id)
        // self dropped here
    }
}

impl<'a, B: BlockData> Deref for BlockMut<'a, B> {
    type Target = B;

    fn deref(&self) -> &B {
        self.lock.deref()
    }
}

impl<'a, B: BlockData> DerefMut for BlockMut<'a, B> {
    fn deref_mut(&mut self) -> &mut B {
        self.lock.deref_mut()
    }
}

pub struct BlockOwned<B: BlockData> {
    file: File<B>,
    txn_id: TxnId,
    block_id: BlockId,
    lock: RwLockReadGuard<B>,
}

impl<B: BlockData> BlockOwned<B> {
    pub fn new(
        file: File<B>,
        txn_id: TxnId,
        block_id: BlockId,
        lock: RwLockReadGuard<B>,
    ) -> BlockOwned<B> {
        BlockOwned {
            file,
            txn_id,
            block_id,
            lock,
        }
    }

    pub async fn upgrade(self) -> TCResult<BlockOwnedMut<B>> {
        let lock = self._upgrade();
        // make sure to drop self before acquiring the write lock
        lock.await
    }

    fn _upgrade(self) -> impl Future<Output = TCResult<BlockOwnedMut<B>>> {
        self.file.get_block_owned_mut(self.txn_id, self.block_id)
        // self dropped here
    }
}

impl<B: BlockData> Deref for BlockOwned<B> {
    type Target = B;

    fn deref(&'_ self) -> &'_ B {
        self.lock.deref()
    }
}

impl<'en, B: BlockData + en::IntoStream<'en>> en::IntoStream<'en> for BlockOwned<B> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream(self.deref().clone(), encoder)
    }
}

pub struct BlockOwnedMut<B: BlockData> {
    file: File<B>,
    txn_id: TxnId,
    block_id: BlockId,
    lock: RwLockWriteGuard<B>,
}

impl<B: BlockData> BlockOwnedMut<B> {
    pub fn new(file: File<B>, txn_id: TxnId, block_id: BlockId, lock: RwLockWriteGuard<B>) -> Self {
        Self {
            file,
            txn_id,
            block_id,
            lock,
        }
    }

    pub async fn downgrade(self) -> TCResult<BlockOwned<B>> {
        let lock = self._downgrade();
        // make sure to drop self (with its write lock) before acquiring the read lock
        lock.await
    }

    fn _downgrade(self) -> impl Future<Output = TCResult<BlockOwned<B>>> {
        self.file.get_block_owned(self.txn_id, self.block_id)
        // self dropped here
    }
}

impl<B: BlockData> Deref for BlockOwnedMut<B> {
    type Target = B;

    fn deref(&self) -> &B {
        self.lock.deref()
    }
}

impl<B: BlockData> DerefMut for BlockOwnedMut<B> {
    fn deref_mut(&mut self) -> &mut B {
        self.lock.deref_mut()
    }
}

pub trait BlockData:
    Clone + TryFrom<Bytes, Error = TCError> + Into<Bytes> + Send + Sync + fmt::Display
{
}
