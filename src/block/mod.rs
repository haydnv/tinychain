use std::convert::TryFrom;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;

use crate::error;
use crate::transaction::lock::{Mutate, TxnLockReadGuard, TxnLockWriteGuard};
use crate::transaction::TxnId;
use crate::value::link::PathSegment;
use crate::value::TCResult;

mod cache;
pub mod dir;
pub mod file;

pub type BlockId = PathSegment;

pub struct Block<'a, T: BlockData> {
    file: &'a file::File<T>,
    block_id: BlockId,
    lock: TxnLockReadGuard<T>,
}

impl<'a, T: BlockData> Block<'a, T> {
    pub fn new(
        file: &'a file::File<T>,
        block_id: BlockId,
        lock: TxnLockReadGuard<T>,
    ) -> Block<'a, T> {
        Block {
            file,
            block_id,
            lock,
        }
    }

    pub async fn upgrade(self) -> TCResult<BlockMut<'a, T>> {
        self.file
            .mutate(self.lock.txn_id().clone(), self.block_id.clone())
            .await?;
        Ok(BlockMut {
            file: self.file,
            block_id: self.block_id,
            lock: self.lock.upgrade().await?,
        })
    }
}

impl<'a, T: BlockData> Deref for Block<'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.lock.deref()
    }
}

pub struct BlockMut<'a, T: BlockData> {
    file: &'a file::File<T>,
    block_id: BlockId,
    lock: TxnLockWriteGuard<T>,
}

impl<'a, T: BlockData> BlockMut<'a, T> {
    pub async fn downgrade(self, txn_id: &'a TxnId) -> TCResult<Block<'a, T>> {
        Ok(Block {
            file: self.file,
            block_id: self.block_id,
            lock: self.lock.downgrade(txn_id).await?,
        })
    }
}

impl<'a, T: BlockData> Deref for BlockMut<'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.lock.deref()
    }
}

impl<'a, T: BlockData> DerefMut for BlockMut<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.lock.deref_mut()
    }
}

pub struct BlockOwned<T: BlockData> {
    file: Arc<file::File<T>>,
    block_id: BlockId,
    lock: TxnLockReadGuard<T>,
}

impl<T: BlockData> BlockOwned<T> {
    pub fn new(
        file: Arc<file::File<T>>,
        block_id: BlockId,
        lock: TxnLockReadGuard<T>,
    ) -> BlockOwned<T> {
        BlockOwned {
            file,
            block_id,
            lock,
        }
    }

    pub async fn upgrade(self) -> TCResult<BlockOwnedMut<T>> {
        self.file
            .mutate(self.lock.txn_id().clone(), self.block_id.clone())
            .await?;

        Ok(BlockOwnedMut {
            file: self.file,
            block_id: self.block_id,
            lock: self.lock.upgrade().await?,
        })
    }
}

impl<T: BlockData> Deref for BlockOwned<T> {
    type Target = T;

    fn deref(&'_ self) -> &'_ T {
        self.lock.deref()
    }
}

pub struct BlockOwnedMut<T: BlockData> {
    file: Arc<file::File<T>>,
    block_id: BlockId,
    lock: TxnLockWriteGuard<T>,
}

impl<T: BlockData> Deref for BlockOwnedMut<T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.lock.deref()
    }
}

impl<T: BlockData> DerefMut for BlockOwnedMut<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.lock.deref_mut()
    }
}

pub trait BlockData:
    Clone + Send + Sync + TryFrom<Bytes, Error = error::TCError> + Into<Bytes>
{
}

#[async_trait]
impl<T: BlockData> Mutate for T {
    type Pending = Self;

    fn diverge(&self, _txn_id: &TxnId) -> Self {
        self.clone()
    }

    async fn converge(&mut self, other: Self) {
        *self = other;
    }
}
