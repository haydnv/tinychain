use std::convert::TryFrom;
use std::fmt;
use std::ops::{Deref, DerefMut};

use bytes::Bytes;
use destream::en;

use error::*;
use generic::PathSegment;
use hostfs;

use super::lock::*;
use super::TxnId;

mod cache;
mod dir;
mod file;

pub use dir::{Dir, DirEntry, FileEntry};
pub use file::File;
pub use hostfs::{mount, Dir as HostDir};

pub type BlockId = PathSegment;

pub struct Block<'a, B: BlockData> {
    file: &'a File<B>,
    block_id: BlockId,
    lock: TxnLockReadGuard<B>,
}

impl<'a, B: BlockData> Block<'a, B> {
    pub fn new(file: &'a File<B>, block_id: BlockId, lock: TxnLockReadGuard<B>) -> Block<'a, B> {
        Block {
            file,
            block_id,
            lock,
        }
    }

    pub async fn upgrade(self) -> TCResult<BlockMut<'a, B>> {
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
    block_id: BlockId,
    lock: TxnLockWriteGuard<B>,
}

impl<'a, B: BlockData> BlockMut<'a, B> {
    pub async fn downgrade(self, txn_id: &'a TxnId) -> TCResult<Block<'a, B>> {
        Ok(Block {
            file: self.file,
            block_id: self.block_id,
            lock: self.lock.downgrade(txn_id).await?,
        })
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
    block_id: BlockId,
    lock: TxnLockReadGuard<B>,
}

impl<B: BlockData> BlockOwned<B> {
    pub fn new(file: File<B>, block_id: BlockId, lock: TxnLockReadGuard<B>) -> BlockOwned<B> {
        BlockOwned {
            file,
            block_id,
            lock,
        }
    }

    pub async fn upgrade(self) -> TCResult<BlockOwnedMut<B>> {
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
    block_id: BlockId,
    lock: TxnLockWriteGuard<B>,
}

impl<B: BlockData> BlockOwnedMut<B> {
    pub async fn downgrade(self, txn_id: &TxnId) -> TCResult<BlockOwned<B>> {
        Ok(BlockOwned {
            file: self.file,
            block_id: self.block_id,
            lock: self.lock.downgrade(txn_id).await?,
        })
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
    Clone + Mutate<Pending = Self> + TryFrom<Bytes, Error = TCError> + Into<Bytes> + Send + fmt::Display
{
}
