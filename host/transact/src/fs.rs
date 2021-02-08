use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::ops::{Deref, DerefMut};

use async_trait::async_trait;
use bytes::Bytes;
use destream::en;
use futures::Future;
use futures_locks::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use error::*;
use generic::{Id, PathSegment};

pub type BlockId = PathSegment;

#[async_trait]
pub trait File: Clone {
    type Block: BlockData;

    async fn create_block(
        &mut self,
        name: Id,
        initial_value: Self::Block,
    ) -> TCResult<BlockOwned<Self>>;

    async fn get_block<'a>(&'a self, name: &'a Id) -> TCResult<Block<'a, Self>>;

    async fn get_block_mut<'a>(&'a self, name: &'a Id) -> TCResult<BlockMut<'a, Self>>;

    async fn get_block_owned(self, name: Id) -> TCResult<BlockOwned<Self>>;

    async fn get_block_owned_mut(self, name: Id) -> TCResult<BlockOwnedMut<Self>>;
}

#[async_trait]
pub trait Dir {
    type Entry;

    async fn create_dir(&mut self, name: PathSegment) -> TCResult<RwLock<Self>>;

    async fn create_file<F: File>(&mut self, name: Id) -> TCResult<RwLock<F>>
    where
        Self::Entry: TryInto<F>;

    async fn get_dir(&self, name: &PathSegment) -> TCResult<Option<RwLock<Self>>>;

    async fn get_file<F: File>(&self, name: &Id) -> TCResult<Option<RwLock<F>>>
    where
        Self::Entry: TryInto<F>;
}

pub struct Block<'a, F: File> {
    file: &'a F,
    block_id: &'a BlockId,
    lock: RwLockReadGuard<F::Block>,
}

impl<'a, F: File> Block<'a, F> {
    pub fn new(
        file: &'a F,
        block_id: &'a BlockId,
        lock: RwLockReadGuard<F::Block>,
    ) -> Block<'a, F> {
        Block {
            file,
            block_id,
            lock,
        }
    }

    pub async fn upgrade(self) -> TCResult<BlockMut<'a, F>> {
        let lock = self._upgrade();
        // make sure to drop self (with its read lock) before acquiring the write lock
        lock.await
    }

    fn _upgrade(self) -> impl Future<Output = TCResult<BlockMut<'a, F>>> {
        self.file.get_block_mut(self.block_id)
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

pub struct BlockMut<'a, F: File> {
    file: &'a F,
    block_id: &'a BlockId,
    lock: RwLockWriteGuard<F::Block>,
}

impl<'a, F: File> BlockMut<'a, F> {
    pub fn new(file: &'a F, block_id: &'a BlockId, lock: RwLockWriteGuard<F::Block>) -> Self {
        Self {
            file,
            block_id,
            lock,
        }
    }

    pub async fn downgrade(self) -> TCResult<Block<'a, F>> {
        let lock = self._downgrade();
        // make sure to drop self (with its write lock) before acquiring the read lock
        lock.await
    }

    fn _downgrade(self) -> impl Future<Output = TCResult<Block<'a, F>>> {
        self.file.get_block(self.block_id)
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

pub struct BlockOwned<F: File> {
    file: F,
    block_id: BlockId,
    lock: RwLockReadGuard<F::Block>,
}

impl<F: File + 'static> BlockOwned<F> {
    pub fn new(file: F, block_id: BlockId, lock: RwLockReadGuard<F::Block>) -> BlockOwned<F> {
        BlockOwned {
            file,
            block_id,
            lock,
        }
    }

    pub async fn upgrade(self) -> TCResult<BlockOwnedMut<F>> {
        let lock = self._upgrade();
        // make sure to drop self before acquiring the write lock
        lock.await
    }

    fn _upgrade(self) -> impl Future<Output = TCResult<BlockOwnedMut<F>>> {
        self.file.get_block_owned_mut(self.block_id)
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

pub struct BlockOwnedMut<F: File> {
    file: F,
    block_id: BlockId,
    lock: RwLockWriteGuard<F::Block>,
}

impl<F: File + 'static> BlockOwnedMut<F> {
    pub fn new(file: F, block_id: BlockId, lock: RwLockWriteGuard<F::Block>) -> Self {
        Self {
            file,
            block_id,
            lock,
        }
    }

    pub async fn downgrade(self) -> TCResult<BlockOwned<F>> {
        let lock = self._downgrade();
        // make sure to drop self (with its write lock) before acquiring the read lock
        lock.await
    }

    fn _downgrade(self) -> impl Future<Output = TCResult<BlockOwned<F>>> {
        self.file.get_block_owned(self.block_id)
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

pub trait BlockData:
    Clone + TryFrom<Bytes, Error = TCError> + Into<Bytes> + Send + Sync + fmt::Display
{}
