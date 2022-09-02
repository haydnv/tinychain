//! A transactional [`File`]

use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use async_trait::async_trait;
use safecast::AsType;
use tokio::sync::broadcast::Sender;
use tokio::sync::{Mutex, OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

use tc_error::*;
use tc_transact::fs::{BlockData, BlockId, FileRead, FileWrite, Store};
use tc_transact::{Transact, TxnId};

use super::CacheBlock;

type Listing = HashMap<BlockId, TxnId>;
pub type FileReadGuard<B> = FileGuard<B, OwnedRwLockReadGuard<Listing>>;
pub type FileWriteGuard<B> = FileGuard<B, OwnedRwLockWriteGuard<Listing>>;

struct Wake;

pub struct BlockReadGuard<B> {
    cache: freqfs::FileReadGuard<CacheBlock, B>,
}

impl<B> Deref for BlockReadGuard<B> {
    type Target = B;

    fn deref(&self) -> &Self::Target {
        self.cache.deref()
    }
}

pub struct BlockWriteGuard<B> {
    cache: freqfs::FileWriteGuard<CacheBlock, B>,
}

impl<B> Deref for BlockWriteGuard<B> {
    type Target = B;

    fn deref(&self) -> &Self::Target {
        self.cache.deref()
    }
}

impl<B> DerefMut for BlockWriteGuard<B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.cache.deref_mut()
    }
}

pub struct FileVersion<B> {
    file: File<B>,
    listing: RwLock<HashMap<BlockId, TxnId>>,
}

pub struct FileGuard<B, L> {
    file: File<B>,
    listing: L,
}

#[async_trait]
impl<B, L> FileRead<B> for FileGuard<B, L>
where
    B: BlockData,
    L: Deref<Target = Listing> + Send + Sync,
{
    type Read = BlockReadGuard<B>;
    type Write = BlockWriteGuard<B>;

    fn block_ids(&self) -> HashSet<&BlockId> {
        todo!()
    }

    fn is_empty(&self) -> bool {
        todo!()
    }

    async fn read_block<I>(&self, name: I) -> TCResult<Self::Read>
    where
        I: Borrow<BlockId> + Send + Sync,
    {
        todo!()
    }

    async fn write_block<I>(&self, name: I) -> TCResult<Self::Write>
    where
        I: Borrow<BlockId> + Send + Sync,
    {
        todo!()
    }
}

#[async_trait]
impl<B, L> FileWrite<B> for FileGuard<B, L>
where
    B: BlockData,
    L: DerefMut<Target = Listing> + Send + Sync,
{
    async fn create_block(
        &mut self,
        name: BlockId,
        initial_value: B,
        size_hint: usize,
    ) -> TCResult<Self::Write> {
        todo!()
    }

    async fn create_block_unique(
        &mut self,
        initial_value: B,
        size_hint: usize,
    ) -> TCResult<(BlockId, Self::Write)> {
        todo!()
    }

    async fn delete_block(&mut self, name: BlockId) -> TCResult<()> {
        todo!()
    }

    async fn truncate(&mut self) -> TCResult<()> {
        todo!()
    }
}

struct FileState<B> {
    cache: freqfs::DirLock<CacheBlock>,
    canon: Listing,
    versions: HashMap<TxnId, FileVersion<B>>,
    tx: Sender<Wake>,
}

#[derive(Clone)]
pub struct File<B> {
    state: Arc<Mutex<FileState<B>>>,
    phantom: PhantomData<B>,
}

impl<B: BlockData> File<B>
where
    CacheBlock: AsType<B>,
{
    pub fn new(canon: freqfs::DirLock<CacheBlock>) -> TCResult<Self> {
        todo!()
    }

    pub(super) async fn load(canon: freqfs::DirLock<CacheBlock>, txn_id: TxnId) -> TCResult<Self> {
        todo!()
    }
}

#[async_trait]
impl<B: BlockData> tc_transact::fs::File<B> for File<B> {
    type Read = FileReadGuard<B>;
    type Write = FileWriteGuard<B>;

    async fn copy_from(&self, txn_id: TxnId, other: &Self, truncate: bool) -> TCResult<()> {
        todo!()
    }

    async fn read(&self, txn_id: TxnId) -> TCResult<Self::Read> {
        todo!()
    }

    async fn write(&self, txn_id: TxnId) -> TCResult<Self::Write> {
        todo!()
    }
}

impl<B: BlockData> Store for File<B> {}

#[async_trait]
impl<B: BlockData> Transact for File<B> {
    async fn commit(&self, txn_id: &TxnId) {
        todo!()
    }

    async fn finalize(&self, txn_id: &TxnId) {
        todo!()
    }
}

impl<B: Send + Sync + 'static> fmt::Display for File<B> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "file of {} blocks", std::any::type_name::<B>())
    }
}
