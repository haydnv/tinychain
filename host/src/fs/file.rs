//! A transactional file

use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fmt::Formatter;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

use async_trait::async_trait;
use log::{debug, trace};
use safecast::AsType;
use uuid::Uuid;

use tc_error::*;
use tc_transact::fs;
use tc_transact::fs::BlockData;
use tc_transact::{Transact, TxnId};

use super::{io_err, CacheBlock, VERSION};

#[derive(Copy, Clone, Eq, PartialEq)]
enum FileDelta {
    Created,
    Modified,
    Deleted,
}

pub struct FileGuard<L> {
    cache: L,
    deltas: HashMap<fs::BlockId, FileDelta>,
}

impl<L> FileGuard<L>
where
    L: Deref<Target = freqfs::Dir<CacheBlock>>,
{
    async fn get_block<I>(&self, name: I) -> TCResult<Option<freqfs::FileLock<CacheBlock>>>
    where
        I: Borrow<fs::BlockId> + Send + Sync,
    {
        todo!()
    }
}

#[async_trait]
impl<B, L> fs::FileRead<B> for FileGuard<L>
where
    B: BlockData,
    L: Deref<Target = freqfs::Dir<CacheBlock>> + Send + Sync,
{
    type Read = freqfs::FileReadGuard<CacheBlock, B>;
    type Write = freqfs::FileWriteGuard<CacheBlock, B>;

    async fn block_ids(&self) -> TCResult<HashSet<fs::BlockId>> {
        todo!()
    }

    async fn is_empty(&self) -> bool {
        todo!()
    }

    async fn read_block<I>(&self, name: I) -> TCResult<Self::Read>
    where
        I: Borrow<fs::BlockId> + Send + Sync,
    {
        todo!()
    }

    async fn write_block<I>(&self, name: I) -> TCResult<Self::Write>
    where
        I: Borrow<fs::BlockId> + Send + Sync,
    {
        todo!()
    }
}

#[async_trait]
impl<B, L> fs::FileWrite<B> for FileGuard<L>
where
    B: BlockData,
    L: DerefMut<Target = freqfs::Dir<CacheBlock>> + Send + Sync,
{
    async fn create_block(
        &mut self,
        name: fs::BlockId,
        initial_value: B,
        size_hint: usize,
    ) -> TCResult<Self::Write> {
        todo!()
    }

    async fn create_block_unique(
        &mut self,
        initial_value: B,
        size_hint: usize,
    ) -> TCResult<(fs::BlockId, Self::Write)> {
        todo!()
    }

    async fn delete_block(&mut self, name: fs::BlockId) -> TCResult<()> {
        todo!()
    }

    async fn truncate(&mut self) -> TCResult<()> {
        todo!()
    }
}

/// A lock on a [`File`]
#[derive(Clone)]
pub struct File<B> {
    canon: freqfs::DirLock<CacheBlock>,
    versions: freqfs::DirLock<CacheBlock>,
    phantom: PhantomData<B>,
}

impl<B: fs::BlockData> File<B> {
    pub async fn new(canon: freqfs::DirLock<CacheBlock>) -> TCResult<Self> {
        todo!()
    }

    pub(super) async fn load(canon: freqfs::DirLock<CacheBlock>, txn_id: TxnId) -> TCResult<Self> {
        todo!()
    }
}

#[async_trait]
impl<B: fs::BlockData> fs::File<B> for File<B> {
    type Read = FileGuard<freqfs::DirReadGuard<CacheBlock>>;
    type Write = FileGuard<freqfs::DirWriteGuard<CacheBlock>>;

    async fn copy_from(&self, txn_id: TxnId, other: &Self, truncate: bool) -> TCResult<()> {
        todo!()
    }

    async fn read(&self, txn_id: TxnId) -> TCResult<Self::Read> {
        todo!()
    }

    async fn write(&self, txn_id: TxnId) -> TCResult<Self::Write> {
        todo!()
    }

    async fn read_block<I>(
        &self,
        txn_id: TxnId,
        name: I,
    ) -> TCResult<freqfs::FileReadGuard<CacheBlock, B>>
    where
        I: Borrow<fs::BlockId> + Send + Sync,
    {
        todo!()
    }

    async fn read_block_owned<I>(
        self,
        txn_id: TxnId,
        name: I,
    ) -> TCResult<freqfs::FileReadGuard<CacheBlock, B>>
    where
        I: Borrow<fs::BlockId> + Send + Sync,
    {
        todo!()
    }

    async fn write_block<I>(
        &self,
        txn_id: TxnId,
        name: I,
    ) -> TCResult<freqfs::FileWriteGuard<CacheBlock, B>>
    where
        I: Borrow<fs::BlockId> + Send + Sync,
    {
        todo!()
    }
}

#[async_trait]
impl<B: fs::BlockData> fs::Store for File<B> {}

#[async_trait]
impl<B: fs::BlockData> Transact for File<B> {
    async fn commit(&self, txn_id: &TxnId) {
        todo!()
    }

    async fn finalize(&self, txn_id: &TxnId) {
        todo!()
    }
}

impl<B: BlockData> fmt::Display for File<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "a file with blocks of type {}", B::ext())
    }
}
