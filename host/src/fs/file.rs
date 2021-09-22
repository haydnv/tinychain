//! A transactional file

use std::collections::hash_map::{Entry, HashMap};
use std::collections::HashSet;
use std::fmt;
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use destream::{de, en};
use freqfs::{DirLock, FileLock, FileReadGuard, FileWriteGuard};
use futures::future::{join_all, try_join_all, FutureExt, TryFutureExt};
use futures::stream::{FuturesUnordered, StreamExt};
use futures::{try_join, TryStreamExt};
use log::debug;
use safecast::AsType;
use tokio::sync::RwLock;
use uuid::Uuid;

use tc_error::*;
use tc_transact::fs;
use tc_transact::lock::TxnLock;
use tc_transact::{Transact, TxnId};

use super::{file_name, io_err, CacheBlock};

#[derive(Clone)]
pub struct Block<B> {
    name: fs::BlockId,
    lock: FileLock<CacheBlock>,
    phantom: PhantomData<B>,
}

#[async_trait]
impl<B> fs::Block<B> for Block<B>
where
    B: Send + Sync + 'static,
    CacheBlock: AsType<B>,
{
    type ReadLock = FileReadGuard<CacheBlock, B>;
    type WriteLock = FileWriteGuard<CacheBlock, B>;

    async fn read(self) -> TCResult<Self::ReadLock> {
        self.lock.read().map_err(io_err).await
    }

    async fn write(self) -> TCResult<Self::WriteLock> {
        self.lock.write().map_err(io_err).await
    }
}

/// A transactional file
pub struct File<B> {
    cache: DirLock<CacheBlock>,
    contents: TxnLock<HashSet<fs::BlockId>>,
    mutated: Arc<RwLock<HashMap<TxnId, HashSet<fs::BlockId>>>>,
    phantom: PhantomData<B>,
}

impl<B> Clone for File<B> {
    fn clone(&self) -> Self {
        Self {
            cache: self.cache.clone(),
            contents: self.contents.clone(),
            mutated: self.mutated.clone(),
            phantom: PhantomData,
        }
    }
}

impl<B> File<B> {
    async fn mutate(&self, txn_id: TxnId, block_id: fs::BlockId) {
        let mut mutated = self.mutated.write().await;
        match mutated.entry(txn_id) {
            Entry::Vacant(entry) => entry.insert(HashSet::new()).insert(block_id),
            Entry::Occupied(mut entry) => entry.get_mut().insert(block_id),
        };
    }

    pub async fn new(cache: DirLock<CacheBlock>) -> TCResult<Self> {
        let fs_dir = cache.read().await;
        let contents = fs_dir
            .iter()
            .map(|(name, _)| name.parse())
            .collect::<TCResult<_>>()?;

        Ok(Self {
            cache,
            contents: TxnLock::new("file contents", contents),
            mutated: Arc::new(RwLock::new(HashMap::new())),
            phantom: PhantomData,
        })
    }
}

#[async_trait]
impl<B> fs::Store for File<B>
where
    B: Send + Sync + 'static,
{
    async fn is_empty(&self, txn_id: TxnId) -> TCResult<bool> {
        self.contents
            .read(txn_id)
            .map_ok(|contents| contents.is_empty())
            .await
    }
}

#[async_trait]
impl<B: Send + Sync + 'static> fs::File<B> for File<B>
where
    CacheBlock: AsType<B>,
{
    type Block = Block<B>;

    async fn block_ids(&self, txn_id: TxnId) -> TCResult<HashSet<fs::BlockId>> {
        let contents = self.contents.read(txn_id).await?;
        Ok((*contents).clone())
    }

    async fn contains_block(&self, txn_id: TxnId, name: &fs::BlockId) -> TCResult<bool> {
        self.contents
            .read(txn_id)
            .map_ok(|contents| contents.contains(name))
            .await
    }

    async fn copy_from(&self, other: &Self, txn_id: TxnId) -> TCResult<()> {
        unimplemented!()
    }

    async fn create_block(
        &self,
        txn_id: TxnId,
        name: fs::BlockId,
        initial_value: B,
    ) -> TCResult<Self::Block> {
        let mut contents = self.contents.write(txn_id).await?;
        if contents.contains(&name) {
            return Err(TCError::bad_request(
                "there is already a block with this ID",
                name,
            ));
        }

        unimplemented!()
    }

    async fn create_block_tmp(
        &self,
        txn_id: TxnId,
        initial_value: B,
    ) -> TCResult<(fs::BlockId, Self::Block)> {
        unimplemented!()
    }

    async fn delete_block(&self, txn_id: TxnId, name: fs::BlockId) -> TCResult<()> {
        let mut contents = self.contents.write(txn_id).await?;
        if !contents.remove(&name) {
            return Err(TCError::not_found(format!("block named {}", name)));
        }

        unimplemented!()
    }

    async fn get_block(&self, txn_id: TxnId, name: fs::BlockId) -> TCResult<Block<B>> {
        debug!("File::get_block {}", name);

        unimplemented!()
    }

    async fn read_block(
        &self,
        txn_id: TxnId,
        name: fs::BlockId,
    ) -> TCResult<FileReadGuard<CacheBlock, B>> {
        debug!("File::read_block {}", name);

        let block = self.get_block(txn_id, name).await?;
        fs::Block::read(block).await
    }

    async fn read_block_owned(
        self,
        txn_id: TxnId,
        name: fs::BlockId,
    ) -> TCResult<FileReadGuard<CacheBlock, B>> {
        debug!("File::read_block_owned {}", name);

        let block = self.get_block(txn_id, name).await?;
        fs::Block::read(block).await
    }

    async fn write_block(
        &self,
        txn_id: TxnId,
        name: fs::BlockId,
    ) -> TCResult<FileWriteGuard<CacheBlock, B>> {
        debug!("File::write_block");
        let block = self.get_block(txn_id, name.clone()).await?;
        self.mutate(txn_id, name).await;
        fs::Block::write(block).await
    }

    async fn truncate(&self, txn_id: TxnId) -> TCResult<()> {
        unimplemented!()
    }
}

#[async_trait]
impl<B: Send + Sync + 'static> Transact for File<B>
where
    CacheBlock: AsType<B>,
{
    async fn commit(&self, txn_id: &TxnId) {
        unimplemented!()
    }

    async fn finalize(&self, txn_id: &TxnId) {
        unimplemented!()
    }
}

impl<B: Send + Sync + 'static> fmt::Display for File<B> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "file of {} blocks", std::any::type_name::<B>())
    }
}
