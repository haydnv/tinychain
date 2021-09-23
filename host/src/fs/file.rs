//! A transactional file

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use freqfs::*;
use futures::future::{join_all, FutureExt, TryFutureExt};
use futures::try_join;
use log::debug;
use safecast::AsType;
use tokio::sync::{RwLock, RwLockWriteGuard};
use uuid::Uuid;

use tc_error::*;
use tc_transact::fs;
use tc_transact::lock::{TxnLock, TxnLockWriteGuard};
use tc_transact::{Transact, TxnId};

use super::{io_err, CacheBlock, VERSION};

#[derive(Clone)]
pub struct Block<B> {
    lock: FileLock<CacheBlock>,
    phantom: PhantomData<B>,
}

impl<B> From<FileLock<CacheBlock>> for Block<B> {
    fn from(lock: FileLock<CacheBlock>) -> Self {
        Self {
            lock,
            phantom: PhantomData,
        }
    }
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

type Blocks = HashMap<fs::BlockId, TxnLock<TxnId>>;

/// A transactional file
pub struct File<B> {
    canon: DirLock<CacheBlock>,

    // don't try to keep the block contents in a TxnLock, since the block versions
    // may need to be backed up to disk--just keep a lock on the TxnId of the last mutation
    // and guard the block content by acquiring a lock on the mutation ID
    // before allowing access to a block
    blocks: Arc<RwLock<Blocks>>,

    present: TxnLock<HashSet<fs::BlockId>>,
    versions: DirLock<CacheBlock>,
    phantom: PhantomData<B>,
}

impl<B> Clone for File<B> {
    fn clone(&self) -> Self {
        Self {
            canon: self.canon.clone(),
            blocks: self.blocks.clone(),
            present: self.present.clone(),
            versions: self.versions.clone(),
            phantom: PhantomData,
        }
    }
}

impl<B: fs::BlockData> File<B>
where
    CacheBlock: AsType<B>,
{
    pub async fn new(canon: DirLock<CacheBlock>) -> TCResult<Self> {
        let mut fs_dir = canon.write().await;
        if fs_dir.len() > 0 {
            return Err(TCError::internal("new file is not empty"));
        }

        Ok(Self {
            canon,
            blocks: Arc::new(RwLock::new(HashMap::new())),
            present: TxnLock::new(format!("block listing for {:?}", &*fs_dir), HashSet::new()),
            versions: fs_dir.create_dir(VERSION.to_string()).map_err(io_err)?,
            phantom: PhantomData,
        })
    }

    async fn get_block(
        &self,
        txn_id: TxnId,
        block_id: fs::BlockId,
        mutate: bool,
    ) -> TCResult<Block<B>> {
        let present = self.present.read(txn_id).await?;
        let blocks = self.blocks.read().await;
        if !present.contains(&block_id) {
            return Err(TCError::not_found(block_id));
        }

        let mut version = self.version(&txn_id).await?;

        if mutate {
            let mut last_mutation = blocks
                .get(&block_id)
                .expect("block last mutation ID")
                .write(txn_id)
                .await?;

            assert!(&*last_mutation <= &txn_id);
            *last_mutation = txn_id;
        } else {
            // just acquire the lock to reserve this TxnId
            blocks
                .get(&block_id)
                .expect("block last mutation ID")
                .read(txn_id)
                .await?;
        }

        let name = Self::file_name(&block_id);

        let block = if let Some(block) = version.get_file(&name) {
            block
        } else {
            let canon = self.canon.read().await;
            let block_canon = canon.get_file(&name).expect("canonical block");
            let size_hint = block_canon.size_hint().await;
            let value = block_canon.read().map_err(io_err).await?;
            let block_version = version
                .create_file(name, B::clone(&*value), size_hint)
                .map_err(io_err)?;

            block_version
        };

        Ok(block.into())
    }

    async fn version(&self, txn_id: &TxnId) -> TCResult<DirWriteGuard<CacheBlock>> {
        let version = {
            let mut versions = self.versions.write().await;
            versions
                .get_or_create_dir(txn_id.to_string())
                .map_err(io_err)?
        };

        version.write().map(Ok).await
    }

    fn file_name(block_id: &fs::BlockId) -> String {
        format!("{}.{}", block_id, B::ext())
    }
}

#[async_trait]
impl<B: fs::BlockData> fs::Store for File<B> {
    async fn is_empty(&self, txn_id: TxnId) -> TCResult<bool> {
        self.present
            .read(txn_id)
            .map_ok(|present| present.is_empty())
            .await
    }
}

#[async_trait]
impl<B> fs::File<B> for File<B>
where
    B: fs::BlockData,
    CacheBlock: AsType<B>,
{
    type Block = Block<B>;

    async fn block_ids(&self, txn_id: TxnId) -> TCResult<HashSet<fs::BlockId>> {
        self.present
            .read(txn_id)
            .map_ok(|present| present.clone())
            .await
    }

    async fn contains_block(&self, txn_id: TxnId, name: &fs::BlockId) -> TCResult<bool> {
        self.present
            .read(txn_id)
            .map_ok(|present| present.contains(name))
            .await
    }

    async fn copy_from(&self, other: &Self, txn_id: TxnId) -> TCResult<()> {
        unimplemented!()
    }

    async fn create_block(
        &self,
        txn_id: TxnId,
        block_id: fs::BlockId,
        initial_value: B,
        size_hint: usize,
    ) -> TCResult<Self::Block> {
        debug!("File::create_block {}", block_id);

        let present = self.present.write(txn_id).await?;
        if present.contains(&block_id) {
            return Err(TCError::bad_request("block already exists", block_id));
        }

        let blocks = self.blocks.write().await;
        let version = self.version(&txn_id).await?;

        create_block_inner(
            present,
            blocks,
            version,
            txn_id,
            block_id,
            initial_value,
            size_hint,
        )
        .await
    }

    async fn create_block_tmp(
        &self,
        txn_id: TxnId,
        initial_value: B,
        size_hint: usize,
    ) -> TCResult<(fs::BlockId, Self::Block)> {
        debug!("File::create_block_tmp");

        let present = self.present.write(txn_id).await?;
        let name = loop {
            let name = Uuid::new_v4().into();
            if !present.contains(&name) {
                break name;
            }
        };

        let blocks = self.blocks.write().await;
        let version = self.version(&txn_id).await?;

        create_block_inner(
            present,
            blocks,
            version,
            txn_id,
            name.clone(),
            initial_value,
            size_hint,
        )
        .map_ok(|block| (name, block))
        .await
    }

    async fn delete_block(&self, txn_id: TxnId, block_id: fs::BlockId) -> TCResult<()> {
        debug!("File::delete_block {}", block_id);

        let mut present = self.present.write(txn_id).await?;
        let mut blocks = self.blocks.write().await;
        if let Some(block) = blocks.get_mut(&block_id) {
            *block.write(txn_id).await? = txn_id;

            // keep the version directory in sync in case create_block is called later
            // with the same block_id
            let mut version = self.version(&txn_id).await?;
            version.delete(Self::file_name(&block_id));
        }

        present.remove(&block_id);
        Ok(())
    }

    async fn read_block(
        &self,
        txn_id: TxnId,
        block_id: fs::BlockId,
    ) -> TCResult<FileReadGuard<CacheBlock, B>> {
        debug!("File::read_block {}", block_id);

        let block = self.get_block(txn_id, block_id, false).await?;
        fs::Block::read(block).await
    }

    async fn read_block_owned(
        self,
        txn_id: TxnId,
        block_id: fs::BlockId,
    ) -> TCResult<FileReadGuard<CacheBlock, B>> {
        debug!("File::read_block_owned {}", block_id);

        let block = self.get_block(txn_id, block_id, false).await?;
        fs::Block::read(block).await
    }

    async fn write_block(
        &self,
        txn_id: TxnId,
        block_id: fs::BlockId,
    ) -> TCResult<FileWriteGuard<CacheBlock, B>> {
        debug!("File::write_block {}", block_id);

        let block = self.get_block(txn_id, block_id, true).await?;
        fs::Block::write(block).await
    }

    async fn truncate(&self, txn_id: TxnId) -> TCResult<()> {
        let mut contents = self.present.write(txn_id).await?;
        let mut version = self.version(&txn_id).await?;
        for block_id in contents.drain() {
            version.delete(Self::file_name(&block_id));
        }

        Ok(())
    }
}

#[async_trait]
impl<B> Transact for File<B>
where
    B: fs::BlockData,
    CacheBlock: AsType<B>,
{
    async fn commit(&self, txn_id: &TxnId) {
        debug!("File::commit");

        let mut blocks = self.blocks.write().await;

        self.present.commit(txn_id).await;
        debug!("File::commit committed block listing");

        join_all(
            blocks
                .values()
                .map(|last_mutation| last_mutation.commit(txn_id)),
        )
        .await;

        {
            let present = self.present.read(*txn_id).await.expect("file block list");
            let version = self.version(txn_id).await.expect("file block versions");
            let mut canon = self.canon.write().await;
            let mut deleted = Vec::with_capacity(blocks.len());
            for block_id in blocks.keys() {
                let name = Self::file_name(block_id);
                if present.contains(block_id) {
                    if let Some(version) = version.get_file(&name) {
                        // these copies can't be run concurrently since `canon` is borrowed mutably
                        canon
                            .copy_from(name, version)
                            .await
                            .expect("copy block version");
                    }
                } else {
                    canon.delete(name);
                    deleted.push(block_id.clone());
                }
            }

            for block_id in deleted.into_iter() {
                assert!(blocks.remove(&block_id).is_some());
            }
        }

        try_join!(self.canon.sync(false), self.versions.sync(false))
            .expect("sync file commit to disk");
    }

    async fn finalize(&self, txn_id: &TxnId) {
        let mut versions = self.versions.write().await;
        versions.delete(txn_id.to_string());

        let blocks = self.blocks.read().await;
        join_all(
            blocks
                .values()
                .map(|last_commit_id| last_commit_id.finalize(txn_id)),
        )
        .await;

        self.present.finalize(txn_id).await
    }
}

impl<B: Send + Sync + 'static> fmt::Display for File<B> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "file of {} blocks", std::any::type_name::<B>())
    }
}

async fn create_block_inner<'a, B: fs::BlockData + 'a>(
    mut present: TxnLockWriteGuard<HashSet<fs::BlockId>>,
    mut blocks: RwLockWriteGuard<'a, Blocks>,
    mut version: DirWriteGuard<CacheBlock>,
    txn_id: TxnId,
    block_id: fs::BlockId,
    value: B,
    size: usize,
) -> TCResult<Block<B>>
where
    CacheBlock: AsType<B>,
{
    debug_assert!(!blocks.contains_key(&block_id));

    blocks.insert(
        block_id.clone(),
        TxnLock::new(format!("block {}", block_id), txn_id),
    );

    let name = format!("{}.{}", block_id, B::ext());
    let block = version
        .create_file(name, value, Some(size))
        .map_err(io_err)?;

    present.insert(block_id);

    Ok(block.into())
}
