//! A transactional file

use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fmt::Formatter;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use async_trait::async_trait;
use futures::{join, try_join, TryFutureExt};
use log::{debug, trace};
use safecast::AsType;
use uuid::Uuid;

use tc_error::*;
use tc_transact::fs::{BlockData, BlockId, FileRead, FileWrite, Store};
use tc_transact::lock::{TxnLock, TxnLockReadGuard, TxnLockWriteGuard};
use tc_transact::{Transact, TxnId};
use tokio::sync::Mutex;

use super::{io_err, CacheBlock, VERSION};

pub type FileReadGuard<B> = FileGuard<B, TxnLockReadGuard<HashSet<BlockId>>>;
pub type FileWriteGuard<B> = FileGuard<B, TxnLockWriteGuard<HashSet<BlockId>>>;

pub struct FileGuard<B, L> {
    file: File<B>,
    listing: L,
}

impl<B, L> FileGuard<B, L>
where
    B: BlockData,
    L: Deref<Target = HashSet<BlockId>> + Send + Sync,
{
    async fn get_block<I>(&self, name: I) -> TCResult<Option<freqfs::FileLock<CacheBlock>>>
    where
        I: Borrow<BlockId> + Send + Sync,
    {
        todo!()
    }
}

#[async_trait]
impl<B, L> FileRead<B> for FileGuard<B, L>
where
    B: BlockData,
    L: Deref<Target = HashSet<BlockId>> + Send + Sync,
{
    type Read = freqfs::FileReadGuard<CacheBlock, B>;
    type Write = freqfs::FileWriteGuard<CacheBlock, B>;

    fn block_ids(&self) -> HashSet<&BlockId> {
        self.listing.iter().collect()
    }

    fn is_empty(&self) -> bool {
        self.listing.is_empty()
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
    L: Deref<Target = HashSet<BlockId>> + Send + Sync,
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

/// A lock on a transactional file
#[derive(Clone)]
pub struct File<B> {
    canon: freqfs::DirLock<CacheBlock>,
    versions: freqfs::DirLock<CacheBlock>,
    listing: TxnLock<HashSet<BlockId>>,
    phantom: PhantomData<B>,
}

impl<B: BlockData> File<B>
where
    CacheBlock: AsType<B>,
{
    pub fn new(canon: freqfs::DirLock<CacheBlock>) -> TCResult<Self> {
        let mut fs_dir = canon.try_write().expect("write lock on new file");
        if fs_dir.len() > 0 {
            return Err(TCError::internal("new file is not empty"));
        }

        let versions = fs_dir.create_dir(VERSION.to_string()).map_err(io_err)?;
        let name = format!("block listing for {:?}", &*fs_dir);

        Ok(Self {
            canon,
            versions,
            listing: TxnLock::new(name, HashSet::new()),
            phantom: PhantomData,
        })
    }

    pub(super) async fn load(canon: freqfs::DirLock<CacheBlock>, txn_id: TxnId) -> TCResult<Self> {
        let mut fs_dir = canon.write().await;
        let versions = fs_dir
            .get_or_create_dir(VERSION.to_string())
            .map_err(io_err)?;

        let mut blocks = HashMap::new();
        let mut present = HashSet::new();
        let mut version = versions
            .write()
            .await
            .create_dir(txn_id.to_string())
            .map_err(io_err)?
            .write()
            .await;

        for (name, block) in fs_dir.iter() {
            if name.starts_with('.') {
                debug!("File::load skipping hidden filesystem entry");
                continue;
            }

            if name.len() < B::ext().len() + 1 || !name.ends_with(B::ext()) {
                return Err(TCError::internal(format!(
                    "block has invalid extension: {}",
                    name
                )));
            }

            let (size_hint, contents) = match block {
                freqfs::DirEntry::File(block) => {
                    let size_hint = block.size_hint().await;
                    let contents = block
                        .read()
                        .map_ok(|contents| B::clone(&*contents))
                        .map_err(io_err)
                        .await?;

                    (size_hint, contents)
                }
                freqfs::DirEntry::Dir(_) => {
                    return Err(TCError::internal(format!(
                        "expected block file but found directory: {}",
                        name
                    )))
                }
            };

            let block_id: BlockId = name[..(name.len() - B::ext().len() - 1)]
                .parse()
                .map_err(TCError::internal)?;

            present.insert(block_id.clone());

            let lock_name = format!("block {}", block_id);
            blocks.insert(block_id, TxnLock::new(lock_name, txn_id));

            version
                .create_file(name.clone(), contents, size_hint)
                .map_err(io_err)?;
        }

        #[cfg(debug_assertions)]
        let lock_name = format!("block list of file {:?}", &*fs_dir);
        #[cfg(not(debug_assertions))]
        let lock_name = "block list of transactional file";

        Ok(Self {
            canon,
            versions,
            listing: TxnLock::new(lock_name, present),
            phantom: PhantomData,
        })
    }

    async fn version(&self, txn_id: &TxnId) -> TCResult<freqfs::DirLock<CacheBlock>> {
        trace!("getting write lock on file dir");

        let (canon, mut versions) = join!(self.canon.write(), self.versions.write());
        trace!("got write lock on file dir");

        versions
            .get_or_create_dir(txn_id.to_string())
            .map_err(io_err)
    }

    async fn version_read(&self, txn_id: &TxnId) -> TCResult<freqfs::DirReadGuard<CacheBlock>> {
        trace!("getting read lock on file dir");
        let version = self.version(txn_id).await?;
        trace!("getting read lock on file version {} dir", txn_id);
        Ok(version.read().await)
    }

    async fn version_write(&self, txn_id: &TxnId) -> TCResult<freqfs::DirWriteGuard<CacheBlock>> {
        trace!("getting write lock on file dir");
        let version = self.version(txn_id).await?;
        trace!("getting write lock on file version {} dir", txn_id);
        Ok(version.write().await)
    }

    async fn with_version_read<F, T>(&self, txn_id: &TxnId, then: F) -> TCResult<T>
    where
        F: FnOnce(freqfs::DirReadGuard<CacheBlock>) -> T,
    {
        let dir = self.version_read(txn_id).await?;
        trace!("got read lock on file version {} dir", txn_id);
        Ok(then(dir))
    }

    async fn with_version_write<F, T>(&self, txn_id: &TxnId, then: F) -> TCResult<T>
    where
        F: FnOnce(freqfs::DirWriteGuard<CacheBlock>) -> T,
    {
        let dir = self.version_write(txn_id).await?;
        trace!("got write lock on file version {} dir", txn_id);
        Ok(then(dir))
    }

    fn file_name(block_id: &BlockId) -> String {
        format!("{}.{}", block_id, B::ext())
    }
}

#[async_trait]
impl<B: BlockData> tc_transact::fs::File<B> for File<B> {
    type Read = FileReadGuard<B>;
    type Write = FileWriteGuard<B>;

    async fn copy_from(&self, txn_id: TxnId, other: &Self, truncate: bool) -> TCResult<()> {
        let (mut dest, source) = try_join!(self.write(txn_id), other.read(txn_id))?;
        if truncate {
            dest.truncate().await?;
        }

        for block_id in source.block_ids() {
            todo!("copy file blocks")
        }

        Ok(())
    }

    async fn read(&self, txn_id: TxnId) -> TCResult<Self::Read> {
        let listing = self.listing.read(txn_id).await?;

        Ok(FileGuard {
            file: self.clone(),
            listing,
        })
    }

    async fn write(&self, txn_id: TxnId) -> TCResult<Self::Write> {
        let listing = self.listing.write(txn_id).await?;

        Ok(FileGuard {
            file: self.clone(),
            listing,
        })
    }
}

#[async_trait]
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

impl<B: BlockData> fmt::Display for File<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "a file with blocks of type {}", B::ext())
    }
}
