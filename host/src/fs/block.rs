use std::convert::TryFrom;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::path::PathBuf;

use async_trait::async_trait;
use futures::join;
use uplock::*;

use tc_error::*;
use tc_transact::fs;

use crate::transact::TxnId;

use super::cache::*;
use tc_transact::fs::BlockData;

pub struct BlockRead<B> {
    guard: RwLockReadGuard<B>,
}

impl<B> Deref for BlockRead<B> {
    type Target = B;

    fn deref(&self) -> &B {
        self.guard.deref()
    }
}

pub struct BlockWrite<B> {
    guard: RwLockWriteGuard<B>,
}

impl<B> Deref for BlockWrite<B> {
    type Target = B;

    fn deref(&self) -> &B {
        self.guard.deref()
    }
}

impl<B> DerefMut for BlockWrite<B> {
    fn deref_mut(&mut self) -> &mut B {
        self.guard.deref_mut()
    }
}

pub struct Block<B> {
    path: PathBuf,
    cache: Cache,
    committed: RwLock<Option<TxnId>>,
    reserved: RwLock<Option<TxnId>>,
    phantom: PhantomData<B>,
}

impl<B: BlockData> Block<B>
where
    CacheBlock: From<CacheLock<B>>,
    CacheLock<B>: TryFrom<CacheBlock, Error = TCError>,
{
    pub fn new(cache: Cache, file_path: &PathBuf, block_id: &fs::BlockId) -> Self {
        let mut path = file_path.clone();
        path.push(block_id.as_str());

        Self {
            cache,
            path,
            committed: RwLock::new(None),
            reserved: RwLock::new(None),
            phantom: PhantomData,
        }
    }

    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    async fn access(&self, txn_id: &TxnId) -> TCResult<()> {
        let (committed, reserved) = join!(self.committed.read(), self.reserved.read());

        if let Some(commit_id) = committed.deref() {
            if txn_id < commit_id {
                return Err(TCError::conflict());
            }
        }

        if let Some(reserved_id) = reserved.deref() {
            if txn_id < reserved_id {
                return Err(TCError::conflict());
            } else {
                *reserved.upgrade().await = Some(*txn_id);
            }
        } else {
            *reserved.upgrade().await = Some(*txn_id);
        }

        Ok(())
    }

    pub async fn commit(&self, txn_id: &TxnId) -> TCResult<()> {
        let committed = self.committed.read().await;

        let version = version_path(&self.path, txn_id);
        if self.cache.sync(&version).await? {
            let cached = self.cache.read(&version).await?;
            let contents = cached.expect("cached block version").read().await;

            self.cache
                .write(self.path.clone(), contents.deref().clone())
                .await?;

            self.cache.sync(&self.path).await?;
        }

        *committed.upgrade().await = Some(*txn_id);
        Ok(())
    }
}

#[async_trait]
impl<B: fs::BlockData> fs::Block<B> for Block<B>
where
    CacheBlock: From<CacheLock<B>>,
    CacheLock<B>: TryFrom<CacheBlock, Error = TCError>,
{
    type ReadLock = BlockRead<B>;
    type WriteLock = BlockWrite<B>;

    async fn read(&self, txn_id: &TxnId) -> TCResult<Self::ReadLock> {
        let path = version_path(&self.path, txn_id);
        if let Some(lock) = self.cache.read(&path).await? {
            let guard = lock.read().await;
            Ok(BlockRead { guard })
        } else if let Some(lock) = self.cache.read(&self.path).await? {
            self.access(txn_id).await?;

            let guard = lock.read().await;
            Ok(BlockRead { guard })
        } else {
            Err(TCError::internal(format!(
                "missing block at {:?}",
                &self.path
            )))
        }
    }

    async fn write(&self, txn_id: &TxnId) -> TCResult<Self::WriteLock> {
        self.access(txn_id).await?;

        let path = version_path(&self.path, txn_id);
        if let Some(lock) = self.cache.read(&path).await? {
            let guard = lock.write().await;
            Ok(BlockWrite { guard })
        } else if let Some(lock) = self.cache.read(&self.path).await? {
            // make sure there are no open read locks on the canonical version
            let contents = lock.write().await;
            let lock = self.cache.write(path, contents.deref().clone()).await?;
            let guard = lock.write().await;
            Ok(BlockWrite { guard })
        } else {
            Err(TCError::internal(format!(
                "missing block at {:?}",
                &self.path
            )))
        }
    }
}

#[inline]
fn version_path(path: &PathBuf, txn_id: &TxnId) -> PathBuf {
    let block_id = path.file_name().expect("block file name");
    let mut path = PathBuf::from(path.parent().expect("file path"));
    path.push(super::VERSION.to_string());
    path.push(txn_id.to_string());
    path.push(block_id);
    path
}
