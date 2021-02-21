//! A transactional file.

use std::collections::HashSet;
use std::convert::TryFrom;
use std::io;
use std::marker::PhantomData;
use std::path::PathBuf;

use async_trait::async_trait;
use futures::future::{join_all, try_join_all, TryFutureExt};
use log::{debug, error};

use tc_error::*;
use tc_generic::Id;
use tc_transact::fs;
use tc_transact::lock::{Mutable, TxnLock};
use tc_transact::{Transact, TxnId};

use super::{file_name, fs_path, io_err, Cache, CacheBlock, CacheLock, DirContents};
use crate::fs::create_parent;

/// A transactional file.
#[derive(Clone)]
pub struct File<B> {
    cache: Cache,
    path: PathBuf,
    listing: TxnLock<Mutable<HashSet<fs::BlockId>>>,
    phantom: PhantomData<B>,
}

impl<B: fs::BlockData + 'static> File<B> {
    fn _new(cache: Cache, path: PathBuf, listing: HashSet<fs::BlockId>) -> Self {
        let listing = TxnLock::new(format!("File {:?}", &path), listing.into());
        let phantom = PhantomData;

        Self {
            cache,
            path,
            listing,
            phantom,
        }
    }

    /// Create a new [`File`] at the given path.
    pub fn new(cache: Cache, mut path: PathBuf, ext: &str) -> Self {
        path.set_extension(ext);
        Self::_new(cache, path, HashSet::new())
    }

    /// Load a saved [`File`] from the given path.
    pub async fn load(cache: Cache, path: PathBuf, contents: DirContents) -> TCResult<Self> {
        if contents.iter().all(|(_, meta)| meta.is_file()) {
            let listing = contents
                .into_iter()
                .map(|(handle, _)| file_name(&handle))
                .collect::<TCResult<HashSet<fs::BlockId>>>()?;

            Ok(Self::_new(cache, path, listing))
        } else {
            Err(TCError::internal(format!(
                "directory at {:?} contains both blocks and subdirectories",
                path
            )))
        }
    }

    async fn get_block_lock(
        &self,
        txn_id: &TxnId,
        block_id: &fs::BlockId,
    ) -> TCResult<Option<CacheLock<B>>>
    where
        CacheLock<B>: TryFrom<CacheBlock, Error = TCError>,
        CacheBlock: From<CacheLock<B>>,
    {
        if !fs::File::block_exists(self, txn_id, block_id).await? {
            return Err(TCError::not_found(block_id));
        }

        let path = block_path(&self.path, txn_id, block_id);
        if !path.exists() {
            let source_path = fs_path(&self.path, block_id);
            create_parent(&path).await?;

            debug!("copy canonical block {:?} to {:?}", source_path, path);

            tokio::fs::copy(&source_path, &path)
                .map_err(|e| io_err(e, (&source_path, &path)))
                .await?;
        }

        debug_assert!(path.exists());
        self.cache.read(&path).await
    }
}

#[async_trait]
impl<B: Send + Sync> fs::Store for File<B> {
    async fn is_empty(&self, txn_id: &TxnId) -> TCResult<bool> {
        self.listing
            .read(txn_id)
            .map_ok(|listing| listing.is_empty())
            .await
    }
}

#[async_trait]
impl<B: fs::BlockData + 'static> fs::File for File<B>
where
    CacheBlock: From<CacheLock<B>>,
    CacheLock<B>: TryFrom<CacheBlock, Error = TCError>,
{
    type Block = B;

    async fn block_exists(&self, txn_id: &TxnId, name: &fs::BlockId) -> TCResult<bool> {
        let listing = self.listing.read(txn_id).await?;
        Ok(listing.contains(name))
    }

    async fn create_block(
        &self,
        txn_id: TxnId,
        name: fs::BlockId,
        initial_value: Self::Block,
    ) -> TCResult<fs::BlockOwned<Self>> {
        let path = block_path(&self.path, &txn_id, &name);
        debug!("create block at {:?}", &path);

        let mut listing = self.listing.write(txn_id).await?;
        let lock = self.cache.write(path, initial_value).await?;
        listing.insert(name.clone());

        let read_lock = lock.read().await;
        Ok(fs::BlockOwned::new(self.clone(), txn_id, name, read_lock))
    }

    async fn get_block<'a>(
        &'a self,
        txn_id: &'a TxnId,
        name: &'a fs::BlockId,
    ) -> TCResult<fs::Block<'a, Self>> {
        if let Some(block) = self.get_block_lock(txn_id, name).await? {
            let lock = block.read().await;
            Ok(fs::Block::new(self, txn_id, name, lock))
        } else {
            Err(TCError::not_found(name))
        }
    }

    async fn get_block_mut<'a>(
        &'a self,
        txn_id: &'a TxnId,
        name: &'a fs::BlockId,
    ) -> TCResult<fs::BlockMut<'a, Self>> {
        if let Some(block) = self.get_block_lock(txn_id, name).await? {
            let lock = block.write().await;
            Ok(fs::BlockMut::new(self, txn_id, name, lock))
        } else {
            Err(TCError::not_found(name))
        }
    }

    async fn get_block_owned(self, txn_id: TxnId, name: Id) -> TCResult<fs::BlockOwned<Self>> {
        if let Some(block) = self.get_block_lock(&txn_id, &name).await? {
            let lock = block.read().await;
            Ok(fs::BlockOwned::new(self, txn_id, name, lock))
        } else {
            Err(TCError::not_found(name))
        }
    }

    async fn get_block_owned_mut(
        self,
        txn_id: TxnId,
        name: fs::BlockId,
    ) -> TCResult<fs::BlockOwnedMut<Self>> {
        if let Some(block) = self.get_block_lock(&txn_id, &name).await? {
            let lock = block.write().await;
            Ok(fs::BlockOwnedMut::new(self, txn_id, name, lock))
        } else {
            Err(TCError::not_found(name))
        }
    }
}

#[async_trait]
impl<B: fs::BlockData> Transact for File<B> {
    async fn commit(&self, txn_id: &TxnId) {
        debug!("commit file {:?} at {}", &self.path, txn_id);

        // get a write lock to prevent any concurrent access to this file
        {
            let listing = self.listing.write(*txn_id).await.unwrap();
            debug!(
                "file {:?} has {} blocks at {}",
                &self.path,
                listing.len(),
                txn_id
            );

            try_join_all(
                listing
                    .iter()
                    .map(|block_id| block_path(&self.path, txn_id, block_id))
                    .map(|path| self.cache.sync(path)),
            )
            .await
            .expect("commit file cache");

            if !listing.is_empty() {
                let version = version_path(&self.path, txn_id);
                debug!("commit version directory {:?}", version);

                if self.path.exists() {
                    tokio::fs::remove_dir_all(&self.path)
                        .await
                        .expect("commit file");
                }

                if version.exists() {
                    tokio::fs::rename(version, &self.path)
                        .await
                        .expect("commit file version")
                }
            }
        }

        self.listing.commit(txn_id).await;
        debug!("committed {:?} at {}", &self.path, txn_id);
    }

    async fn finalize(&self, txn_id: &TxnId) {
        let listing = self.listing.write(*txn_id).await.unwrap();
        join_all(
            listing
                .iter()
                .map(|block_id| block_path(&self.path, txn_id, block_id))
                .map(|path| self.cache.remove(path)),
        )
        .await;

        let version = version_path(&self.path, txn_id);
        if version.exists() {
            debug!("removing old version directory {:?}", version);
            if let Err(cause) = tokio::fs::remove_dir_all(&version).await {
                if cause.kind() != io::ErrorKind::NotFound {
                    error!(
                        "failed to remove old version directory {:?}: {}",
                        version, cause
                    );
                }
            }
        }

        self.listing.finalize(txn_id).await;
        debug!("finalized {:?} at {}", &self.path, txn_id);
    }
}

#[inline]
fn block_path(file_path: &PathBuf, txn_id: &TxnId, block_id: &fs::BlockId) -> PathBuf {
    let mut path = version_path(file_path, txn_id);
    path.push(block_id.to_string());
    path
}

#[inline]
fn version_path(file_path: &PathBuf, txn_id: &TxnId) -> PathBuf {
    let mut path = PathBuf::from(file_path.parent().unwrap());
    path.push(super::VERSION.to_string());
    path.push(txn_id.to_string());
    path
}
