//! A transactional file.

use std::collections::HashSet;
use std::convert::TryFrom;
use std::marker::PhantomData;
use std::ops::Deref;
use std::path::PathBuf;

use async_trait::async_trait;
use futures::future::{join_all, try_join_all, FutureExt, TryFutureExt};
use futures::try_join;
use log::{debug, error};

use tc_error::*;
use tc_transact::fs;
use tc_transact::lock::{Mutable, TxnLock};
use tc_transact::{Transact, TxnId};
use tcgeneric::Id;

use super::{file_name, Cache, CacheBlock, CacheLock, DirContents};

/// A transactional file.
#[derive(Clone)]
pub struct File<B> {
    cache: Cache,
    path: PathBuf,
    listing: TxnLock<Mutable<HashSet<fs::BlockId>>>,
    // TODO: replace with a custom lock
    mutated: TxnLock<Mutable<HashSet<fs::BlockId>>>,
    phantom: PhantomData<B>,
}

impl<B: fs::BlockData + 'static> File<B> {
    fn _new(cache: Cache, path: PathBuf, listing: HashSet<fs::BlockId>) -> Self {
        let listing = TxnLock::new(format!("File {:?} listing", &path), listing.into());
        let mutated = TxnLock::new(format!("File {:?} mutated", &path), HashSet::new().into());
        let phantom = PhantomData;

        Self {
            cache,
            path,
            listing,
            mutated,
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
        let path = block_path(&self.path, txn_id, block_id);
        if let Some(lock) = self.cache.read(&path).await? {
            Ok(Some(lock))
        } else if let Some(block) = self.cache.read(&canonical(&self.path, block_id)).await? {
            let contents = block.read().await;
            self.cache
                .write(path, contents.deref().clone())
                .map_ok(Some)
                .await
        } else {
            Ok(None)
        }
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
        let (mut listing, mut mutated) =
            try_join!(self.listing.write(txn_id), self.mutated.write(txn_id))?;

        if listing.contains(&name) {
            return Err(TCError::bad_request("block already exists", name));
        }

        let path = block_path(&self.path, &txn_id, &name);
        debug!("create block at {:?}", &path);

        let lock = self.cache.write(path, initial_value).await?;

        listing.insert(name.clone());
        mutated.insert(name.clone());

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
            Err(TCError::not_found(format!("block {}", name)))
        }
    }

    async fn get_block_mut<'a>(
        &'a self,
        txn_id: &'a TxnId,
        name: &'a fs::BlockId,
    ) -> TCResult<fs::BlockMut<'a, Self>> {
        if let Some(block) = self.get_block_lock(txn_id, name).await? {
            let (mut mutated, lock) =
                try_join!(self.mutated.write(*txn_id), block.write().map(Ok))?;
            if !mutated.contains(name) {
                mutated.insert(name.clone());
            }

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
            let (mut mutated, lock) = try_join!(self.mutated.write(txn_id), block.write().map(Ok))?;
            if !mutated.contains(&name) {
                mutated.insert(name.clone());
            }

            Ok(fs::BlockOwnedMut::new(self, txn_id, name, lock))
        } else {
            Err(TCError::not_found(name))
        }
    }
}

#[async_trait]
impl<B: fs::BlockData> Transact for File<B>
where
    CacheBlock: From<CacheLock<B>>,
    CacheLock<B>: TryFrom<CacheBlock, Error = TCError>,
{
    async fn commit(&self, txn_id: &TxnId) {
        debug!("commit file {:?} at {}", &self.path, txn_id);

        if let Some(mutated) = Some(
            self.mutated
                .read(txn_id)
                .await
                .expect("file mutated block list"),
        ) {
            let _listing = self.listing.read(txn_id).await.expect("lock file listing");

            if !self.path.exists() {
                tokio::fs::create_dir(&self.path)
                    .await
                    .expect("create file dir");
            }

            let backup = mutated.iter().map(|block_id| async move {
                let version = block_path(&self.path, txn_id, block_id);
                self.cache.sync(&version).await?;
                debug_assert!(version.exists());
                TCResult::Ok(())
            });

            try_join_all(backup).await.expect("backup mutated blocks");

            let commit = mutated.iter().map(|block_id| async move {
                let version = block_path(&self.path, txn_id, block_id);
                let dest = canonical(&self.path, block_id);

                let block = self
                    .cache
                    .read(&version)
                    .await?
                    .expect("read mutated block to commit");

                let contents = block.read().await;
                self.cache
                    .write(dest.clone(), contents.deref().clone())
                    .await?;

                self.cache.sync(&dest).await?;
                debug_assert!(dest.exists());
                TCResult::Ok(())
            });

            try_join_all(commit).await.expect("commit mutated blocks");
        }

        self.listing.commit(txn_id).await;
        self.mutated.commit(txn_id).await;
        debug!("committed {:?} at {}", &self.path, txn_id);
    }

    async fn finalize(&self, txn_id: &TxnId) {
        join_all(
            self.listing
                .read(txn_id)
                .await
                .expect("lock file list")
                .iter()
                .map(|block_id| block_path(&self.path, txn_id, block_id))
                .map(|path| self.cache.remove(path)),
        )
        .await;

        let version = version_path(&self.path, txn_id);
        if version.exists() {
            debug!("removing old version directory {:?}", version);
            if let Err(cause) = tokio::fs::remove_dir_all(&version).await {
                error!(
                    "failed to remove old version directory {:?}: {}",
                    version, cause
                );
            }
        }

        self.listing.finalize(txn_id).await;
        self.mutated.finalize(txn_id).await;
        debug!("finalized {:?} at {}", &self.path, txn_id);
    }
}

#[inline]
fn canonical(file_path: &PathBuf, block_id: &fs::BlockId) -> PathBuf {
    let mut path = file_path.clone();
    path.push(block_id.to_string());
    path
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
