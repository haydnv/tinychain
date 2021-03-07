//! A transactional file.

use std::collections::hash_map::{Entry, HashMap};
use std::collections::HashSet;
use std::convert::TryFrom;
use std::iter;
use std::path::PathBuf;

use async_trait::async_trait;
use futures::future::{try_join_all, TryFutureExt};
use futures::join;
use log::debug;
use uplock::RwLock;

use tc_error::*;
use tc_transact::fs;
use tc_transact::lock::{Mutable, TxnLock};
use tc_transact::{Transact, TxnId};

use super::block::*;
use super::cache::*;
use super::{file_name, DirContents};

/// A transactional file.
#[derive(Clone)]
pub struct File<B> {
    path: PathBuf,
    cache: Cache,
    listing: TxnLock<Mutable<HashSet<fs::BlockId>>>,
    touched: RwLock<HashMap<TxnId, HashMap<fs::BlockId, Block<B>>>>,
}

impl<B: fs::BlockData + 'static> File<B> {
    fn _new(cache: Cache, path: PathBuf, listing: HashSet<fs::BlockId>) -> Self {
        let lock_name = format!("file {:?} block list", &path);

        Self {
            cache,
            path,
            listing: TxnLock::new(lock_name, listing.into()),
            touched: RwLock::new(HashMap::new()),
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
impl<B: fs::BlockData + 'static> fs::File<B> for File<B>
where
    CacheBlock: From<CacheLock<B>>,
    CacheLock<B>: TryFrom<CacheBlock, Error = TCError>,
{
    type Block = Block<B>;

    async fn contains_block(&self, txn_id: &TxnId, name: &fs::BlockId) -> TCResult<bool> {
        let listing = self.listing.read(txn_id).await?;
        Ok(listing.contains(name))
    }

    async fn create_block(
        &self,
        txn_id: TxnId,
        name: fs::BlockId,
        initial_value: B,
    ) -> TCResult<BlockRead<B>> {
        let (listing, mut touched) = join!(self.listing.write(txn_id), self.touched.write());
        let mut listing = listing?;

        if listing.contains(&name) {
            return Err(TCError::bad_request("block already exists", name));
        }

        let version = block_version_path(&self.path, &name, &txn_id);
        self.cache.write(version, initial_value).await?;

        let block = Block::new(self.cache.clone(), &self.path, &name);
        let lock = fs::Block::read(&block, &txn_id).await?;

        touch(&mut touched, txn_id, name.clone(), block);
        listing.insert(name);
        Ok(lock)
    }

    async fn delete_block(&self, txn_id: &TxnId, name: fs::BlockId) -> TCResult<()> {
        let (listing, mut touched) = join!(self.listing.write(*txn_id), self.touched.write());
        let mut listing = listing?;

        if !listing.remove(&name) {
            return Ok(());
        }

        if !touched.contains_key(txn_id) {
            touched.insert(*txn_id, HashMap::new());
        }

        let blocks = touched.get_mut(txn_id).expect("file block changelist");
        match blocks.entry(name) {
            Entry::Vacant(entry) => {
                let block = Block::new(self.cache.clone(), &self.path, entry.key());
                entry.insert(block);
                Ok(())
            }
            Entry::Occupied(_) => Ok(()),
        }
    }

    async fn get_block(&self, txn_id: &TxnId, name: fs::BlockId) -> TCResult<BlockRead<B>> {
        let mut touched = self.touched.write().await;

        if !touched.contains_key(txn_id) {
            touched.insert(*txn_id, HashMap::new());
        }

        let blocks = touched.get_mut(txn_id).expect("file block changelist");
        match blocks.entry(name) {
            Entry::Vacant(entry) => {
                let block = Block::new(self.cache.clone(), &self.path, entry.key());
                let block = entry.insert(block);
                fs::Block::read(block, txn_id).await
            }
            Entry::Occupied(entry) => fs::Block::read(entry.get(), txn_id).await,
        }
    }

    async fn get_block_mut(&self, txn_id: &TxnId, name: fs::BlockId) -> TCResult<BlockWrite<B>> {
        let mut touched = self.touched.write().await;

        if !touched.contains_key(txn_id) {
            touched.insert(*txn_id, HashMap::new());
        }

        let blocks = touched.get_mut(txn_id).expect("file block changelist");
        match blocks.entry(name) {
            Entry::Vacant(entry) => {
                let block = Block::new(self.cache.clone(), &self.path, entry.key());
                let block = entry.insert(block);
                fs::Block::write(block, txn_id).await
            }
            Entry::Occupied(entry) => fs::Block::write(entry.get(), txn_id).await,
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

        {
            let (listing, touched) = join!(self.listing.read(txn_id), self.touched.read());
            let listing = &listing.expect("file listing");
            if let Some(blocks) = touched.get(txn_id) {
                try_join_all(blocks.values().map(|block| block.prepare(txn_id)))
                    .await
                    .expect("prepare blocks to commit");

                let commits = blocks.iter().map(|(block_id, block)| async move {
                    if listing.contains(block_id) {
                        block.commit(txn_id).await?;
                    } else {
                        self.cache.remove_and_delete(block.path()).await?;
                    }

                    TCResult::Ok(())
                });

                try_join_all(commits).await.expect("commit file blocks");
            }
        }

        self.listing.commit(txn_id).await;
        debug!("committed {:?} at {}", &self.path, txn_id);
    }

    async fn finalize(&self, txn_id: &TxnId) {
        let version = version_path(&self.path, txn_id);
        if version.exists() {
            tokio::fs::remove_dir_all(version)
                .await
                .expect("delete file version");
        }

        let mut touched = self.touched.write().await;
        touched.remove(txn_id);

        self.listing.finalize(txn_id).await;
        debug!("finalized {:?} at {}", &self.path, txn_id);
    }
}

fn touch<B>(
    touched: &mut HashMap<TxnId, HashMap<fs::BlockId, Block<B>>>,
    txn_id: TxnId,
    block_id: fs::BlockId,
    block: Block<B>,
) {
    match touched.entry(txn_id) {
        Entry::Occupied(mut entry) => {
            entry.get_mut().insert(block_id, block);
        }
        Entry::Vacant(entry) => {
            entry.insert(iter::once((block_id, block)).collect());
        }
    }
}

#[inline]
fn block_version_path(file_path: &PathBuf, block_id: &fs::BlockId, txn_id: &TxnId) -> PathBuf {
    let mut path = version_path(file_path, txn_id);
    path.push(block_id.to_string());
    path
}

#[inline]
fn version_path(file_path: &PathBuf, txn_id: &TxnId) -> PathBuf {
    let mut path = file_path.clone();
    path.push(super::VERSION.to_string());
    path.push(txn_id.to_string());
    path
}
