//! A transactional [`File`]

use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{join_all, try_join_all, FutureExt, TryFutureExt};
use log::{debug, trace};
use safecast::AsType;
use tokio::sync::RwLock;
use uuid::Uuid;

use tc_error::*;
use tc_transact::fs::{BlockData, BlockId, FileRead, FileWrite, Store};
use tc_transact::lock::{TxnLock, TxnLockReadGuard, TxnLockWriteGuard};
use tc_transact::{Transact, TxnId};

use super::{io_err, CacheBlock, VERSION};

type Listing = HashSet<BlockId>;
pub type FileReadGuard<B> = FileGuard<B, TxnLockReadGuard<Listing>>;
pub type FileWriteGuard<B> = FileGuard<B, TxnLockWriteGuard<Listing>>;

pub struct BlockReadGuard<B> {
    cache: freqfs::FileReadGuard<CacheBlock, B>,
    #[allow(unused)]
    modified: TxnLockReadGuard<TxnId>,
}

impl<B> Deref for BlockReadGuard<B> {
    type Target = B;

    fn deref(&self) -> &Self::Target {
        self.cache.deref()
    }
}

pub struct BlockWriteGuard<B> {
    cache: freqfs::FileWriteGuard<CacheBlock, B>,
    #[allow(unused)]
    modified: TxnLockWriteGuard<TxnId>,
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

#[derive(Clone)]
pub struct FileGuard<B, L> {
    file: File<B>,
    txn_id: TxnId,
    listing: L,
}

#[async_trait]
impl<B, L> FileRead<B> for FileGuard<B, L>
where
    B: BlockData,
    L: Deref<Target = Listing> + Send + Sync,
    CacheBlock: AsType<B>,
{
    type Read = BlockReadGuard<B>;
    type Write = BlockWriteGuard<B>;

    fn block_ids(&self) -> HashSet<&BlockId> {
        self.listing.iter().collect()
    }

    fn contains(&self, block_id: &BlockId) -> bool {
        self.listing.contains(block_id)
    }

    fn is_empty(&self) -> bool {
        self.listing.is_empty()
    }

    async fn read_block<I>(&self, block_id: I) -> TCResult<Self::Read>
    where
        I: Borrow<BlockId> + Send + Sync,
    {
        let block_id = block_id.borrow();

        if !self.listing.contains(block_id) {
            #[cfg(debug_assertions)]
            panic!("{} is missing block: {}", self.file, block_id);

            #[cfg(not(debug_assertions))]
            return Err(TCError::not_found(block_id));
        }

        trace!("locking block {} for reading...", block_id);

        let modified = {
            let block = {
                let modified = self.file.modified.read().await;
                modified
                    .get(block_id)
                    .expect("block last mutation ID")
                    .clone()
            };

            block.read(self.txn_id).await?
        };

        let name = file_name::<B>(block_id);
        if let Some(block) = self
            .file
            .with_version_read(&self.txn_id, |version| version.get_file(&name))
            .await?
        {
            trace!(
                "block {} already has a version at {}",
                block_id,
                self.txn_id
            );

            let cache = block.read().map_err(io_err).await?;
            let guard = BlockReadGuard { cache, modified };

            trace!("locked block {} for writing at {}", block_id, self.txn_id);

            return Ok(guard);
        } else {
            trace!(
                "creating new version of block {} at {}...",
                block_id,
                self.txn_id
            );
        }

        assert!(*modified < self.txn_id);

        let (size_hint, value) = {
            let block_version = self
                .file
                .with_version_read(&modified, |version| version.get_file(&name))
                .await?
                .expect("block prior value");

            let size_hint = block_version.size_hint().await;

            let value = {
                let value = block_version.read().map_err(io_err).await?;
                B::clone(&*value)
            };

            trace!(
                "got canonical version of block {} to copy at {}...",
                block_id,
                self.txn_id
            );

            (size_hint, value)
        };

        let block = self
            .file
            .with_version_write(&self.txn_id, |mut version| {
                version.create_file(name, value, size_hint).map_err(io_err)
            })
            .await??;

        trace!(
            "created new version of block {} at {}",
            block_id,
            self.txn_id
        );

        let cache = block.read().map_err(io_err).await?;
        trace!("locked block {} for reading...", block_id);
        Ok(BlockReadGuard { cache, modified })
    }

    async fn write_block<I>(&self, block_id: I) -> TCResult<Self::Write>
    where
        I: Borrow<BlockId> + Send + Sync,
    {
        let block_id = block_id.borrow();

        if !self.listing.contains(block_id) {
            #[cfg(debug_assertions)]
            panic!("{} is missing block: {}", self.file, block_id);

            #[cfg(not(debug_assertions))]
            return Err(TCError::not_found(block_id));
        }

        let txn_id = self.txn_id;

        trace!("locking block {} for writing at {}...", block_id, txn_id);

        let mut modified = {
            {
                let block = {
                    let blocks = self.file.modified.read().await;

                    blocks
                        .get(block_id)
                        .expect("block last mutation ID")
                        .clone()
                };

                block.write(txn_id).await?
            }
        };

        trace!("block {} was last modified at {}...", block_id, *modified);
        (*modified) = self.txn_id;

        let name = file_name::<B>(block_id);

        if let Some(block) = self
            .file
            .with_version_read(&txn_id, |version| version.get_file(&name))
            .await?
        {
            trace!("block {} already has a version at {}", block_id, txn_id);

            let cache = block.write().map_err(io_err).await?;
            let guard = BlockWriteGuard { cache, modified };

            trace!("locked block {} for writing at {}", block_id, txn_id);

            return Ok(guard);
        }

        // a write can only happen before a commit
        // therefore the canonical version must be current

        let block_canon = {
            let canon = self.file.canon.read().await;
            canon.get_file(&name).expect("canonical block")
        };

        let size_hint = block_canon.size_hint().await;
        let value = {
            let value = block_canon.read().map_err(io_err).await?;
            B::clone(&*value)
        };

        trace!(
            "got canonical version of block {} to copy at {}",
            block_id,
            txn_id
        );

        let block = self
            .file
            .with_version_write(&txn_id, |mut version| {
                version.create_file(name, value, size_hint).map_err(io_err)
            })
            .await??;

        let cache = block.write().map_err(io_err).await?;
        let guard = BlockWriteGuard { cache, modified };

        trace!("locked block {} for writing at {}", block_id, txn_id);

        Ok(guard)
    }
}

#[async_trait]
impl<B, L> FileWrite<B> for FileGuard<B, L>
where
    B: BlockData,
    L: DerefMut<Target = Listing> + Send + Sync,
    CacheBlock: AsType<B>,
{
    async fn create_block(
        &mut self,
        block_id: BlockId,
        initial_value: B,
        size_hint: usize,
    ) -> TCResult<Self::Write> {
        if self.listing.contains(&block_id) {
            #[cfg(debug_assertions)]
            panic!("{} already has a block with ID {}", self.file, block_id);

            #[cfg(not(debug_assertions))]
            return Err(TCError::bad_request("block already exists", block_id));
        }

        let txn_id = self.txn_id;

        let (block, modified) = {
            let mut modified = self.file.modified.write().await;
            let mut version = self.file.version_write(&txn_id).await?;

            let lock = TxnLock::new(format!("block {}", block_id), txn_id);
            let write_lock = lock.try_write(txn_id).expect("block last modified");
            modified.insert(block_id.clone(), lock);

            let name = format!("{}.{}", block_id, B::ext());
            let block = version
                .create_file(name, initial_value, Some(size_hint))
                .map_err(io_err)?;

            self.listing.insert(block_id);

            (block, write_lock)
        };

        block
            .write()
            .map_ok(move |cache| BlockWriteGuard { cache, modified })
            .map_err(io_err)
            .await
    }

    async fn create_block_unique(
        &mut self,
        initial_value: B,
        size_hint: usize,
    ) -> TCResult<(BlockId, Self::Write)> {
        let block_id: BlockId = loop {
            let name = Uuid::new_v4().into();
            if !self.listing.contains(&name) {
                break name;
            }
        };

        self.create_block(block_id.clone(), initial_value, size_hint)
            .map_ok(move |block| (block_id, block))
            .await
    }

    async fn delete_block(&mut self, block_id: BlockId) -> TCResult<()> {
        let mut modified = self.file.modified.write().await;
        if let Some(last_mutation) = modified.get_mut(&block_id) {
            *last_mutation.write(self.txn_id).await? = self.txn_id;

            // keep the version directory in sync in case create_block is called later
            // with the same block_id
            self.file
                .with_version_write(&self.txn_id, |mut version| {
                    version.delete(file_name::<B>(&block_id))
                })
                .await?;
        }

        self.listing.remove(&block_id);
        Ok(())
    }

    async fn copy_from<O: FileRead<B>>(&mut self, other: &O, truncate: bool) -> TCResult<()> {
        if truncate {
            self.truncate().await?;
        }

        for block_id in other.block_ids() {
            // TODO: provide a better size hint
            let block = other.read_block(block_id).map_ok(|b| (*b).clone()).await?;
            if self.contains(block_id) {
                let mut dest = self.write_block(block_id).await?;
                *dest = block;
            } else {
                self.create_block(block_id.clone(), block, 0).await?;
            }
        }

        Ok(())
    }

    async fn truncate(&mut self) -> TCResult<()> {
        let mut version = self.file.version_write(&self.txn_id).await?;
        for block_id in self.listing.drain() {
            version.delete(file_name::<B>(&block_id));
        }

        Ok(())
    }
}

#[derive(Clone)]
pub struct File<B> {
    canon: freqfs::DirLock<CacheBlock>,
    versions: freqfs::DirLock<CacheBlock>,
    listing: TxnLock<Listing>,
    modified: Arc<RwLock<HashMap<BlockId, TxnLock<TxnId>>>>,
    phantom: PhantomData<B>,
}

impl<B: BlockData> File<B>
where
    CacheBlock: AsType<B>,
{
    #[cfg(debug_assertions)]
    fn lock_name(fs_dir: &freqfs::Dir<CacheBlock>) -> String {
        format!("block list of file {:?}", &*fs_dir)
    }

    #[cfg(not(debug_assertions))]
    fn lock_name(_fs_dir: &freqfs::Dir<CacheBlock>) -> String {
        "block list of transactional file".to_string()
    }

    pub fn new(canon: freqfs::DirLock<CacheBlock>) -> TCResult<Self> {
        let mut fs_dir = canon
            .try_write()
            .map_err(|cause| TCError::internal(format!("new file is already in use: {}", cause)))?;

        if fs_dir.len() > 0 {
            return Err(TCError::internal("new file is not empty"));
        }

        Ok(Self {
            canon,
            listing: TxnLock::new(Self::lock_name(&fs_dir), Listing::new()),
            modified: Arc::new(Default::default()),
            versions: fs_dir.create_dir(VERSION.to_string()).map_err(io_err)?,
            phantom: PhantomData,
        })
    }

    pub(super) async fn load(canon: freqfs::DirLock<CacheBlock>, txn_id: TxnId) -> TCResult<Self> {
        let mut fs_dir = canon.write().await;

        debug!("File::load {:?}", &*fs_dir);

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

        Ok(Self {
            canon,
            versions,
            listing: TxnLock::new(Self::lock_name(&fs_dir), present),
            modified: Arc::new(RwLock::new(blocks)),
            phantom: Default::default(),
        })
    }

    pub fn into_inner(self) -> freqfs::DirLock<CacheBlock> {
        self.canon
    }

    async fn version(&self, txn_id: &TxnId) -> TCResult<freqfs::DirLock<CacheBlock>> {
        let mut versions = self.versions.write().await;
        versions
            .get_or_create_dir(txn_id.to_string())
            .map_err(io_err)
    }

    async fn version_read(&self, txn_id: &TxnId) -> TCResult<freqfs::DirReadGuard<CacheBlock>> {
        let version = self.version(txn_id).await?;
        version.read().map(Ok).await
    }

    async fn version_write(&self, txn_id: &TxnId) -> TCResult<freqfs::DirWriteGuard<CacheBlock>> {
        let version = self.version(txn_id).await?;
        version.write().map(Ok).await
    }

    async fn with_version_read<F, T>(&self, txn_id: &TxnId, then: F) -> TCResult<T>
    where
        F: FnOnce(freqfs::DirReadGuard<CacheBlock>) -> T,
    {
        self.version_read(txn_id).map_ok(then).await
    }

    async fn with_version_write<F, T>(&self, txn_id: &TxnId, then: F) -> TCResult<T>
    where
        F: FnOnce(freqfs::DirWriteGuard<CacheBlock>) -> T,
    {
        self.version_write(txn_id).map_ok(then).await
    }
}

#[async_trait]
impl<B> tc_transact::fs::File<B> for File<B>
where
    B: BlockData,
    CacheBlock: AsType<B>,
{
    type Read = FileReadGuard<B>;
    type Write = FileWriteGuard<B>;

    async fn read(&self, txn_id: TxnId) -> TCResult<Self::Read> {
        debug!("File::read");

        self.listing
            .read(txn_id)
            .map_ok(move |listing| {
                trace!("locked file for reading at {}", txn_id);

                FileGuard {
                    file: self.clone(),
                    txn_id,
                    listing,
                }
            })
            .await
    }

    async fn write(&self, txn_id: TxnId) -> TCResult<Self::Write> {
        debug!("File::write");

        self.listing
            .write(txn_id)
            .map_ok(move |listing| {
                trace!("locked file for writing at {}", txn_id);

                FileGuard {
                    file: self.clone(),
                    txn_id,
                    listing,
                }
            })
            .await
    }
}

impl<B: BlockData> Store for File<B> {}

#[async_trait]
impl<B: BlockData> Transact for File<B>
where
    CacheBlock: AsType<B>,
{
    async fn commit(&self, txn_id: &TxnId) {
        debug!("File::commit");

        let modified = self.modified.read().await;

        self.listing.commit(txn_id).await;
        trace!("File::commit committed block listing");

        let block_commits = modified
            .values()
            .map(|last_mutation| last_mutation.commit(txn_id));

        join_all(block_commits).await;

        {
            let present = self.listing.read(*txn_id).await.expect("file block list");

            let version = {
                let fs_dir = self.versions.read().await;
                if let Some(version) = fs_dir.get_dir(&txn_id.to_string()) {
                    Some(version.read().await)
                } else {
                    None
                }
            };

            let mut canon = self.canon.write().await;
            let mut synchronize = Vec::with_capacity(present.len());
            for block_id in modified.keys() {
                let name = file_name::<B>(block_id);
                if present.contains(block_id) {
                    if let Some(version) = &version {
                        if let Some(version) = version.get_file(&name) {
                            let block: freqfs::FileReadGuard<CacheBlock, B> =
                                version.read().await.expect("block version");

                            let canon = if let Some(canon) = canon.get_file(&name) {
                                *canon.write().await.expect("canonical block") = (*block).clone();
                                canon
                            } else {
                                let size_hint = version.size_hint().await;
                                canon
                                    .create_file(name, (*block).clone(), size_hint)
                                    .expect("new canonical block")
                            };

                            synchronize.push(async move { canon.sync(true).await });
                        } else {
                            trace!("block {} has no version to commit at {}", block_id, txn_id);
                        }
                    }
                } else {
                    canon.delete(name);
                }
            }

            try_join_all(synchronize)
                .await
                .expect("sync block contents to disk");
        }

        self.canon
            .sync(false)
            .await
            .expect("sync file content to disk");
    }

    async fn finalize(&self, txn_id: &TxnId) {
        debug!("File::finalize");

        {
            let modified = self.modified.read().await;
            join_all(
                modified
                    .values()
                    .map(|last_commit_id| last_commit_id.finalize(txn_id)),
            )
            .await;
        }

        self.listing.finalize(txn_id).await;

        self.versions
            .write()
            .map(|mut version| version.delete(txn_id.to_string()))
            .await;
    }
}

impl<B: Send + Sync + 'static> fmt::Display for File<B> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "file of {} blocks", std::any::type_name::<B>())
    }
}

fn file_name<B: BlockData>(block_id: &BlockId) -> String {
    format!("{}.{}", block_id, B::ext())
}
