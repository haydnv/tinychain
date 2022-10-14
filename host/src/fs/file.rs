//! A transactional [`File`]

use std::borrow::Borrow;
use std::collections::HashSet;
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

use async_trait::async_trait;
use futures::future::{join_all, FutureExt, TryFutureExt};
use log::{debug, trace};
use safecast::AsType;
use uuid::Uuid;

use tc_error::*;
use tc_transact::fs::{
    BlockData, BlockId, BlockRead, BlockReadExclusive, BlockWrite, FileRead, FileReadExclusive,
    FileWrite, Store,
};
use tc_transact::lock::*;
use tc_transact::{Transact, TxnId};
use tcgeneric::Map;

use super::{io_err, CacheBlock, VERSION};

pub type FileReadGuard<B> = FileGuard<B, TxnMapLockReadGuard<TxnLock<TxnId>>>;
pub type FileReadGuardExclusive<B> = FileGuard<B, TxnMapLockReadGuardExclusive<TxnLock<TxnId>>>;
pub type FileWriteGuard<B> = FileGuard<B, TxnMapLockWriteGuard<TxnLock<TxnId>>>;

/// A read lock guard for a block in a [`File`]
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

impl<B: BlockData> BlockRead<B> for BlockReadGuard<B> {}

/// An exclusive read lock guard for a block in a [`File`]
pub struct BlockReadGuardExclusive<B> {
    cache: freqfs::FileWriteGuard<CacheBlock, B>,
    txn_id: TxnId,
    #[allow(unused)]
    modified: TxnLockReadGuardExclusive<TxnId>,
}

impl<B> Deref for BlockReadGuardExclusive<B> {
    type Target = B;

    fn deref(&self) -> &Self::Target {
        &self.cache
    }
}

impl<B: BlockData> BlockReadExclusive<B> for BlockReadGuardExclusive<B>
where
    CacheBlock: AsType<B>,
{
    type File = File<B>;

    fn upgrade(self) -> <Self::File as tc_transact::fs::File<B>>::BlockWrite {
        let mut modified = self.modified.upgrade();
        assert!(*modified <= self.txn_id);
        *modified = self.txn_id;

        BlockWriteGuard {
            cache: self.cache,
            txn_id: self.txn_id,
            modified,
        }
    }
}

/// A write lock guard for a block in a [`File`]
pub struct BlockWriteGuard<B> {
    cache: freqfs::FileWriteGuard<CacheBlock, B>,
    txn_id: TxnId,
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

impl<B: BlockData> BlockWrite<B> for BlockWriteGuard<B>
where
    CacheBlock: AsType<B>,
{
    type File = File<B>;

    fn downgrade(self) -> <Self::File as tc_transact::fs::File<B>>::BlockReadExclusive {
        BlockReadGuardExclusive {
            cache: self.cache,
            txn_id: self.txn_id,
            modified: self.modified.downgrade(),
        }
    }
}

/// A lock guard for the contents (i.e. block listing) of a [`File`]
#[derive(Clone)]
pub struct FileGuard<B, L> {
    file: File<B>,
    txn_id: TxnId,
    blocks: L,
}

impl<B, L> FileGuard<B, L>
where
    B: BlockData,
    L: TxnMapRead<TxnLock<TxnId>>,
    CacheBlock: AsType<B>,
{
    async fn last_modified(&self, block_id: &BlockId) -> TCResult<TxnLock<TxnId>> {
        self.blocks
            .get(block_id)
            .ok_or_else(|| TCError::not_found(block_id))
    }

    async fn block_version(
        &self,
        last_modified: &TxnId,
        block_id: &BlockId,
    ) -> TCResult<freqfs::FileLock<CacheBlock>> {
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

            return Ok(block);
        } else {
            trace!(
                "creating new version of block {} at {}...",
                block_id,
                self.txn_id
            );
        }

        let (size_hint, value) = {
            let block_version = self
                .file
                .with_version_read(last_modified, |version| version.get_file(&name))
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

        Ok(block)
    }
}

#[async_trait]
impl<B, L> FileRead<B> for FileGuard<B, L>
where
    B: BlockData,
    L: TxnMapRead<TxnLock<TxnId>> + Send + Sync,
    CacheBlock: AsType<B>,
{
    type File = File<B>;

    fn block_ids(&self) -> HashSet<&BlockId> {
        self.blocks.iter().map(|(key, _)| key).collect()
    }

    fn contains(&self, block_id: &BlockId) -> bool {
        self.blocks.contains_key(block_id)
    }

    fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    async fn read_block<I>(&self, block_id: I) -> TCResult<BlockReadGuard<B>>
    where
        I: Borrow<BlockId> + Send + Sync,
    {
        let block_id = block_id.borrow();
        debug!("FileGuard::read_block {}", block_id);

        let modified = {
            let modified = self.last_modified(block_id).await?;
            modified.read(self.txn_id).await?
        };

        let block = self.block_version(&modified, block_id).await?;
        let cache = block.read().map_err(io_err).await?;
        trace!("locked block {} for reading...", block_id);
        Ok(BlockReadGuard { cache, modified })
    }

    async fn read_block_exclusive<I>(
        &self,
        block_id: I,
    ) -> TCResult<<Self::File as tc_transact::fs::File<B>>::BlockReadExclusive>
    where
        I: Borrow<BlockId> + Send + Sync,
    {
        let block_id = block_id.borrow();
        let txn_id = self.txn_id;

        debug!("FileGuard::read_block_exclusive {}", block_id);

        let modified = {
            let modified = self.last_modified(block_id).await?;
            modified.read_exclusive(txn_id).await?
        };

        let block = self.block_version(&modified, block_id).await?;
        let cache = block.write().map_err(io_err).await?;
        trace!("locked block {} for reading...", block_id);

        Ok(BlockReadGuardExclusive {
            cache,
            txn_id,
            modified,
        })
    }

    async fn write_block<I>(&self, block_id: I) -> TCResult<BlockWriteGuard<B>>
    where
        I: Borrow<BlockId> + Send + Sync,
    {
        let block_id = block_id.borrow();
        let txn_id = self.txn_id;

        debug!("FileGuard::write_block {}", block_id);

        let mut modified = {
            let modified = self.last_modified(block_id).await?;
            modified.write(txn_id).await?
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
            let guard = BlockWriteGuard {
                cache,
                txn_id,
                modified,
            };

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
        let guard = BlockWriteGuard {
            cache,
            txn_id,
            modified,
        };

        trace!("locked block {} for writing at {}", block_id, txn_id);

        Ok(guard)
    }
}

impl<B> FileReadExclusive<B> for FileReadGuardExclusive<B>
where
    B: BlockData,
    CacheBlock: AsType<B>,
{
    fn upgrade(self) -> <Self::File as tc_transact::fs::File<B>>::Write {
        FileGuard {
            file: self.file,
            txn_id: self.txn_id,
            blocks: self.blocks.upgrade(),
        }
    }
}

#[async_trait]
impl<B> FileWrite<B> for FileWriteGuard<B>
where
    B: BlockData,
    CacheBlock: AsType<B>,
{
    fn downgrade(self) -> <Self::File as tc_transact::fs::File<B>>::ReadExclusive {
        FileGuard {
            file: self.file,
            txn_id: self.txn_id,
            blocks: self.blocks.downgrade(),
        }
    }

    async fn create_block(
        &mut self,
        block_id: BlockId,
        initial_value: B,
        size_hint: usize,
    ) -> TCResult<BlockWriteGuard<B>> {
        if self.blocks.contains_key(&block_id) {
            #[cfg(debug_assertions)]
            panic!("{} already has a block with ID {}", self.file, block_id);

            #[cfg(not(debug_assertions))]
            return Err(TCError::bad_request("block already exists", block_id));
        }

        let txn_id = self.txn_id;

        let (block, modified) = {
            let mut version = self.file.version_write(&txn_id).await?;

            let lock = TxnLock::new(format!("block {}", block_id), txn_id);
            let write_lock = lock.try_write(txn_id).expect("block last modified");
            self.blocks.insert(block_id.clone(), lock);

            let name = format!("{}.{}", block_id, B::ext());
            let block = version
                .create_file(name, initial_value, Some(size_hint))
                .map_err(io_err)?;

            (block, write_lock)
        };

        block
            .write()
            .map_ok(move |cache| BlockWriteGuard {
                cache,
                txn_id,
                modified,
            })
            .map_err(io_err)
            .await
    }

    async fn create_block_unique(
        &mut self,
        initial_value: B,
        size_hint: usize,
    ) -> TCResult<(BlockId, BlockWriteGuard<B>)> {
        let block_id: BlockId = loop {
            let name = Uuid::new_v4().into();
            if !self.blocks.contains_key(&name) {
                break name;
            }
        };

        self.create_block(block_id.clone(), initial_value, size_hint)
            .map_ok(move |block| (block_id, block))
            .await
    }

    async fn delete_block(&mut self, block_id: BlockId) -> TCResult<()> {
        if let Some(last_mutation) = self.blocks.get(&block_id) {
            *last_mutation.write(self.txn_id).await? = self.txn_id;

            // keep the version directory in sync in case create_block is called later
            // with the same block_id
            self.file
                .with_version_write(&self.txn_id, |mut version| {
                    version.delete(file_name::<B>(&block_id))
                })
                .await?;
        }

        self.blocks.remove(&block_id);
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
        for (block_id, _) in self.blocks.drain() {
            version.delete(file_name::<B>(&block_id));
        }

        Ok(())
    }
}

/// A transactional file
#[derive(Clone)]
pub struct File<B> {
    canon: freqfs::DirLock<CacheBlock>,
    versions: freqfs::DirLock<CacheBlock>,
    blocks: TxnMapLock<TxnLock<TxnId>>,
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
            versions: fs_dir.create_dir(VERSION.to_string()).map_err(io_err)?,
            blocks: TxnMapLock::new(Self::lock_name(&fs_dir)),
            phantom: PhantomData,
        })
    }

    pub(super) async fn load(canon: freqfs::DirLock<CacheBlock>, txn_id: TxnId) -> TCResult<Self> {
        let mut fs_dir = canon.write().await;

        debug!("File::load {:?}", &*fs_dir);

        let versions = fs_dir
            .get_or_create_dir(VERSION.to_string())
            .map_err(io_err)?;

        let mut blocks = Map::new();
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

            let lock_name = format!("block {}", block_id);
            blocks.insert(block_id, TxnLock::new(lock_name, txn_id));

            version
                .create_file(name.clone(), contents, size_hint)
                .map_err(io_err)?;
        }

        Ok(Self {
            canon,
            versions,
            blocks: TxnMapLock::with_contents(Self::lock_name(&fs_dir), blocks),
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
    type ReadExclusive = FileReadGuardExclusive<B>;
    type Write = FileWriteGuard<B>;
    type BlockRead = BlockReadGuard<B>;
    type BlockReadExclusive = BlockReadGuardExclusive<B>;
    type BlockWrite = BlockWriteGuard<B>;

    async fn read(&self, txn_id: TxnId) -> TCResult<Self::Read> {
        debug!("File::read");

        self.blocks
            .read(txn_id)
            .map_ok(move |blocks| {
                trace!("locked file for reading at {}", txn_id);

                FileGuard {
                    file: self.clone(),
                    txn_id,
                    blocks,
                }
            })
            .await
    }

    async fn read_exclusive(&self, txn_id: TxnId) -> TCResult<Self::ReadExclusive> {
        debug!("File::read_exclusive");

        self.blocks
            .read_exclusive(txn_id)
            .map_ok(move |blocks| {
                trace!("locked file for reading at {}", txn_id);

                FileGuard {
                    file: self.clone(),
                    txn_id,
                    blocks,
                }
            })
            .await
    }

    async fn write(&self, txn_id: TxnId) -> TCResult<Self::Write> {
        debug!("File::write");

        self.blocks
            .write(txn_id)
            .map_ok(move |blocks| {
                trace!("locked file for writing at {}", txn_id);

                FileGuard {
                    file: self.clone(),
                    txn_id,
                    blocks,
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
    type Commit = TxnMapLockCommitGuard<TxnLock<TxnId>>;

    async fn commit(&self, txn_id: &TxnId) -> Self::Commit {
        debug!("File::commit");

        let blocks = self.blocks.commit(txn_id).await;
        trace!("File::commit committed block listing");

        {
            let version = {
                let fs_dir = self.versions.read().await;
                if let Some(version) = fs_dir.get_dir(&txn_id.to_string()) {
                    version.read().await
                } else {
                    // in this case no blocks have been modified, so there's nothing to commit
                    return blocks;
                }
            };

            let mut canon = self.canon.write().await;

            for (block_id, last_modified) in blocks.iter() {
                let last_modified = last_modified.commit(txn_id).await;

                if &*last_modified == txn_id {
                    let name = file_name::<B>(block_id);
                    let version = version.get_file(&name).expect("block version lock");

                    let block = version.read().await.expect("block version");

                    let canon = if let Some(canon) = canon.get_file(&name) {
                        *canon.write().await.expect("canonical block") = (*block).clone();
                        canon
                    } else {
                        let size_hint = version.size_hint().await;

                        canon
                            .create_file(name, (*block).clone(), size_hint)
                            .expect("new canonical block")
                    };

                    canon.sync(true).await.expect("sync canonical block");
                    trace!("File::commit canonical block {}", block_id);
                } else {
                    trace!(
                        "File::commit skipping block {} since it was not modified",
                        block_id
                    );
                }
            }

            for (name, _) in version.iter() {
                let block_id = block_id(name).expect("block ID");
                if !blocks.contains_key(&block_id) {
                    trace!("File::commit delete block {}", block_id);
                    canon.delete(name.clone());
                }
            }
        }

        self.canon
            .sync(false)
            .await
            .expect("sync file content to disk");

        blocks
    }

    async fn finalize(&self, txn_id: &TxnId) {
        debug!("File::finalize");

        {
            let blocks = self.blocks.read(*txn_id).await.expect("file block listing");
            let finalize = blocks
                .iter()
                .map(|(_, last_commit_id)| async move { last_commit_id.finalize(txn_id).await });

            join_all(finalize).await;
        }

        self.blocks.finalize(txn_id).await;

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

#[inline]
fn block_id(name: &str) -> TCResult<BlockId> {
    let i = name
        .rfind('.')
        .ok_or_else(|| TCError::bad_request("invalid block name", name))?;
    let name = std::str::from_utf8(&name.as_bytes()[..i]).map_err(TCError::internal)?;
    name.parse()
        .map_err(|cause| TCError::bad_request("invalid block name", cause))
}

#[inline]
fn file_name<B: BlockData>(block_id: &BlockId) -> String {
    format!("{}.{}", block_id, B::ext())
}
