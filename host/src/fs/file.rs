//! A transactional [`File`]

use std::borrow::Borrow;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::str::FromStr;

use async_trait::async_trait;
use destream::de::Error;
use futures::future::{join_all, FutureExt, TryFutureExt};
use log::{debug, trace};
use safecast::AsType;
use uuid::Uuid;

use tc_error::*;
use tc_transact::fs::{
    BlockData, BlockRead, BlockReadExclusive, BlockWrite, FileRead, FileReadExclusive, FileWrite,
    Store,
};
use tc_transact::lock::*;
use tc_transact::{Transact, TxnId};

use super::{io_err, CacheBlock, VERSION};

/// A read guard on a [`File`]
pub type FileReadGuard<K, B> = FileGuard<K, B, TxnLockReadGuard<BTreeMap<K, TxnLock<TxnId>>>>;

/// An exclusive read guard on a [`File`]
pub type FileReadGuardExclusive<K, B> =
    FileGuard<K, B, TxnLockReadGuardExclusive<BTreeMap<K, TxnLock<TxnId>>>>;

/// A write guard on a [`File`]
pub type FileWriteGuard<K, B> = FileGuard<K, B, TxnLockWriteGuard<BTreeMap<K, TxnLock<TxnId>>>>;

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
pub struct BlockReadGuardExclusive<K, B> {
    cache: freqfs::FileWriteGuard<CacheBlock, B>,
    txn_id: TxnId,
    #[allow(unused)]
    modified: TxnLockReadGuardExclusive<TxnId>,
    phantom: PhantomData<K>,
}

impl<K, B> Deref for BlockReadGuardExclusive<K, B> {
    type Target = B;

    fn deref(&self) -> &Self::Target {
        &self.cache
    }
}

impl<K, B: BlockData> BlockReadExclusive for BlockReadGuardExclusive<K, B>
where
    K: FromStr + fmt::Display + Ord + PartialEq + Clone + Send + Sync + 'static,
    <K as FromStr>::Err: std::error::Error + fmt::Display,
    CacheBlock: AsType<B>,
{
    type File = File<K, B>;

    fn upgrade(self) -> BlockWriteGuard<K, B> {
        let mut modified = self.modified.upgrade();
        assert!(*modified <= self.txn_id);
        *modified = self.txn_id;

        BlockWriteGuard {
            cache: self.cache,
            txn_id: self.txn_id,
            modified,
            phantom: PhantomData,
        }
    }
}

/// A write lock guard for a block in a [`File`]
pub struct BlockWriteGuard<K, B> {
    cache: freqfs::FileWriteGuard<CacheBlock, B>,
    txn_id: TxnId,
    modified: TxnLockWriteGuard<TxnId>,
    phantom: PhantomData<K>,
}

impl<K, B> Deref for BlockWriteGuard<K, B> {
    type Target = B;

    fn deref(&self) -> &Self::Target {
        self.cache.deref()
    }
}

impl<K, B> DerefMut for BlockWriteGuard<K, B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.cache.deref_mut()
    }
}

impl<K, B> BlockWrite for BlockWriteGuard<K, B>
where
    K: FromStr + fmt::Display + Ord + Clone + Send + Sync + 'static,
    B: BlockData,
    <K as FromStr>::Err: std::error::Error + fmt::Display,
    CacheBlock: AsType<B>,
{
    type File = File<K, B>;

    fn downgrade(self) -> BlockReadGuardExclusive<K, B> {
        BlockReadGuardExclusive {
            cache: self.cache,
            txn_id: self.txn_id,
            modified: self.modified.downgrade(),
            phantom: PhantomData,
        }
    }
}

/// A lock guard for the contents (i.e. block listing) of a [`File`]
#[derive(Clone)]
pub struct FileGuard<K, B, L> {
    file: File<K, B>,
    txn_id: TxnId,
    blocks: L,
}

impl<K, B, L> FileGuard<K, B, L>
where
    K: FromStr + fmt::Display + Ord + Clone,
    B: BlockData,
    L: Deref<Target = BTreeMap<K, TxnLock<TxnId>>> + Send + Sync,
    <K as FromStr>::Err: std::error::Error + fmt::Display,
    CacheBlock: AsType<B>,
{
    fn last_modified(&self, block_id: &K) -> TCResult<TxnLock<TxnId>> {
        self.blocks
            .get(block_id)
            .cloned()
            .ok_or_else(|| TCError::not_found(block_id))
    }

    async fn block_version(
        &self,
        last_modified: &TxnId,
        block_id: &K,
    ) -> TCResult<freqfs::FileLock<CacheBlock>> {
        let name = file_name::<_, K, B>(block_id);

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

        let value = {
            let block_version = self
                .file
                .with_version_read(last_modified, |version| version.get_file(&name))
                .await?
                .expect("block prior value");

            let value = {
                let value = block_version.read().map_err(io_err).await?;
                B::clone(&*value)
            };

            trace!(
                "got canonical version of block {} to copy at {}...",
                block_id,
                self.txn_id
            );

            value
        };

        let size_hint = value.get_size();
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

    fn try_block_version(
        &self,
        last_modified: &TxnId,
        block_id: &K,
    ) -> TCResult<freqfs::FileLock<CacheBlock>> {
        let name = file_name::<_, K, B>(block_id);

        if let Some(block) = self
            .file
            .with_try_version_read(&self.txn_id, |version| version.get_file(&name))?
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

        let value = {
            let block_version = self
                .file
                .with_try_version_read(last_modified, |version| version.get_file(&name))?
                .expect("block prior value");

            {
                let value = block_version.try_read().map_err(io_err)?;
                B::clone(&*value)
            }
        };

        let size_hint = value.get_size();
        let block = self
            .file
            .with_try_version_write(&self.txn_id, |mut version| {
                version.create_file(name, value, size_hint).map_err(io_err)
            })??;

        trace!(
            "created new version of block {} at {}",
            block_id,
            self.txn_id
        );

        Ok(block)
    }
}

#[async_trait]
impl<K, B, L> FileRead for FileGuard<K, B, L>
where
    K: FromStr + fmt::Display + Ord + PartialEq + Clone + Send + Sync + 'static,
    B: BlockData,
    L: Deref<Target = BTreeMap<K, TxnLock<TxnId>>> + Send + Sync,
    <K as FromStr>::Err: std::error::Error + fmt::Display,
    CacheBlock: AsType<B>,
{
    type File = File<K, B>;

    fn block_ids(&self) -> BTreeSet<K> {
        self.blocks.keys().cloned().collect()
    }

    fn contains<Q: Borrow<K>>(&self, block_id: Q) -> bool {
        self.blocks.contains_key(block_id.borrow())
    }

    fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    async fn read_block<Q>(&self, block_id: Q) -> TCResult<BlockReadGuard<B>>
    where
        Q: Borrow<K> + Send + Sync,
    {
        let block_id = block_id.borrow();
        debug!("FileGuard::read_block {}", block_id);

        let modified = {
            let modified = self.last_modified(block_id)?;
            modified.read(self.txn_id).await?
        };

        let block = self.block_version(&modified, block_id).await?;
        let cache = block.read().map_err(io_err).await?;
        trace!("locked block {} for reading...", block_id);
        Ok(BlockReadGuard { cache, modified })
    }

    fn try_read_block<Q>(
        &self,
        block_id: Q,
    ) -> TCResult<<Self::File as tc_transact::fs::File>::BlockRead>
    where
        Q: Borrow<<Self::File as tc_transact::fs::File>::Key> + Send + Sync,
    {
        let block_id = block_id.borrow();
        debug!("FileGuard::try_read_block {}", block_id);

        let modified = {
            let modified = self.last_modified(block_id)?;
            modified.try_read(self.txn_id)?
        };

        let block = self.try_block_version(&modified, block_id)?;
        let cache = block.try_read().map_err(io_err)?;
        trace!("locked block {} for reading...", block_id);
        Ok(BlockReadGuard { cache, modified })
    }

    async fn read_block_exclusive<Q>(&self, block_id: Q) -> TCResult<BlockReadGuardExclusive<K, B>>
    where
        Q: Borrow<K> + Send + Sync,
    {
        let block_id = block_id.borrow();
        let txn_id = self.txn_id;

        debug!("FileGuard::read_block_exclusive {}", block_id);

        let modified = {
            let modified = self.last_modified(block_id)?;
            modified.read_exclusive(txn_id).await?
        };

        let block = self.block_version(&modified, block_id).await?;
        let cache = block.write().map_err(io_err).await?;
        trace!("locked block {} for exclusive reading...", block_id);

        Ok(BlockReadGuardExclusive {
            cache,
            txn_id,
            modified,
            phantom: PhantomData,
        })
    }

    fn try_read_block_exclusive<Q>(
        &self,
        block_id: Q,
    ) -> TCResult<<Self::File as tc_transact::fs::File>::BlockReadExclusive>
    where
        Q: Borrow<<Self::File as tc_transact::fs::File>::Key> + Send + Sync,
    {
        let block_id = block_id.borrow();
        let txn_id = self.txn_id;

        debug!("FileGuard::try_read_block_exclusive {}", block_id);

        let modified = {
            let modified = self.last_modified(block_id)?;
            modified.try_read_exclusive(self.txn_id)?
        };

        let block = self.try_block_version(&modified, block_id)?;
        let cache = block.try_write().map_err(io_err)?;
        trace!("locked block {} for exclusive reading...", block_id);

        Ok(BlockReadGuardExclusive {
            cache,
            txn_id,
            modified,
            phantom: PhantomData,
        })
    }

    async fn write_block<Q>(&self, block_id: Q) -> TCResult<BlockWriteGuard<K, B>>
    where
        Q: Borrow<K> + Send + Sync,
    {
        let block_id = block_id.borrow();
        let txn_id = self.txn_id;

        debug!("FileGuard::write_block {}", block_id);

        let mut modified = {
            let modified = self.last_modified(block_id)?;
            modified.write(txn_id).await?
        };

        trace!("block {} was last modified at {}...", block_id, *modified);
        (*modified) = self.txn_id;

        let name = file_name::<_, K, B>(block_id);

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
                phantom: PhantomData,
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

        let value = {
            let value = block_canon.read().map_err(io_err).await?;
            B::clone(&*value)
        };

        trace!(
            "got canonical version of block {} to copy at {}",
            block_id,
            txn_id
        );

        let size_hint = value.get_size();
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
            phantom: PhantomData,
        };

        trace!("locked block {} for writing at {}", block_id, txn_id);

        Ok(guard)
    }

    fn try_write_block<Q>(
        &self,
        block_id: Q,
    ) -> TCResult<<Self::File as tc_transact::fs::File>::BlockWrite>
    where
        Q: Borrow<<Self::File as tc_transact::fs::File>::Key> + Send + Sync,
    {
        let block_id = block_id.borrow();
        let txn_id = self.txn_id;

        debug!("FileGuard::try_write_block {}", block_id);

        let mut modified = {
            let modified = self.last_modified(block_id)?;
            modified.try_write(txn_id)?
        };

        trace!("block {} was last modified at {}...", block_id, *modified);
        (*modified) = self.txn_id;

        let name = file_name::<_, K, B>(block_id);

        if let Some(block) = self
            .file
            .with_try_version_read(&txn_id, |version| version.get_file(&name))?
        {
            trace!("block {} already has a version at {}", block_id, txn_id);

            let cache = block.try_write().map_err(io_err)?;
            let guard = BlockWriteGuard {
                cache,
                txn_id,
                modified,
                phantom: PhantomData,
            };

            trace!("locked block {} for writing at {}", block_id, txn_id);

            return Ok(guard);
        }

        // a write can only happen before a commit
        // therefore the canonical version must be current

        let block_canon = {
            let canon = self.file.canon.try_read().map_err(io_err)?;
            canon.get_file(&name).expect("canonical block")
        };

        let value = {
            let value = block_canon.try_read().map_err(io_err)?;
            B::clone(&*value)
        };

        trace!(
            "got canonical version of block {} to copy at {}",
            block_id,
            txn_id
        );

        let size_hint = value.get_size();
        let block = self.file.with_try_version_write(&txn_id, |mut version| {
            version.create_file(name, value, size_hint).map_err(io_err)
        })??;

        let cache = block.try_write().map_err(io_err)?;
        let guard = BlockWriteGuard {
            cache,
            txn_id,
            modified,
            phantom: PhantomData,
        };

        trace!("locked block {} for writing at {}", block_id, txn_id);

        Ok(guard)
    }
}

impl<K, B> FileReadExclusive for FileReadGuardExclusive<K, B>
where
    K: FromStr + fmt::Display + Ord + PartialEq + Clone + Send + Sync + 'static,
    B: BlockData,
    <K as FromStr>::Err: std::error::Error + fmt::Display,
    CacheBlock: AsType<B>,
{
    fn upgrade(self) -> FileWriteGuard<K, B> {
        FileGuard {
            file: self.file,
            txn_id: self.txn_id,
            blocks: self.blocks.upgrade(),
        }
    }
}

impl<K: Ord + Clone + fmt::Display, B: BlockData> FileWriteGuard<K, B>
where
    CacheBlock: AsType<B>,
{
    fn create_block_inner(
        &mut self,
        mut version: freqfs::DirWriteGuard<CacheBlock>,
        block_id: K,
        initial_value: B,
    ) -> TCResult<BlockWriteGuard<K, B>> {
        if self.blocks.contains_key(&block_id) {
            #[cfg(debug_assertions)]
            panic!("{} already has a block with ID {}", self.file, block_id);

            #[cfg(not(debug_assertions))]
            return Err(bad_request!("block {} already exists", block_id));
        }

        let txn_id = self.txn_id;
        let lock = TxnLock::new("block last commit ID", txn_id, txn_id);
        let modified = lock.try_write(txn_id).expect("block last modified");

        trace!("creating new block {}...", block_id);
        let name = format!("{}.{}", block_id, B::ext());
        self.blocks.insert(block_id, lock);

        let size_hint = initial_value.get_size();
        let block = version
            .create_file(name, initial_value, size_hint)
            .map_err(io_err)?;

        std::mem::drop(version);

        block
            .try_write()
            .map(move |cache| BlockWriteGuard {
                cache,
                txn_id,
                modified,
                phantom: PhantomData,
            })
            .map_err(io_err)
    }
}

#[async_trait]
impl<K, B> FileWrite for FileWriteGuard<K, B>
where
    K: FromStr + fmt::Display + Ord + PartialEq + Clone + Send + Sync + 'static,
    B: BlockData,
    <K as FromStr>::Err: std::error::Error + fmt::Display,
    CacheBlock: AsType<B>,
{
    fn downgrade(self) -> FileReadGuardExclusive<K, B> {
        FileGuard {
            file: self.file,
            txn_id: self.txn_id,
            blocks: self.blocks.downgrade(),
        }
    }

    async fn create_block(
        &mut self,
        block_id: K,
        initial_value: B,
    ) -> TCResult<BlockWriteGuard<K, B>> {
        let version = self.file.version_write(&self.txn_id).await?;
        self.create_block_inner(version, block_id, initial_value)
    }

    fn try_create_block(
        &mut self,
        block_id: <Self::File as tc_transact::fs::File>::Key,
        initial_value: <Self::File as tc_transact::fs::File>::Block,
    ) -> TCResult<<Self::File as tc_transact::fs::File>::BlockWrite> {
        let version = self.file.try_version_write(&self.txn_id)?;
        self.create_block_inner(version, block_id, initial_value)
    }

    async fn create_block_unique(
        &mut self,
        initial_value: B,
    ) -> TCResult<(K, BlockWriteGuard<K, B>)> {
        let uuid = Uuid::new_v4();
        let block_id: <Self::File as tc_transact::fs::File>::Key = uuid
            .to_string()
            .parse()
            .map_err(|cause| unexpected!("invalid block id: {}", uuid).consume(cause))?;

        self.create_block(block_id.clone(), initial_value)
            .map_ok(move |block| (block_id, block))
            .await
    }

    fn try_create_block_unique(
        &mut self,
        initial_value: <Self::File as tc_transact::fs::File>::Block,
    ) -> TCResult<(
        <Self::File as tc_transact::fs::File>::Key,
        <Self::File as tc_transact::fs::File>::BlockWrite,
    )> {
        let uuid = Uuid::new_v4();
        let block_id: <Self::File as tc_transact::fs::File>::Key = uuid
            .to_string()
            .parse()
            .map_err(|cause| unexpected!("invalid block ID {}", uuid).consume(cause))?;

        self.try_create_block(block_id.clone(), initial_value)
            .map(move |block| (block_id, block))
    }

    async fn delete_block<Q>(&mut self, block_id: Q) -> TCResult<()>
    where
        Q: Borrow<K> + Send + Sync,
    {
        if let Some(last_mutation) = self.blocks.get(block_id.borrow()) {
            *last_mutation.write(self.txn_id).await? = self.txn_id;

            // keep the version directory in sync in case create_block is called later
            // with the same block_id
            let mut version = self.file.version_write(&self.txn_id).await?;
            let name = file_name::<_, K, B>(block_id.borrow());
            version.delete(name).await;
        }

        self.blocks.remove(block_id.borrow());
        Ok(())
    }

    fn try_delete_block<Q>(&mut self, _name: Q) -> TCResult<()>
    where
        Q: Borrow<<Self::File as tc_transact::fs::File>::Key> + Send + Sync,
    {
        Err(not_implemented!("FileWriteGuard::try_delete_block"))
    }

    async fn copy_from<O>(&mut self, other: &O, truncate: bool) -> TCResult<()>
    where
        O: FileRead,
        O::File: tc_transact::fs::File<Key = K, Block = B>,
    {
        if truncate {
            self.truncate().await?;
        }

        for block_id in other.block_ids() {
            let block = other.read_block(&block_id).map_ok(|b| (*b).clone()).await?;
            if self.contains(&block_id) {
                let mut dest = self.write_block(&block_id).await?;
                *dest = block;
            } else {
                self.create_block(block_id, block).await?;
            }
        }

        Ok(())
    }

    fn try_copy_from<O>(&mut self, _other: &O, _truncate: bool) -> TCResult<()>
    where
        O: FileRead,
        O::File: tc_transact::fs::File<Key = K, Block = B>,
    {
        Err(not_implemented!("FileWriteGuard::try_copy_from"))
    }

    async fn truncate(&mut self) -> TCResult<()> {
        let mut version = self.file.version_write(&self.txn_id).await?;

        for block_id in self.block_ids() {
            let name = file_name::<_, K, B>(&block_id);
            version.delete(name).await;
        }

        self.blocks.clear();

        Ok(())
    }

    fn try_truncate(&mut self) -> TCResult<()> {
        Err(not_implemented!("FileWriteGuard::try_truncate"))
    }
}

/// A transactional file
#[derive(Clone)]
pub struct File<K, B> {
    canon: freqfs::DirLock<CacheBlock>,
    versions: freqfs::DirLock<CacheBlock>,
    blocks: TxnLock<BTreeMap<K, TxnLock<TxnId>>>,
    phantom: PhantomData<B>,
}

impl<K, B: BlockData> File<K, B>
where
    CacheBlock: AsType<B>,
    K: FromStr + fmt::Display + Ord + Clone,
    <K as FromStr>::Err: std::error::Error + fmt::Display,
{
    #[cfg(debug_assertions)]
    fn lock_name(fs_dir: &freqfs::Dir<CacheBlock>) -> String {
        format!("block list of file {:?}", &*fs_dir)
    }

    #[cfg(not(debug_assertions))]
    fn lock_name(_fs_dir: &freqfs::Dir<CacheBlock>) -> String {
        "block list of transactional file".to_string()
    }

    pub fn new(canon: freqfs::DirLock<CacheBlock>, txn_id: TxnId) -> TCResult<Self> {
        let mut fs_dir = canon
            .try_write()
            .map_err(|cause| unexpected!("new file is already in use: {}", cause))?;

        if fs_dir.len() > 0 {
            return Err(unexpected!("new file is not empty"));
        }

        Ok(Self {
            canon,
            versions: fs_dir.create_dir(VERSION.to_string()).map_err(io_err)?,
            blocks: TxnLock::new(Self::lock_name(&fs_dir), txn_id, BTreeMap::new()),
            phantom: PhantomData,
        })
    }

    pub(super) async fn load(canon: freqfs::DirLock<CacheBlock>, txn_id: TxnId) -> TCResult<Self> {
        let mut fs_dir = canon.write().await;

        debug!("File::load {:?}", &*fs_dir);

        let versions = fs_dir
            .get_or_create_dir(VERSION.to_string())
            .map_err(io_err)?;

        let mut blocks = BTreeMap::new();
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

            trace!("File::load found block {}", name);

            if name.len() < B::ext().len() + 1 || !name.ends_with(B::ext()) {
                return Err(unexpected!(
                    "block has invalid extension: {} (expected {})",
                    name,
                    B::ext()
                ));
            }

            let contents = match block {
                freqfs::DirEntry::File(block) => {
                    block
                        .read()
                        .map_ok(|contents| B::clone(&*contents))
                        .map_err(io_err)
                        .await?
                }
                freqfs::DirEntry::Dir(_) => {
                    return Err(unexpected!(
                        "expected block file but found directory: {}",
                        name
                    ))
                }
            };

            let block_id = &name[..(name.len() - B::ext().len() - 1)];
            let block_id: K = block_id
                .parse()
                .map_err(|cause| unexpected!("invalid block ID: {}", block_id).consume(cause))?;

            blocks.insert(
                block_id,
                TxnLock::new("block last commit ID", txn_id, txn_id),
            );

            let size_hint = contents.get_size();
            version
                .create_file(name.clone(), contents, size_hint)
                .map_err(io_err)?;
        }

        Ok(Self {
            canon,
            versions,
            blocks: TxnLock::new(Self::lock_name(&fs_dir), txn_id, blocks),
            phantom: Default::default(),
        })
    }

    async fn version(&self, txn_id: &TxnId) -> TCResult<freqfs::DirLock<CacheBlock>> {
        let mut versions = self.versions.write().await;
        versions
            .get_or_create_dir(txn_id.to_string())
            .map_err(io_err)
    }

    fn try_version(&self, txn_id: &TxnId) -> TCResult<freqfs::DirLock<CacheBlock>> {
        let mut versions = self.versions.try_write().map_err(io_err)?;
        versions
            .get_or_create_dir(txn_id.to_string())
            .map_err(io_err)
    }

    async fn version_read(&self, txn_id: &TxnId) -> TCResult<freqfs::DirReadGuard<CacheBlock>> {
        let version = self.version(txn_id).await?;
        version.read().map(Ok).await
    }

    fn try_version_read(&self, txn_id: &TxnId) -> TCResult<freqfs::DirReadGuard<CacheBlock>> {
        let version = self.try_version(txn_id)?;
        version.try_read().map_err(io_err)
    }

    async fn version_write(&self, txn_id: &TxnId) -> TCResult<freqfs::DirWriteGuard<CacheBlock>> {
        let version = self.version(txn_id).await?;
        version.write().map(Ok).await
    }

    fn try_version_write(&self, txn_id: &TxnId) -> TCResult<freqfs::DirWriteGuard<CacheBlock>> {
        self.try_version(txn_id)
            .and_then(|version| version.try_write().map_err(io_err))
    }

    async fn with_version_read<F, T>(&self, txn_id: &TxnId, then: F) -> TCResult<T>
    where
        F: FnOnce(freqfs::DirReadGuard<CacheBlock>) -> T,
    {
        self.version_read(txn_id).map_ok(then).await
    }

    fn with_try_version_read<F, T>(&self, txn_id: &TxnId, then: F) -> TCResult<T>
    where
        F: FnOnce(freqfs::DirReadGuard<CacheBlock>) -> T,
    {
        self.try_version_read(txn_id).map(then)
    }

    async fn with_version_write<F, T>(&self, txn_id: &TxnId, then: F) -> TCResult<T>
    where
        F: FnOnce(freqfs::DirWriteGuard<CacheBlock>) -> T,
    {
        self.version_write(txn_id).map_ok(then).await
    }

    fn with_try_version_write<F, T>(&self, txn_id: &TxnId, then: F) -> TCResult<T>
    where
        F: FnOnce(freqfs::DirWriteGuard<CacheBlock>) -> T,
    {
        self.try_version_write(txn_id).map(then)
    }
}

#[async_trait]
impl<K, B> tc_transact::fs::File for File<K, B>
where
    K: FromStr + fmt::Display + Ord + PartialEq + Clone + Send + Sync + 'static,
    B: BlockData,
    <K as FromStr>::Err: std::error::Error + fmt::Display,
    CacheBlock: AsType<B>,
{
    type Key = K;
    type Block = B;
    type Read = FileReadGuard<K, B>;
    type ReadExclusive = FileReadGuardExclusive<K, B>;
    type Write = FileWriteGuard<K, B>;
    type BlockRead = BlockReadGuard<B>;
    type BlockReadExclusive = BlockReadGuardExclusive<K, B>;
    type BlockWrite = BlockWriteGuard<K, B>;
    type Inner = freqfs::DirLock<CacheBlock>;

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
            .map_err(TCError::from)
            .await
    }

    fn try_read(&self, txn_id: TxnId) -> TCResult<Self::Read> {
        self.blocks
            .try_read(txn_id)
            .map(move |blocks| {
                trace!("locked file for reading at {}", txn_id);

                FileGuard {
                    file: self.clone(),
                    txn_id,
                    blocks,
                }
            })
            .map_err(TCError::from)
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
            .map_err(TCError::from)
            .await
    }

    async fn try_read_exclusive(&self, txn_id: TxnId) -> TCResult<Self::ReadExclusive> {
        self.blocks
            .try_read_exclusive(txn_id)
            .map(move |blocks| {
                trace!("locked file for reading at {}", txn_id);

                FileGuard {
                    file: self.clone(),
                    txn_id,
                    blocks,
                }
            })
            .map_err(TCError::from)
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
            .map_err(TCError::from)
            .await
    }

    fn try_write(&self, txn_id: TxnId) -> TCResult<Self::Write> {
        self.blocks
            .try_write(txn_id)
            .map(move |blocks| {
                trace!("locked file for writing at {}", txn_id);

                FileGuard {
                    file: self.clone(),
                    txn_id,
                    blocks,
                }
            })
            .map_err(TCError::from)
    }

    fn into_inner(self) -> freqfs::DirLock<CacheBlock> {
        self.canon
    }
}

impl<K, B: BlockData> Store for File<K, B>
where
    K: FromStr + fmt::Display + Ord + PartialEq + Clone + Send + Sync + 'static,
    B: BlockData,
    <K as FromStr>::Err: std::error::Error + fmt::Display,
    CacheBlock: AsType<B>,
{
    fn is_empty(&self, txn_id: TxnId) -> TCResult<bool> {
        tc_transact::fs::File::try_read(self, txn_id).map(|guard| guard.is_empty())
    }
}

#[async_trait]
impl<K, B: BlockData> Transact for File<K, B>
where
    K: FromStr + fmt::Display + PartialEq + Ord + Clone + Send + Sync + 'static,
    <K as FromStr>::Err: std::error::Error + fmt::Display,
    CacheBlock: AsType<B>,
{
    type Commit = Option<TxnLockCommit<BTreeMap<K, TxnLock<TxnId>>>>;

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        debug!("commit {}", self);

        let blocks = self.blocks.commit(txn_id).await?;

        trace!("committed block listing");

        {
            let version = {
                let fs_dir = self.versions.read().await;
                let version = fs_dir.get_dir(&txn_id.to_string())?;
                version.read().await
            };

            let mut canon = self.canon.write().await;

            for (block_id, last_modified) in blocks.iter() {
                trace!("commit last modified ID of block {}...", block_id);
                let last_modified = if let Some(last_modified) = last_modified.commit(txn_id).await
                {
                    last_modified
                } else {
                    trace!(
                        "File::commit skipping block {} since it was not accessed",
                        block_id
                    );

                    continue;
                };

                if &*last_modified == &txn_id {
                    let name = file_name::<_, K, B>(block_id);
                    let version = version.get_file(&name).expect("block version lock");

                    let block = version.read().await.expect("block version");

                    let canon = if let Some(canon) = canon.get_file(&name) {
                        *canon.write().await.expect("canonical block") = (*block).clone();
                        canon
                    } else {
                        let size_hint = (&*block).get_size();
                        canon
                            .create_file(name, (*block).clone(), size_hint)
                            .expect("new canonical block")
                    };

                    canon.sync().await.expect("sync canonical block");
                    trace!("File::commit canonical block {}", block_id);
                } else {
                    trace!(
                        "File::commit skipping block {} since it was not modified",
                        block_id
                    );
                }
            }

            trace!("commit blocks deleted in file version {}", txn_id);
            for (name, _) in version.iter() {
                let block_id = block_id(name).expect("block ID");
                if !blocks.contains_key(&block_id) {
                    trace!("File::commit delete block {}", block_id);
                    canon.delete(name.clone()).await;
                }
            }
        }

        trace!("sync canonical file contents to disk...");
        self.canon.sync().await.expect("sync canonical file");
        trace!("sync'd canonical file contents to disk");
        Some(blocks)
    }

    async fn rollback(&self, txn_id: &TxnId) {
        debug!("roll back {}", self);

        if let Some(blocks) = self.blocks.rollback(txn_id).await {
            join_all(
                blocks
                    .values()
                    .map(|last_commit_id| last_commit_id.rollback(txn_id)),
            )
            .await;

            let mut versions = self.versions.write().await;
            versions.delete(txn_id.to_string()).await;
        }
    }

    async fn finalize(&self, txn_id: &TxnId) {
        debug!("File::finalize");

        if let Some(blocks) = self.blocks.finalize(txn_id).await {
            let cleanups = blocks
                .values()
                .map(|last_commit_id| last_commit_id.finalize(txn_id));

            join_all(cleanups).await;

            let mut versions = self.versions.write().await;
            versions.delete(txn_id.to_string()).await;
        }
    }
}

impl<K, B: Send + Sync + 'static> fmt::Display for File<K, B> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        #[cfg(debug_assertions)]
        {
            if let Ok(dir) = self.canon.try_read() {
                write!(f, "file at {}", dir.path().to_string_lossy().to_string())
            } else {
                write!(f, "file of {} blocks", std::any::type_name::<B>())
            }
        }

        #[cfg(not(debug_assertions))]
        write!(f, "file of {} blocks", std::any::type_name::<B>())
    }
}

#[inline]
fn block_id<K>(name: &str) -> TCResult<K>
where
    K: FromStr,
    <K as FromStr>::Err: std::error::Error + fmt::Display,
{
    let i = name
        .rfind('.')
        .ok_or_else(|| TCError::invalid_value(name, "a block name"))?;

    // make sure the name is valid unicode
    let name = &name[..i];
    let name = std::str::from_utf8(name.as_bytes())
        .map_err(|cause| unexpected!("invalid block ID {}", name).consume(cause))?;

    name.parse()
        .map_err(|cause| TCError::invalid_value(name, "a block name").consume(cause))
}

#[inline]
fn file_name<Q, K, B>(block_id: Q) -> String
where
    K: fmt::Display,
    B: BlockData,
    Q: Borrow<K>,
{
    format!("{}.{}", block_id.borrow(), B::ext())
}
