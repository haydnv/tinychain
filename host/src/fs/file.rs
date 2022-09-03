//! A transactional [`File`]

use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use async_trait::async_trait;
use futures::{FutureExt, TryFutureExt};
use safecast::AsType;
use tokio::sync::RwLock;

use tc_error::*;
use tc_transact::fs::{BlockData, BlockId, FileRead, FileWrite, Store};
use tc_transact::lock::{TxnLock, TxnLockReadGuard, TxnLockWriteGuard};
use tc_transact::{Transact, TxnId};

use super::{io_err, CacheBlock, VERSION};

type Listing = HashSet<BlockId>;
pub type FileReadGuard<B> = FileGuard<B, TxnLockReadGuard<Listing>>;
pub type FileWriteGuard<B> = FileGuard<B, TxnLockWriteGuard<Listing>>;

struct Wake;

pub struct BlockReadGuard<B> {
    cache: freqfs::FileReadGuard<CacheBlock, B>,
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

pub struct FileGuard<B, L> {
    file: File<B>,
    txn_id: TxnId,
    listing: L,
}

impl<B, L> FileGuard<B, L>
where
    B: BlockData,
{
    fn file_name(block_id: &BlockId) -> String {
        format!("{}.{}", block_id, B::ext())
    }
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
        todo!()
    }

    fn is_empty(&self) -> bool {
        todo!()
    }

    async fn read_block<I>(&self, block_id: I) -> TCResult<Self::Read>
    where
        I: Borrow<BlockId> + Send + Sync,
    {
        if !self.listing.contains(block_id.borrow()) {
            return Err(TCError::not_found(block_id.borrow()));
        }

        let modified = {
            let block = {
                let modified = self.file.modified.read().await;
                modified
                    .get(block_id.borrow())
                    .expect("block last mutation ID")
                    .clone()
            };

            block.read(self.txn_id).await?
        };

        let name = Self::file_name(block_id.borrow());
        if let Some(block) = self
            .file
            .with_version_read(&self.txn_id, |version| version.get_file(&name))
            .await?
        {
            return block
                .read()
                .map_ok(|cache| BlockReadGuard { cache, modified })
                .map_err(io_err)
                .await;
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

            (size_hint, value)
        };

        let block = self
            .file
            .with_version_write(&self.txn_id, |mut version| {
                version.create_file(name, value, size_hint).map_err(io_err)
            })
            .await??;

        block
            .read()
            .map_ok(move |cache| BlockReadGuard { cache, modified })
            .map_err(io_err)
            .await
    }

    async fn write_block<I>(&self, block_id: I) -> TCResult<Self::Write>
    where
        I: Borrow<BlockId> + Send + Sync,
    {
        if !self.listing.contains(block_id.borrow()) {
            return Err(TCError::not_found(block_id.borrow()));
        }

        let mut modified = {
            {
                let block = {
                    let blocks = self.file.modified.read().await;

                    blocks
                        .get(block_id.borrow())
                        .expect("block last mutation ID")
                        .clone()
                };

                block.write(self.txn_id).await?
            }
        };

        *modified = self.txn_id;

        let name = Self::file_name(block_id.borrow());

        if let Some(block) = self
            .file
            .with_version_read(&self.txn_id, |version| version.get_file(&name))
            .await?
        {
            return block
                .write()
                .map_ok(move |cache| BlockWriteGuard { cache, modified })
                .map_err(io_err)
                .await;
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

        let block = self
            .file
            .with_version_write(&self.txn_id, |mut version| {
                version.create_file(name, value, size_hint).map_err(io_err)
            })
            .await??;

        block
            .write()
            .map_ok(move |cache| BlockWriteGuard { cache, modified })
            .map_err(io_err)
            .await
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
    fn lock_name(fs_dir: &freqfs::Dir<CacheBlock>) -> String {
        "block list of transactional file"
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

    async fn copy_from(&self, txn_id: TxnId, other: &Self, truncate: bool) -> TCResult<()> {
        todo!()
    }

    async fn read(&self, txn_id: TxnId) -> TCResult<Self::Read> {
        self.listing
            .read(txn_id)
            .map_ok(move |listing| FileGuard {
                file: self.clone(),
                txn_id,
                listing,
            })
            .await
    }

    async fn write(&self, txn_id: TxnId) -> TCResult<Self::Write> {
        self.listing
            .write(txn_id)
            .map_ok(move |listing| FileGuard {
                file: self.clone(),
                txn_id,
                listing,
            })
            .await
    }
}

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

impl<B: Send + Sync + 'static> fmt::Display for File<B> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "file of {} blocks", std::any::type_name::<B>())
    }
}
