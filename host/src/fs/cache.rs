//! The filesystem [`Cache`], with LFU eviction

use std::collections::HashSet;
use std::convert::{TryFrom, TryInto};
use std::path::PathBuf;
use std::sync::Arc;

#[cfg(feature = "tensor")]
use afarray::Array;
use async_trait::async_trait;
use freqache::LFUCache;
use futures::future::{Future, TryFutureExt};
use log::{debug, info};
use tokio::fs;
use tokio::io::AsyncWrite;
use tokio::sync::{Mutex, OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

use tc_btree::Node;
use tc_error::*;
use tc_transact::fs::BlockData;
use tc_value::Value;
use tcgeneric::TCBoxTryFuture;

use crate::chain::ChainBlock;

use super::{create_parent, io_err, TMP};
use std::ops::{Deref, DerefMut};

struct Policy;

#[async_trait]
impl freqache::Policy<PathBuf, CacheBlock> for Policy {
    fn can_evict(&self, _block: &CacheBlock) -> bool {
        unimplemented!()
    }

    async fn evict(&self, _path: PathBuf, _block: &CacheBlock) {
        unimplemented!()
    }
}

struct CacheState {
    lfu: LFUCache<PathBuf, CacheBlock, Policy>,
    deleted: HashSet<PathBuf>,
}

/// The filesystem block cache.
///
/// **IMPORTANT**: for performance reasons, the cache listing is not locked during file I/O.
/// This means that it's up to the developer to explicitly call `sync` between inserts and deletes
/// of the same block path.
#[derive(Clone)]
pub struct Cache {
    state: Arc<Mutex<CacheState>>,
}

impl Cache {
    /// Construct a new cache with the given size.
    pub fn new(max_size: u64) -> Self {
        let state = CacheState {
            lfu: LFUCache::new(max_size, Policy),
            deleted: HashSet::new(),
        };

        Self {
            state: Arc::new(Mutex::new(state)),
        }
    }

    /// Delete a block from the cache.
    pub async fn delete(&self, path: PathBuf) {
        debug!("Cache::delete {:?}", path);

        let mut state = self.state.lock().await;
        if let Some(block) = state.lfu.remove(&path) {
            block.delete().await;
            state.deleted.insert(path.clone());
        } else if path.exists() {
            state.deleted.insert(path.clone());
        }
    }

    /// Read a block from the cache if possible, otherwise from the filesystem.
    pub async fn read<B>(&self, path: &PathBuf) -> TCResult<Option<CacheLock<B>>>
    where
        B: BlockData,
        CacheBlock: From<CacheLock<B>>,
        CacheLock<B>: TryFrom<CacheBlock, Error = TCError>,
    {
        debug!("Cache::read {:?}", path);

        let mut state = self.state.lock().await;

        if state.deleted.contains(path) {
            Ok(None)
        } else if let Some(block) = state.lfu.get(path).cloned() {
            block.try_into().map(Some)
        } else if path.exists() {
            info!("cache miss: {:?}", path);
            let lock = CacheLock::<B>::pending(path.clone());
            let block = CacheBlock::from(lock.clone());
            state.lfu.insert(path.clone(), block);
            Ok(Some(lock))
        } else {
            Ok(None)
        }
    }

    /// Synchronize a cached block with the filesystem.
    pub async fn sync(&self, path: &PathBuf) -> TCResult<bool> {
        debug!("Cache::sync {:?}", path);

        let mut state = self.state.lock().await;
        if state.deleted.contains(path) {
            assert!(!state.lfu.contains_key(path));
            std::mem::drop(state);

            let cache = self.clone();
            delete_block_at(path)
                .and_then(|()| async move {
                    let mut state = cache.state.lock().await;
                    state.deleted.remove(path);
                    Ok(false)
                })
                .await
        } else if let Some(block) = state.lfu.get(path).cloned() {
            std::mem::drop(state);
            persist(block).map_ok(|_num_bytes| true).await
        } else {
            Ok(path.exists())
        }
    }

    /// Update a block in the cache.
    pub async fn write<B: BlockData>(&self, path: PathBuf, value: B) -> TCResult<CacheLock<B>>
    where
        CacheBlock: From<CacheLock<B>>,
        CacheLock<B>: TryFrom<CacheBlock, Error = TCError>,
    {
        debug!("Cache::write {:?}", path);

        let mut state = self.state.lock().await;
        state.deleted.remove(&path);

        if let Some(block) = state.lfu.get(&path).cloned() {
            std::mem::drop(state);

            let block = CacheLock::<B>::try_from(block)?;
            *block.write().await = value;
            Ok(block)
        } else {
            let block = CacheLock::new(path.clone(), value);
            state.lfu.insert(path, block.clone().into());
            Ok(block)
        }
    }
}

/// A [`CacheLock`] representing a single filesystem block.
#[derive(Clone)]
pub enum CacheBlock {
    BTree(CacheLock<Node>),
    Chain(CacheLock<ChainBlock>),
    #[cfg(feature = "tensor")]
    Tensor(CacheLock<Array>),
    Value(CacheLock<Value>),
}

impl CacheBlock {
    async fn delete(self) {
        match self {
            Self::BTree(lock) => lock.delete().await,
            Self::Chain(lock) => lock.delete().await,
            #[cfg(feature = "tensor")]
            Self::Tensor(lock) => lock.delete().await,
            Self::Value(lock) => lock.delete().await,
        }
    }

    fn path(&self) -> &PathBuf {
        match self {
            Self::BTree(lock) => lock.path(),
            Self::Chain(lock) => lock.path(),
            #[cfg(feature = "tensor")]
            Self::Tensor(lock) => lock.path(),
            Self::Value(lock) => lock.path(),
        }
    }

    async fn persist<W: AsyncWrite + Send + Unpin>(&self, sink: &mut W) -> TCResult<u64> {
        match self {
            Self::BTree(block) => {
                let contents = block.read().await;
                contents.persist(sink).await
            }
            Self::Chain(block) => {
                let contents = block.read().await;
                contents.persist(sink).await
            }
            Self::Value(block) => {
                let contents = block.read().await;
                contents.persist(sink).await
            }
            #[cfg(feature = "tensor")]
            Self::Tensor(block) => {
                let contents = block.read().await;
                contents.persist(sink).await
            }
        }
    }
}

impl freqache::Entry for CacheBlock {
    fn weight(&self) -> u64 {
        match self {
            Self::BTree(_) => Node::max_size(),
            Self::Chain(_) => ChainBlock::max_size(),
            #[cfg(feature = "tensor")]
            Self::Tensor(_) => Array::max_size(),
            Self::Value(_) => Value::max_size(),
        }
    }
}

#[cfg(feature = "tensor")]
impl From<CacheLock<Array>> for CacheBlock {
    fn from(lock: CacheLock<Array>) -> CacheBlock {
        Self::Tensor(lock)
    }
}

impl From<CacheLock<ChainBlock>> for CacheBlock {
    fn from(lock: CacheLock<ChainBlock>) -> CacheBlock {
        Self::Chain(lock)
    }
}

impl From<CacheLock<Node>> for CacheBlock {
    fn from(lock: CacheLock<Node>) -> CacheBlock {
        Self::BTree(lock)
    }
}

impl From<CacheLock<Value>> for CacheBlock {
    fn from(lock: CacheLock<Value>) -> CacheBlock {
        Self::Value(lock)
    }
}

#[cfg(feature = "tensor")]
impl TryFrom<CacheBlock> for CacheLock<Array> {
    type Error = TCError;

    fn try_from(block: CacheBlock) -> TCResult<Self> {
        match block {
            CacheBlock::Tensor(block) => Ok(block),
            _ => Err(TCError::unsupported("unexpected block type")),
        }
    }
}

impl TryFrom<CacheBlock> for CacheLock<ChainBlock> {
    type Error = TCError;

    fn try_from(block: CacheBlock) -> TCResult<Self> {
        match block {
            CacheBlock::Chain(block) => Ok(block),
            _ => Err(TCError::unsupported("unexpected block type")),
        }
    }
}

impl TryFrom<CacheBlock> for CacheLock<Node> {
    type Error = TCError;

    fn try_from(block: CacheBlock) -> TCResult<Self> {
        match block {
            CacheBlock::BTree(block) => Ok(block),
            _ => Err(TCError::unsupported("unexpected block type")),
        }
    }
}

impl TryFrom<CacheBlock> for CacheLock<Value> {
    type Error = TCError;

    fn try_from(block: CacheBlock) -> TCResult<Self> {
        match block {
            CacheBlock::Value(block) => Ok(block),
            _ => Err(TCError::unsupported("unexpected block type")),
        }
    }
}

enum CacheLockState<B> {
    Pending,
    Active(Arc<RwLock<B>>),
    Deleted,
}

impl<B> CacheLockState<B> {
    fn is_deleted(&self) -> bool {
        match self {
            Self::Deleted => true,
            _ => false,
        }
    }

    fn is_pending(&self) -> bool {
        match self {
            Self::Pending => true,
            _ => false,
        }
    }

    fn active_lock(&self) -> &Arc<RwLock<B>> {
        if let Self::Active(lock) = self {
            lock
        } else {
            unreachable!("lock inactive cache block")
        }
    }
}

/// A type-specific lock on a block in the [`Cache`].
#[derive(Clone)]
pub struct CacheLock<B> {
    path: Arc<PathBuf>,
    state: Arc<RwLock<CacheLockState<B>>>,
}

impl<B: BlockData> CacheLock<B> {
    fn new(path: PathBuf, value: B) -> Self {
        let state = CacheLockState::Active(Arc::new(RwLock::new(value)));
        Self {
            path: Arc::new(path),
            state: Arc::new(RwLock::new(state)),
        }
    }

    fn pending(path: PathBuf) -> Self {
        Self {
            path: Arc::new(path),
            state: Arc::new(RwLock::new(CacheLockState::Pending)),
        }
    }

    fn path(&self) -> &PathBuf {
        &self.path
    }

    async fn delete(self) {
        let mut state = self.state.write().await;
        *state = CacheLockState::Deleted;
    }

    async fn get_lock(&self) -> OwnedRwLockReadGuard<CacheLockState<B>> {
        let mut state = self.state.clone().write_owned().await;
        assert!(!state.is_deleted());

        if state.is_pending() {
            let block_file = read_file(&self.path).await.expect("a block file");
            let block = B::load(block_file).await.expect("a decoded block");
            let lock = Arc::new(RwLock::new(block));
            *state = CacheLockState::Active(lock.clone());
        }

        state.downgrade()
    }

    pub async fn read(&self) -> CacheLockReadGuard<B> {
        let state = self.get_lock().await;
        let value = state.active_lock().clone().read_owned().await;
        CacheLockReadGuard {
            block: self.clone(),
            value,
        }
    }

    pub async fn write(&self) -> CacheLockWriteGuard<B> {
        let state = self.get_lock().await;
        let value = state.active_lock().clone().write_owned().await;
        CacheLockWriteGuard {
            block: self.clone(),
            value,
        }
    }
}

pub struct CacheLockReadGuard<B> {
    block: CacheLock<B>,
    value: OwnedRwLockReadGuard<B>,
}

impl<B> Deref for CacheLockReadGuard<B> {
    type Target = B;

    fn deref(&self) -> &Self::Target {
        self.value.deref()
    }
}

pub struct CacheLockWriteGuard<B> {
    block: CacheLock<B>,
    value: OwnedRwLockWriteGuard<B>,
}

impl<B> Deref for CacheLockWriteGuard<B> {
    type Target = B;

    fn deref(&self) -> &Self::Target {
        self.value.deref()
    }
}

impl<B> DerefMut for CacheLockWriteGuard<B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.value.deref_mut()
    }
}

fn delete_block_at(path: &PathBuf) -> TCBoxTryFuture<()> {
    Box::pin(async move {
        let tmp = path.with_extension(TMP);
        if tmp.exists() {
            tokio::fs::remove_file(&tmp)
                .map_err(|e| io_err(e, &tmp))
                .await?;
        }

        if path.exists() {
            tokio::fs::remove_file(&path)
                .map_err(|e| io_err(e, &path))
                .await?;
        }

        Ok(())
    })
}

async fn persist(block: CacheBlock) -> TCResult<u64> {
    let path = block.path();
    let tmp = path.with_extension(TMP);

    let size = {
        let mut tmp_file = if tmp.exists() {
            write_file(&tmp).await?
        } else {
            create_parent(&tmp).await?;
            create_file(&tmp).await?
        };

        let size = block.persist(&mut tmp_file).await?;
        tmp_file.sync_all().map_err(|e| io_err(e, &tmp)).await?;
        size
    };

    tokio::fs::rename(&tmp, path)
        .map_err(|e| io_err(e, &tmp))
        .await?;

    Ok(size)
}

#[inline]
fn create_file(path: &PathBuf) -> impl Future<Output = TCResult<fs::File>> + '_ {
    tokio::fs::File::create(path).map_err(move |e| io_err(e, path))
}

#[inline]
fn read_file(path: &PathBuf) -> impl Future<Output = TCResult<fs::File>> + '_ {
    fs::File::open(path).map_err(move |e| io_err(e, path))
}

async fn write_file(path: &PathBuf) -> TCResult<fs::File> {
    fs::OpenOptions::new()
        .truncate(true)
        .write(true)
        .open(path)
        .map_err(move |e| io_err(e, path))
        .await
}
