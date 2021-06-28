//! The filesystem cache, with LFU eviction

use std::convert::{TryFrom, TryInto};
use std::path::PathBuf;

#[cfg(feature = "tensor")]
use afarray::Array;
use async_trait::async_trait;
use destream::IntoStream;
use freqache::Entry;
use futures::{Future, TryFutureExt};
use log::{debug, error, info, warn};
use tokio::fs;
use tokio::io::AsyncWrite;
use tokio::sync::mpsc;
use uplock::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use tc_btree::Node;
use tc_error::*;
use tc_transact::fs::BlockData;

use crate::chain::ChainBlock;
use crate::scalar::Value;

use super::{create_parent, io_err, TMP};

struct Policy;

#[async_trait]
impl freqache::Policy<PathBuf, CacheBlock> for Policy {
    fn can_evict(&self, block: &CacheBlock) -> bool {
        block.ref_count() <= 1
    }

    async fn evict(&self, path: PathBuf, block: &CacheBlock) {
        debug!("evict block at {:?} from cache", path);

        let size = persist(&path, block)
            .await
            .expect("persist cache block to disk");

        debug!("block at {:?} evicted, wrote {} bytes to disk", path, size);
    }
}

type LFU = freqache::LFUCache<PathBuf, CacheBlock, Policy>;

/// A [`CacheLock`] representing a single filesystem block.
#[derive(Clone)]
pub enum CacheBlock {
    BTree(CacheLock<Node>),
    Chain(CacheLock<ChainBlock>),
    Value(CacheLock<Value>),

    #[cfg(feature = "tensor")]
    Tensor(CacheLock<Array>),
}

impl CacheBlock {
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

    fn ref_count(&self) -> usize {
        match self {
            Self::BTree(block) => block.ref_count(),
            Self::Chain(block) => block.ref_count(),
            Self::Value(block) => block.ref_count(),
            #[cfg(feature = "tensor")]
            Self::Tensor(block) => block.ref_count(),
        }
    }
}

impl Entry for CacheBlock {
    fn weight(&self) -> u64 {
        match self {
            Self::BTree(_) => Node::max_size(),
            Self::Chain(_) => ChainBlock::max_size(),
            Self::Value(_) => Value::max_size(),
            #[cfg(feature = "tensor")]
            Self::Tensor(_) => Array::max_size(),
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

/// A filesystem cache lock.
pub struct CacheLock<T> {
    lock: RwLock<T>,
}

impl<T> CacheLock<T> {
    fn new(value: T) -> Self {
        Self {
            lock: RwLock::new(value),
        }
    }

    pub async fn read(&self) -> RwLockReadGuard<T> {
        debug!(
            "CacheLock got read lock request on a lock with {} refs...",
            self.lock.ref_count()
        );

        self.lock.read().await
    }

    pub async fn write(&self) -> RwLockWriteGuard<T> {
        debug!(
            "CacheLock got write lock request on a lock with {} refs...",
            self.lock.ref_count()
        );

        self.lock.write().await
    }

    pub fn ref_count(&self) -> usize {
        self.lock.ref_count()
    }
}

impl<T> Clone for CacheLock<T> {
    fn clone(&self) -> Self {
        Self {
            lock: self.lock.clone(),
        }
    }
}

struct Evict;

/// The filesystem cache.
#[derive(Clone)]
pub struct Cache {
    tx: mpsc::Sender<Evict>,
    lfu: RwLock<LFU>,
}

impl Cache {
    /// Construct a new cache with the given size.
    pub fn new(max_size: u64) -> Self {
        let (tx, rx) = mpsc::channel(1024);
        let cache = Self {
            tx,
            lfu: RwLock::new(LFU::new(max_size, Policy)),
        };

        spawn_cleanup_thread(cache.lfu.clone(), rx);
        cache
    }

    async fn _read_and_insert<B: BlockData>(
        mut cache: RwLockWriteGuard<LFU>,
        path: PathBuf,
    ) -> TCResult<CacheLock<B>>
    where
        CacheLock<B>: TryFrom<CacheBlock, Error = TCError>,
        CacheBlock: From<CacheLock<B>>,
    {
        let block_file = read_file(&path).await?;
        let block = B::load(block_file).await?;

        debug!("cache insert: {:?}", path);

        let block = CacheLock::new(block);
        cache.insert(path, block.clone().into()).await;

        Ok(block)
    }

    /// Read a block from the cache if possible, or else fetch it from the filesystem.
    pub async fn read<B: BlockData>(&self, path: &PathBuf) -> TCResult<Option<CacheLock<B>>>
    where
        CacheLock<B>: TryFrom<CacheBlock, Error = TCError>,
        CacheBlock: From<CacheLock<B>>,
    {
        debug!("Cache::read {:?}", path);

        let mut cache = self.lfu.write().await;

        if let Some(block) = cache.get(path).await {
            debug!("cache hit: {:?}", path);
            let block = block.clone().try_into()?;
            return Ok(Some(block));
        } else if !path.exists() {
            return Ok(None);
        } else {
            info!("cache miss: {:?}", path);
        }

        Self::_read_and_insert(cache, path.clone())
            .map_ok(Some)
            .await
    }

    /// Delete a block from the cache.
    pub async fn delete(&self, path: &PathBuf) -> Option<CacheBlock> {
        debug!("Cache::delete {:?}", path);

        let mut cache = self.lfu.write().await;
        cache.remove(path).await
    }

    /// Remove a block from the cache and delete it from the filesystem.
    pub async fn delete_and_sync(&self, path: PathBuf) -> TCResult<()> {
        debug!("Cache::delete_and_sync {:?}", path);

        let mut cache = self.lfu.write().await;
        cache.remove(&path).await;

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
    }

    /// Lock the cache for writing before deleting the given filesystem directory.
    pub async fn delete_dir(&self, path: PathBuf) -> TCResult<()> {
        let _lock = self.lfu.write().await;
        tokio::fs::remove_dir_all(&path)
            .map_err(|e| io_err(e, &path))
            .await
    }

    async fn _sync(cache: &mut LFU, path: &PathBuf) -> TCResult<bool> {
        debug!("sync block at {:?} with filesystem", &path);

        if let Some(block) = cache.get(path).await {
            let size = persist(path, &block).await?;
            debug!("sync'd block at {:?}, wrote {} bytes", path, size);
            Ok(true)
        } else {
            info!("cache sync miss: {:?}", path);
            Ok(path.exists())
        }
    }

    /// Synchronize a cached block with the filesystem.
    pub async fn sync(&self, path: &PathBuf) -> TCResult<bool> {
        debug!("sync block at {:?} with filesystem", &path);

        let mut cache = self.lfu.write().await;
        Self::_sync(&mut cache, path).await
    }

    /// Sync the source block with the filesystem and then copy it to the destination.
    pub async fn sync_and_copy<'en, B: BlockData + IntoStream<'en> + 'en>(
        &self,
        source: PathBuf,
        dest: PathBuf,
    ) -> TCResult<CacheLock<B>>
    where
        CacheLock<B>: TryFrom<CacheBlock, Error = TCError>,
        CacheBlock: From<CacheLock<B>>,
    {
        debug!("cache sync + copy from {:?} to {:?}", source, dest);
        let mut cache = self.lfu.write().await;
        Self::_sync(&mut cache, &source).await?;

        tokio::fs::copy(&source, &dest)
            .map_err(|e| io_err(e, format!("copy from {:?} to {:?}", source, dest)))
            .await?;

        Self::_read_and_insert(cache, dest).await
    }

    async fn _write<'en, B: BlockData + IntoStream<'en> + 'en>(
        cache: &mut LFU,
        tx: &mpsc::Sender<Evict>,
        path: PathBuf,
        block: CacheLock<B>,
    ) where
        CacheBlock: From<CacheLock<B>>,
    {
        cache.insert(path, block.into()).await;

        if cache.is_full() {
            debug!("the block cache is full, triggering garbage collection...");
            if let Err(err) = tx.send(Evict).await {
                error!("the cache cleanup thread is dead! {}", err);
            }
        }
    }

    /// Update a block in the cache.
    pub async fn write<'en, B: BlockData + IntoStream<'en> + 'en>(
        &self,
        path: PathBuf,
        block: B,
    ) -> TCResult<CacheLock<B>>
    where
        CacheBlock: From<CacheLock<B>>,
    {
        debug!("cache insert: {:?}", &path);
        let block = CacheLock::new(block);
        let mut cache = self.lfu.write().await;
        Self::_write(&mut cache, &self.tx, path, block.clone()).await;
        Ok(block)
    }
}

fn spawn_cleanup_thread(cache: RwLock<LFU>, mut rx: mpsc::Receiver<Evict>) {
    tokio::spawn(async move {
        info!("cache cleanup thread is running...");

        while rx.recv().await.is_some() {
            debug!("got Evict message");
            let mut lfu = cache.write().await;
            debug!("running cache eviction with {} entries...", lfu.len());
            lfu.evict().await;
            debug!("cache eviction complete, {} entries remain", lfu.len());
        }

        warn!("cache cleanup thread shutting down");
    });
}

async fn persist(path: &PathBuf, block: &CacheBlock) -> TCResult<u64> {
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
