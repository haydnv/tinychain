//! The filesystem cache, with LFU eviction

use std::convert::{TryFrom, TryInto};
use std::path::PathBuf;

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

use super::{create_parent, io_err};

struct Policy;

#[async_trait]
impl freqache::Policy<PathBuf, CacheBlock> for Policy {
    fn can_evict(&self, block: &CacheBlock) -> bool {
        block.ref_count() < 2
    }

    async fn evict(&self, path: PathBuf, block: &CacheBlock) {
        persist(&path, block)
            .await
            .expect("persist cache block to disk")
    }
}

type LFU = freqache::LFUCache<PathBuf, CacheBlock, Policy>;

/// A [`CacheLock`] representing a single filesystem block.
#[derive(Clone)]
pub enum CacheBlock {
    BTree(CacheLock<Node>),
    Chain(CacheLock<ChainBlock>),
    Value(CacheLock<Value>),
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
        }
    }

    fn ref_count(&self) -> usize {
        match self {
            Self::BTree(block) => block.ref_count(),
            Self::Chain(block) => block.ref_count(),
            Self::Value(block) => block.ref_count(),
        }
    }
}

impl Entry for CacheBlock {
    fn weight(&self) -> u64 {
        match self {
            Self::BTree(_) => Node::max_size(),
            Self::Chain(_) => ChainBlock::max_size(),
            Self::Value(_) => Value::max_size(),
        }
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

        spawn_cleanup_thread(cache.clone(), rx);
        cache
    }

    /// Read a block from the cache if possible, or else fetch it from the filesystem.
    pub async fn read<B: BlockData>(&self, path: &PathBuf) -> TCResult<Option<CacheLock<B>>>
    where
        CacheLock<B>: TryFrom<CacheBlock, Error = TCError>,
        CacheBlock: From<CacheLock<B>>,
    {
        let mut cache = self.lfu.write().await;

        if let Some(block) = cache.get(path).await {
            debug!("cache hit: {:?}", path);
            let block = block.clone().try_into()?;
            return Ok(Some(block));
        } else if !path.exists() {
            return Ok(None);
        } else {
            log::info!("cache miss: {:?}", path);
        }

        let block_file = read_file(&path).await?;
        let block = B::load(block_file).await?;

        debug!("cache insert: {:?}", path);

        let block = CacheLock::new(block);
        cache.insert(path.clone(), block.clone().into()).await;

        Ok(Some(block))
    }

    /// Delete a block from the cache.
    pub async fn delete(&self, path: &PathBuf) -> Option<CacheBlock> {
        let mut cache = self.lfu.write().await;
        cache.remove(path).await
    }

    /// Remove a block from the cache and delete it from the filesystem.
    pub async fn delete_and_sync(&self, path: &PathBuf) -> TCResult<()> {
        let mut cache = self.lfu.write().await;
        cache.remove(path).await;

        if path.exists() {
            tokio::fs::remove_file(path)
                .map_err(|e| io_err(e, path))
                .await
        } else {
            Ok(())
        }
    }

    async fn _sync(cache: &mut LFU, path: &PathBuf) -> TCResult<bool> {
        debug!("sync block at {:?} with filesystem", &path);

        if let Some(block) = cache.get(path).await {
            persist(path, &block).await?;
            Ok(true)
        } else {
            log::info!("cache sync miss: {:?}", path);
            Ok(path.exists())
        }
    }

    /// Synchronize a cached block with the filesystem.
    pub async fn sync(&self, path: &PathBuf) -> TCResult<bool> {
        debug!("sync block at {:?} with filesystem", &path);

        let mut cache = self.lfu.write().await;
        Self::_sync(&mut cache, path).await
    }

    async fn _write<'en, B: BlockData + IntoStream<'en> + 'en>(
        cache: &mut LFU,
        tx: &mpsc::Sender<Evict>,
        path: PathBuf,
        block: CacheLock<B>,
    ) where
        CacheBlock: From<CacheLock<B>>,
    {
        cache.insert(path, block.clone().into()).await;

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

    /// Update a block in the cache and then sync it with the filesystem.
    pub async fn write_and_sync<'en, B: BlockData + IntoStream<'en> + 'en>(
        &self,
        path: PathBuf,
        block: B,
    ) -> TCResult<bool>
    where
        CacheBlock: From<CacheLock<B>>,
    {
        debug!("cache insert + sync: {:?}", &path);
        let block = CacheLock::new(block);
        let mut cache = self.lfu.write().await;
        Self::_write(&mut cache, &self.tx, path.clone(), block).await;
        let exists = Self::_sync(&mut cache, &path).await?;
        Ok(exists)
    }
}

fn spawn_cleanup_thread(cache: Cache, mut rx: mpsc::Receiver<Evict>) {
    tokio::spawn(async move {
        info!("cache cleanup thread is running...");

        while rx.recv().await.is_some() {
            debug!("got Evict message, running cache eviction...");
            let mut lfu = cache.lfu.write().await;
            lfu.evict().await;
        }

        warn!("cache cleanup thread is shutting down...");
    });
}

async fn persist(path: &PathBuf, block: &CacheBlock) -> TCResult<()> {
    let mut block_file = if path.exists() {
        debug!("open block file at {:?} for sync", path);
        write_file(path).await?
    } else {
        debug!("create new filesystem block at {:?}", path);
        create_parent(path).await?;
        create_file(path).await?
    };

    block.persist(&mut block_file).map_ok(|_size| ()).await
}

#[inline]
fn create_file(path: &PathBuf) -> impl Future<Output = TCResult<fs::File>> + '_ {
    tokio::fs::File::create(path).map_err(move |e| io_err(e, path))
}

#[inline]
fn read_file(path: &PathBuf) -> impl Future<Output = TCResult<fs::File>> + '_ {
    fs::File::open(path).map_err(move |e| {
        debug!("io error: {}", e);
        io_err(e, path)
    })
}

async fn write_file(path: &PathBuf) -> TCResult<fs::File> {
    fs::OpenOptions::new()
        .write(true)
        .open(path)
        .map_err(move |e| {
            debug!("io error: {}", e);
            io_err(e, path)
        })
        .await
}
