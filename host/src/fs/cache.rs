//! The filesystem cache, with LFU eviction. INCOMPLETE.

use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use std::hash::Hash;
use std::io;
use std::path::PathBuf;

use bytes::Bytes;
use futures::TryFutureExt;
use log::debug;
use tokio::fs;
use tokio::sync::mpsc;
use uplock::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use tc_error::*;
use tc_transact::fs::BlockData;

use crate::chain::ChainBlock;

use super::{create_parent, io_err};

/// A [`CacheLock`] representing a single filesystem block.
#[derive(Clone)]
pub enum CacheBlock {
    Bin(CacheLock<Bytes>),
    Chain(CacheLock<ChainBlock>),
}

impl CacheBlock {
    async fn into_bytes(self) -> Bytes {
        match self {
            Self::Bin(block) => (*block.read().await).clone(),
            Self::Chain(block) => (*block.read().await).clone().into(),
        }
    }

    fn ref_count(&self) -> usize {
        match self {
            Self::Bin(block) => block.ref_count(),
            Self::Chain(block) => block.ref_count(),
        }
    }
}

impl From<CacheLock<Bytes>> for CacheBlock {
    fn from(lock: CacheLock<Bytes>) -> CacheBlock {
        Self::Bin(lock)
    }
}

impl From<CacheLock<ChainBlock>> for CacheBlock {
    fn from(lock: CacheLock<ChainBlock>) -> CacheBlock {
        Self::Chain(lock)
    }
}

impl TryFrom<CacheBlock> for CacheLock<Bytes> {
    type Error = TCError;

    fn try_from(block: CacheBlock) -> TCResult<Self> {
        match block {
            CacheBlock::Bin(block) => Ok(block),
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

    /// Lock this value immutably for reading.
    pub async fn read(&self) -> RwLockReadGuard<T> {
        self.lock.read().await
    }

    /// Lock this value mutably and exclusively for writing.
    pub async fn write(&self) -> RwLockWriteGuard<T> {
        self.lock.write().await
    }

    /// Return the number of references to this cache entry.
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

struct Inner {
    tx: mpsc::UnboundedSender<Evict>,
    size: usize,
    max_size: usize,
    entries: HashMap<PathBuf, CacheBlock>,
    lfu: LFU<PathBuf>,
}

impl Inner {
    async fn remove(&mut self, path: &PathBuf) -> TCResult<()> {
        if let Some(old_block) = self.entries.remove(path) {
            let as_bytes = old_block.into_bytes().await;
            let block_size = as_bytes.len();

            fs::write(path, as_bytes)
                .map_err(|e| io_err(e, &path))
                .await?;

            if self.size > block_size {
                self.size -= block_size;
            }

            self.lfu.remove(&path);
        }

        Ok(())
    }
}

/// The filesystem cache.
#[derive(Clone)]
pub struct Cache {
    inner: RwLock<Inner>,
}

impl Cache {
    /// Construct a new cache with the given size.
    pub fn new(max_size: usize) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();

        let cache = Self {
            inner: RwLock::new(Inner {
                tx,
                size: 0,
                max_size,
                entries: HashMap::new(),
                lfu: LFU::new(),
            }),
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
        let mut inner = self.inner.write().await;
        if let Some(lock) = inner.entries.get(path) {
            debug!("cache hit: {:?}", path);
            let lock = lock.clone().try_into()?;
            inner.lfu.bump(path);
            return Ok(Some(lock));
        } else {
            log::info!("cache miss: {:?}", path);
        }

        let block = match fs::read(path).await {
            Ok(block) => Bytes::from(block),
            Err(err) if err.kind() == io::ErrorKind::NotFound => {
                debug!("no such block: {:?}", path);
                return Ok(None);
            }
            Err(err) => return Err(io_err(err, path)),
        };

        debug!("cache insert: {:?}", path);

        let size = block.len();
        let block = B::try_from(block).map_err(|_| TCError::internal("unable to decode block"))?;
        let lock = CacheLock::new(block);
        let cached = CacheBlock::from(lock.clone());

        inner.size += size;
        inner.lfu.insert(path.clone());
        inner.entries.insert(path.clone(), cached);

        Ok(Some(lock))
    }

    /// Update a block in the cache.
    pub async fn write<B: BlockData>(&self, path: PathBuf, block: B) -> TCResult<CacheLock<B>>
    where
        CacheBlock: From<CacheLock<B>>,
    {
        let size = {
            let as_bytes: Bytes = block.clone().into();
            as_bytes.len()
        };

        let mut inner = self.inner.write().await;

        if let Some(old_block) = inner.entries.remove(&path) {
            let old_size = old_block.into_bytes().await.len();
            if old_size > inner.size {
                inner.size -= old_size;
            }
        } else {
            inner.lfu.insert(path.clone());
        }

        let block = CacheLock::new(block);
        inner.lfu.bump(&path);
        inner.entries.insert(path, block.clone().into());
        inner.size += size;
        if inner.size > inner.max_size {
            inner.tx.send(Evict).map_err(TCError::internal)?;
        }

        Ok(block)
    }

    /// Remove a block from the cache.
    pub async fn remove(&self, path: PathBuf) -> TCResult<()> {
        let mut inner = self.inner.write().await;
        inner.remove(&path).await
    }

    /// Synchronize a cached block with the filesystem.
    pub async fn sync(&self, path: &PathBuf) -> TCResult<()> {
        debug!("sync block at {:?} with filesystem", &path);

        let inner = self.inner.read().await;
        if let Some(block) = inner.entries.get(path) {
            let as_bytes = block.clone().into_bytes().await;

            create_parent(path).await?;

            fs::write(path, as_bytes)
                .map_err(|e| io_err(e, path))
                .await?;
        } else {
            log::warn!("no such block! {:?}", path);
        }

        Ok(())
    }
}

struct LFU<T: Hash> {
    entries: HashMap<T, usize>,
    priority: Vec<T>,
}

impl<T: Clone + Eq + Hash> LFU<T> {
    fn new() -> Self {
        LFU {
            entries: HashMap::new(),
            priority: Vec::new(),
        }
    }

    fn bump(&mut self, id: &T) {
        let (r_id, r) = self.entries.remove_entry(id).unwrap();
        if r == 0 {
            self.entries.insert(r_id, r);
        } else {
            let (l_id, l) = self.entries.remove_entry(&self.priority[r - 1]).unwrap();
            self.priority.swap(l, r);
            self.entries.insert(l_id, r);
            self.entries.insert(r_id, l);
        }
    }

    fn insert(&mut self, id: T) {
        assert!(!self.entries.contains_key(&id));

        self.entries.insert(id.clone(), self.priority.len());
        self.priority.push(id.clone());
    }

    fn remove(&mut self, id: &T) {
        if let Some(i) = self.entries.remove(id) {
            self.priority.remove(i);
        }
    }
}

fn spawn_cleanup_thread(cache: Cache, mut rx: mpsc::UnboundedReceiver<Evict>) {
    tokio::spawn(async move {
        while rx.recv().await.is_some() {
            let mut cache = cache.inner.write().await;
            let mut priority = cache.lfu.priority.clone().into_iter();
            while cache.size > cache.max_size {
                if let Some(block_id) = priority.next() {
                    let evict = {
                        let block = cache.entries.get(&block_id).expect("cache internal");
                        block.ref_count() == 1
                    };

                    if evict {
                        cache.remove(&block_id).await.expect("cache block sync");
                    }
                } else {
                    break;
                }
            }
        }
    });
}
