//! The filesystem cache, with LFU eviction

use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use std::hash::Hash;
use std::ops::Deref;
use std::path::PathBuf;

use destream::IntoStream;
use futures::{Future, TryFutureExt};
use log::debug;
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

    async fn size(&self) -> TCResult<u64> {
        match self {
            Self::BTree(block) => block.read().await.deref().size().await,
            Self::Chain(block) => block.read().await.deref().size().await,
            Self::Value(block) => block.read().await.deref().size().await,
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
        self.lock.read().await
    }

    pub async fn write(&self) -> RwLockWriteGuard<T> {
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

struct Inner {
    tx: mpsc::UnboundedSender<Evict>,
    size: u64,
    max_size: u64,
    entries: HashMap<PathBuf, CacheBlock>,
    lfu: LFU<PathBuf>,
}

impl Inner {
    async fn remove(&mut self, path: &PathBuf) -> TCResult<()> {
        if let Some(block) = self.entries.remove(path) {
            let mut block_file = write_file(path).await?;
            let new_size = block.persist(&mut block_file).await?;

            // TODO: keep track of the difference between new_size and old_size
            if self.size > new_size {
                self.size -= new_size;
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
    pub fn new(max_size: u64) -> Self {
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
        } else if !path.exists() {
            return Ok(None);
        } else {
            log::info!("cache miss: {:?}", path);
        }

        let block_file = read_file(&path).await?;
        let metadata = block_file.metadata().map_err(|e| io_err(e, &path)).await?;
        let block = B::load(block_file).await?;

        debug!("cache insert: {:?}", path);

        let lock = CacheLock::new(block);
        let cached = CacheBlock::from(lock.clone());

        inner.size += metadata.len();
        inner.lfu.insert(path.clone());
        inner.entries.insert(path.clone(), cached);

        Ok(Some(lock))
    }

    /// Remove a block from the cache.
    pub async fn remove(&self, path: &PathBuf) -> TCResult<()> {
        let mut inner = self.inner.write().await;
        inner.remove(path).await
    }

    /// Remove a block from the cache and delete it from the filesystem.
    pub async fn remove_and_delete(&self, path: &PathBuf) -> TCResult<()> {
        let mut inner = self.inner.write().await;
        inner.remove(path).await?;
        tokio::fs::remove_file(path)
            .map_err(|e| io_err(e, path))
            .await
    }

    async fn _sync(inner: RwLockReadGuard<Inner>, path: &PathBuf) -> TCResult<bool> {
        debug!("sync block at {:?} with filesystem", &path);

        if let Some(block) = inner.entries.get(path) {
            let mut block_file = if path.exists() {
                debug!("open block file at {:?} for sync", path);
                write_file(path).await?
            } else {
                debug!("create new filesystem block at {:?}", path);
                create_parent(path).await?;
                create_file(path).await?
            };

            block.persist(&mut block_file).await?;

            Ok(true)
        } else {
            log::info!("cache sync miss: {:?}", path);
            Ok(path.exists())
        }
    }

    /// Synchronize a cached block with the filesystem.
    pub async fn sync(&self, path: &PathBuf) -> TCResult<bool> {
        debug!("sync block at {:?} with filesystem", &path);

        let inner = self.inner.read().await;
        Self::_sync(inner, path).await
    }

    async fn _write<'en, B: BlockData + IntoStream<'en> + 'en>(
        inner: &mut RwLockWriteGuard<Inner>,
        path: PathBuf,
        block: B,
    ) -> TCResult<CacheLock<B>>
    where
        CacheBlock: From<CacheLock<B>>,
    {
        if let Some(old_block) = inner.entries.remove(&path) {
            inner.size -= old_block.size().await?;
            inner.lfu.bump(&path);
        } else {
            inner.lfu.insert(path.clone());
        }

        let size = block.clone().into_size().await?;
        let block = CacheLock::new(block);
        inner.entries.insert(path, block.clone().into());
        inner.size += size;
        if inner.size > inner.max_size {
            inner.tx.send(Evict).map_err(TCError::internal)?;
        }

        Ok(block)
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

        let mut inner = self.inner.write().await;
        Self::_write(&mut inner, path, block).await
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
        let mut inner = self.inner.write().await;
        Self::_write(&mut inner, path.clone(), block).await?;
        let inner = inner.downgrade().await;
        Self::_sync(inner, &path).await
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
