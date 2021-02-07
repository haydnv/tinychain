use std::collections::hash_map::{Entry, HashMap};
use std::convert::TryInto;
use std::fmt;
use std::hash::Hash;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use bytes::Bytes;
use futures::TryFutureExt;
use futures_locks::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use tokio::fs;

use error::*;
use generic::{Id, PathSegment};

use crate::chain::ChainBlock;

use super::BlockData;

pub struct CacheLock<T> {
    ref_count: Arc<std::sync::RwLock<usize>>,
    lock: RwLock<T>,
}

impl<T> CacheLock<T> {
    fn new(value: T) -> Self {
        Self {
            ref_count: Arc::new(std::sync::RwLock::new(0)),
            lock: RwLock::new(value),
        }
    }

    async fn read(&self) -> RwLockReadGuard<T> {
        self.lock.read().await
    }

    async fn write(&self) -> RwLockWriteGuard<T> {
        self.lock.write().await
    }
}

impl<T> Clone for CacheLock<T> {
    fn clone(&self) -> Self {
        *self.ref_count.write().unwrap() += 1;

        Self {
            ref_count: self.ref_count.clone(),
            lock: self.lock.clone(),
        }
    }
}

impl<T> Drop for CacheLock<T> {
    fn drop(&mut self) {
        *self.ref_count.write().unwrap() -= 1;
    }
}

pub struct CacheDir {
    mount_point: PathBuf,
    contents: HashMap<PathSegment, CacheDirEntry>,
    cache: Cache,
}

enum CacheDirEntry {
    Dir(Arc<CacheDirEntry>),
    File(Arc<CacheFileEntry>),
}

pub struct CacheFile<B: BlockData> {
    mount_point: PathBuf,
    blocks: RwLock<HashMap<Id, Option<CacheLock<B>>>>,
    cache: Cache,
}

enum CacheFileEntry {
    Chain(CacheFile<ChainBlock>),
}

impl<B: BlockData> CacheFile<B> {
    async fn create_block(&self, block_id: Id, value: B) -> TCResult<CacheLock<B>> {
        let mut blocks = self.blocks.write().await;
        match blocks.entry(block_id) {
            Entry::Occupied(entry) => Err(TCError::bad_request(
                "There is already a block at",
                entry.key(),
            )),
            Entry::Vacant(entry) => {
                self.cache
                    .insert(fs_path(&self.mount_point, entry.key()), value.size())
                    .await;

                let lock = CacheLock::new(value);
                entry.insert(Some(lock.clone()));
                Ok(lock)
            }
        }
    }

    async fn get_block(&self, block_id: &Id) -> TCResult<Option<CacheLock<B>>> {
        {
            let blocks = self.blocks.read().await;

            if let Some(entry) = blocks.get(block_id) {
                if let Some(block) = entry {
                    return Ok(Some(block.clone()));
                }
            } else {
                return Ok(None);
            }
        }

        let path = fs_path(&self.mount_point, block_id);
        let block = fs::read(&path)
            .map_ok(Bytes::from)
            .map_err(|e| io_err(e, &path))
            .await?;

        let block = CacheLock::new(block.try_into()?);

        self.cache.bump(&path).await;
        let mut blocks = self.blocks.write().await;
        blocks.insert(block_id.clone(), Some(block.clone()));

        Ok(Some(block))
    }
}

struct LFU<T: Hash> {
    entries: HashMap<T, usize>,
    priority: Vec<T>,
    size: usize,
}

impl<T: Clone + Eq + Hash> LFU<T> {
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

    fn insert(&mut self, id: T, size: usize) {
        if self.entries.contains_key(&id) {
            self.bump(&id);
        } else {
            self.entries.insert(id.clone(), self.priority.len());
            self.priority.push(id);
            self.size += size;
        }
    }
}

struct Inner {
    max_size: usize,
    root: RwLock<CacheDir>,
    lfu: RwLock<LFU<PathBuf>>,
}

#[derive(Clone)]
pub struct Cache {
    inner: Arc<Inner>,
}

impl Cache {
    async fn bump(&self, path: &PathBuf) {
        let mut lfu = self.inner.lfu.write().await;
        lfu.bump(path);
    }

    async fn evict(&mut self) {
        // TODO
    }

    async fn insert(&self, path: PathBuf, size: usize) {
        let mut lfu = self.inner.lfu.write().await;
        lfu.insert(path, size);

        if lfu.size > self.inner.max_size {
            // TODO: evict
        }
    }
}

fn file_name(handle: &fs::DirEntry) -> TCResult<PathSegment> {
    if let Some(name) = handle.file_name().to_str() {
        name.parse()
    } else {
        Err(TCError::internal("Cannot load file with no name!"))
    }
}

fn fs_path(mount_point: &PathBuf, name: &PathSegment) -> PathBuf {
    let mut path = mount_point.clone();
    path.push(name.to_string());
    path
}

fn io_err<I: fmt::Debug>(err: io::Error, info: I) -> TCError {
    match err.kind() {
        io::ErrorKind::NotFound => {
            TCError::unsupported(format!("There is no directory at {:?}", info))
        }
        io::ErrorKind::PermissionDenied => TCError::internal(format!(
            "Tinychain does not have permission to access the host filesystem: {:?}",
            info
        )),
        other => TCError::internal(format!("host filesystem error: {:?}", other)),
    }
}
