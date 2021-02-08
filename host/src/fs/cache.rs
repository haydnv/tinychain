use std::collections::hash_map::{Entry, HashMap};
use std::convert::TryInto;
use std::fmt;
use std::fs::Metadata;
use std::hash::Hash;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use bytes::Bytes;
use futures::TryFutureExt;
use futures_locks::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use tokio::fs;

use error::*;
use generic::{label, Label, NativeClass, PathSegment, TCPathBuf};
use transact::fs::{BlockData, BlockId};

use crate::chain::ChainBlock;
use crate::state::StateType;

const CLASS: Label = label(".class");

struct Inner {
    clients: RwLock<HashMap<PathBuf, RwLock<CacheDir>>>,
    lfu: RwLock<LFU<PathBuf>>,
    max_size: usize,
}

#[derive(Clone)]
pub struct Cache {
    inner: Arc<Inner>,
}

impl Cache {
    pub fn new(max_size: usize) -> Self {
        let inner = Inner {
            clients: RwLock::new(HashMap::new()),
            lfu: RwLock::new(LFU::new()),
            max_size,
        };

        Self {
            inner: Arc::new(inner),
        }
    }

    pub async fn register(&self, path: PathBuf, dir: RwLock<CacheDir>) {
        let mut clients = self.inner.clients.write().await;
        clients.insert(path, dir);
    }

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

#[derive(Clone)]
pub struct CacheFile<B> {
    cache: Cache,
    mount_point: PathBuf,
    blocks: HashMap<BlockId, Option<CacheLock<B>>>,
}

impl<B: BlockData> CacheFile<B> {
    fn new(cache: Cache, mount_point: PathBuf) -> Self {
        Self {
            cache,
            mount_point,
            blocks: HashMap::new(),
        }
    }

    async fn load(
        cache: Cache,
        mount_point: PathBuf,
        contents: Vec<(fs::DirEntry, Metadata)>,
    ) -> TCResult<Self> {
        let mut blocks = HashMap::new();

        for (handle, meta) in contents.into_iter() {
            let name = file_name(&handle)?;
            let path = fs_path(&mount_point, &name);

            if !meta.is_file() {
                return Err(TCError::internal(format!(
                    "block {:?} is not a file",
                    &path
                )));
            }

            blocks.insert(name, None);
        }

        Ok(Self {
            cache,
            mount_point,
            blocks,
        })
    }
}

pub enum CacheFileEntry {
    Chain(RwLock<CacheFile<ChainBlock>>),
}

impl CacheFileEntry {
    async fn load(
        cache: Cache,
        mount_point: PathBuf,
        contents: Vec<(fs::DirEntry, Metadata)>,
    ) -> TCResult<Self> {
        if contents.iter().all(|(_, meta)| meta.is_file()) {
            let classfile = fs::read_to_string(fs_path(&mount_point, &CLASS.into()))
                .map_err(|e| io_err(e, &mount_point))
                .await?;

            let classpath: TCPathBuf = classfile.parse()?;
            if let Some(StateType::Chain(_)) = StateType::from_path(&classpath) {
                let file = CacheFile::load(cache, mount_point, contents).await?;
                Ok(Self::Chain(RwLock::new(file)))
            } else {
                Err(TCError::internal(format!(
                    "unrecognized file class: {}",
                    classpath
                )))
            }
        } else {
            Err(TCError::internal(format!(
                "file at {:?} must not contain subdirectories",
                &mount_point
            )))
        }
    }
}

impl<B: BlockData> CacheFile<B> {
    pub async fn create_block(&mut self, block_id: BlockId, value: B) -> TCResult<CacheLock<B>> {
        match self.blocks.entry(block_id) {
            Entry::Occupied(entry) => Err(TCError::bad_request(
                "There is already a block at",
                entry.key(),
            )),
            Entry::Vacant(entry) => {
                let size = value.clone().into().len();
                self.cache
                    .insert(fs_path(&self.mount_point, entry.key()), size)
                    .await;

                let lock = CacheLock::new(value);
                entry.insert(Some(lock.clone()));
                Ok(lock)
            }
        }
    }

    pub async fn get_block(
        file: &RwLock<Self>,
        block_id: &BlockId,
    ) -> TCResult<Option<CacheLock<B>>> {
        let path = {
            let this = file.read().await;
            let path = fs_path(&this.mount_point, block_id);
            this.cache.bump(&path).await;

            if let Some(entry) = this.blocks.get(block_id) {
                if let Some(block) = entry {
                    return Ok(Some(block.clone()));
                }
            }

            path
        };

        let mut this = file.write().await;
        let block = fs::read(&path)
            .map_ok(Bytes::from)
            .map_err(|e| io_err(e, &path))
            .await?;

        let block = CacheLock::new(block.try_into()?);

        this.blocks.insert(block_id.clone(), Some(block.clone()));

        Ok(Some(block))
    }
}

struct LFU<T: Hash> {
    entries: HashMap<T, usize>,
    priority: Vec<T>,
    size: usize,
}

impl<T: Clone + Eq + Hash> LFU<T> {
    fn new() -> Self {
        LFU {
            entries: HashMap::new(),
            priority: Vec::new(),
            size: 0,
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

    fn insert(&mut self, id: T, size: usize) {
        self.entries.insert(id.clone(), self.priority.len());
        self.priority.push(id.clone());
        self.size += size;
    }
}

pub enum CacheDirEntry {
    Dir(RwLock<CacheDir>),
    File(CacheFileEntry),
}

impl From<RwLock<CacheFile<ChainBlock>>> for CacheDirEntry {
    fn from(file: RwLock<CacheFile<ChainBlock>>) -> Self {
        Self::File(CacheFileEntry::Chain(file))
    }
}

pub struct CacheDir {
    cache: Cache,
    mount_point: PathBuf,
    contents: HashMap<PathSegment, CacheDirEntry>,
}

impl CacheDir {
    fn new(cache: Cache, mount_point: PathBuf) -> Self {
        Self {
            cache,
            mount_point,
            contents: HashMap::new(),
        }
    }

    pub async fn load(cache: Cache, mount_point: PathBuf) -> TCResult<Self> {
        let entries = dir_contents(&mount_point).await?;
        Self::_load(cache, mount_point, entries).await
    }

    async fn _load(
        cache: Cache,
        mount_point: PathBuf,
        entries: Vec<(fs::DirEntry, Metadata)>,
    ) -> TCResult<Self> {
        let mut contents = HashMap::new();

        if entries.iter().all(|(_, meta)| meta.is_dir()) {
            for (handle, _) in entries.into_iter() {
                let name = file_name(&handle)?;
                let path = fs_path(&mount_point, &name);
                let entry_contents = dir_contents(&path).await?;
                if entry_contents.iter().all(|(_, meta)| meta.is_file()) {
                    let file = CacheFileEntry::load(cache.clone(), path, entry_contents).await?;
                    contents.insert(name, CacheDirEntry::File(file));
                }
            }

            Ok(Self {
                cache,
                mount_point,
                contents,
            })
        } else {
            Err(TCError::internal(format!(
                "expected directory but found file blocks in {:?}",
                &mount_point
            )))
        }
    }

    pub async fn create_file<B: BlockData>(
        &mut self,
        name: PathSegment,
    ) -> TCResult<RwLock<CacheFile<B>>>
    where
        CacheDirEntry: From<RwLock<CacheFile<B>>>,
    {
        match self.contents.entry(name) {
            Entry::Occupied(entry) => Err(TCError::bad_request(
                "there is already a directory at",
                entry.key(),
            )),
            Entry::Vacant(entry) => {
                let file = CacheFile::<B>::new(
                    self.cache.clone(),
                    fs_path(&self.mount_point, entry.key()),
                );
                let file = RwLock::new(file);
                entry.insert(file.clone().into());
                Ok(file)
            }
        }
    }

    pub async fn create_dir(&mut self, name: PathSegment) -> TCResult<RwLock<Self>> {
        match self.contents.entry(name) {
            Entry::Occupied(entry) => Err(TCError::bad_request(
                "there is already a directory at",
                entry.key(),
            )),
            Entry::Vacant(entry) => {
                let dir = Self::new(self.cache.clone(), fs_path(&self.mount_point, entry.key()));
                let dir = RwLock::new(dir);
                entry.insert(CacheDirEntry::Dir(dir.clone()));
                Ok(dir)
            }
        }
    }
}
async fn dir_contents(dir_path: &PathBuf) -> TCResult<Vec<(fs::DirEntry, Metadata)>> {
    let mut contents = vec![];
    let mut handles = fs::read_dir(dir_path)
        .map_err(|e| io_err(e, dir_path))
        .await?;

    while let Some(handle) = handles
        .next_entry()
        .map_err(|e| io_err(e, dir_path))
        .await?
    {
        let meta = handle
            .metadata()
            .map_err(|e| io_err(e, handle.path()))
            .await?;

        contents.push((handle, meta));
    }

    Ok(contents)
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
