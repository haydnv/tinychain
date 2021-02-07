use std::collections::hash_map::{Entry, HashMap};
use std::collections::HashSet;
use std::convert::TryInto;
use std::fmt;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use bytes::Bytes;
use futures::TryFutureExt;
use futures_locks::RwLock;
use tokio::fs;

use error::*;
use generic::{Id, PathSegment};

use crate::chain::ChainBlock;

use super::BlockData;

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
    blocks: RwLock<HashMap<Id, Option<RwLock<B>>>>,
    cache: Cache,
}

enum CacheFileEntry {
    Chain(CacheFile<ChainBlock>),
}

impl<B: BlockData> CacheFile<B> {
    async fn create_block(&self, block_id: Id, value: B) -> TCResult<RwLock<B>> {
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

                let lock = RwLock::new(value);
                entry.insert(Some(lock.clone()));
                Ok(lock)
            }
        }
    }

    async fn get_block(&self, block_id: &Id) -> TCResult<Option<RwLock<B>>> {
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

        let block = RwLock::new(block.try_into()?);

        self.cache.bump(&path).await;
        let mut blocks = self.blocks.write().await;
        blocks.insert(block_id.clone(), Some(block.clone()));

        Ok(Some(block))
    }
}

struct Inner {
    size: usize,
    max_size: usize,
    priority: RwLock<Vec<PathBuf>>,
    entries: HashSet<PathBuf>,
    root: Arc<CacheDir>,
}

#[derive(Clone)]
pub struct Cache {
    inner: Arc<Inner>,
}

impl Cache {
    async fn bump(&self, _path: &PathBuf) {
        unimplemented!()
    }

    async fn insert(&self, _path: PathBuf, _size: usize) {
        unimplemented!()
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
