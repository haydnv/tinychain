use std::collections::hash_map::{Entry, HashMap};
use std::mem;
use std::path::PathBuf;

use bytes::{BufMut, Bytes, BytesMut};

use crate::error;
use crate::value::link::PathSegment;
use crate::value::TCResult;

use super::lock::{RwLock, RwLockReadGuard, RwLockWriteGuard};

#[derive(Eq, PartialEq)]
enum BlockDelta {
    None,
    Append(usize),
    Rewrite,
}

struct CachedBlock {
    delta: BlockDelta,
    contents: BytesMut,
}

impl CachedBlock {
    fn new() -> CachedBlock {
        CachedBlock {
            delta: BlockDelta::None,
            contents: BytesMut::new(),
        }
    }

    fn append(&mut self, data: Bytes) {
        if data.is_empty() {
            return;
        }

        if self.delta == BlockDelta::None {
            self.delta = BlockDelta::Append(self.contents.len())
        }

        self.contents.put(data);
    }

    fn insert(&mut self, offset: usize, data: Bytes) {
        if data.is_empty() {
            return;
        }

        self.delta = BlockDelta::Rewrite;

        let tail = self.contents.split_to(offset);
        self.contents.put(data);
        self.contents.put(tail);
    }
}

pub struct Block {
    name: PathSegment,
    data: Option<CachedBlock>,
}

impl Block {
    pub fn new(name: PathSegment) -> Block {
        Block {
            name,
            data: Some(CachedBlock::new()),
        }
    }

    pub async fn copy_from(&mut self, mut other: Block) {
        if other.data.is_some() {
            mem::swap(&mut self.data, &mut other.data);
        } else {
            // TODO: replace this block on the filesystem
            panic!("NOT IMPLEMENTED")
        }
    }

    pub async fn append(&mut self, data: Bytes) {
        if let Some(cached) = &mut self.data {
            cached.append(data);
        } else {
            // TODO: append to this block on the filesystem
            panic!("NOT IMPLEMENTED")
        }
    }

    pub async fn insert(&mut self, offset: usize, data: Bytes) {
        if let Some(cached) = &mut self.data {
            cached.insert(offset, data);
        } else {
            // TODO: read data from filesystem
            panic!("NOT IMPLEMENTED")
        }
    }
}

pub enum DirEntry {
    Block(Block),
    Dir(Dir),
}

pub struct Dir {
    name: PathSegment,
    contents: HashMap<PathSegment, RwLock<DirEntry>>,
    exists_on_fs: bool,
}

impl Dir {
    fn new(name: PathSegment) -> Dir {
        Dir {
            name,
            contents: HashMap::new(),
            exists_on_fs: false,
        }
    }

    pub fn create_block(&mut self, name: PathSegment) -> TCResult<()> {
        match self.contents.entry(name) {
            Entry::Occupied(entry) => Err(error::bad_request(
                "The filesystem already has an entry at",
                entry.key(),
            )),
            Entry::Vacant(entry) => {
                let name = entry.key().clone();
                entry.insert(RwLock::new(DirEntry::Block(Block::new(name))));
                Ok(())
            }
        }
    }

    pub fn create_dir(&mut self, name: PathSegment) -> TCResult<()> {
        match self.contents.entry(name) {
            Entry::Occupied(entry) => Err(error::bad_request(
                "The filesystem already has an entry at",
                entry.key(),
            )),
            Entry::Vacant(entry) => {
                let name = entry.key().clone();
                entry.insert(RwLock::new(DirEntry::Dir(Dir::new(name))));
                Ok(())
            }
        }
    }

    pub async fn get(&self, name: &PathSegment) -> Option<RwLockReadGuard<DirEntry>> {
        match self.contents.get(name) {
            None => None,
            Some(lock) => Some(lock.read().await),
        }
    }

    pub async fn get_mut(&self, name: &PathSegment) -> Option<RwLockWriteGuard<DirEntry>> {
        match self.contents.get(name) {
            None => None,
            Some(lock) => Some(lock.write().await),
        }
    }
}

pub struct FileSystem {
    mount_point: PathBuf,
    contents: HashMap<PathSegment, Dir>,
}

impl FileSystem {
    pub async fn new(mount_point: PathBuf) -> FileSystem {
        // TODO: load from disk
        FileSystem {
            mount_point,
            contents: HashMap::new(),
        }
    }

    pub fn create_dir<'a>(&'a mut self, name: PathSegment) -> TCResult<&'a Dir> {
        match self.contents.entry(name) {
            Entry::Occupied(entry) => Err(error::bad_request(
                "The filesystem cache already has a directory at",
                entry.key(),
            )),
            Entry::Vacant(entry) => {
                let name = entry.key().clone();
                Ok(entry.insert(Dir::new(name)))
            }
        }
    }

    pub fn get_dir<'a>(&'a self, name: &PathSegment) -> Option<&'a Dir> {
        self.contents.get(name)
    }

    pub fn get_or_create_dir<'a>(&'a mut self, name: &'a PathSegment) -> Option<&'a Dir> {
        if self.contents.contains_key(name) {
            self.get_dir(name)
        } else {
            Some(self.create_dir(name.clone()).unwrap())
        }
    }
}
