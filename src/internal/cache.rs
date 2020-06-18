use std::collections::hash_map::{Entry, HashMap};
use std::path::PathBuf;

use bytes::{BufMut, Bytes, BytesMut};

use crate::error;
use crate::value::link::PathSegment;
use crate::value::TCResult;

use super::lock::RwLock;

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

    fn copy_from_slice(slice: &[u8]) -> CachedBlock {
        let mut contents = BytesMut::with_capacity(slice.len());
        contents.extend_from_slice(slice);

        CachedBlock {
            delta: BlockDelta::Rewrite,
            contents,
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

    pub async fn as_bytes(&'_ self) -> &'_ [u8] {
        if let Some(data) = &self.data {
            &data.contents[..]
        } else {
            // TODO: read from filesystem
            panic!("NOT IMPLEMENTED")
        }
    }

    pub fn name(&'_ self) -> &'_ PathSegment {
        &self.name
    }

    pub async fn copy_from(&mut self, other: &Block) {
        self.data = Some(CachedBlock::copy_from_slice(other.as_bytes().await))
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
    Block(RwLock<Block>),
    Dir(RwLock<Dir>),
}

pub struct Dir {
    name: PathSegment,
    contents: HashMap<PathSegment, DirEntry>,
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

    pub fn create_block(&mut self, name: PathSegment) -> TCResult<RwLock<Block>> {
        match self.contents.entry(name) {
            Entry::Occupied(entry) => Err(error::bad_request(
                "The filesystem already has an entry at",
                entry.key(),
            )),
            Entry::Vacant(entry) => {
                let name = entry.key().clone();
                let block = RwLock::new(Block::new(name));
                entry.insert(DirEntry::Block(block.clone()));
                Ok(block)
            }
        }
    }

    pub fn create_dir(&mut self, name: PathSegment) -> TCResult<RwLock<Dir>> {
        match self.contents.entry(name) {
            Entry::Occupied(entry) => Err(error::bad_request(
                "The filesystem already has an entry at",
                entry.key(),
            )),
            Entry::Vacant(entry) => {
                let name = entry.key().clone();
                let dir = RwLock::new(Dir::new(name));
                entry.insert(DirEntry::Dir(dir.clone()));
                Ok(dir)
            }
        }
    }

    pub fn create_or_get_dir(&mut self, name: &PathSegment) -> TCResult<RwLock<Dir>> {
        match self.get_dir(name) {
            Ok(Some(dir)) => Ok(dir),
            Err(cause) => Err(cause),
            Ok(None) => self.create_dir(name.clone()),
        }
    }

    pub fn get_block(&self, name: &PathSegment) -> TCResult<Option<RwLock<Block>>> {
        match self.contents.get(name) {
            None => Ok(None),
            Some(DirEntry::Block(block)) => Ok(Some(block.clone())),
            Some(DirEntry::Dir(_)) => Err(error::bad_request(
                "Expected filesystem block but found",
                "(directory)",
            )),
        }
    }

    pub fn get_dir(&self, name: &PathSegment) -> TCResult<Option<RwLock<Dir>>> {
        match self.contents.get(name) {
            None => Ok(None),
            Some(DirEntry::Dir(dir)) => Ok(Some(dir.clone())),
            Some(DirEntry::Block(_)) => Err(error::bad_request(
                "Expected filesystem directory but found",
                "(block)",
            )),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.contents.is_empty()
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
