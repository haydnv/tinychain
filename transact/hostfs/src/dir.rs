use std::collections::hash_map::{Entry, HashMap};
use std::ops::DerefMut;
use std::path::PathBuf;

use bytes::Bytes;
use futures_locks::RwLock;

use error::*;
use generic::PathSegment;

pub enum DirEntry {
    Block(RwLock<Bytes>),
    Dir(RwLock<Dir>),
}

pub struct Dir {
    mount_point: PathBuf,
    contents: HashMap<PathSegment, DirEntry>,
}

impl Dir {
    pub fn new(mount_point: PathBuf) -> Dir {
        Dir {
            mount_point,
            contents: HashMap::new(),
        }
    }

    pub fn block_ids(&'_ self) -> impl Iterator<Item = &'_ PathSegment> + '_ {
        self.contents
            .iter()
            .filter_map(|(name, entry)| match entry {
                DirEntry::Block(_) => Some(name),
                _ => None,
            })
    }

    pub fn create_block(
        &mut self,
        name: PathSegment,
        initial_value: Bytes,
    ) -> TCResult<RwLock<Bytes>> {
        match self.contents.entry(name) {
            Entry::Occupied(entry) => Err(TCError::bad_request(
                "The filesystem already has an entry at",
                entry.key(),
            )),
            Entry::Vacant(entry) => {
                let block = RwLock::new(initial_value);
                entry.insert(DirEntry::Block(block.clone()));
                Ok(block)
            }
        }
    }

    pub async fn create_or_get_block(
        &mut self,
        name: PathSegment,
        data: Bytes,
    ) -> TCResult<RwLock<Bytes>> {
        match self.get_block(&name) {
            Ok(Some(block)) => {
                *(block.write().await.deref_mut()) = data;
                Ok(block)
            }
            Err(cause) => Err(cause),
            Ok(None) => self.create_block(name.clone(), data),
        }
    }

    pub fn copy_all(&mut self, source: &mut Dir) -> TCResult<()> {
        for block_id in source.block_ids().cloned() {
            self.copy_block(block_id, source)?;
        }

        Ok(())
    }

    pub fn copy_block(&mut self, name: PathSegment, source: &Dir) -> TCResult<()> {
        let block = source
            .get_block(&name)?
            .ok_or_else(|| TCError::not_found(&name))?;

        self.contents.insert(name, DirEntry::Block(block));
        Ok(())
    }

    pub async fn create_dir(&mut self, name: PathSegment) -> TCResult<RwLock<Dir>> {
        match self.contents.entry(name) {
            Entry::Occupied(entry) => Err(TCError::bad_request(
                "The filesystem already has an entry at",
                entry.key(),
            )),
            Entry::Vacant(entry) => {
                let name = entry.key().clone();
                let dir = RwLock::new(Dir::new(fs_path(&self.mount_point, &name)));
                entry.insert(DirEntry::Dir(dir.clone()));
                Ok(dir)
            }
        }
    }

    pub async fn create_or_get_dir(&mut self, name: &PathSegment) -> TCResult<RwLock<Dir>> {
        match self.get_dir(name) {
            Ok(Some(dir)) => Ok(dir),
            Err(cause) => Err(cause),
            Ok(None) => self.create_dir(name.clone()).await,
        }
    }

    pub fn delete(&mut self, name: &PathSegment) -> TCResult<()> {
        self.contents.remove(name);
        Ok(())
    }

    pub fn delete_block(&mut self, name: &PathSegment) -> TCResult<RwLock<Bytes>> {
        match self.contents.remove(name) {
            None => Err(TCError::not_found(name)),
            Some(DirEntry::Block(block)) => Ok(block),
            Some(entry) => {
                self.contents.insert(name.clone(), entry);
                Err(TCError::bad_request(
                    "Expected filesystem block but found",
                    "(directory)",
                ))
            }
        }
    }

    pub fn delete_dir(&mut self, name: &PathSegment) -> TCResult<Option<RwLock<Dir>>> {
        match self.contents.remove(name) {
            None => Ok(None),
            Some(DirEntry::Dir(dir)) => Ok(Some(dir)),
            Some(entry) => {
                self.contents.insert(name.clone(), entry);
                Err(TCError::bad_request(
                    "Expected filesystem directory but found",
                    "(block)",
                ))
            }
        }
    }

    pub fn get_block(&self, name: &PathSegment) -> TCResult<Option<RwLock<Bytes>>> {
        match self.contents.get(name) {
            None => Ok(None),
            Some(DirEntry::Block(block)) => Ok(Some(block.clone())),
            Some(DirEntry::Dir(_)) => Err(TCError::bad_request(
                "Expected filesystem block but found",
                "(directory)",
            )),
        }
    }

    pub fn get_dir(&self, name: &PathSegment) -> TCResult<Option<RwLock<Dir>>> {
        match self.contents.get(name) {
            None => Ok(None),
            Some(DirEntry::Dir(dir)) => Ok(Some(dir.clone())),
            Some(DirEntry::Block(_)) => Err(TCError::bad_request(
                "Expected filesystem directory but found",
                "(block)",
            )),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.contents.is_empty()
    }
}

fn fs_path(mount_point: &PathBuf, name: &PathSegment) -> PathBuf {
    let mut path = mount_point.clone();
    path.push(name.to_string());
    path
}
