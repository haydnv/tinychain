use std::collections::hash_map::{Entry, HashMap};
use std::fs::Metadata;
use std::path::PathBuf;
use std::pin::Pin;

use futures::future::{Future, TryFutureExt};
use futures_locks::RwLock;
use tokio::fs;

use error::*;
use generic::{Id, PathSegment};

use super::{file_name, fs_path, io_err, File};
use crate::dir_contents;

pub enum DirEntry {
    Dir(RwLock<Dir>),
    File(RwLock<File>),
}

impl DirEntry {
    async fn load(mount_point: PathBuf, entries: Vec<(fs::DirEntry, Metadata)>) -> TCResult<Self> {
        if entries.iter().all(|(_, meta)| meta.is_dir()) {
            let dir = Dir::load(mount_point, entries).await?;
            Ok(DirEntry::Dir(RwLock::new(dir)))
        } else if entries.iter().all(|(_, meta)| meta.is_file()) {
            let file = File::load(mount_point, entries)?;
            Ok(DirEntry::File(RwLock::new(file)))
        } else {
            Err(TCError::internal(format!(
                "host directory {:?} contains both blocks and subdirectories",
                &mount_point
            )))
        }
    }
}

pub struct Dir {
    mount_point: PathBuf,
    contents: HashMap<PathSegment, DirEntry>,
}

impl Dir {
    fn load(
        mount_point: PathBuf,
        entries: Vec<(fs::DirEntry, Metadata)>,
    ) -> Pin<Box<dyn Future<Output = TCResult<Self>>>> {
        Box::pin(async move {
            if entries.iter().all(|(_, meta)| meta.is_dir()) {
                let mut contents = HashMap::new();
                for (handle, _) in entries.into_iter() {
                    let name = file_name(&handle)?;

                    let sub_path = fs_path(&mount_point, &name);
                    let sub_entries = dir_contents(&sub_path).await?;
                    let entry = DirEntry::load(sub_path, sub_entries).await?;

                    contents.insert(name, entry);
                }

                Ok(Self {
                    mount_point,
                    contents,
                })
            } else {
                Err(TCError::internal(format!(
                    "host directory {:?} represents a Tinychain directory, not a file",
                    mount_point
                )))
            }
        })
    }

    pub async fn new(mount_point: PathBuf) -> TCResult<Self> {
        let entries = dir_contents(&mount_point).await?;
        Self::load(mount_point, entries).await
    }

    pub fn contents(&'_ self) -> &'_ HashMap<PathSegment, DirEntry> {
        &self.contents
    }

    pub async fn create_dir(&mut self, name: PathSegment) -> TCResult<RwLock<Dir>> {
        match self.contents.entry(name) {
            Entry::Occupied(entry) => Err(TCError::bad_request(
                "the filesystem already has an entry at",
                entry.key(),
            )),
            Entry::Vacant(entry) => {
                let name = entry.key().clone();
                let dir = Dir::new(fs_path(&self.mount_point, &name)).await?;
                let dir = RwLock::new(dir);
                entry.insert(DirEntry::Dir(dir.clone()));
                Ok(dir)
            }
        }
    }

    pub async fn create_file(&mut self, name: PathSegment) -> TCResult<RwLock<File>> {
        match self.contents.entry(name) {
            Entry::Occupied(entry) => Err(TCError::bad_request(
                "The filesystem already has an entry at",
                entry.key(),
            )),
            Entry::Vacant(entry) => {
                let name = entry.key().clone();
                let file = File::create(fs_path(&self.mount_point, &name));
                let file = RwLock::new(file);
                entry.insert(DirEntry::File(file.clone()));
                Ok(file)
            }
        }
    }

    pub async fn create_or_get_dir(&mut self, name: &PathSegment) -> TCResult<RwLock<Dir>> {
        match self.get_dir(name) {
            Ok(Some(dir)) => Ok(dir),
            Ok(None) => self.create_dir(name.clone()).await,
            Err(cause) => Err(cause),
        }
    }

    pub async fn create_or_get_file(&mut self, name: &PathSegment) -> TCResult<RwLock<File>> {
        match self.get_file(name) {
            Ok(Some(file)) => Ok(file),
            Ok(None) => self.create_file(name.clone()).await,
            Err(cause) => Err(cause),
        }
    }

    pub async fn delete(&mut self, name: &PathSegment) -> TCResult<()> {
        if self.contents.contains_key(name) {
            let path = fs_path(&self.mount_point, name);
            fs::remove_file(&path).map_err(|e| io_err(e, &path)).await?;
            self.contents.remove(name);
            Ok(())
        } else {
            Err(TCError::not_found(name))
        }
    }

    pub fn get_dir(&self, name: &PathSegment) -> TCResult<Option<RwLock<Dir>>> {
        match self.contents.get(name) {
            None => Ok(None),
            Some(DirEntry::Dir(dir)) => Ok(Some(dir.clone())),
            Some(DirEntry::File(_)) => Err(TCError::bad_request(
                "Expected filesystem directory but found",
                "(file)",
            )),
        }
    }

    pub fn get_file(&self, name: &Id) -> TCResult<Option<RwLock<File>>> {
        match self.contents.get(name) {
            None => Ok(None),
            Some(DirEntry::File(file)) => Ok(Some(file.clone())),
            Some(DirEntry::Dir(_)) => Err(TCError::bad_request(
                "Expected filesystem file but found",
                "(directory)",
            )),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.contents.is_empty()
    }
}
