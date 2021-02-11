//! A transactional filesystem directory.

use std::collections::HashMap;
use std::convert::TryFrom;
use std::path::PathBuf;
use std::pin::Pin;

use async_trait::async_trait;
use bytes::Bytes;
use futures::future::{join_all, Future, TryFutureExt};
use log::{debug, warn};

use error::*;
use generic::{Id, PathSegment};
use transact::lock::{Mutable, TxnLock};
use transact::TxnId;
use transact::{fs, Transact};

use crate::chain::{self, ChainBlock};
use crate::scalar::ScalarType;
use crate::state::StateType;

use super::{dir_contents, file_ext, file_name, fs_path, Cache, DirContents, File};

pub const BIN_EXT: &str = "bin";

/// A file in a [`Dir`].
#[derive(Clone)]
pub enum FileEntry {
    Bin(File<Bytes>),
    Chain(File<ChainBlock>),
}

impl FileEntry {
    fn new(cache: Cache, path: PathBuf, class: StateType) -> TCResult<Self> {
        match class {
            StateType::Chain(_) => Ok(Self::Chain(File::new(cache, path, chain::EXT))),
            StateType::Scalar(st) => match st {
                ScalarType::Value(_) => Ok(Self::Bin(File::new(cache, path, BIN_EXT))),
                other => Err(TCError::bad_request("cannot create file for", other)),
            },
            other => Err(TCError::bad_request("cannot create file for", other)),
        }
    }

    async fn load(cache: Cache, path: PathBuf, contents: DirContents) -> TCResult<Self> {
        let ext = file_ext(&path)
            .ok_or_else(|| TCError::unsupported(format!("file at {:?} has no extension", &path)))?;

        match ext {
            BIN_EXT => {
                let file = File::load(cache.clone(), path, contents).await?;
                Ok(FileEntry::Bin(file))
            }
            chain::EXT => {
                let file = File::load(cache.clone(), path, contents).await?;
                Ok(FileEntry::Chain(file))
            }
            other => Err(TCError::internal(format!(
                "file at {:?} has invalid extension {}",
                &path, other
            ))),
        }
    }
}

impl TryFrom<FileEntry> for File<ChainBlock> {
    type Error = TCError;

    fn try_from(file: FileEntry) -> TCResult<Self> {
        match file {
            FileEntry::Chain(file) => Ok(file),
            _ => Err(TCError::unsupported("this is not a Chain file!")),
        }
    }
}

impl TryFrom<FileEntry> for File<Bytes> {
    type Error = TCError;

    fn try_from(file: FileEntry) -> TCResult<Self> {
        match file {
            FileEntry::Bin(file) => Ok(file),
            _ => Err(TCError::unsupported("this is not a binary file!")),
        }
    }
}

/// An entry in a [`Dir`], either a [`FileEntry`] or sub-[`Dir`].
#[derive(Clone)]
pub enum DirEntry {
    Dir(Dir),
    File(FileEntry),
}

#[derive(Clone)]
pub struct Dir {
    cache: Cache,
    path: PathBuf,
    entries: TxnLock<Mutable<HashMap<PathSegment, DirEntry>>>,
}

impl Dir {
    fn new(cache: Cache, path: PathBuf, entries: HashMap<PathSegment, DirEntry>) -> Self {
        let entries = TxnLock::new(entries.into());
        Self {
            cache,
            path,
            entries,
        }
    }

    fn load(
        cache: Cache,
        path: PathBuf,
        contents: DirContents,
    ) -> Pin<Box<dyn Future<Output = TCResult<Self>>>> {
        Box::pin(async move {
            if contents.iter().all(|(_, meta)| meta.is_dir()) {
                let mut entries = HashMap::new();

                for (handle, _) in contents.into_iter() {
                    let name = file_name(&handle)?;
                    let path = handle.path();

                    if name == super::VERSION {
                        let path = handle.path();
                        warn!("skipping old version file at {:?}", path);
                        continue;
                    } else if name.starts_with(".") {
                        debug!("skip loading hidden file {:?}", path);
                    }

                    let contents = dir_contents(&path).await?;

                    if is_empty(&contents) {
                        if file_ext(&path).is_some() {
                            let file = FileEntry::load(cache.clone(), path, contents).await?;
                            entries.insert(name, DirEntry::File(file));
                        } else {
                            let dir = Dir::load(cache.clone(), path, contents).await?;
                            entries.insert(name, DirEntry::Dir(dir));
                        }
                    } else if contents.iter().all(|(_, meta)| meta.is_file()) {
                        let file = FileEntry::load(cache.clone(), path, contents).await?;
                        entries.insert(name, DirEntry::File(file));
                    } else if contents.iter().all(|(_, meta)| meta.is_dir()) {
                        let dir = Dir::load(cache.clone(), path, contents).await?;
                        entries.insert(name, DirEntry::Dir(dir));
                    } else {
                        return Err(TCError::internal(format!(
                            "directory at {:?} contains both blocks and subdirectories",
                            path
                        )));
                    }
                }

                Ok(Self::new(cache, path, entries))
            } else {
                Err(TCError::internal(format!(
                    "directory at {:?} contains both blocks and subdirectories",
                    path
                )))
            }
        })
    }

    /// Return the [`DirEntry`] at the given path if there is one, otherwise
    pub fn find<'a>(
        &'a self,
        txn_id: &'a TxnId,
        path: &'a [PathSegment],
    ) -> Pin<Box<dyn Future<Output = TCResult<Option<DirEntry>>> + 'a>> {
        Box::pin(async move {
            if path.is_empty() {
                return Ok(None);
            }

            let entries = self.entries.read(txn_id).await?;
            if path.len() == 1 {
                Ok(entries.get(&path[0]).cloned())
            } else {
                match entries.get(&path[0]) {
                    Some(DirEntry::Dir(dir)) => dir.find(txn_id, &path[1..]).await,
                    _ => Ok(None),
                }
            }
        })
    }
}

#[async_trait]
impl fs::Store for Dir {
    async fn is_empty(&self, txn_id: &TxnId) -> TCResult<bool> {
        self.entries
            .read(txn_id)
            .map_ok(|entries| entries.is_empty())
            .await
    }
}

#[async_trait]
impl fs::Dir for Dir {
    type File = FileEntry;
    type FileClass = StateType;

    async fn create_dir(&self, txn_id: TxnId, name: PathSegment) -> TCResult<Self> {
        let path = fs_path(&self.path, &name);
        let dir = Dir::new(self.cache.clone(), path, HashMap::new());

        let mut entries = self.entries.write(txn_id).await?;
        entries.insert(name, DirEntry::Dir(dir.clone()));

        Ok(dir)
    }

    async fn create_file(&self, txn_id: TxnId, name: Id, class: StateType) -> TCResult<Self::File> {
        let path = fs_path(&self.path, &name);
        let file = FileEntry::new(self.cache.clone(), path, class)?;

        let mut entries = self.entries.write(txn_id).await?;
        entries.insert(name, DirEntry::File(file.clone()));

        Ok(file)
    }

    async fn get_dir(&self, txn_id: &TxnId, name: &PathSegment) -> TCResult<Option<Self>> {
        let entries = self.entries.read(txn_id).await?;
        match entries.get(&name) {
            Some(DirEntry::Dir(dir)) => Ok(Some(dir.clone())),
            Some(_) => Err(TCError::bad_request("not a dir", name)),
            None => Ok(None),
        }
    }

    async fn get_file(&self, txn_id: &TxnId, name: &Id) -> TCResult<Option<Self::File>> {
        let entries = self.entries.read(txn_id).await?;
        match entries.get(&name) {
            Some(DirEntry::File(file)) => Ok(Some(file.clone())),
            Some(_) => Err(TCError::bad_request("not a file", name)),
            None => Ok(None),
        }
    }
}

#[async_trait]
impl Transact for Dir {
    async fn commit(&self, txn_id: &TxnId) {
        debug!("commit dir {:?} at {}", &self.path, txn_id);

        {
            let entries = self.entries.read(&txn_id).await.unwrap();
            join_all(entries.values().map(|entry| match entry {
                DirEntry::Dir(dir) => dir.commit(txn_id),
                DirEntry::File(file) => match file {
                    FileEntry::Bin(file) => file.commit(txn_id),
                    FileEntry::Chain(file) => file.commit(txn_id),
                },
            }))
            .await;
        }

        self.entries.commit(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        {
            let entries = self.entries.read(&txn_id).await.unwrap();
            join_all(entries.values().map(|entry| match entry {
                DirEntry::Dir(dir) => dir.finalize(txn_id),
                DirEntry::File(file) => match file {
                    FileEntry::Bin(file) => file.finalize(txn_id),
                    FileEntry::Chain(file) => file.finalize(txn_id),
                },
            }))
            .await;
        }

        self.entries.finalize(txn_id).await
    }
}

/// Load a [`Dir`] from the given filesystem path.
pub async fn load(cache: Cache, mount_point: PathBuf) -> TCResult<Dir> {
    let contents = dir_contents(&mount_point).await?;
    Dir::load(cache, mount_point, contents).await
}

fn is_empty(contents: &DirContents) -> bool {
    for (handle, _) in contents {
        if !handle.file_name().to_str().unwrap().starts_with('.') {
            return false;
        }
    }

    true
}
