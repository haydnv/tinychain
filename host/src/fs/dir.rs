//! A transactional filesystem directory.

use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::fmt;
use std::path::PathBuf;
use std::pin::Pin;

use async_trait::async_trait;
use futures::future::{join_all, Future, TryFutureExt};
use log::debug;

use tc_error::*;
use tc_transact::fs::{self, BlockData};
use tc_transact::lock::{Mutable, TxnLock};
use tc_transact::{Transact, TxnId};
use tcgeneric::{Id, PathSegment};

use crate::chain::{self, ChainBlock};
use crate::scalar::{ScalarType, Value};
use crate::state::StateType;

use super::{dir_contents, file_ext, file_name, fs_path, io_err, Cache, DirContents, File};

const VALUE_EXT: &'static str = "value";

#[derive(Clone)]
pub enum FileEntry {
    Chain(File<ChainBlock>),
    Value(File<Value>),
}

impl FileEntry {
    fn new(cache: Cache, path: PathBuf, class: StateType) -> TCResult<Self> {
        match class {
            StateType::Chain(_) => Ok(Self::Chain(File::new(cache, path, ChainBlock::ext()))),
            StateType::Scalar(st) => match st {
                ScalarType::Value(_) => Ok(Self::Value(File::new(cache, path, Value::ext()))),
                other => Err(TCError::bad_request("cannot create file for", other)),
            },
            other => Err(TCError::bad_request("cannot create file for", other)),
        }
    }

    async fn load(cache: Cache, path: PathBuf, contents: DirContents) -> TCResult<Self> {
        let ext = file_ext(&path)
            .ok_or_else(|| TCError::unsupported(format!("file at {:?} has no extension", &path)))?;

        match ext {
            chain::EXT => {
                let file = File::load(cache.clone(), path, contents)?;
                Ok(FileEntry::Chain(file))
            }
            VALUE_EXT => {
                let file = File::load(cache.clone(), path, contents)?;
                Ok(FileEntry::Value(file))
            }
            other => Err(TCError::internal(format!(
                "file at {:?} has invalid extension {}",
                &path, other
            ))),
        }
    }
}

impl From<File<ChainBlock>> for FileEntry {
    fn from(file: File<ChainBlock>) -> Self {
        Self::Chain(file)
    }
}

impl TryFrom<FileEntry> for File<ChainBlock> {
    type Error = TCError;

    fn try_from(entry: FileEntry) -> TCResult<Self> {
        match entry {
            FileEntry::Chain(file) => Ok(file),
            other => Err(TCError::bad_request(
                "expected a Chain file but found",
                other,
            )),
        }
    }
}

impl From<File<Value>> for FileEntry {
    fn from(file: File<Value>) -> Self {
        Self::Value(file)
    }
}

impl TryFrom<FileEntry> for File<Value> {
    type Error = TCError;

    fn try_from(entry: FileEntry) -> TCResult<Self> {
        match entry {
            FileEntry::Value(file) => Ok(file),
            other => Err(TCError::bad_request(
                "expected a Chain file but found",
                other,
            )),
        }
    }
}

impl fmt::Display for FileEntry {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Chain(chain) => fmt::Display::fmt(chain, f),
            Self::Value(value) => fmt::Display::fmt(value, f),
        }
    }
}

#[derive(Clone)]
enum DirEntry {
    Dir(Dir),
    File(FileEntry),
}

impl fmt::Display for DirEntry {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Dir(dir) => fmt::Display::fmt(dir, f),
            Self::File(file) => fmt::Display::fmt(file, f),
        }
    }
}

#[derive(Clone)]
pub struct Dir {
    path: PathBuf,
    cache: Cache,
    contents: TxnLock<Mutable<HashMap<PathSegment, DirEntry>>>,
}

impl Dir {
    pub fn load(
        cache: Cache,
        path: PathBuf,
        entries: DirContents,
    ) -> Pin<Box<dyn Future<Output = TCResult<Self>>>> {
        Box::pin(async move {
            if entries.iter().all(|(_, meta)| meta.is_dir()) {
                let mut contents = HashMap::new();

                for (handle, _) in entries.into_iter() {
                    let name = file_name(&handle)?;
                    let path = handle.path();
                    let entries = dir_contents(&path).await?;
                    if is_empty(&entries) {
                        if file_ext(&path).is_some() {
                            let file = FileEntry::load(cache.clone(), path, entries).await?;
                            contents.insert(name, DirEntry::File(file));
                        } else {
                            let dir = Dir::load(cache.clone(), path, entries).await?;
                            contents.insert(name, DirEntry::Dir(dir));
                        }
                    } else if entries.iter().all(|(_, meta)| meta.is_file()) {
                        let file = FileEntry::load(cache.clone(), path, entries).await?;
                        contents.insert(name, DirEntry::File(file));
                    } else if entries.iter().all(|(_, meta)| meta.is_dir()) {
                        let dir = Dir::load(cache.clone(), path, entries).await?;
                        contents.insert(name, DirEntry::Dir(dir));
                    } else {
                        return Err(TCError::internal(format!(
                            "directory at {:?} contains both blocks and subdirectories",
                            path
                        )));
                    }
                }

                let lock_name = format!("contents of {:?}", path);
                Ok(Dir {
                    path,
                    cache,
                    contents: TxnLock::new(lock_name, contents.into()),
                })
            } else {
                Err(TCError::internal(format!(
                    "directory at {:?} contains both blocks and subdirectories",
                    path
                )))
            }
        })
    }

    pub async fn entry_ids(&self, txn_id: &TxnId) -> TCResult<HashSet<PathSegment>> {
        let contents = self.contents.read(txn_id).await?;
        Ok(contents.keys().cloned().collect())
    }

    pub async fn get_or_create_dir(&self, txn_id: TxnId, name: PathSegment) -> TCResult<Self> {
        if let Some(dir) = fs::Dir::get_dir(self, &txn_id, &name).await? {
            Ok(dir)
        } else {
            fs::Dir::create_dir(self, txn_id, name).await
        }
    }
}

#[async_trait]
impl fs::Store for Dir {
    async fn is_empty(&self, txn_id: &TxnId) -> TCResult<bool> {
        self.contents
            .read(txn_id)
            .map_ok(|contents| contents.is_empty())
            .await
    }
}

#[async_trait]
impl fs::Dir for Dir {
    type File = FileEntry;
    type FileClass = StateType;

    async fn contains(&self, txn_id: &TxnId, name: &PathSegment) -> TCResult<bool> {
        self.contents
            .read(txn_id)
            .map_ok(|contents| contents.contains_key(name))
            .await
    }

    async fn create_dir(&self, txn_id: TxnId, name: PathSegment) -> TCResult<Self> {
        let mut contents = self.contents.write(txn_id).await?;
        if contents.contains_key(&name) {
            return Err(TCError::bad_request(
                "filesystem entry already exists",
                name,
            ));
        }

        let subdir = Dir {
            path: fs_path(&self.path, &name),
            cache: self.cache.clone(),
            contents: TxnLock::new(
                format!("transactional subdirectory at {}", name),
                HashMap::new().into(),
            ),
        };

        contents.insert(name, DirEntry::Dir(subdir.clone()));
        Ok(subdir)
    }

    async fn create_file(&self, txn_id: TxnId, name: Id, class: StateType) -> TCResult<Self::File> {
        let mut contents = self.contents.write(txn_id).await?;
        if contents.contains_key(&name) {
            return Err(TCError::bad_request(
                "filesystem entry already exists",
                name,
            ));
        }

        let path = fs_path(&self.path, &name);
        let file = FileEntry::new(self.cache.clone(), path, class)?;
        contents.insert(name, DirEntry::File(file.clone()));
        Ok(file.into())
    }

    async fn get_dir(&self, txn_id: &TxnId, name: &PathSegment) -> TCResult<Option<Self>> {
        let contents = self.contents.read(txn_id).await?;
        match contents.get(name) {
            Some(DirEntry::Dir(dir)) => Ok(Some(dir.clone())),
            Some(other) => Err(TCError::bad_request("expected a directory, not", other)),
            None => Ok(None),
        }
    }

    async fn get_file(&self, txn_id: &TxnId, name: &Id) -> TCResult<Option<Self::File>> {
        let contents = self.contents.read(txn_id).await?;
        match contents.get(name) {
            Some(DirEntry::File(file)) => Ok(Some(file.clone())),
            Some(other) => Err(TCError::bad_request("expected a directory, not", other)),
            None => Ok(None),
        }
    }
}

#[async_trait]
impl Transact for Dir {
    async fn commit(&self, txn_id: &TxnId) {
        debug!("commit dir {:?} at {}", &self.path, txn_id);

        if !self.path.exists() {
            tokio::fs::create_dir(&self.path)
                .map_err(|e| io_err(e, &self.path))
                .await
                .expect("create filesystem dir");

            debug!("created filesystem dir {:?}", &self.path);
        }

        {
            let contents = self.contents.read(&txn_id).await.unwrap();
            join_all(contents.values().map(|entry| match entry {
                DirEntry::Dir(dir) => dir.commit(txn_id),
                DirEntry::File(file) => match file {
                    FileEntry::Chain(file) => file.commit(txn_id),
                    FileEntry::Value(file) => file.commit(txn_id),
                },
            }))
            .await;
        }

        self.contents.commit(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        {
            let contents = self.contents.read(&txn_id).await.unwrap();
            join_all(contents.values().map(|entry| match entry {
                DirEntry::Dir(dir) => dir.finalize(txn_id),
                DirEntry::File(file) => match file {
                    FileEntry::Chain(file) => file.finalize(txn_id),
                    FileEntry::Value(file) => file.finalize(txn_id),
                },
            }))
            .await;
        }

        self.contents.finalize(txn_id).await
    }
}

impl fmt::Display for Dir {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "directory at {:?}", self.path)
    }
}

fn is_empty(contents: &DirContents) -> bool {
    for (handle, _) in contents {
        if !handle.file_name().to_str().unwrap().starts_with('.') {
            return false;
        }
    }

    true
}
