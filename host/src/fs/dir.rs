//! A transactional filesystem directory.

use std::collections::{HashMap, HashSet};
use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::path::PathBuf;
use std::pin::Pin;

#[cfg(feature = "tensor")]
use afarray::Array;
use async_trait::async_trait;
use futures::future::{join_all, Future, TryFutureExt};
use log::debug;
use uuid::Uuid;

use tc_btree::Node;
use tc_error::*;
#[cfg(feature = "tensor")]
use tc_tensor::TensorType;
use tc_transact::fs::{self, BlockData};
use tc_transact::lock::TxnLock;
use tc_transact::{Transact, TxnId};
use tc_value::Value;
use tcgeneric::{Id, PathSegment};

use crate::chain::{self, ChainBlock};
use crate::collection::CollectionType;
use crate::scalar::ScalarType;
use crate::state::StateType;

use super::{dir_contents, file_ext, file_name, fs_path, Cache, DirContents, File};

const VALUE_EXT: &'static str = "value";

#[derive(Clone, Eq, PartialEq)]
pub enum FileEntry {
    BTree(File<Node>),
    Chain(File<ChainBlock>),
    Value(File<Value>),

    #[cfg(feature = "tensor")]
    Tensor(File<Array>),
}

impl FileEntry {
    fn new<C>(cache: Cache, path: PathBuf, class: C) -> TCResult<Self>
    where
        StateType: From<C>,
    {
        fn err<T: fmt::Display>(class: T) -> TCError {
            TCError::bad_request("cannot create file for", class)
        }

        match StateType::from(class) {
            StateType::Collection(ct) => match ct {
                CollectionType::BTree(_) => Ok(Self::BTree(File::new(cache, path, Node::ext()))),
                CollectionType::Table(tt) => Err(err(tt)),

                #[cfg(feature = "tensor")]
                CollectionType::Tensor(tt) => match tt {
                    TensorType::Dense => Ok(Self::Tensor(File::new(cache, path, Array::ext()))),
                    TensorType::Sparse => {
                        Err(TCError::unsupported("cannot create File for SparseTensor"))
                    }
                },
            },
            StateType::Chain(_) => Ok(Self::Chain(File::new(cache, path, ChainBlock::ext()))),
            StateType::Scalar(st) => match st {
                ScalarType::Value(_) => Ok(Self::Value(File::new(cache, path, Value::ext()))),
                other => Err(err(other)),
            },
            other => Err(err(other)),
        }
    }

    async fn load(cache: Cache, path: PathBuf, contents: DirContents) -> TCResult<Self> {
        let ext = file_ext(&path)
            .ok_or_else(|| TCError::unsupported(format!("file at {:?} has no extension", &path)))?;

        match ext {
            tc_btree::EXT => {
                let file = File::load(cache.clone(), path, contents)?;
                Ok(FileEntry::BTree(file))
            }
            chain::EXT => {
                let file = File::load(cache.clone(), path, contents)?;
                Ok(FileEntry::Chain(file))
            }
            #[cfg(feature = "tensor")]
            tc_tensor::EXT => {
                let file = File::load(cache.clone(), path, contents)?;
                Ok(FileEntry::Tensor(file))
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

#[cfg(feature = "tensor")]
impl TryFrom<FileEntry> for File<Array> {
    type Error = TCError;

    fn try_from(entry: FileEntry) -> TCResult<Self> {
        match entry {
            FileEntry::Tensor(file) => Ok(file),
            other => Err(TCError::bad_request(
                "expected a Tensor file but found",
                other,
            )),
        }
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

impl TryFrom<FileEntry> for File<Node> {
    type Error = TCError;

    fn try_from(entry: FileEntry) -> TCResult<Self> {
        match entry {
            FileEntry::BTree(file) => Ok(file),
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
            Self::BTree(btree) => fmt::Display::fmt(btree, f),
            Self::Chain(chain) => fmt::Display::fmt(chain, f),
            Self::Value(value) => fmt::Display::fmt(value, f),

            #[cfg(feature = "tensor")]
            Self::Tensor(tensor) => fmt::Display::fmt(tensor, f),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
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
    contents: TxnLock<HashMap<PathSegment, DirEntry>>,
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
                    contents: TxnLock::new(lock_name, contents),
                })
            } else {
                Err(TCError::internal(format!(
                    "directory at {:?} contains both blocks and subdirectories",
                    path
                )))
            }
        })
    }

    pub async fn entry_ids(&self, txn_id: TxnId) -> TCResult<HashSet<PathSegment>> {
        let contents = self.contents.read(txn_id).await?;
        Ok(contents.keys().cloned().collect())
    }

    pub async fn get_or_create_dir(&self, txn_id: TxnId, name: PathSegment) -> TCResult<Self> {
        if let Some(dir) = fs::Dir::get_dir(self, txn_id, &name).await? {
            Ok(dir)
        } else {
            fs::Dir::create_dir(self, txn_id, name).await
        }
    }

    pub async fn unique_id(&self, txn_id: TxnId) -> TCResult<PathSegment> {
        let existing_ids = self.entry_ids(txn_id).await?;
        loop {
            let id: PathSegment = Uuid::new_v4().into();
            if !existing_ids.contains(&id) {
                break Ok(id);
            }
        }
    }
}

impl Eq for Dir {}

impl PartialEq for Dir {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
    }
}

#[async_trait]
impl fs::Store for Dir {
    async fn is_empty(&self, txn_id: TxnId) -> TCResult<bool> {
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

    async fn contains(&self, txn_id: TxnId, name: &PathSegment) -> TCResult<bool> {
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
                HashMap::new(),
            ),
        };

        contents.insert(name, DirEntry::Dir(subdir.clone()));
        Ok(subdir)
    }

    async fn create_dir_tmp(&self, txn_id: TxnId) -> TCResult<Dir> {
        self.unique_id(txn_id)
            .and_then(|id| self.create_dir(txn_id, id))
            .await
    }

    async fn create_file<F: TryFrom<FileEntry, Error = TCError>, C: Send>(
        &self,
        txn_id: TxnId,
        name: Id,
        class: C,
    ) -> TCResult<F>
    where
        StateType: From<C>,
    {
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
        file.try_into()
    }

    async fn create_file_tmp<F: TryFrom<FileEntry, Error = TCError>, C: Send>(
        &self,
        txn_id: TxnId,
        class: C,
    ) -> TCResult<F>
    where
        StateType: From<C>,
    {
        self.unique_id(txn_id)
            .and_then(|id| self.create_file(txn_id, id, class))
            .await
    }

    async fn get_dir(&self, txn_id: TxnId, name: &PathSegment) -> TCResult<Option<Self>> {
        let contents = self.contents.read(txn_id).await?;
        match contents.get(name) {
            Some(DirEntry::Dir(dir)) => Ok(Some(dir.clone())),
            Some(other) => Err(TCError::bad_request("expected a directory, not", other)),
            None => Ok(None),
        }
    }

    async fn get_file<F: TryFrom<Self::File, Error = TCError>>(
        &self,
        txn_id: TxnId,
        name: &Id,
    ) -> TCResult<Option<F>> {
        let contents = self.contents.read(txn_id).await?;
        match contents.get(name) {
            Some(DirEntry::File(file)) => Ok(Some(file.clone().try_into()?)),
            Some(other) => Err(TCError::bad_request("expected a file, not", other)),
            None => Ok(None),
        }
    }
}

#[async_trait]
impl Transact for Dir {
    async fn commit(&self, txn_id: &TxnId) {
        debug!("commit dir {:?} at {}", &self.path, txn_id);

        if !self.path.exists() {
            let dir_create = tokio::fs::create_dir(&self.path).await;

            match dir_create {
                Ok(_) => {
                    debug!("created filesystem dir {:?}", &self.path);
                }
                Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
                    debug!("another thread created filesystem dir {:?}", &self.path);
                }
                Err(err) => panic!("error creating dir {:?}: {}", &self.path, err),
            }
        }

        {
            let contents = self.contents.read(*txn_id).await.unwrap();

            join_all(contents.values().map(|entry| match entry {
                DirEntry::Dir(dir) => dir.commit(txn_id),
                DirEntry::File(file) => match file {
                    FileEntry::BTree(file) => file.commit(txn_id),
                    FileEntry::Chain(file) => file.commit(txn_id),
                    FileEntry::Value(file) => file.commit(txn_id),

                    #[cfg(feature = "tensor")]
                    FileEntry::Tensor(file) => file.commit(txn_id),
                },
            }))
            .await;
        }

        self.contents.commit(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        debug!("finalize dir {:?} at {}", &self.path, txn_id);

        {
            let contents = self.contents.read(*txn_id).await.unwrap();
            join_all(contents.values().map(|entry| match entry {
                DirEntry::Dir(dir) => dir.finalize(txn_id),
                DirEntry::File(file) => match file {
                    FileEntry::BTree(file) => file.finalize(txn_id),
                    FileEntry::Chain(file) => file.finalize(txn_id),
                    FileEntry::Value(file) => file.finalize(txn_id),

                    #[cfg(feature = "tensor")]
                    FileEntry::Tensor(file) => file.finalize(txn_id),
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
