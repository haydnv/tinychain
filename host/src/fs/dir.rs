//! A transactional filesystem directory.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use freqfs::{DirLock, DirWriteGuard};
use futures::future::{join_all, Future, FutureExt, TryFutureExt};
use log::debug;
use safecast::AsType;
use tokio::sync::RwLock;
use uuid::Uuid;

use tc_error::*;
#[cfg(feature = "tensor")]
use tc_tensor::TensorType;
use tc_transact::fs;
use tc_transact::lock::{TxnLock, TxnLockWriteGuard};
use tc_transact::{Transact, TxnId};
use tc_value::Value;
use tcgeneric::{Id, PathSegment};

use crate::chain::ChainBlock;
use crate::collection::CollectionType;
use crate::scalar::ScalarType;
use crate::state::StateType;

use super::{file_ext, file_name, io_err, CacheBlock, File};

#[derive(Clone)]
pub enum DirEntry {
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
pub enum FileEntry {
    BTree(File<tc_btree::Node>),
    Chain(File<crate::chain::ChainBlock>),
    #[cfg(feature = "tensor")]
    Tensor(File<afarray::Array>),
    Value(File<tc_value::Value>),
}

impl FileEntry {
    fn new<C>(cache: DirLock<CacheBlock>, class: C) -> TCResult<Self>
    where
        StateType: From<C>,
    {
        match class.into() {
            StateType::Collection(ct) => match ct {
                CollectionType::BTree(_) | CollectionType::Table(_) => todo!(),

                #[cfg(feature = "tensor")]
                CollectionType::Tensor(_) => todo!(),
            },
            StateType::Chain(_) => todo!(),
            other => Err(TCError::bad_request("invalid file class", other)),
        }
    }
}

impl AsType<File<tc_btree::Node>> for FileEntry {
    fn as_type(&self) -> Option<&File<tc_btree::Node>> {
        if let Self::BTree(file) = self {
            Some(file)
        } else {
            None
        }
    }

    fn as_type_mut(&mut self) -> Option<&mut File<tc_btree::Node>> {
        if let Self::BTree(file) = self {
            Some(file)
        } else {
            None
        }
    }

    fn into_type(self) -> Option<File<tc_btree::Node>> {
        if let Self::BTree(file) = self {
            Some(file)
        } else {
            None
        }
    }
}

impl AsType<File<ChainBlock>> for FileEntry {
    fn as_type(&self) -> Option<&File<ChainBlock>> {
        if let Self::Chain(file) = self {
            Some(file)
        } else {
            None
        }
    }

    fn as_type_mut(&mut self) -> Option<&mut File<ChainBlock>> {
        if let Self::Chain(file) = self {
            Some(file)
        } else {
            None
        }
    }

    fn into_type(self) -> Option<File<ChainBlock>> {
        if let Self::Chain(file) = self {
            Some(file)
        } else {
            None
        }
    }
}

#[cfg(feature = "tensor")]
impl AsType<File<afarray::Array>> for FileEntry {
    fn as_type(&self) -> Option<&File<afarray::Array>> {
        if let Self::Tensor(file) = self {
            Some(file)
        } else {
            None
        }
    }

    fn as_type_mut(&mut self) -> Option<&mut File<afarray::Array>> {
        if let Self::Tensor(file) = self {
            Some(file)
        } else {
            None
        }
    }

    fn into_type(self) -> Option<File<afarray::Array>> {
        if let Self::Tensor(file) = self {
            Some(file)
        } else {
            None
        }
    }
}

impl AsType<File<Value>> for FileEntry {
    fn as_type(&self) -> Option<&File<Value>> {
        if let Self::Value(file) = self {
            Some(file)
        } else {
            None
        }
    }

    fn as_type_mut(&mut self) -> Option<&mut File<Value>> {
        if let Self::Value(file) = self {
            Some(file)
        } else {
            None
        }
    }

    fn into_type(self) -> Option<File<Value>> {
        if let Self::Value(file) = self {
            Some(file)
        } else {
            None
        }
    }
}

impl From<File<tc_btree::Node>> for FileEntry {
    fn from(file: File<tc_btree::Node>) -> Self {
        Self::BTree(file)
    }
}

impl From<File<ChainBlock>> for FileEntry {
    fn from(file: File<ChainBlock>) -> Self {
        Self::Chain(file)
    }
}

#[cfg(feature = "tensor")]
impl From<File<afarray::Array>> for FileEntry {
    fn from(file: File<afarray::Array>) -> Self {
        Self::Tensor(file)
    }
}

impl From<File<Value>> for FileEntry {
    fn from(file: File<Value>) -> Self {
        Self::Value(file)
    }
}

impl fmt::Display for FileEntry {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BTree(_) => f.write_str("a BTree file"),
            Self::Chain(_) => f.write_str("a Chain file"),
            #[cfg(feature = "tensor")]
            Self::Tensor(_) => f.write_str("a Tensor file"),
            Self::Value(_) => f.write_str("a Value file"),
        }
    }
}

#[derive(Clone)]
pub struct Dir {
    cache: DirLock<CacheBlock>,
    contents: TxnLock<HashMap<PathSegment, DirEntry>>,
}

impl Dir {
    pub async fn new(cache: DirLock<CacheBlock>) -> Self {
        let lock_name = cache
            .read()
            .map(|dir| format!("contents of {:?}", &*dir))
            .await;

        Self {
            cache,
            contents: TxnLock::new(lock_name, HashMap::new()),
        }
    }

    pub async fn get_or_create_dir(&self, txn_id: TxnId, name: PathSegment) -> TCResult<Dir> {
        unimplemented!()
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

// TODO: transactional versioning on the filesystem itself
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
            return Err(TCError::bad_request("directory already exists", name));
        }

        let cache = self.cache.write().await;
        create_dir_inner(contents, cache, name).await
    }

    async fn create_dir_tmp(&self, txn_id: TxnId) -> TCResult<Dir> {
        let mut contents = self.contents.write(txn_id).await?;
        let name = loop {
            let id: PathSegment = Uuid::new_v4().into();
            if !contents.contains_key(&id) {
                break id;
            }
        };

        let cache = self.cache.write().await;
        create_dir_inner(contents, cache, name).await
    }

    async fn create_file<C, F>(&self, txn_id: TxnId, name: Id, class: C) -> TCResult<F>
    where
        C: Send,
        F: Clone,
        StateType: From<C>,
        FileEntry: AsType<F>,
    {
        let mut contents = self.contents.write(txn_id).await?;
        if contents.contains_key(&name) {
            return Err(TCError::bad_request("file already exists", name));
        }

        let mut cache = self.cache.write().await;
        create_file_inner(contents, cache, name, class).await
    }

    async fn create_file_tmp<C, F>(&self, txn_id: TxnId, class: C) -> TCResult<F>
    where
        C: Send,
        F: Clone,
        StateType: From<C>,
        FileEntry: AsType<F>,
    {
        let mut contents = self.contents.write(txn_id).await?;
        let name = loop {
            let id: PathSegment = Uuid::new_v4().into();
            if !contents.contains_key(&id) {
                break id;
            }
        };

        let cache = self.cache.write().await;
        create_file_inner(contents, cache, name, class).await
    }

    async fn get_dir(&self, txn_id: TxnId, name: &PathSegment) -> TCResult<Option<Self>> {
        let contents = self.contents.read(txn_id).await?;
        match contents.get(name) {
            Some(DirEntry::Dir(dir)) => Ok(Some(dir.clone())),
            Some(other) => Err(TCError::bad_request("expected a directory, not", other)),
            None => Ok(None),
        }
    }

    async fn get_file<F: Clone>(&self, txn_id: TxnId, name: &Id) -> TCResult<Option<F>>
    where
        FileEntry: AsType<F>,
    {
        let contents = self.contents.read(txn_id).await?;
        match contents.get(name) {
            Some(DirEntry::File(file)) => file
                .as_type()
                .cloned()
                .ok_or_else(|| TCError::bad_request("unexpected file type", file))
                .map(Some),
            Some(other) => Err(TCError::bad_request("expected a file, not", other)),
            None => Ok(None),
        }
    }
}

#[async_trait]
impl Transact for Dir {
    async fn commit(&self, txn_id: &TxnId) {
        // let the TxnLock panic first, if a panic is going to happen, so we don't overwrite files
        self.contents.commit(txn_id).await;

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

    async fn finalize(&self, txn_id: &TxnId) {
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
        f.write_str("a directory")
    }
}

async fn create_dir_inner(
    mut contents: TxnLockWriteGuard<HashMap<PathSegment, DirEntry>>,
    mut cache: DirWriteGuard<CacheBlock>,
    name: PathSegment,
) -> TCResult<Dir> {
    let cache = cache.create_dir(name.to_string()).map_err(io_err)?;

    let subdir = Dir {
        cache,
        contents: TxnLock::new(
            format!("transactional subdirectory at {}", name),
            HashMap::new(),
        ),
    };

    contents.insert(name, DirEntry::Dir(subdir.clone()));
    Ok(subdir)
}

async fn create_file_inner<C, F>(
    mut contents: TxnLockWriteGuard<HashMap<PathSegment, DirEntry>>,
    mut cache: DirWriteGuard<CacheBlock>,
    name: PathSegment,
    class: C,
) -> TCResult<F>
where
    StateType: From<C>,
    FileEntry: AsType<F>,
{
    let cache = cache.create_dir(name.to_string()).map_err(io_err)?;
    let file = FileEntry::new(cache, class)?;
    contents.insert(name, DirEntry::File(file.clone()));
    file.into_type()
        .ok_or_else(|| TCError::internal("wrong file class"))
}
