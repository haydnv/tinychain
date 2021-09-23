//! A transactional filesystem directory.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::ops::{Deref, DerefMut};

#[cfg(feature = "tensor")]
use afarray::Array;
use async_trait::async_trait;
use freqfs::DirLock;
use futures::future::{join_all, TryFutureExt};
use safecast::AsType;
use uuid::Uuid;

use tc_btree::Node;
use tc_error::*;
#[cfg(feature = "tensor")]
use tc_tensor::TensorType;
use tc_transact::fs;
use tc_transact::lock::TxnLock;
use tc_transact::{Transact, TxnId};
use tc_value::Value;
use tcgeneric::{Id, PathSegment};

use crate::chain::ChainBlock;
use crate::collection::CollectionType;
use crate::scalar::ScalarType;
use crate::state::StateType;

use super::{io_err, CacheBlock, File};

#[derive(Clone)]
pub enum FileEntry {
    BTree(File<Node>),
    Chain(File<ChainBlock>),
    Value(File<Value>),

    #[cfg(feature = "tensor")]
    Tensor(File<Array>),
}

impl FileEntry {
    async fn new<C>(cache: DirLock<CacheBlock>, class: C) -> TCResult<Self>
    where
        StateType: From<C>,
    {
        fn err<T: fmt::Display>(class: T) -> TCError {
            TCError::bad_request("cannot create file for", class)
        }

        match StateType::from(class) {
            StateType::Collection(ct) => match ct {
                CollectionType::BTree(_) => File::new(cache).map_ok(Self::BTree).await,
                CollectionType::Table(tt) => Err(err(tt)),

                #[cfg(feature = "tensor")]
                CollectionType::Tensor(tt) => match tt {
                    TensorType::Dense => File::new(cache).map_ok(Self::Tensor).await,
                    TensorType::Sparse => Err(err(TensorType::Sparse)),
                },
            },
            StateType::Chain(_) => File::new(cache).map_ok(Self::Chain).await,
            StateType::Scalar(st) => match st {
                ScalarType::Value(_) => File::new(cache).map_ok(Self::Value).await,
                other => Err(err(other)),
            },
            other => Err(err(other)),
        }
    }
}

impl AsType<File<Node>> for FileEntry {
    fn as_type(&self) -> Option<&File<Node>> {
        if let Self::BTree(file) = self {
            Some(file)
        } else {
            None
        }
    }

    fn as_type_mut(&mut self) -> Option<&mut File<Node>> {
        if let Self::BTree(file) = self {
            Some(file)
        } else {
            None
        }
    }

    fn into_type(self) -> Option<File<Node>> {
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
impl AsType<File<Array>> for FileEntry {
    fn as_type(&self) -> Option<&File<Array>> {
        if let Self::Tensor(file) = self {
            Some(file)
        } else {
            None
        }
    }

    fn as_type_mut(&mut self) -> Option<&mut File<Array>> {
        if let Self::Tensor(file) = self {
            Some(file)
        } else {
            None
        }
    }

    fn into_type(self) -> Option<File<Array>> {
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

impl From<File<Node>> for FileEntry {
    fn from(file: File<Node>) -> Self {
        Self::BTree(file)
    }
}

impl From<File<ChainBlock>> for FileEntry {
    fn from(file: File<ChainBlock>) -> Self {
        Self::Chain(file)
    }
}

#[cfg(feature = "tensor")]
impl From<File<Array>> for FileEntry {
    fn from(file: File<Array>) -> Self {
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
            Self::BTree(btree) => fmt::Display::fmt(btree, f),
            Self::Chain(chain) => fmt::Display::fmt(chain, f),
            Self::Value(value) => fmt::Display::fmt(value, f),

            #[cfg(feature = "tensor")]
            Self::Tensor(tensor) => fmt::Display::fmt(tensor, f),
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
struct Contents {
    inner: HashMap<PathSegment, DirEntry>,
}

impl Contents {
    fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }
}

impl Deref for Contents {
    type Target = HashMap<PathSegment, DirEntry>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for Contents {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl PartialEq for Contents {
    fn eq(&self, other: &Self) -> bool {
        let this: HashSet<_> = self.inner.keys().collect();
        let that: HashSet<_> = other.inner.keys().collect();
        this == that
    }
}

impl Eq for Contents {}

#[derive(Clone)]
pub struct Dir {
    cache: DirLock<CacheBlock>,
    contents: TxnLock<Contents>,
}

impl Dir {
    pub async fn new(cache: DirLock<CacheBlock>) -> TCResult<Self> {
        let fs_dir = cache.read().await;
        let lock_name = format!("contents of {:?}", &*fs_dir);

        if fs_dir.len() > 0 {
            return Err(TCError::unsupported("cannot re-create existing directory"));
        }

        Ok(Self {
            cache,
            contents: TxnLock::new(lock_name, Contents::new()),
        })
    }

    pub async fn get_or_create_dir(&self, txn_id: TxnId, name: PathSegment) -> TCResult<Self> {
        if let Some(dir) = fs::Dir::get_dir(self, txn_id, &name).await? {
            Ok(dir)
        } else {
            fs::Dir::create_dir(self, txn_id, name).await
        }
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

        let mut cache = self.cache.write().await;
        let dir_cache = cache.create_dir(name.to_string()).map_err(io_err)?;
        let subdir = Dir::new(dir_cache).await?;
        contents.insert(name, DirEntry::Dir(subdir.clone()));
        Ok(subdir)
    }

    async fn create_dir_tmp(&self, txn_id: TxnId) -> TCResult<Dir> {
        let mut contents = self.contents.write(txn_id).await?;
        let name = loop {
            let name = Uuid::new_v4().into();
            if !contents.contains_key(&name) {
                break name;
            }
        };

        let mut cache = self.cache.write().await;
        let dir_cache = cache.create_dir(name.to_string()).map_err(io_err)?;
        let subdir = Dir::new(dir_cache).await?;
        contents.insert(name, DirEntry::Dir(subdir.clone()));
        Ok(subdir)
    }

    async fn create_file<C, F>(&self, txn_id: TxnId, name: Id, class: C) -> TCResult<F>
    where
        C: Copy + Send + fmt::Display,
        StateType: From<C>,
        FileEntry: AsType<F>,
    {
        let mut contents = self.contents.write(txn_id).await?;
        if contents.contains_key(&name) {
            return Err(TCError::bad_request(
                "filesystem entry already exists",
                name,
            ));
        }

        let mut cache = self.cache.write().await;
        let file_cache = cache.create_dir(name.to_string()).map_err(io_err)?;
        let file = FileEntry::new(file_cache, class).await?;
        contents.insert(name, DirEntry::File(file.clone()));
        file.into_type()
            .ok_or_else(|| TCError::bad_request("expected file type", class))
    }

    async fn create_file_tmp<C, F>(&self, txn_id: TxnId, class: C) -> TCResult<F>
    where
        C: Copy + Send + fmt::Display,
        StateType: From<C>,
        FileEntry: AsType<F>,
    {
        let mut contents = self.contents.write(txn_id).await?;
        let name = loop {
            let name = Uuid::new_v4().into();
            if !contents.contains_key(&name) {
                break name;
            }
        };

        let mut cache = self.cache.write().await;
        let file_cache = cache.create_dir(name.to_string()).map_err(io_err)?;
        let file = FileEntry::new(file_cache, class).await?;
        contents.insert(name, DirEntry::File(file.clone()));
        file.into_type()
            .ok_or_else(|| TCError::bad_request("expected file type", class))
    }

    async fn get_dir(&self, txn_id: TxnId, name: &PathSegment) -> TCResult<Option<Self>> {
        let contents = self.contents.read(txn_id).await?;
        match contents.get(name) {
            Some(DirEntry::Dir(dir)) => Ok(Some(dir.clone())),
            Some(other) => Err(TCError::bad_request("expected a directory, not", other)),
            None => Ok(None),
        }
    }

    async fn get_file<F>(&self, txn_id: TxnId, name: &Id) -> TCResult<Option<F>>
    where
        FileEntry: AsType<F>,
    {
        let contents = self.contents.read(txn_id).await?;
        match contents.get(name) {
            Some(DirEntry::File(file)) => file
                .clone()
                .into_type()
                .map(Some)
                .ok_or_else(|| TCError::bad_request("unexpected file type", file)),

            Some(other) => Err(TCError::bad_request("expected a file, not", other)),
            None => Ok(None),
        }
    }
}

#[async_trait]
impl Transact for Dir {
    async fn commit(&self, txn_id: &TxnId) {
        let contents = self.contents.write(*txn_id).await.expect("dir contents");

        self.contents.commit(txn_id).await;

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
            let contents = self.contents.read(*txn_id).await.expect("dir contents");
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
        f.write_str("a transactional directory")
    }
}
