//! A transactional filesystem directory.

use std::collections::HashMap;
use std::fmt;
use std::fmt::Display;
use std::ops::{Deref, DerefMut};

use async_trait::async_trait;
use futures::future::{join_all, TryFutureExt};
use log::debug;
use safecast::AsType;
use uuid::Uuid;

use tc_btree::{BTreeType, Node};
use tc_error::*;
#[cfg(feature = "tensor")]
use tc_tensor::{Array, TensorType};
use tc_transact::fs;
use tc_transact::lock::{TxnLock, TxnLockReadGuard, TxnLockWriteGuard};
use tc_transact::{Transact, TxnId};
use tcgeneric::{Id, PathSegment, TCBoxTryFuture};

use crate::chain::{ChainBlock, ChainType};
use crate::collection::CollectionType;
use crate::scalar::{Scalar, ScalarType};
use crate::state::StateType;
use crate::transact::fs::BlockData;

use super::{io_err, CacheBlock, File};

/// A file in a directory
#[derive(Clone)]
pub enum FileEntry {
    BTree(File<Node>),
    Chain(File<ChainBlock>),
    Scalar(File<Scalar>),

    #[cfg(feature = "tensor")]
    Tensor(File<Array>),
}

impl FileEntry {
    fn new<ST>(cache: freqfs::DirLock<CacheBlock>, class: ST) -> TCResult<Self>
    where
        StateType: From<ST>,
    {
        fn err<T: fmt::Display>(class: T) -> TCError {
            TCError::bad_request("cannot create file for", class)
        }

        match StateType::from(class) {
            StateType::Collection(ct) => match ct {
                CollectionType::BTree(_) => File::new(cache).map(Self::BTree),
                CollectionType::Table(tt) => Err(err(tt)),

                #[cfg(feature = "tensor")]
                CollectionType::Tensor(tt) => match tt {
                    TensorType::Dense => File::new(cache).map(Self::Tensor),
                    TensorType::Sparse => Err(err(TensorType::Sparse)),
                },
            },
            StateType::Chain(_) => File::new(cache).map(Self::Chain),
            StateType::Scalar(_) => File::new(cache).map(Self::Scalar),
            other => Err(err(other)),
        }
    }

    async fn load<ST>(
        cache: freqfs::DirLock<CacheBlock>,
        class: ST,
        txn_id: TxnId,
    ) -> TCResult<Self>
    where
        StateType: From<ST>,
    {
        fn err<T: fmt::Display>(class: T) -> TCError {
            TCError::bad_request("cannot load file for", class)
        }

        match StateType::from(class) {
            StateType::Collection(ct) => match ct {
                CollectionType::BTree(_) => File::load(cache, txn_id).map_ok(Self::BTree).await,
                CollectionType::Table(tt) => Err(err(tt)),

                #[cfg(feature = "tensor")]
                CollectionType::Tensor(tt) => match tt {
                    TensorType::Dense => File::load(cache, txn_id).map_ok(Self::Tensor).await,
                    TensorType::Sparse => Err(err(TensorType::Sparse)),
                },
            },
            StateType::Chain(_) => File::load(cache, txn_id).map_ok(Self::Chain).await,
            StateType::Scalar(_) => File::load(cache, txn_id).map_ok(Self::Scalar).await,
            other => Err(err(other)),
        }
    }
}

// TODO: generate this with a macro
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

impl AsType<File<Scalar>> for FileEntry {
    fn as_type(&self) -> Option<&File<Scalar>> {
        if let Self::Scalar(file) = self {
            Some(file)
        } else {
            None
        }
    }

    fn as_type_mut(&mut self) -> Option<&mut File<Scalar>> {
        if let Self::Scalar(file) = self {
            Some(file)
        } else {
            None
        }
    }

    fn into_type(self) -> Option<File<Scalar>> {
        if let Self::Scalar(file) = self {
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

impl From<File<Scalar>> for FileEntry {
    fn from(file: File<Scalar>) -> Self {
        Self::Scalar(file)
    }
}

impl fmt::Display for FileEntry {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BTree(btree) => fmt::Display::fmt(btree, f),
            Self::Chain(chain) => fmt::Display::fmt(chain, f),
            Self::Scalar(scalar) => fmt::Display::fmt(scalar, f),

            #[cfg(feature = "tensor")]
            Self::Tensor(tensor) => fmt::Display::fmt(tensor, f),
        }
    }
}

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
pub struct Contents {
    inner: HashMap<PathSegment, DirEntry>,
}

impl PartialEq for Contents {
    fn eq(&self, other: &Self) -> bool {
        if self.len() == other.len() {
            self.keys().zip(other.keys()).all(|(l, r)| l == r)
        } else {
            false
        }
    }
}

impl Eq for Contents {}

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

impl From<HashMap<PathSegment, DirEntry>> for Contents {
    fn from(inner: HashMap<PathSegment, DirEntry>) -> Self {
        Self { inner }
    }
}

pub struct DirGuard<C, L> {
    cache: C,
    contents: L,
}

impl<C, L> fs::DirRead for DirGuard<C, L>
where
    C: Deref<Target = freqfs::Dir<CacheBlock>> + Send + Sync,
    L: Deref<Target = Contents> + Send + Sync,
{
    type FileEntry = FileEntry;
    type Lock = Dir;

    fn contains(&self, name: &PathSegment) -> bool {
        self.contents.get(name).is_some()
    }

    fn get_dir(&self, name: &PathSegment) -> TCResult<Option<Self::Lock>> {
        match self.contents.get(name) {
            Some(DirEntry::Dir(dir)) => Ok(Some(dir.clone())),
            Some(other) => Err(TCError::bad_request("expected a directory, not", other)),
            None => Ok(None),
        }
    }

    fn get_file<F, B>(&self, name: &Id) -> TCResult<Option<F>>
    where
        Self::FileEntry: AsType<F>,
        B: BlockData,
        F: fs::File<B>,
    {
        match self.contents.get(name) {
            Some(DirEntry::File(file)) => file
                .clone()
                .into_type()
                .map(Some)
                .ok_or_else(|| TCError::bad_request("unexpected file type", file)),

            Some(other) => Err(TCError::bad_request("expected a file, not", other)),
            None => Ok(None),
        }
    }

    fn is_empty(&self) -> bool {
        self.contents.is_empty()
    }
}

impl<C, L> fs::DirWrite for DirGuard<C, L>
where
    C: DerefMut<Target = freqfs::Dir<CacheBlock>> + Send + Sync,
    L: DerefMut<Target = Contents> + Send + Sync,
{
    type FileClass = StateType;

    fn create_dir(&mut self, name: PathSegment) -> TCResult<Self::Lock> {
        if self.contents.contains_key(&name) {
            return Err(TCError::bad_request("directory already exists", name));
        }

        let dir = self
            .cache
            .create_dir(name.to_string())
            .map(Dir::new)
            .map_err(io_err)?;

        self.contents.insert(name, DirEntry::Dir(dir.clone()));

        Ok(dir)
    }

    fn create_dir_unique(&mut self) -> TCResult<Self::Lock> {
        let name: PathSegment = loop {
            let name = Uuid::new_v4().into();
            if !self.contents.contains_key(&name) {
                break name;
            }
        };

        self.create_dir(name)
    }

    fn create_file<ST, F, B>(&mut self, name: Id, class: ST) -> TCResult<F>
    where
        Self::FileEntry: AsType<F>,
        StateType: From<ST>,
        ST: Copy + Send + Display,
        B: BlockData,
        F: fs::File<B>,
    {
        if self.contents.contains_key(&name) {
            return Err(TCError::bad_request("file already exists", name));
        }

        let file = self
            .cache
            .create_dir(format!("{}.{}", name, B::ext()))
            .map_err(io_err)
            .and_then(|cache| FileEntry::new(cache, class))?;

        self.contents.insert(name, DirEntry::File(file.clone()));

        file.as_type()
            .cloned()
            .ok_or_else(|| TCError::internal(format!("wrong file type for {}: {}", file, class)))
    }

    fn create_file_unique<ST, F, B>(&mut self, class: ST) -> TCResult<F>
    where
        Self::FileEntry: AsType<F>,
        StateType: From<ST>,
        ST: Copy + Send + Display,
        B: BlockData,
        F: fs::File<B>,
    {
        let name: PathSegment = loop {
            let name = Uuid::new_v4().into();
            if !self.contents.contains_key(&name) {
                break name;
            }
        };

        self.create_file(name, class)
    }
}

pub type DirReadGuard = DirGuard<freqfs::DirReadGuard<CacheBlock>, TxnLockReadGuard<Contents>>;
pub type DirWriteGuard = DirGuard<freqfs::DirWriteGuard<CacheBlock>, TxnLockWriteGuard<Contents>>;

/// A filesystem directory.
#[derive(Clone)]
pub struct Dir {
    cache: freqfs::DirLock<CacheBlock>,
    listing: TxnLock<Contents>,
}

impl Dir {
    pub fn new(cache: freqfs::DirLock<CacheBlock>) -> Self {
        let lock_name = "contents of a transactional filesystem directory";

        Self {
            cache,
            listing: TxnLock::new(lock_name, Contents::from(HashMap::new())),
        }
    }

    /// Load an existing [`Dir`] from the filesystem.
    pub fn load<'a>(cache: freqfs::DirLock<CacheBlock>, txn_id: TxnId) -> TCBoxTryFuture<'a, Self> {
        Box::pin(async move {
            let fs_dir = cache.read().await;

            debug!("Dir::load {:?}", &*fs_dir);

            let mut listing = HashMap::new();
            for (name, fs_cache) in fs_dir.iter() {
                if name.starts_with('.') {
                    debug!("Dir::load skipping hidden filesystem entry {}", name);
                    continue;
                }

                let fs_cache = match fs_cache {
                    freqfs::DirEntry::Dir(dir_lock) => dir_lock.clone(),
                    _ => return Err(TCError::internal(format!("{} is not a directory", name))),
                };

                let (name, entry) = if is_file(name, &fs_cache).await {
                    let (name, class) = file_class(name)?;
                    let entry = FileEntry::load(fs_cache, class, txn_id).await?;
                    (name, DirEntry::File(entry))
                } else if is_dir(&fs_cache).await {
                    let subdir = Dir::load(fs_cache, txn_id).await?;
                    let name = name.parse().map_err(TCError::internal)?;
                    (name, DirEntry::Dir(subdir))
                } else {
                    #[cfg(debug_assertions)]
                    let fs_path = format!("{:?} entry \"{}\"", &*fs_dir, name);
                    #[cfg(not(debug_assertions))]
                    let fs_path = name;

                    return Err(TCError::internal(format!(
                        "directory {} contains both blocks and subdirectories",
                        fs_path
                    )));
                };

                listing.insert(name, entry);
            }

            #[cfg(debug_assertions)]
            let lock_name = format!("contents of {:?}", &*fs_dir);
            #[cfg(not(debug_assertions))]
            let lock_name = "contents of transactional filesystem directory";

            Ok(Self {
                cache,
                listing: TxnLock::new(lock_name, Contents::from(listing)),
            })
        })
    }

    pub fn into_inner(self) -> freqfs::DirLock<CacheBlock> {
        self.cache
    }
}

#[async_trait]
impl fs::Dir for Dir {
    type Read = DirReadGuard;
    type Write = DirWriteGuard;

    async fn read(&self, txn_id: TxnId) -> TCResult<Self::Read> {
        let contents = self.listing.read(txn_id).await?;
        // TODO: support versioning on the filesystem itself
        let cache = self.cache.read().await;
        Ok(DirGuard { cache, contents })
    }

    async fn write(&self, txn_id: TxnId) -> TCResult<Self::Write> {
        let contents = self.listing.write(txn_id).await?;
        // TODO: support versioning on the filesystem itself
        let cache = self.cache.write().await;
        Ok(DirGuard { cache, contents })
    }
}

#[async_trait]
impl fs::Store for Dir {}

#[async_trait]
impl Transact for Dir {
    async fn commit(&self, txn_id: &TxnId) {
        self.listing.commit(txn_id).await;

        let listing = self.listing.read(*txn_id).await.expect("dir listing");

        join_all(listing.values().map(|entry| match entry {
            DirEntry::Dir(dir) => dir.commit(txn_id),
            DirEntry::File(file) => match file {
                FileEntry::BTree(file) => file.commit(txn_id),
                FileEntry::Chain(file) => file.commit(txn_id),
                FileEntry::Scalar(file) => file.commit(txn_id),

                #[cfg(feature = "tensor")]
                FileEntry::Tensor(file) => file.commit(txn_id),
            },
        }))
        .await;
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.listing.finalize(txn_id).await;

        {
            let listing = self.listing.read(*txn_id).await.expect("dir listing");

            join_all(listing.values().map(|entry| match entry {
                DirEntry::Dir(dir) => dir.finalize(txn_id),
                DirEntry::File(file) => match file {
                    FileEntry::BTree(file) => file.finalize(txn_id),
                    FileEntry::Chain(file) => file.finalize(txn_id),
                    FileEntry::Scalar(file) => file.finalize(txn_id),

                    #[cfg(feature = "tensor")]
                    FileEntry::Tensor(file) => file.finalize(txn_id),
                },
            }))
            .await;
        }
    }
}

impl fmt::Display for Dir {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a transactional directory")
    }
}

async fn is_dir(fs_cache: &freqfs::DirLock<CacheBlock>) -> bool {
    for (name, entry) in fs_cache.read().await.iter() {
        if name.starts_with('.') {
            continue;
        }

        if let freqfs::DirEntry::File(_) = entry {
            return false;
        }
    }

    true
}

async fn is_file(name: &str, fs_cache: &freqfs::DirLock<CacheBlock>) -> bool {
    if ext_class(name).is_none() {
        return false;
    }

    for (name, entry) in fs_cache.read().await.iter() {
        if name.starts_with('.') {
            continue;
        }

        if let freqfs::DirEntry::Dir(_) = entry {
            return false;
        }
    }

    true
}

fn file_class(name: &str) -> TCResult<(PathSegment, StateType)> {
    let i = name
        .rfind('.')
        .ok_or_else(|| TCError::internal(format!("invalid file name {}", name)))?;

    let stem = name[..i].parse().map_err(TCError::internal)?;
    let class = ext_class(&name[i..])
        .ok_or_else(|| TCError::internal(format!("invalid file extension {}", name)))?;

    Ok((stem, class))
}

fn ext_class(name: &str) -> Option<StateType> {
    if name.ends_with('.') {
        return None;
    }

    let i = name.rfind('.').map(|i| i + 1).unwrap_or(0);

    match &name[i..] {
        "node" => Some(BTreeType::default().into()),
        "chain_block" => Some(ChainType::default().into()),
        #[cfg(feature = "tensor")]
        "array" => Some(TensorType::Dense.into()),
        "scalar" => Some(ScalarType::default().into()),
        _ => None,
    }
}
