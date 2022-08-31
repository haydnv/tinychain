//! A transactional filesystem directory.

use std::collections::HashMap;
use std::fmt;
use std::fmt::Display;
use std::ops::{Deref, DerefMut};

use async_trait::async_trait;
use futures::future::TryFutureExt;
use log::debug;
use safecast::AsType;
use uuid::Uuid;

use tc_btree::{BTreeType, Node};
use tc_error::*;
#[cfg(feature = "tensor")]
use tc_tensor::{Array, TensorType};
use tc_transact::fs;
use tc_transact::lock::TxnLock;
use tc_transact::{Transact, TxnId};
use tcgeneric::{Id, PathSegment, TCBoxTryFuture};

use crate::chain::{ChainBlock, ChainType};
use crate::collection::CollectionType;
use crate::scalar::{Scalar, ScalarType};
use crate::state::StateType;

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
    async fn new<C>(cache: freqfs::DirLock<CacheBlock>, class: C) -> TCResult<Self>
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
            StateType::Scalar(_) => File::new(cache).map_ok(Self::Scalar).await,
            other => Err(err(other)),
        }
    }

    async fn load<C>(cache: freqfs::DirLock<CacheBlock>, class: C, txn_id: TxnId) -> TCResult<Self>
    where
        StateType: From<C>,
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

pub struct DirGuard<L> {
    lock: L,
}

#[async_trait]
impl<L> fs::DirRead for DirGuard<L>
where
    L: Deref<Target = freqfs::Dir<CacheBlock>> + Send + Sync,
{
    type FileEntry = FileEntry;
    type FileClass = StateType;
    type Lock = Dir;

    async fn contains(&self, name: &PathSegment) -> TCResult<bool> {
        todo!()
    }

    async fn get_dir(&self, name: &PathSegment) -> TCResult<Option<Self::Lock>> {
        todo!()
    }

    async fn get_file<F, B>(&self, name: &Id) -> TCResult<Option<F>>
    where
        Self::FileEntry: AsType<F>,
        B: fs::BlockData,
        F: fs::File<B>,
    {
        todo!()
    }

    async fn is_empty(&self) -> bool {
        todo!()
    }
}

#[async_trait]
impl<L> fs::DirWrite for DirGuard<L>
where
    L: DerefMut<Target = freqfs::Dir<CacheBlock>> + Send + Sync,
{
    async fn create_dir(&mut self, name: PathSegment) -> TCResult<Self::Lock> {
        todo!()
    }

    async fn create_dir_unique(&mut self) -> TCResult<Self::Lock> {
        todo!()
    }

    async fn create_file<C, F, B>(&mut self, name: Id, class: C) -> TCResult<F>
    where
        Self::FileEntry: AsType<F>,
        C: Copy + Send + Display,
        B: fs::BlockData,
        F: fs::File<B>,
    {
        todo!()
    }

    async fn create_file_unique<C, F, B>(&mut self, class: C) -> TCResult<F>
    where
        Self::FileEntry: AsType<F>,
        C: Copy + Send + Display,
        B: fs::BlockData,
        F: fs::File<B>,
    {
        todo!()
    }
}

#[async_trait]
impl fs::Dir for Dir {
    type Read = DirGuard<freqfs::DirReadGuard<CacheBlock>>;
    type Write = DirGuard<freqfs::DirWriteGuard<CacheBlock>>;

    async fn read(&self, txn_id: TxnId) -> TCResult<Self::Read> {
        todo!()
    }

    async fn write(&self, txn_id: TxnId) -> TCResult<Self::Write> {
        todo!()
    }
}

#[async_trait]
impl fs::Store for Dir {}

#[async_trait]
impl Transact for Dir {
    async fn commit(&self, txn_id: &TxnId) {
        todo!()
    }

    async fn finalize(&self, txn_id: &TxnId) {
        todo!()
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
