//! A transactional filesystem directory.

use std::collections::HashMap;
use std::fmt;
use std::ops::{Deref, DerefMut};

use async_trait::async_trait;
use freqfs::DirLock;
use futures::future::{self, join_all, TryFutureExt};
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
            StateType::Scalar(_) => File::new(cache).map_ok(Self::Scalar).await,
            other => Err(err(other)),
        }
    }

    async fn load<C>(cache: DirLock<CacheBlock>, class: C, txn_id: TxnId) -> TCResult<Self>
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
    Raw(DirLock<CacheBlock>),
}

impl fmt::Display for DirEntry {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Dir(dir) => fmt::Display::fmt(dir, f),
            Self::File(file) => fmt::Display::fmt(file, f),
            Self::Raw(_) => {
                f.write_str("a directory entry which manages its own transactional state")
            }
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
    cache: DirLock<CacheBlock>,
    listing: TxnLock<Contents>,
}

impl Dir {
    pub fn new(cache: DirLock<CacheBlock>) -> Self {
        let lock_name = "contents of a transactional filesystem directory";

        Self {
            cache,
            listing: TxnLock::new(lock_name, Contents::from(HashMap::new())),
        }
    }

    /// Load an existing [`Dir`] from the filesystem.
    pub fn load<'a>(cache: DirLock<CacheBlock>, txn_id: TxnId) -> TCBoxTryFuture<'a, Self> {
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
                    return Err(TCError::internal(format!(
                        "directory {} contains both blocks and subdirectories",
                        name
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

    /// Get the child directory with the given `name` or create a new one.
    pub async fn get_or_create_dir(&self, txn_id: TxnId, name: PathSegment) -> TCResult<Self> {
        if let Some(dir) = fs::Dir::get_dir(self, txn_id, &name).await? {
            Ok(dir)
        } else {
            fs::Dir::create_dir(self, txn_id, name).await
        }
    }

    /// Create a new raw directory descriptor.
    ///
    /// The transactional state of a raw directory descriptor must be managed by the calling code.
    pub(crate) async fn create_raw(
        &self,
        txn_id: TxnId,
        name: PathSegment,
    ) -> TCResult<DirLock<CacheBlock>> {
        let mut listing = self.listing.write(txn_id).await?;
        if listing.contains_key(&name) {
            return Err(TCError::bad_request(
                "filesystem entry already exists",
                name,
            ));
        }

        let descriptor = {
            let mut cache = self.cache.write().await;
            cache.create_dir(name.to_string()).map_err(io_err)?
        };

        listing.insert(name, DirEntry::Raw(descriptor.clone()));
        Ok(descriptor)
    }

    /// Get a raw directory descriptor.
    ///
    /// The transactional state of a raw directory descriptor must be managed by the calling code.
    pub(crate) async fn get_raw(
        &self,
        txn_id: TxnId,
        name: &PathSegment,
    ) -> TCResult<Option<DirLock<CacheBlock>>> {
        let listing = self.listing.read(txn_id).await?;
        if !listing.contains_key(name) {
            return Ok(None);
        }

        match listing.get(name) {
            Some(DirEntry::Raw(dir)) => Ok(Some(dir.clone())),
            Some(other) => Err(TCError::bad_request("expected a directory, not", other)),
            None => Ok(None),
        }
    }
}

#[async_trait]
impl fs::Store for Dir {
    async fn is_empty(&self, txn_id: TxnId) -> TCResult<bool> {
        self.listing
            .read(txn_id)
            .map_ok(|listing| listing.is_empty())
            .await
    }
}

#[async_trait]
impl fs::Dir for Dir {
    type File = FileEntry;
    type FileClass = StateType;

    async fn contains(&self, txn_id: TxnId, name: &PathSegment) -> TCResult<bool> {
        self.listing
            .read(txn_id)
            .map_ok(|listing| listing.contains_key(name))
            .await
    }

    async fn create_dir(&self, txn_id: TxnId, name: PathSegment) -> TCResult<Self> {
        let mut listing = self.listing.write(txn_id).await?;
        if listing.contains_key(&name) {
            return Err(TCError::bad_request(
                "filesystem entry already exists",
                name,
            ));
        }

        let dir_cache = {
            let mut cache = self.cache.write().await;
            cache.create_dir(name.to_string()).map_err(io_err)?
        };

        let subdir = Dir::new(dir_cache);
        listing.insert(name, DirEntry::Dir(subdir.clone()));
        Ok(subdir)
    }

    async fn create_dir_unique(&self, txn_id: TxnId) -> TCResult<Dir> {
        let mut listing = self.listing.write(txn_id).await?;
        let name: PathSegment = loop {
            let name = Uuid::new_v4().into();
            if !listing.contains_key(&name) {
                break name;
            }
        };

        let dir_cache = {
            let mut cache = self.cache.write().await;
            cache.create_dir(name.to_string()).map_err(io_err)?
        };

        let subdir = Dir::new(dir_cache);
        listing.insert(name, DirEntry::Dir(subdir.clone()));
        Ok(subdir)
    }

    async fn create_file<C, F, B>(&self, txn_id: TxnId, file_id: Id, class: C) -> TCResult<F>
    where
        C: Copy + Send + fmt::Display,
        StateType: From<C>,
        FileEntry: AsType<F>,
        F: fs::File<B>,
        B: fs::BlockData,
    {
        let mut listing = self.listing.write(txn_id).await?;
        if listing.contains_key(&file_id) {
            return Err(TCError::bad_request(
                "filesystem entry already exists",
                file_id,
            ));
        }

        let file = {
            let mut cache = self.cache.write().await;
            let name = format!("{}.{}", file_id, B::ext());
            debug!("create file at {}", name);

            assert!(!listing.contains_key(&file_id));

            let file_cache = cache.create_dir(name).map_err(io_err)?;
            FileEntry::new(file_cache, class).await?
        };

        listing.insert(file_id, DirEntry::File(file.clone()));
        file.into_type()
            .ok_or_else(|| TCError::bad_request("expected file type", class))
    }

    async fn create_file_unique<C, F, B>(&self, txn_id: TxnId, class: C) -> TCResult<F>
    where
        C: Copy + Send + fmt::Display,
        StateType: From<C>,
        FileEntry: AsType<F>,
        F: fs::File<B>,
        B: fs::BlockData,
    {
        let mut listing = self.listing.write(txn_id).await?;
        let file_id: PathSegment = loop {
            let name = Uuid::new_v4().into();
            if !listing.contains_key(&name) {
                break name;
            }
        };

        let file = {
            let mut cache = self.cache.write().await;
            let name = format!("{}.{}", file_id, B::ext());
            debug!("create file at {}", name);

            assert!(!listing.contains_key(&file_id));

            let file_cache = cache.create_dir(name).map_err(io_err)?;
            FileEntry::new(file_cache, class).await?
        };

        listing.insert(file_id, DirEntry::File(file.clone()));
        file.into_type()
            .ok_or_else(|| TCError::bad_request("expected file type", class))
    }

    async fn get_dir(&self, txn_id: TxnId, name: &PathSegment) -> TCResult<Option<Self>> {
        let listing = self.listing.read(txn_id).await?;
        if !listing.contains_key(name) {
            return Ok(None);
        }

        match listing.get(name) {
            Some(DirEntry::Dir(dir)) => Ok(Some(dir.clone())),
            Some(other) => Err(TCError::bad_request("expected a directory, not", other)),
            None => Ok(None),
        }
    }

    async fn get_file<F, B>(&self, txn_id: TxnId, file_id: &Id) -> TCResult<Option<F>>
    where
        FileEntry: AsType<F>,
        F: fs::File<B>,
        B: fs::BlockData,
    {
        let listing = self.listing.read(txn_id).await?;
        if !listing.contains_key(file_id) {
            return Ok(None);
        }

        match listing.get(file_id) {
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
            DirEntry::Raw(_) => Box::pin(future::ready(())), // no-op
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
                DirEntry::Raw(_) => Box::pin(future::ready(())), // no-op
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

async fn is_dir(fs_cache: &DirLock<CacheBlock>) -> bool {
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

async fn is_file(name: &str, fs_cache: &DirLock<CacheBlock>) -> bool {
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
