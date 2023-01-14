//! A transactional filesystem directory.

use std::collections::BTreeMap;
use std::convert::TryFrom;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::str::FromStr;

use async_trait::async_trait;
use futures::future::{self, join_all, FutureExt, TryFutureExt};
use futures::stream::{FuturesUnordered, StreamExt};
use log::debug;
use safecast::{as_type, AsType};
use uuid::Uuid;

use tc_btree::{Node, NodeId};
use tc_error::*;
#[cfg(feature = "tensor")]
use tc_tensor::Array;
use tc_transact::fs;
use tc_transact::lock::*;
use tc_transact::{Transact, TxnId};
use tc_value::Version as VersionNumber;
use tcgeneric::{Id, PathSegment, TCBoxFuture, TCBoxTryFuture};

use crate::chain::ChainBlock;
use crate::cluster::library;
use crate::object::InstanceClass;
use crate::transact::fs::BlockData;

use super::{io_err, CacheBlock, File};

pub enum EntryType {
    BTree,
    Chain,
    Class,
    Library,
    Service,

    #[cfg(feature = "tensor")]
    Tensor,
}

impl EntryType {
    fn from_ext(ext: &str) -> Option<Self> {
        match ext {
            "btree" => Some(Self::BTree),
            "chain" => Some(Self::Chain),
            "class" => Some(Self::Class),
            "lib" => Some(Self::Library),
            "service" => Some(Self::Service),

            #[cfg(feature = "tensor")]
            "tensor" => Some(Self::Tensor),

            _ => None,
        }
    }

    fn ext(&self) -> &'static str {
        match self {
            Self::BTree => "btree",
            Self::Chain => "chain",
            Self::Class => "class",
            Self::Library => "lib",
            Self::Service => "service",

            #[cfg(feature = "tensor")]
            Self::Tensor => "tensor",
        }
    }
}

pub trait FileExt {
    fn ext() -> &'static str;
}

macro_rules! file_ext {
    ($k:ty, $b:ty, $v:ident) => {
        impl FileExt for File<$k, $b> {
            fn ext() -> &'static str {
                EntryType::$v.ext()
            }
        }
    };
}

file_ext!(NodeId, Node, BTree);
file_ext!(Id, ChainBlock, Chain);
file_ext!(Id, InstanceClass, Class);
file_ext!(VersionNumber, library::Version, Library);
file_ext!(VersionNumber, InstanceClass, Service);
#[cfg(feature = "tensor")]
file_ext!(u64, Array, Tensor);

/// A file in a directory
#[derive(Clone)]
pub enum FileEntry {
    BTree(File<NodeId, Node>),
    Chain(File<Id, ChainBlock>),
    Class(File<Id, InstanceClass>),
    Library(File<VersionNumber, library::Version>),
    Service(File<VersionNumber, InstanceClass>),

    #[cfg(feature = "tensor")]
    Tensor(File<u64, Array>),
}

impl FileEntry {
    async fn load(
        cache: freqfs::DirLock<CacheBlock>,
        class: EntryType,
        txn_id: TxnId,
    ) -> TCResult<Self> {
        match class {
            EntryType::BTree => File::load(cache, txn_id).map_ok(Self::BTree).await,
            EntryType::Chain => File::load(cache, txn_id).map_ok(Self::Chain).await,
            EntryType::Class => File::load(cache, txn_id).map_ok(Self::Class).await,
            EntryType::Library => File::load(cache, txn_id).map_ok(Self::Library).await,
            EntryType::Service => File::load(cache, txn_id).map_ok(Self::Service).await,

            #[cfg(feature = "tensor")]
            EntryType::Tensor => File::load(cache, txn_id).map_ok(Self::Tensor).await,
        }
    }
}

as_type!(FileEntry, BTree, File<NodeId, Node>);
as_type!(FileEntry, Chain, File<Id, ChainBlock>);
as_type!(FileEntry, Class, File<Id, InstanceClass>);
as_type!(FileEntry, Library, File<VersionNumber, library::Version>);
as_type!(FileEntry, Service, File<VersionNumber, InstanceClass>);
#[cfg(feature = "tensor")]
as_type!(FileEntry, Tensor, File<u64, Array>);

impl fmt::Display for FileEntry {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BTree(btree) => fmt::Display::fmt(btree, f),
            Self::Chain(chain) => fmt::Display::fmt(chain, f),
            Self::Class(class) => fmt::Display::fmt(class, f),
            Self::Library(library) => fmt::Display::fmt(library, f),
            Self::Service(service) => fmt::Display::fmt(service, f),

            #[cfg(feature = "tensor")]
            Self::Tensor(tensor) => fmt::Display::fmt(tensor, f),
        }
    }
}

/// An entry in a [`Dir`]
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

pub enum Store {
    Create(DirWriteGuard, PathSegment),
    Get(DirReadGuard, PathSegment),
    GetOrCreate(DirWriteGuard, PathSegment),
    Dir(Dir),
    File(FileEntry),
}

impl From<Dir> for Store {
    fn from(dir: Dir) -> Self {
        Self::Dir(dir)
    }
}

impl<K, V> From<File<K, V>> for Store
where
    FileEntry: From<File<K, V>>,
{
    fn from(file: File<K, V>) -> Self {
        Self::File(FileEntry::from(file))
    }
}

impl TryFrom<Store> for Dir {
    type Error = TCError;

    fn try_from(store: Store) -> TCResult<Self> {
        match store {
            Store::Create(mut parent, name) => fs::DirCreate::create_dir(&mut parent, name),
            Store::GetOrCreate(mut parent, name) => {
                fs::DirCreate::get_or_create_dir(&mut parent, name)
            }
            Store::Get(parent, name) => {
                fs::DirRead::get_dir(&parent, &name)?.ok_or_else(|| TCError::not_found(name))
            }
            Store::Dir(dir) => Ok(dir),
            Store::File(file) => Err(TCError::bad_request("expected a directory but found", file)),
        }
    }
}

impl<K, B> TryFrom<Store> for File<K, B>
where
    K: FromStr + fmt::Display + Ord + Clone + Send + Sync + 'static,
    B: BlockData,
    CacheBlock: AsType<B>,
    DirWriteGuard: fs::DirCreateFile<File<K, B>>,
    <K as FromStr>::Err: std::error::Error + fmt::Display,
    FileEntry: AsType<Self>,
{
    type Error = TCError;

    fn try_from(store: Store) -> TCResult<Self> {
        match store {
            Store::Dir(dir) => Err(TCError::bad_request("expected a file but found", dir)),
            Store::File(file) => file
                .into_type()
                .ok_or_else(|| TCError::unsupported("file is of unexpected type")),
            Store::Create(mut parent, name) => fs::DirCreateFile::create_file(&mut parent, name),
            Store::GetOrCreate(mut parent, name) => {
                fs::DirCreateFile::get_or_create_file(&mut parent, name)
            }
            Store::Get(parent, name) => {
                fs::DirReadFile::get_file(&parent, &name)?.ok_or_else(|| TCError::not_found(name))
            }
        }
    }
}

#[async_trait]
impl fs::Store for Store {
    fn is_empty(&self, txn_id: TxnId) -> TCResult<bool> {
        let entry = match self {
            Self::Create(parent, name) => match parent.contents.get(name) {
                None => return Ok(true),
                Some(entry) => entry.clone(),
            },
            Self::GetOrCreate(parent, name) => match parent.contents.get(name) {
                None => return Ok(true),
                Some(entry) => entry.clone(),
            },
            Self::Get(parent, name) => match parent.contents.get(name) {
                None => return Ok(true),
                Some(entry) => entry.clone(),
            },
            Self::Dir(dir) => DirEntry::Dir(dir.clone()),
            Self::File(file) => DirEntry::File(file.clone()),
        };

        match entry {
            DirEntry::Dir(dir) => dir.is_empty(txn_id),
            DirEntry::File(file) => match file {
                FileEntry::BTree(file) => file.is_empty(txn_id),
                FileEntry::Chain(file) => file.is_empty(txn_id),
                FileEntry::Class(file) => file.is_empty(txn_id),
                FileEntry::Library(file) => file.is_empty(txn_id),
                FileEntry::Service(file) => file.is_empty(txn_id),
                #[cfg(feature = "tensor")]
                FileEntry::Tensor(file) => file.is_empty(txn_id),
            },
        }
    }
}

/// A lock guard for a [`Dir`]
pub struct DirGuard<C, L> {
    txn_id: TxnId,
    cache: C,
    contents: L,
}

impl<C, L> DirGuard<C, L>
where
    C: Deref<Target = freqfs::Dir<CacheBlock>> + Send + Sync,
    L: Deref<Target = BTreeMap<PathSegment, DirEntry>> + Send + Sync,
{
    // Iterate over the contents of this directory
    pub fn file_names(&self) -> impl Iterator<Item = &PathSegment> {
        self.contents.keys()
    }

    // Iterate over the contents of this directory
    pub fn iter(&self) -> impl Iterator<Item = (&PathSegment, &DirEntry)> {
        self.contents.iter()
    }
}

impl<C, L> fs::DirRead for DirGuard<C, L>
where
    C: Deref<Target = freqfs::Dir<CacheBlock>> + Send + Sync,
    L: Deref<Target = BTreeMap<PathSegment, DirEntry>> + Send + Sync,
{
    type Lock = Dir;

    fn contains(&self, name: &PathSegment) -> bool {
        self.contents.get(name).is_some()
    }

    fn get_dir(&self, name: &PathSegment) -> TCResult<Option<Self::Lock>> {
        match self.contents.get(name) {
            Some(DirEntry::Dir(dir)) => Ok(Some(dir.clone())),
            Some(_) => Err(TCError::bad_request(
                "expected a directory but found a file at",
                name,
            )),
            None => Ok(None),
        }
    }

    fn is_empty(&self) -> bool {
        self.contents.is_empty()
    }

    fn len(&self) -> usize {
        self.contents.len()
    }
}

impl<C, L, K, B> fs::DirReadFile<File<K, B>> for DirGuard<C, L>
where
    K: FromStr + fmt::Display + Ord + Clone + Send + Sync + 'static,
    C: Deref<Target = freqfs::Dir<CacheBlock>> + Send + Sync,
    L: Deref<Target = BTreeMap<PathSegment, DirEntry>> + Send + Sync,
    B: BlockData,
    <K as FromStr>::Err: std::error::Error + fmt::Display,
    CacheBlock: AsType<B>,
    FileEntry: AsType<File<K, B>>,
{
    fn get_file(&self, name: &Id) -> TCResult<Option<File<K, B>>> {
        match self.contents.get(name) {
            Some(DirEntry::File(file)) => file
                .clone()
                .into_type()
                .map(Some)
                .ok_or_else(|| TCError::bad_request("unexpected file type", file)),

            Some(_) => {
                #[cfg(debug_assertions)]
                let name = format!("{} in {}", name, self.cache.path().to_str().expect("path"));

                Err(TCError::bad_request(
                    "expected a file but found a directory at",
                    name,
                ))
            }
            None => Ok(None),
        }
    }
}

impl<C, L> fs::DirCreate for DirGuard<C, L>
where
    C: DerefMut<Target = freqfs::Dir<CacheBlock>> + Send + Sync,
    L: DerefMut<Target = BTreeMap<PathSegment, DirEntry>> + Send + Sync,
{
    fn create_dir(&mut self, name: PathSegment) -> TCResult<Self::Lock> {
        if self.contents.contains_key(&name) {
            return Err(TCError::bad_request("directory already exists", name));
        }

        let fs_name = name.to_string();
        if ext_class(&fs_name).is_some() {
            return Err(TCError::bad_request(
                "a directory name may not end with a file extension",
                name,
            ));
        }

        let dir = self
            .cache
            .create_dir(fs_name)
            .map(|dir| Dir::new(dir, self.txn_id))
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
}

impl<C, L, K, B> fs::DirCreateFile<File<K, B>> for DirGuard<C, L>
where
    C: DerefMut<Target = freqfs::Dir<CacheBlock>> + Send + Sync,
    L: DerefMut<Target = BTreeMap<PathSegment, DirEntry>> + Send + Sync,
    B: BlockData,
    K: FromStr + fmt::Display + Ord + Clone + Send + Sync + 'static,
    <K as FromStr>::Err: std::error::Error + fmt::Display,
    File<K, B>: FileExt,
    FileEntry: AsType<File<K, B>>,
    CacheBlock: AsType<B>,
{
    fn create_file(&mut self, name: Id) -> TCResult<File<K, B>> {
        if self.contents.contains_key(&name) {
            return Err(TCError::bad_request("file already exists", name));
        }

        let canon = self
            .cache
            .create_dir(format!("{}.{}", name, File::<K, B>::ext()))
            .map_err(io_err)?;

        let file = File::<K, B>::new(canon, self.txn_id)?;

        self.contents
            .insert(name, DirEntry::File(file.clone().into()));

        Ok(file)
    }

    fn create_file_unique(&mut self) -> TCResult<File<K, B>> {
        let name: PathSegment = loop {
            let name = Uuid::new_v4().into();
            if !self.contents.contains_key(&name) {
                break name;
            }
        };

        self.create_file(name)
    }

    fn get_or_create_file(&mut self, name: PathSegment) -> TCResult<File<K, B>> {
        use fs::DirReadFile;

        if let Some(file) = self.get_file(&name)? {
            Ok(file)
        } else {
            self.create_file(name)
        }
    }
}

pub type DirReadGuard =
    DirGuard<freqfs::DirReadGuard<CacheBlock>, TxnLockReadGuard<BTreeMap<PathSegment, DirEntry>>>;
pub type DirWriteGuard =
    DirGuard<freqfs::DirWriteGuard<CacheBlock>, TxnLockWriteGuard<BTreeMap<PathSegment, DirEntry>>>;

impl DirReadGuard {
    /// Get an entry in this directory without resolving it to a [`Dir`] or [`File`].
    pub fn get_store(self, name: PathSegment) -> Option<Store> {
        if self.contents.contains_key(&name) {
            Some(Store::Get(self, name))
        } else {
            None
        }
    }
}

impl DirWriteGuard {
    /// Create a new [`Store`] in this [`Dir`].
    pub fn create_store(self, name: PathSegment) -> Store {
        Store::Create(self, name)
    }

    /// Access a [`Store`] in this [`Dir`] which can be resolved to either a [`Dir`] or [`File`].
    pub fn get_or_create_store(self, name: PathSegment) -> Store {
        Store::GetOrCreate(self, name)
    }

    /// Create a new [`Store`] in this [`Dir`] with a unique name.
    pub fn create_store_unique(self) -> Store {
        let name: PathSegment = loop {
            let name = Uuid::new_v4().into();
            if !self.contents.contains_key(&name) {
                break name;
            }
        };

        Store::Create(self, name)
    }
}

/// A filesystem directory.
#[derive(Clone)]
pub struct Dir {
    cache: freqfs::DirLock<CacheBlock>,
    listing: TxnLock<BTreeMap<PathSegment, DirEntry>>,
}

impl Dir {
    pub(crate) fn new(cache: freqfs::DirLock<CacheBlock>, txn_id: TxnId) -> Self {
        #[cfg(debug_assertions)]
        let lock_name = {
            cache
                .try_read()
                .expect("dir lock")
                .path()
                .to_string_lossy()
                .to_string()
        };

        #[cfg(not(debug_assertions))]
        let lock_name = "filesystem dir";

        Self {
            cache,
            listing: TxnLock::new(lock_name, txn_id, BTreeMap::new()),
        }
    }

    /// Load an existing [`Dir`] from the filesystem.
    pub fn load<'a>(cache: freqfs::DirLock<CacheBlock>, txn_id: TxnId) -> TCBoxTryFuture<'a, Self> {
        Box::pin(async move {
            let fs_dir = cache.read().await;

            debug!("Dir::load {:?}", &*fs_dir);

            let mut listing = BTreeMap::new();
            for (name, fs_cache) in fs_dir.iter() {
                if name.starts_with('.') {
                    debug!("Dir::load skipping hidden filesystem entry {}", name);
                    continue;
                } else {
                    debug!("Dir::load entry {}", name);
                }

                let fs_cache = match fs_cache {
                    freqfs::DirEntry::Dir(dir_lock) => dir_lock.clone(),
                    _ => return Err(TCError::internal(format!("{} is not a directory", name))),
                };

                let (name, entry) = if is_file(name).await {
                    let (name, class) = file_class(name)?;
                    let entry = FileEntry::load(fs_cache, class, txn_id).await?;
                    (name, DirEntry::File(entry))
                } else if is_dir(&fs_cache).await {
                    assert!(ext_class(name).is_none());
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

            Ok(Self {
                cache,
                listing: TxnLock::new("filesystem dir", txn_id, listing),
            })
        })
    }

    /// Convenience method create a new [`Store`]
    pub async fn create_store(&self, txn_id: TxnId, name: PathSegment) -> TCResult<Store> {
        fs::Dir::write(self, txn_id)
            .map_ok(|lock| lock.create_store(name))
            .await
    }

    /// Convenience method to assign a unique name to a [`Store`] before creating it
    pub async fn create_store_unique(&self, txn_id: TxnId) -> TCResult<Store> {
        fs::Dir::write(self, txn_id)
            .map_ok(|lock| lock.create_store_unique())
            .await
    }

    /// Convenience method to get or create a new [`Store`]
    pub async fn get_or_create_store(&self, txn_id: TxnId, name: PathSegment) -> TCResult<Store> {
        fs::Dir::write(self, txn_id)
            .map_ok(|lock| lock.get_or_create_store(name))
            .await
    }
}

#[async_trait]
impl fs::Dir for Dir {
    type Read = DirReadGuard;
    type Write = DirWriteGuard;
    type Store = Store;
    type Inner = freqfs::DirLock<CacheBlock>;

    async fn read(&self, txn_id: TxnId) -> TCResult<Self::Read> {
        let contents = self.listing.read(txn_id).await?;
        let cache = self.cache.read().await;

        Ok(DirGuard {
            cache,
            contents,
            txn_id,
        })
    }

    fn try_read(&self, txn_id: TxnId) -> TCResult<Self::Read> {
        let contents = self.listing.try_read(txn_id)?;
        let cache = self.cache.try_read().map_err(io_err)?;

        Ok(DirGuard {
            cache,
            contents,
            txn_id,
        })
    }

    async fn write(&self, txn_id: TxnId) -> TCResult<Self::Write> {
        let contents = self.listing.write(txn_id).await?;
        let cache = self.cache.write().await;

        Ok(DirGuard {
            cache,
            contents,
            txn_id,
        })
    }

    fn try_write(&self, txn_id: TxnId) -> TCResult<Self::Write> {
        let contents = self.listing.try_write(txn_id)?;
        let cache = self.cache.try_write().map_err(io_err)?;

        Ok(DirGuard {
            cache,
            contents,
            txn_id,
        })
    }

    fn into_inner(self) -> freqfs::DirLock<CacheBlock> {
        self.cache
    }
}

impl fs::Store for Dir {
    fn is_empty(&self, txn_id: TxnId) -> TCResult<bool> {
        tc_transact::fs::Dir::try_read(self, txn_id).map(|guard| fs::DirRead::is_empty(&guard))
    }
}

#[async_trait]
impl Transact for Dir {
    type Commit = Option<TxnLockCommit<BTreeMap<PathSegment, DirEntry>>>;

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        let listing = self.listing.commit(txn_id).await?;

        let commits: FuturesUnordered<TCBoxFuture<()>> = FuturesUnordered::new();

        for entry in listing.values() {
            let commit: TCBoxFuture<()> = match entry {
                DirEntry::Dir(dir) => Box::pin(dir.commit(txn_id).map(|_| ())),
                DirEntry::File(file) => match file {
                    FileEntry::BTree(file) => Box::pin(file.commit(txn_id).map(|_| ())),
                    FileEntry::Chain(file) => Box::pin(file.commit(txn_id).map(|_| ())),
                    FileEntry::Class(file) => Box::pin(file.commit(txn_id).map(|_| ())),
                    FileEntry::Library(file) => Box::pin(file.commit(txn_id).map(|_| ())),
                    FileEntry::Service(file) => Box::pin(file.commit(txn_id).map(|_| ())),
                    #[cfg(feature = "tensor")]
                    FileEntry::Tensor(file) => Box::pin(file.commit(txn_id).map(|_| ())),
                },
            };

            commits.push(commit);
        }

        commits.fold((), |(), ()| future::ready(())).await;

        Some(listing)
    }

    async fn rollback(&self, txn_id: &TxnId) {
        debug!("roll back {}", self);

        if let Some(listing) = self.listing.rollback(txn_id).await {
            let rollbacks = listing.values().map(|entry| match entry {
                DirEntry::Dir(dir) => dir.rollback(txn_id),
                DirEntry::File(file) => match file {
                    FileEntry::BTree(file) => file.rollback(txn_id),
                    FileEntry::Chain(file) => file.rollback(txn_id),
                    FileEntry::Class(file) => file.rollback(txn_id),
                    FileEntry::Library(file) => file.rollback(txn_id),
                    FileEntry::Service(file) => file.rollback(txn_id),
                    #[cfg(feature = "tensor")]
                    FileEntry::Tensor(file) => file.rollback(txn_id),
                },
            });

            join_all(rollbacks).await;
        }
    }

    async fn finalize(&self, txn_id: &TxnId) {
        if let Some(listing) = self.listing.finalize(txn_id).await {
            let cleanups = listing.values().map(|entry| match entry {
                DirEntry::Dir(dir) => dir.finalize(txn_id),
                DirEntry::File(file) => match file {
                    FileEntry::BTree(file) => file.finalize(txn_id),
                    FileEntry::Chain(file) => file.finalize(txn_id),
                    FileEntry::Class(file) => file.finalize(txn_id),
                    FileEntry::Library(file) => file.finalize(txn_id),
                    FileEntry::Service(file) => file.finalize(txn_id),
                    #[cfg(feature = "tensor")]
                    FileEntry::Tensor(file) => file.finalize(txn_id),
                },
            });

            join_all(cleanups).await;
        }
    }
}

impl fmt::Display for Dir {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        #[cfg(debug_assertions)]
        {
            if let Ok(dir) = self.cache.try_read() {
                return write!(f, "a transactional directory at {:?}", dir.path());
            }
        }

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

async fn is_file(name: &str) -> bool {
    ext_class(name).is_some()
}

fn file_class(name: &str) -> TCResult<(PathSegment, EntryType)> {
    let i = name
        .rfind('.')
        .ok_or_else(|| TCError::internal(format!("invalid file name {}", name)))?;

    let stem = name[..i].parse().map_err(TCError::internal)?;
    let class = ext_class(&name[i..])
        .ok_or_else(|| TCError::internal(format!("invalid file extension {}", name)))?;

    Ok((stem, class))
}

fn ext_class(name: &str) -> Option<EntryType> {
    if name.ends_with('.') {
        return None;
    }

    let i = name.rfind('.').map(|i| i + 1)?;

    EntryType::from_ext(&name[i..])
}
