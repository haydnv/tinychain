//! A transactional filesystem directory.

use std::convert::TryFrom;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::str::FromStr;

use async_trait::async_trait;
use futures::future::{join_all, TryFutureExt};
use log::debug;
use safecast::{as_type, AsType};
use uuid::Uuid;

use tc_btree::{BTreeType, Node, NodeId};
use tc_error::*;
#[cfg(feature = "tensor")]
use tc_tensor::{Array, TensorType};
use tc_transact::fs;
use tc_transact::lock::*;
use tc_transact::{Transact, TxnId};
use tc_value::Version as VersionNumber;
use tcgeneric::{Id, Map, PathSegment, TCBoxTryFuture};

use crate::chain::{ChainBlock, ChainType};
use crate::cluster::library;
use crate::collection::CollectionType;
use crate::state::StateType;
use crate::transact::fs::BlockData;

use super::{io_err, CacheBlock, File};

/// A file in a directory
#[derive(Clone)]
pub enum FileEntry {
    BTree(File<NodeId, Node>),
    Chain(File<Id, ChainBlock>),
    Library(File<VersionNumber, library::Version>),

    #[cfg(feature = "tensor")]
    Tensor(File<u64, Array>),
}

impl FileEntry {
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
            other => Err(err(other)),
        }
    }
}

as_type!(FileEntry, BTree, File<NodeId, Node>);
as_type!(FileEntry, Chain, File<Id, ChainBlock>);
as_type!(FileEntry, Library, File<VersionNumber, library::Version>);
#[cfg(feature = "tensor")]
as_type!(FileEntry, Tensor, File<u64, Array>);

impl fmt::Display for FileEntry {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BTree(btree) => fmt::Display::fmt(btree, f),
            Self::Chain(chain) => fmt::Display::fmt(chain, f),
            Self::Library(library) => fmt::Display::fmt(library, f),

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

pub struct Store {
    name: PathSegment,
    parent: DirWriteGuard,
}

impl TryFrom<Store> for Dir {
    type Error = TCError;

    fn try_from(mut store: Store) -> TCResult<Self> {
        fs::DirCreate::get_or_create_dir(&mut store.parent, store.name)
    }
}

impl<K, B> TryFrom<Store> for File<K, B>
where
    K: FromStr + fmt::Display + Ord + Clone + Send + Sync + 'static,
    B: BlockData,
    CacheBlock: AsType<B>,
    DirWriteGuard: fs::DirCreateFile<File<K, B>>,
    <K as FromStr>::Err: std::error::Error + fmt::Display,
{
    type Error = TCError;

    fn try_from(mut store: Store) -> TCResult<Self> {
        fs::DirCreateFile::get_or_create_file(&mut store.parent, store.name)
    }
}

#[async_trait]
impl fs::Store for Store {
    async fn is_empty(&self, txn_id: TxnId) -> TCResult<bool> {
        match self.parent.contents.get(&self.name) {
            Some(DirEntry::Dir(dir)) => fs::Store::is_empty(&dir, txn_id).await,
            Some(DirEntry::File(file)) => match file {
                FileEntry::BTree(file) => fs::Store::is_empty(&file, txn_id).await,
                FileEntry::Chain(file) => fs::Store::is_empty(&file, txn_id).await,
                FileEntry::Library(file) => fs::Store::is_empty(&file, txn_id).await,
                #[cfg(feature = "tensor")]
                FileEntry::Tensor(file) => fs::Store::is_empty(&file, txn_id).await,
            },
            None => Ok(true),
        }
    }
}

/// A lock guard for a [`Dir`]
pub struct DirGuard<C, L> {
    cache: C,
    contents: L,
}

impl<C, L> DirGuard<C, L>
where
    C: Deref<Target = freqfs::Dir<CacheBlock>> + Send + Sync,
    L: TxnMapRead<PathSegment, DirEntry> + Send + Sync,
{
    // Iterate over the contents of this directory
    pub fn iter(&self) -> tc_transact::lock::Iter<PathSegment, DirEntry> {
        self.contents.iter()
    }
}

impl<C, L> fs::DirRead for DirGuard<C, L>
where
    C: Deref<Target = freqfs::Dir<CacheBlock>> + Send + Sync,
    L: TxnMapRead<PathSegment, DirEntry> + Send + Sync,
{
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

    fn is_empty(&self) -> bool {
        self.contents.is_empty()
    }
}

impl<C, L, K, B> fs::DirReadFile<File<K, B>> for DirGuard<C, L>
where
    K: FromStr + fmt::Display + Ord + Clone + Send + Sync + 'static,
    C: Deref<Target = freqfs::Dir<CacheBlock>> + Send + Sync,
    L: TxnMapRead<PathSegment, DirEntry> + Send + Sync,
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

            Some(other) => Err(TCError::bad_request("expected a file, not", other)),
            None => Ok(None),
        }
    }
}

impl<C, L> fs::DirCreate for DirGuard<C, L>
where
    C: DerefMut<Target = freqfs::Dir<CacheBlock>> + Send + Sync,
    L: TxnMapWrite<PathSegment, DirEntry> + Send + Sync,
{
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
}

impl<C, L, K, B> fs::DirCreateFile<File<K, B>> for DirGuard<C, L>
where
    C: DerefMut<Target = freqfs::Dir<CacheBlock>> + Send + Sync,
    L: TxnMapWrite<PathSegment, DirEntry> + Send + Sync,
    B: BlockData,
    K: FromStr + fmt::Display + Ord + Clone + Send + Sync + 'static,
    <K as FromStr>::Err: std::error::Error + fmt::Display,
    FileEntry: AsType<File<K, B>>,
    CacheBlock: AsType<B>,
{
    fn create_file(&mut self, name: Id) -> TCResult<File<K, B>> {
        if self.contents.contains_key(&name) {
            return Err(TCError::bad_request("file already exists", name));
        }

        let canon = self
            .cache
            .create_dir(format!("{}.{}", name, B::ext()))
            .map_err(io_err)?;
        let file = File::<K, B>::new(canon)?;

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
    DirGuard<freqfs::DirReadGuard<CacheBlock>, TxnMapLockReadGuard<PathSegment, DirEntry>>;
pub type DirWriteGuard =
    DirGuard<freqfs::DirWriteGuard<CacheBlock>, TxnMapLockWriteGuard<PathSegment, DirEntry>>;

impl DirWriteGuard {
    /// Access a [`Store`] in this [`Dir`] which can be resolved to either a [`Dir`] or [`File`].
    pub fn get_or_create_store(self, name: PathSegment) -> Store {
        Store { name, parent: self }
    }
}

/// A filesystem directory.
#[derive(Clone)]
pub struct Dir {
    cache: freqfs::DirLock<CacheBlock>,
    listing: TxnMapLock<PathSegment, DirEntry>,
}

impl Dir {
    pub(crate) fn new(cache: freqfs::DirLock<CacheBlock>) -> Self {
        let lock_name = "contents of a transactional filesystem directory";

        Self {
            cache,
            listing: TxnMapLock::new(lock_name),
        }
    }

    /// Load an existing [`Dir`] from the filesystem.
    pub fn load<'a>(cache: freqfs::DirLock<CacheBlock>, txn_id: TxnId) -> TCBoxTryFuture<'a, Self> {
        Box::pin(async move {
            let fs_dir = cache.read().await;

            debug!("Dir::load {:?}", &*fs_dir);

            let mut listing = Map::new();
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
                listing: TxnMapLock::with_contents(lock_name, listing),
            })
        })
    }

    /// Get this [`Dir`]'s underlying [`freqfs::DirLock`].
    ///
    /// Callers of this method must explicitly manage the transactional state of this [`Dir`].
    pub fn into_inner(self) -> freqfs::DirLock<CacheBlock> {
        self.cache
    }

    /// Get an [`DirEntry`] from this [`Dir`] which can be a [`File`] or a sub-[`Dir`].
    pub async fn get(&self, txn_id: TxnId, name: &PathSegment) -> TCResult<Option<DirEntry>> {
        let listing = self.listing.read(txn_id).await?;
        Ok(listing.get(name).clone())
    }
}

#[async_trait]
impl fs::Dir for Dir {
    type Read = DirReadGuard;
    type Write = DirWriteGuard;

    async fn read(&self, txn_id: TxnId) -> TCResult<Self::Read> {
        let contents = self.listing.read(txn_id).await?;
        let cache = self.cache.read().await;
        Ok(DirGuard { cache, contents })
    }

    async fn write(&self, txn_id: TxnId) -> TCResult<Self::Write> {
        let contents = self.listing.write(txn_id).await?;
        let cache = self.cache.write().await;
        Ok(DirGuard { cache, contents })
    }
}

#[async_trait]
impl fs::Store for Dir {
    async fn is_empty(&self, txn_id: TxnId) -> TCResult<bool> {
        tc_transact::fs::Dir::read(self, txn_id)
            .map_ok(|guard| fs::DirRead::is_empty(&guard))
            .await
    }
}

#[async_trait]
impl Transact for Dir {
    type Commit = TxnMapLockCommitGuard<PathSegment, DirEntry>;

    async fn commit(&self, txn_id: &TxnId) -> Self::Commit {
        let listing = self.listing.commit(txn_id).await;

        join_all(listing.iter().map(|(_, entry)| async move {
            match entry {
                DirEntry::Dir(dir) => {
                    dir.commit(txn_id).await;
                }
                DirEntry::File(file) => match file {
                    FileEntry::BTree(file) => {
                        file.commit(txn_id).await;
                    }
                    FileEntry::Chain(file) => {
                        file.commit(txn_id).await;
                    }
                    FileEntry::Library(file) => {
                        file.commit(txn_id).await;
                    }
                    #[cfg(feature = "tensor")]
                    FileEntry::Tensor(file) => {
                        file.commit(txn_id).await;
                    }
                },
            }
        }))
        .await;

        listing
    }

    async fn finalize(&self, txn_id: &TxnId) {
        {
            let listing = self.listing.read(*txn_id).await.expect("dir listing");

            join_all(listing.iter().map(|(_, entry)| async move {
                match entry {
                    DirEntry::Dir(dir) => dir.finalize(txn_id).await,
                    DirEntry::File(file) => match file {
                        FileEntry::BTree(file) => file.finalize(txn_id).await,
                        FileEntry::Chain(file) => file.finalize(txn_id).await,
                        FileEntry::Library(file) => file.finalize(txn_id).await,
                        #[cfg(feature = "tensor")]
                        FileEntry::Tensor(file) => file.finalize(txn_id).await,
                    },
                }
            }))
            .await;
        }

        self.listing.finalize(txn_id).await;
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
        _ => None,
    }
}
