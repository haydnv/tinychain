use std::collections::hash_map::{Entry, HashMap};
use std::collections::HashSet;
use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::slice;
use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

use crate::chain;
use crate::class::{TCBoxTryFuture, TCResult};
use crate::collection::btree;
use crate::collection::tensor;
use crate::error;
use crate::lock::RwLock;
use crate::scalar::value::link::{PathSegment, TCPath};
use crate::transaction::lock::{Mutate, TxnLock};
use crate::transaction::{Transact, TxnId};

use super::file::File;
use super::hostfs;
use super::BlockData;

#[derive(Clone)]
pub enum DirEntry {
    Dir(Arc<Dir>),
    BTree(Arc<File<btree::Node>>),
    Chain(Arc<File<chain::ChainBlock>>),
    Tensor(Arc<File<tensor::Array>>),
}

impl From<Arc<File<btree::Node>>> for DirEntry {
    fn from(file: Arc<File<btree::Node>>) -> DirEntry {
        DirEntry::BTree(file)
    }
}

impl From<Arc<File<chain::ChainBlock>>> for DirEntry {
    fn from(file: Arc<File<chain::ChainBlock>>) -> DirEntry {
        DirEntry::Chain(file)
    }
}

impl From<Arc<File<tensor::Array>>> for DirEntry {
    fn from(file: Arc<File<tensor::Array>>) -> DirEntry {
        DirEntry::Tensor(file)
    }
}

impl TryFrom<DirEntry> for Arc<Dir> {
    type Error = error::TCError;

    fn try_from(entry: DirEntry) -> TCResult<Arc<Dir>> {
        match entry {
            DirEntry::Dir(dir) => Ok(dir),
            other => Err(error::bad_request("Expected Dir but found", other)),
        }
    }
}

impl TryFrom<DirEntry> for Arc<File<btree::Node>> {
    type Error = error::TCError;

    fn try_from(entry: DirEntry) -> TCResult<Arc<File<btree::Node>>> {
        match entry {
            DirEntry::BTree(btree) => Ok(btree),
            other => Err(error::bad_request("Expected Dir but found", other)),
        }
    }
}

impl TryFrom<DirEntry> for Arc<File<chain::ChainBlock>> {
    type Error = error::TCError;

    fn try_from(entry: DirEntry) -> TCResult<Arc<File<chain::ChainBlock>>> {
        match entry {
            DirEntry::Chain(chain) => Ok(chain),
            other => Err(error::bad_request("Expected Dir but found", other)),
        }
    }
}

impl TryFrom<DirEntry> for Arc<File<tensor::Array>> {
    type Error = error::TCError;

    fn try_from(entry: DirEntry) -> TCResult<Arc<File<tensor::Array>>> {
        match entry {
            DirEntry::Tensor(tensor) => Ok(tensor),
            other => Err(error::bad_request("Expected Dir but found", other)),
        }
    }
}

impl fmt::Display for DirEntry {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DirEntry::Dir(_) => write!(f, "(directory)"),
            DirEntry::BTree(_) => write!(f, "(BTree file)"),
            DirEntry::Chain(_) => write!(f, "(Chain file)"),
            DirEntry::Tensor(_) => write!(f, "(Tensor file)"),
        }
    }
}

#[derive(Clone, Default)]
struct DirContents {
    entries: HashMap<PathSegment, DirEntry>,
    deleted: HashSet<PathSegment>,
}

#[async_trait]
impl Mutate for DirContents {
    type Pending = Self;

    fn diverge(&self, _txn_id: &TxnId) -> Self::Pending {
        DirContents {
            entries: self.entries.clone(),
            deleted: HashSet::new(),
        }
    }

    async fn converge(&mut self, other: Self::Pending) {
        self.entries = other.entries;
    }
}

pub struct Dir {
    cache: RwLock<hostfs::Dir>,
    contents: TxnLock<DirContents>,
}

impl Dir {
    pub fn create<I: fmt::Display>(cache: RwLock<hostfs::Dir>, name: I) -> Arc<Dir> {
        Arc::new(Dir {
            cache,
            contents: TxnLock::new(name.to_string(), DirContents::default()),
        })
    }

    pub async fn unique_id(&self, txn_id: &TxnId) -> TCResult<PathSegment> {
        let existing_ids = self.entry_ids(txn_id).await?;
        loop {
            let id: PathSegment = Uuid::new_v4().into();
            if !existing_ids.contains(&id) {
                return Ok(id);
            }
        }
    }

    async fn entry_ids(&self, txn_id: &TxnId) -> TCResult<HashSet<PathSegment>> {
        Ok(self
            .contents
            .read(txn_id)
            .await?
            .entries
            .keys()
            .cloned()
            .collect())
    }

    pub async fn delete(&self, txn_id: TxnId, name: PathSegment) -> TCResult<()> {
        let mut contents = self.contents.write(txn_id).await?;
        contents.entries.remove(&name);
        contents.deleted.insert(name);
        Ok(())
    }

    pub fn get_dir<'a>(
        &'a self,
        txn_id: &'a TxnId,
        path: &'a [PathSegment],
    ) -> TCBoxTryFuture<'a, Option<Arc<Dir>>> {
        Box::pin(async move {
            if path.is_empty() {
                Err(error::bad_request(
                    "Cannot get Dir at empty path",
                    TCPath::from(path),
                ))
            } else if path.len() == 1 {
                if let Some(entry) = self.contents.read(txn_id).await?.entries.get(&path[0]) {
                    match entry {
                        DirEntry::Dir(dir) => Ok(Some(dir.clone())),
                        other => Err(error::bad_request("Not a Dir", other)),
                    }
                } else {
                    Ok(None)
                }
            } else if let Some(dir) = self.get_dir(txn_id, slice::from_ref(&path[0])).await? {
                dir.get_dir(txn_id, &path[1..]).await
            } else {
                Ok(None)
            }
        })
    }

    pub async fn get_entry<T: TryFrom<DirEntry, Error = error::TCError>>(
        &self,
        txn_id: &TxnId,
        name: &PathSegment,
    ) -> TCResult<Option<T>> {
        if let Some(entry) = self.contents.read(txn_id).await?.entries.get(name) {
            let entry: T = entry.clone().try_into()?;
            Ok(Some(entry))
        } else {
            Ok(None)
        }
    }

    pub fn create_dir<'a>(
        &'a self,
        txn_id: TxnId,
        path: &'a [PathSegment],
    ) -> TCBoxTryFuture<'a, Arc<Dir>> {
        Box::pin(async move {
            if path.is_empty() {
                Err(error::bad_request(
                    "Not a valid directory name",
                    TCPath::from(path),
                ))
            } else if path.len() == 1 {
                let mut contents = self.contents.write(txn_id.clone()).await?;
                match contents.entries.entry(path[0].clone()) {
                    Entry::Vacant(entry) => {
                        let fs_dir = self.cache.write().await.create_dir(path[0].clone())?;
                        let new_dir = Dir::create(fs_dir, &path[0]);
                        entry.insert(DirEntry::Dir(new_dir.clone()));
                        Ok(new_dir)
                    }
                    _ => Err(error::bad_request(
                        "Tried to create a new Dir but there is already an entry at",
                        TCPath::from(path),
                    )),
                }
            } else {
                let dir = self
                    .get_or_create_dir(&txn_id, slice::from_ref(&path[0]))
                    .await?;
                dir.create_dir(txn_id, &path[1..]).await
            }
        })
    }

    pub async fn create_file<T: BlockData>(
        &self,
        txn_id: TxnId,
        name: PathSegment,
    ) -> TCResult<Arc<File<T>>>
    where
        Arc<File<T>>: Into<DirEntry>,
    {
        let mut contents = self.contents.write(txn_id.clone()).await?;
        match contents.entries.entry(name) {
            Entry::Vacant(entry) => {
                let fs_cache = self.cache.write().await.create_dir(entry.key().clone())?;
                let file: Arc<File<T>> = File::create(entry.key().as_str(), fs_cache).await?;
                entry.insert(file.clone().into());
                Ok(file)
            }
            Entry::Occupied(entry) => Err(error::bad_request(
                "Tried to create a new File but there is already an entry at",
                entry.key(),
            )),
        }
    }

    pub fn get_or_create_dir<'a>(
        &'a self,
        txn_id: &'a TxnId,
        path: &'a [PathSegment],
    ) -> TCBoxTryFuture<'a, Arc<Dir>> {
        Box::pin(async move {
            if let Some(dir) = self.get_dir(txn_id, path).await? {
                Ok(dir)
            } else {
                self.create_dir(txn_id.clone(), path).await
            }
        })
    }

    pub async fn is_empty(&self, txn_id: &TxnId) -> TCResult<bool> {
        Ok(self.contents.read(txn_id).await?.entries.is_empty())
    }
}

#[async_trait]
impl Transact for Dir {
    async fn commit(&self, txn_id: &TxnId) {
        self.contents.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.contents.rollback(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        let contents = self.contents.read(txn_id).await.unwrap();
        let mut cache = self.cache.write().await;
        for name in contents.deleted.iter() {
            cache.delete(name).unwrap();
        }

        self.contents.finalize(txn_id).await;
    }
}
