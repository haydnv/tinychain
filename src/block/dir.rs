use std::collections::hash_map::{Entry, HashMap};
use std::collections::HashSet;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

use crate::class::{TCBoxTryFuture, TCResult};
use crate::collection::btree;
use crate::collection::tensor;
use crate::error;
use crate::transaction::lock::{Mutate, TxnLock};
use crate::transaction::{Transact, TxnId};
use crate::value::link::{PathSegment, TCPath};

use super::file::File;
use super::hostfs;

#[derive(Clone)]
enum DirEntry {
    Dir(Arc<Dir>),
    BTree(Arc<File<btree::Node>>),
    Tensor(Arc<File<tensor::Array>>),
}

impl fmt::Display for DirEntry {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DirEntry::Dir(_) => write!(f, "(directory)"),
            DirEntry::BTree(_) => write!(f, "(BTree file)"),
            DirEntry::Tensor(_) => write!(f, "(Tensor file)"),
        }
    }
}

struct DirContents(HashMap<PathSegment, DirEntry>);

#[async_trait]
impl Mutate for DirContents {
    type Pending = HashMap<PathSegment, DirEntry>;

    fn diverge(&self, _txn_id: &TxnId) -> Self::Pending {
        self.0.clone()
    }

    async fn converge(&mut self, other: Self::Pending) {
        self.0 = other;
    }
}

pub struct Dir {
    cache: hostfs::RwLock<hostfs::Dir>,
    temporary: bool,
    contents: TxnLock<DirContents>,
}

impl Dir {
    pub fn create(txn_id: TxnId, cache: hostfs::RwLock<hostfs::Dir>, temporary: bool) -> Arc<Dir> {
        Arc::new(Dir {
            cache,
            temporary,
            contents: TxnLock::new(txn_id, DirContents(HashMap::new())),
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
            .deref()
            .keys()
            .cloned()
            .collect())
    }

    pub async fn delete_file<'a>(&'a self, txn_id: TxnId, name: &'a PathSegment) -> TCResult<()> {
        self.contents.write(txn_id).await?.deref_mut().remove(&name);
        Ok(())
    }

    pub fn get_dir<'a>(
        &'a self,
        txn_id: &'a TxnId,
        path: &'a TCPath,
    ) -> TCBoxTryFuture<'a, Option<Arc<Dir>>> {
        Box::pin(async move {
            if path.is_empty() {
                Err(error::bad_request("Cannot get Dir at empty path", path))
            } else if path.len() == 1 {
                if let Some(entry) = self.contents.read(txn_id).await?.deref().get(&path[0]) {
                    match entry {
                        DirEntry::Dir(dir) => Ok(Some(dir.clone())),
                        other => Err(error::bad_request("Not a Dir", other)),
                    }
                } else {
                    Ok(None)
                }
            } else if let Some(dir) = self.get_dir(txn_id, &path[0].clone().into()).await? {
                dir.get_dir(txn_id, &path.slice_from(1)).await
            } else {
                Ok(None)
            }
        })
    }

    pub async fn get_btree(
        &self,
        txn_id: &TxnId,
        name: &PathSegment,
    ) -> TCResult<Option<Arc<File<btree::Node>>>> {
        if let Some(entry) = self.contents.read(txn_id).await?.deref().get(name) {
            match entry {
                DirEntry::BTree(file) => Ok(Some(file.clone())),
                other => Err(error::bad_request("Not a File", other)),
            }
        } else {
            Ok(None)
        }
    }

    pub async fn get_tensor(
        &self,
        txn_id: &TxnId,
        name: &PathSegment,
    ) -> TCResult<Option<Arc<File<tensor::Array>>>> {
        if let Some(entry) = self.contents.read(txn_id).await?.deref().get(name) {
            match entry {
                DirEntry::Tensor(file) => Ok(Some(file.clone())),
                other => Err(error::bad_request("Not a File", other)),
            }
        } else {
            Ok(None)
        }
    }

    pub fn create_dir<'a>(
        &'a self,
        txn_id: &'a TxnId,
        path: &'a TCPath,
    ) -> TCBoxTryFuture<'a, Arc<Dir>> {
        Box::pin(async move {
            if path.is_empty() {
                Err(error::bad_request("Not a valid directory name", path))
            } else if path.len() == 1 {
                let mut contents = self.contents.write(txn_id.clone()).await?;
                match contents.entry(path[0].clone()) {
                    Entry::Vacant(entry) => {
                        let fs_dir = self.cache.write().await.create_dir(path[0].clone())?;
                        let new_dir = Dir::create(txn_id.clone(), fs_dir, self.temporary);
                        entry.insert(DirEntry::Dir(new_dir.clone()));
                        Ok(new_dir)
                    }
                    _ => Err(error::bad_request(
                        "Tried to create a new Dir but there is already an entry at",
                        path,
                    )),
                }
            } else {
                let dir = self
                    .get_or_create_dir(&txn_id, &path[0].clone().into())
                    .await?;
                dir.create_dir(txn_id, &path.slice_from(1)).await
            }
        })
    }

    pub async fn create_btree(
        &self,
        txn_id: TxnId,
        name: PathSegment,
    ) -> TCResult<Arc<File<btree::Node>>> {
        let mut contents = self.contents.write(txn_id.clone()).await?;
        match contents.entry(name) {
            Entry::Vacant(entry) => {
                let fs_cache = self.cache.write().await.create_dir(entry.key().clone())?;
                let file = File::create(txn_id, fs_cache).await?;
                entry.insert(DirEntry::BTree(file.clone()));
                Ok(file)
            }
            Entry::Occupied(entry) => Err(error::bad_request(
                "Tried to create a new File but there is already an entry at",
                entry.key(),
            )),
        }
    }

    pub async fn create_tensor(
        &self,
        txn_id: TxnId,
        name: PathSegment,
    ) -> TCResult<Arc<File<tensor::Array>>> {
        let mut contents = self.contents.write(txn_id.clone()).await?;
        match contents.entry(name) {
            Entry::Vacant(entry) => {
                let fs_cache = self.cache.write().await.create_dir(entry.key().clone())?;
                let file = File::create(txn_id, fs_cache).await?;
                entry.insert(DirEntry::Tensor(file.clone()));
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
        path: &'a TCPath,
    ) -> TCBoxTryFuture<'a, Arc<Dir>> {
        Box::pin(async move {
            if let Some(dir) = self.get_dir(txn_id, path).await? {
                Ok(dir)
            } else {
                self.create_dir(txn_id, path).await
            }
        })
    }

    pub async fn is_empty(&self, txn_id: &TxnId) -> TCResult<bool> {
        Ok(self.contents.read(txn_id).await?.deref().is_empty())
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
}
