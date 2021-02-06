use std::collections::hash_map::{Entry, HashMap};
use std::collections::HashSet;
use std::fmt;
use std::pin::Pin;
use std::slice;
use std::sync::Arc;

use async_trait::async_trait;
use futures::Future;
use futures_locks::RwLock;
use uuid::Uuid;

use error::*;
use generic::{PathSegment, TCPath};

use crate::lock::*;
use crate::{Transact, TxnId};

use super::file::File;
use super::hostfs;
use super::BlockData;

pub trait FileEntry: Transact + fmt::Display + Clone + Send + Sync {}

#[derive(Clone)]
pub enum DirEntry<F: Clone + Send + Sync> {
    Dir(Arc<Dir<F>>),
    File(F),
}

impl<B: BlockData, F: FileEntry> From<File<B>> for DirEntry<F>
where
    F: From<File<B>>,
{
    fn from(file: File<B>) -> Self {
        Self::File(file.into())
    }
}

impl<F: FileEntry> fmt::Display for DirEntry<F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DirEntry::Dir(_) => write!(f, "(directory)"),
            DirEntry::File(file) => fmt::Display::fmt(file, f),
        }
    }
}

#[derive(Clone)]
struct DirContents<F: Clone + Send + Sync> {
    entries: HashMap<PathSegment, DirEntry<F>>,
    deleted: HashSet<PathSegment>,
}

impl<F: Clone + Send + Sync> Default for DirContents<F> {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            deleted: HashSet::new(),
        }
    }
}

#[async_trait]
impl<F: Clone + Send + Sync> Mutate for DirContents<F> {
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

pub struct Dir<F: Clone + Send + Sync> {
    cache: RwLock<hostfs::Dir>,
    contents: TxnLock<DirContents<F>>,
}

impl<F: FileEntry> Dir<F> {
    pub fn create<I: fmt::Display>(cache: RwLock<hostfs::Dir>, name: I) -> Arc<Self> {
        Arc::new(Dir {
            cache,
            contents: TxnLock::new(name.to_string(), DirContents::default()),
        })
    }

    pub async fn unique_id(&self, txn_id: &TxnId) -> TCResult<PathSegment> {
        let existing_ids = self.entry_ids(txn_id).await?;
        loop {
            let id: PathSegment = Uuid::new_v4().to_string().parse()?;
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
    ) -> Pin<Box<dyn Future<Output = TCResult<Option<Arc<Self>>>> + Send + 'a>> {
        Box::pin(async move {
            if path.is_empty() {
                Err(TCError::bad_request(
                    "Cannot get Dir at empty path",
                    TCPath::from(path),
                ))
            } else if path.len() == 1 {
                if let Some(entry) = self.contents.read(txn_id).await?.entries.get(&path[0]) {
                    match entry {
                        DirEntry::Dir(dir) => Ok(Some(dir.clone())),
                        other => Err(TCError::bad_request("Expected Dir but found", other)),
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

    pub fn create_dir<'a>(
        &'a self,
        txn_id: TxnId,
        path: &'a [PathSegment],
    ) -> Pin<Box<dyn Future<Output = TCResult<Arc<Self>>> + Send + 'a>> {
        Box::pin(async move {
            if path.is_empty() {
                Err(TCError::bad_request(
                    "Not a valid directory name",
                    TCPath::from(path),
                ))
            } else if path.len() == 1 {
                let mut contents = self.contents.write(txn_id).await?;
                match contents.entries.entry(path[0].clone()) {
                    Entry::Vacant(entry) => {
                        let fs_dir = self.cache.write().await.create_dir(path[0].clone()).await?;
                        let new_dir = Dir::create(fs_dir, &path[0]);
                        entry.insert(DirEntry::Dir(new_dir.clone()));
                        Ok(new_dir)
                    }
                    _ => Err(TCError::bad_request(
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

    pub async fn create_file<B: BlockData>(
        &self,
        txn_id: TxnId,
        name: PathSegment,
    ) -> TCResult<File<B>>
    where
        File<B>: Into<DirEntry<F>>,
    {
        let mut contents = self.contents.write(txn_id).await?;
        match contents.entries.entry(name) {
            Entry::Vacant(entry) => {
                let fs_cache = self
                    .cache
                    .write()
                    .await
                    .create_dir(entry.key().clone())
                    .await?;
                let file: File<B> = File::create(entry.key().as_str(), fs_cache).await?;
                entry.insert(file.clone().into());
                Ok(file)
            }
            Entry::Occupied(entry) => Err(TCError::bad_request(
                "Tried to create a new File but there is already an entry at",
                entry.key(),
            )),
        }
    }

    pub fn get_or_create_dir<'a>(
        &'a self,
        txn_id: &'a TxnId,
        path: &'a [PathSegment],
    ) -> Pin<Box<dyn Future<Output = TCResult<Arc<Self>>> + Send + 'a>> {
        Box::pin(async move {
            if let Some(dir) = self.get_dir(txn_id, path).await? {
                Ok(dir)
            } else {
                self.create_dir(*txn_id, path).await
            }
        })
    }

    pub async fn is_empty(&self, txn_id: &TxnId) -> TCResult<bool> {
        Ok(self.contents.read(txn_id).await?.entries.is_empty())
    }
}

#[async_trait]
impl<F: FileEntry> Transact for Dir<F> {
    async fn commit(&self, txn_id: &TxnId) {
        self.contents.commit(txn_id).await
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
