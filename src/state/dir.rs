use std::collections::hash_map::{Entry, HashMap};
use std::collections::HashSet;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::BoxFuture;

use crate::error;
use crate::internal::File;
use crate::transaction::lock::{Mutate, TxnLock};
use crate::transaction::{Transact, TxnId};
use crate::value::link::{PathSegment, TCPath};
use crate::value::TCResult;

#[derive(Clone)]
enum DirEntry {
    Dir(Arc<Dir>),
    File(Arc<File>),
}

impl fmt::Display for DirEntry {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DirEntry::Dir(_) => write!(f, "(directory)"),
            DirEntry::File(_) => write!(f, "(file)"),
        }
    }
}

#[derive(Clone)]
struct DirContents(HashMap<PathSegment, DirEntry>);

#[async_trait]
impl Mutate for DirContents {
    async fn commit(&mut self, mut new_value: DirContents) {
        let existing: HashSet<PathSegment> = self.0.keys().cloned().collect();
        let new: HashSet<PathSegment> = new_value.0.keys().cloned().collect();
        let deleted = existing.difference(&new);

        self.0.extend(new_value.0.drain());

        for name in deleted {
            self.0.remove(name);
        }
    }
}

pub struct Dir {
    context: PathBuf,
    temporary: bool,
    contents: TxnLock<DirContents>,
}

impl Dir {
    pub fn create(txn_id: TxnId, mount_point: PathBuf, temporary: bool) -> Arc<Dir> {
        Arc::new(Dir {
            context: mount_point,
            temporary,
            contents: TxnLock::new(txn_id, DirContents(HashMap::new())),
        })
    }

    pub fn get_dir<'a>(
        &'a self,
        txn_id: &'a TxnId,
        path: &'a TCPath,
    ) -> BoxFuture<'a, TCResult<Option<Arc<Dir>>>> {
        Box::pin(async move {
            if path.is_empty() {
                Err(error::bad_request("Cannot get Dir at empty path", path))
            } else if path.len() == 1 {
                if let Some(entry) = self.contents.read(txn_id).await?.0.get(&path[0]) {
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

    pub async fn get_file(
        &self,
        txn_id: &TxnId,
        name: &PathSegment,
    ) -> TCResult<Option<Arc<File>>> {
        if let Some(entry) = self.contents.read(txn_id).await?.0.get(name) {
            match entry {
                DirEntry::File(file) => Ok(Some(file.clone())),
                other => Err(error::bad_request("Not a File", other)),
            }
        } else {
            Ok(None)
        }
    }

    pub fn create_dir<'a>(
        &'a self,
        txn_id: &'a TxnId,
        path: TCPath,
    ) -> BoxFuture<'a, TCResult<Arc<Dir>>> {
        Box::pin(async move {
            if path.is_empty() {
                Err(error::bad_request("Not a valid directory name", path))
            } else if path.len() == 1 {
                let mut contents = self.contents.write(txn_id.clone()).await?;
                match contents.0.entry(path[0].clone()) {
                    Entry::Vacant(entry) => {
                        let dir =
                            Dir::create(txn_id.clone(), self.fs_path(&path[0]), self.temporary);
                        entry.insert(DirEntry::Dir(dir.clone()));
                        Ok(dir)
                    }
                    _ => Err(error::bad_request(
                        "Tried to create a new Dir but there is already an entry at",
                        path,
                    )),
                }
            } else {
                let dir = self
                    .get_or_create_dir(txn_id, &path[0].clone().into())
                    .await?;
                dir.create_dir(txn_id, path.slice_from(1)).await
            }
        })
    }

    pub async fn create_file(&self, txn_id: &TxnId, name: PathSegment) -> TCResult<Arc<File>> {
        let mut contents = self.contents.write(txn_id.clone()).await?;
        match contents.0.entry(name) {
            Entry::Vacant(entry) => {
                let file = File::new();
                entry.insert(DirEntry::File(file.clone()));
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
    ) -> BoxFuture<'a, TCResult<Arc<Dir>>> {
        Box::pin(async move {
            if path.is_empty() {
                Err(error::bad_request("Not a valid directory name", path))
            } else if path.len() == 1 {
                let mut contents = self.contents.write(txn_id.clone()).await?;
                match contents.0.get(&path[0]) {
                    Some(DirEntry::Dir(dir)) => Ok(dir.clone()),
                    Some(other) => Err(error::bad_request("Expected a Dir but found", other)),
                    None => {
                        let dir =
                            Dir::create(txn_id.clone(), self.fs_path(&path[0]), self.temporary);
                        contents
                            .0
                            .insert(path[0].clone(), DirEntry::Dir(dir.clone()));
                        Ok(dir)
                    }
                }
            } else {
                self.get_or_create_dir(txn_id, &path[0].clone().into())
                    .await?
                    .get_or_create_dir(txn_id, &path.slice_from(1))
                    .await
            }
        })
    }

    pub async fn is_empty(&self, txn_id: &TxnId) -> TCResult<bool> {
        Ok(self.contents.read(txn_id).await?.0.is_empty())
    }

    fn fs_path(&self, name: &PathSegment) -> PathBuf {
        let mut path = self.context.clone();
        path.push(name.to_string());
        path
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
