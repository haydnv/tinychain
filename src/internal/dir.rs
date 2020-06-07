use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::BoxFuture;
use futures::lock::Mutex;

use crate::error;
use crate::transaction::{Transact, TxnId};
use crate::value::link::{PathSegment, TCPath};
use crate::value::TCResult;

use super::store::Store;

enum DirEntry {
    Dir(Arc<Dir>),
    Store(Arc<Store>),
}

#[async_trait]
impl Transact for DirEntry {
    async fn commit(&self, txn_id: &TxnId) {
        match self {
            DirEntry::Dir(dir) => dir.commit(txn_id).await,
            DirEntry::Store(store) => store.commit(txn_id).await,
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        match self {
            DirEntry::Dir(dir) => dir.rollback(txn_id).await,
            DirEntry::Store(store) => store.rollback(txn_id).await,
        }
    }
}

impl fmt::Display for DirEntry {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DirEntry::Dir(_) => write!(f, "(directory)"),
            DirEntry::Store(_) => write!(f, "(block store)"),
        }
    }
}

struct DirState {
    children: HashMap<PathSegment, DirEntry>,
    txn_cache: HashMap<TxnId, HashMap<PathSegment, DirEntry>>,
}

impl DirState {
    fn new() -> DirState {
        DirState {
            children: HashMap::new(),
            txn_cache: HashMap::new(),
        }
    }

    async fn get_dir(&self, txn_id: &TxnId, name: &PathSegment) -> TCResult<Option<Arc<Dir>>> {
        if let Some(Some(entry)) = self.txn_cache.get(txn_id).map(|data| data.get(name)) {
            match entry {
                DirEntry::Dir(dir) => Ok(Some(dir.clone())),
                other => Err(error::bad_request("Not a Dir", other)),
            }
        } else if let Some(entry) = self.children.get(name) {
            match entry {
                DirEntry::Dir(dir) => Ok(Some(dir.clone())),
                other => Err(error::bad_request("Not a Dir", other)),
            }
        } else {
            Ok(None)
        }
    }

    async fn get_store(&self, txn_id: &TxnId, name: &PathSegment) -> TCResult<Option<Arc<Store>>> {
        if let Some(Some(entry)) = self.txn_cache.get(txn_id).map(|data| data.get(name)) {
            match entry {
                DirEntry::Store(store) => Ok(Some(store.clone())),
                other => Err(error::bad_request("Not a Store", other)),
            }
        } else if let Some(entry) = self.children.get(name) {
            match entry {
                DirEntry::Store(store) => Ok(Some(store.clone())),
                other => Err(error::bad_request("Not a Store", other)),
            }
        } else {
            Ok(None)
        }
    }

    fn print_status(&self) {
        let mut dir_count = 0;
        let mut store_count = 0;
        for (_, entry) in &self.children {
            match entry {
                DirEntry::Dir(_) => dir_count += 1,
                DirEntry::Store(_) => store_count += 1,
            }
        }

        println!(
            "DirState has {} subdirectories and {} block stores",
            dir_count, store_count
        );
    }
}

pub struct Dir {
    context: PathBuf,
    state: Mutex<DirState>,
    temporary: bool,
}

impl Dir {
    pub fn new(mount_point: PathBuf) -> Arc<Dir> {
        Arc::new(Dir {
            context: mount_point,
            state: Mutex::new(DirState::new()),
            temporary: false,
        })
    }

    pub fn new_tmp(mount_point: PathBuf) -> Arc<Dir> {
        Arc::new(Dir {
            context: mount_point,
            state: Mutex::new(DirState::new()),
            temporary: true,
        })
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
                let path = path[0].clone();
                let mut state = self.state.lock().await;
                if state.children.contains_key(&path) {
                    Err(error::bad_request("Tried to create a new directory but there is already an entry at this path", &path))
                } else if let Some(txn_data) = state.txn_cache.get_mut(txn_id) {
                    if txn_data.contains_key(&path) {
                        Err(error::bad_request(
                            "Tried to create the same directory twice",
                            &path,
                        ))
                    } else {
                        println!("Created new Dir: {}", &path);
                        let dir = Dir::new(self.fs_path(&path));
                        txn_data.insert(path, DirEntry::Dir(dir.clone()));
                        Ok(dir)
                    }
                } else {
                    println!("Created new Dir {} in new Txn {}", &path, txn_id);
                    let dir = Dir::new(self.fs_path(&path));
                    let mut txn_data = HashMap::new();
                    txn_data.insert(path, DirEntry::Dir(dir.clone()));
                    state.txn_cache.insert(txn_id.clone(), txn_data);
                    Ok(dir)
                }
            } else {
                let dir = self
                    .get_or_create_dir(txn_id, &path[0].clone().into())
                    .await?;
                dir.create_dir(txn_id, path.slice_from(1)).await
            }
        })
    }

    pub async fn create_store(&self, txn_id: &TxnId, name: PathSegment) -> TCResult<Arc<Store>> {
        let mut state = self.state.lock().await;
        if let Some(txn_data) = state.txn_cache.get_mut(txn_id) {
            if txn_data.contains_key(&name) {
                Err(error::bad_request(
                    "Tried to create a new block store but there is already an entry at",
                    name,
                ))
            } else {
                let store = Store::new();
                txn_data.insert(name, DirEntry::Store(store.clone()));
                Ok(store)
            }
        } else if state.children.contains_key(&name) {
            Err(error::bad_request(
                "Tried to create a new block store but there is already an entry at",
                name,
            ))
        } else {
            let mut txn_data = HashMap::new();
            let store = Store::new();
            txn_data.insert(name, DirEntry::Store(store.clone()));
            state.txn_cache.insert(txn_id.clone(), txn_data);
            Ok(store)
        }
    }

    pub fn get_dir<'a>(
        &'a self,
        txn_id: &'a TxnId,
        path: &'a TCPath,
    ) -> BoxFuture<'a, TCResult<Option<Arc<Dir>>>> {
        Box::pin(async move {
            if path.is_empty() {
                Err(error::bad_request("Not a valid directory name", path))
            } else if let Some(dir) = self.state.lock().await.get_dir(txn_id, &path[0]).await? {
                if path.len() == 1 {
                    Ok(Some(dir))
                } else {
                    dir.get_dir(txn_id, &path.slice_from(1)).await
                }
            } else {
                Ok(None)
            }
        })
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
                let mut state = self.state.lock().await;
                if let Some(dir) = state.get_dir(txn_id, &path[0]).await? {
                    Ok(dir)
                } else {
                    let dir = Dir::new(self.fs_path(&path[0]));
                    state
                        .txn_cache
                        .get_mut(&txn_id)
                        .unwrap()
                        .insert(path[0].clone(), DirEntry::Dir(dir.clone()));

                    Ok(dir)
                }
            } else {
                self.get_or_create_dir(txn_id, &path[0].clone().into())
                    .await?
                    .get_or_create_dir(txn_id, &path.slice_from(1))
                    .await
            }
        })
    }

    pub async fn get_store(&self, txn_id: &TxnId, name: &PathSegment) -> TCResult<Arc<Store>> {
        self.state
            .lock()
            .await
            .get_store(txn_id, name)
            .await?
            .ok_or_else(|| error::not_found(name))
    }

    pub async fn is_empty(&self) -> bool {
        self.state.lock().await.children.is_empty()
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
        println!("Dir::commit {}", txn_id);

        let mut state = self.state.lock().await;
        if let Some(mut txn_data) = state.txn_cache.remove(txn_id) {
            for (name, entry) in txn_data.drain() {
                entry.commit(txn_id).await;
                state.children.insert(name, entry);
            }

            println!("Dir::commit!");
        } else {
            println!("Dir::commit has no data!");
        }

        state.print_status();
    }

    async fn rollback(&self, txn_id: &TxnId) {
        if let Some(mut txn_data) = self.state.lock().await.txn_cache.remove(txn_id) {
            for (_name, entry) in txn_data.drain() {
                entry.rollback(txn_id).await;
            }
        }

        println!("Dir::rollback!");
    }
}
