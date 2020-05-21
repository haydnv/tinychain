use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use std::path::PathBuf;
use std::sync::Arc;

use bytes::{BufMut, Bytes, BytesMut};
use futures::future::BoxFuture;
use futures::lock::Mutex;

use crate::error;
use crate::transaction::TxnId;
use crate::value::link::{PathSegment, TCPath};
use crate::value::TCResult;

pub type Checksum = [u8; 32];

pub trait Block:
    Clone + Into<Bytes> + Into<Checksum> + TryFrom<Bytes, Error = error::TCError>
{
}

struct TxnCache {
    subdirs: HashMap<TxnId, HashMap<PathSegment, Arc<Store>>>,
    blocks: HashMap<TxnId, HashMap<PathSegment, BytesMut>>,
}

impl TxnCache {
    fn new() -> TxnCache {
        TxnCache {
            subdirs: HashMap::new(),
            blocks: HashMap::new(),
        }
    }
}

struct StoreState {
    cache: TxnCache,
    subdirs: HashMap<PathSegment, Arc<Store>>,
    blocks: HashMap<PathSegment, BytesMut>,
}

impl StoreState {
    fn new() -> StoreState {
        StoreState {
            cache: TxnCache::new(),
            subdirs: HashMap::new(),
            blocks: HashMap::new(),
        }
    }
}

pub struct Store {
    mount_point: PathBuf,
    context: Option<PathSegment>,
    tmp: bool,
    state: Mutex<StoreState>,
}

impl Store {
    pub fn new(mount_point: PathBuf) -> Arc<Store> {
        Arc::new(Store {
            mount_point,
            context: None,
            tmp: false,
            state: Mutex::new(StoreState::new()),
        })
    }

    pub fn new_tmp(mount_point: PathBuf) -> Arc<Store> {
        Arc::new(Store {
            mount_point,
            context: None,
            tmp: true,
            state: Mutex::new(StoreState::new()),
        })
    }

    fn subdir(&self, context: PathSegment) -> Arc<Store> {
        Arc::new(Store {
            mount_point: self.fs_path(&context),
            context: Some(context.clone()),
            tmp: self.tmp,
            state: Mutex::new(StoreState::new()),
        })
    }

    pub async fn append(
        &self,
        txn_id: &TxnId,
        block_id: &PathSegment,
        data: Bytes,
    ) -> TCResult<()> {
        println!("append to block {} in {}", block_id, txn_id);
        let mut state = self.state.lock().await;

        let block_exists = state.blocks.contains_key(block_id);
        if let Some(blocks) = state.cache.blocks.get_mut(txn_id) {
            if let Some(block) = blocks.get_mut(block_id) {
                println!("found {} in txn {}", block_id, txn_id);
                block.put(data);
                Ok(())
            } else if block_exists {
                println!("found {}, appending in txn {}", block_id, txn_id);
                blocks.insert(block_id.clone(), data[..].into());
                Ok(())
            } else {
                Err(error::internal(format!(
                    "Tried to append to nonexistent block {} in txn {}",
                    block_id, txn_id
                )))
            }
        } else if block_exists {
            println!("found {}, creating new cache for txn {}", block_id, txn_id);
            let mut blocks = HashMap::new();
            blocks.insert(block_id.clone(), data[..].into());
            state.cache.blocks.insert(txn_id.clone(), blocks);
            Ok(())
        } else {
            Err(error::internal(format!(
                "Tried to append to nonexistent block: {}",
                block_id
            )))
        }
    }

    pub fn commit<'a>(&'a self, txn_id: &'a TxnId) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            let mut state = self.state.lock().await;

            if let Some(subdirs) = state.cache.subdirs.remove(txn_id) {
                state.subdirs.extend(subdirs);
                println!("{:?} subdirs after commit:", self.context);
                for (name, store) in state.subdirs.iter() {
                    store.commit(txn_id).await;
                    println!("\t{}", name);
                }
            }

            if let Some(mut blocks) = state.cache.blocks.remove(txn_id) {
                for (name, block) in blocks.drain() {
                    if let Some(existing_block) = state.blocks.get_mut(&name) {
                        existing_block.put(block);
                        println!("appended committed data to {}", name);
                    } else {
                        println!("committing new block {}", name);
                        state.blocks.insert(name, block);
                    }
                }
            }
        })
    }

    pub async fn contains_block(&self, txn_id: &TxnId, block_id: &PathSegment) -> bool {
        let state = self.state.lock().await;
        if let Some(blocks) = state.cache.blocks.get(txn_id) {
            if blocks.contains_key(block_id) {
                return true;
            }
        }

        state.blocks.contains_key(block_id)
    }

    pub async fn get_block<B: Block>(
        self: Arc<Self>,
        txn_id: TxnId,
        block_id: PathSegment,
    ) -> TCResult<B> {
        // TODO: read from filesystem

        if let Some(buffer) = self.get_bytes(txn_id, block_id.clone()).await {
            buffer.try_into()
        } else {
            Err(error::not_found(block_id))
        }
    }

    pub async fn get_bytes(self: Arc<Self>, txn_id: TxnId, block_id: PathSegment) -> Option<Bytes> {
        let state = self.state.lock().await;

        let block = if let Some(block) = state.blocks.get(&block_id) {
            Bytes::copy_from_slice(&block[..])
        } else {
            Bytes::from(&[][..])
        };

        let append = if let Some(blocks) = state.cache.blocks.get(&txn_id) {
            if let Some(block) = blocks.get(&block_id) {
                Bytes::copy_from_slice(&block[..])
            } else {
                Bytes::from(&[][..])
            }
        } else {
            Bytes::from(&[][..])
        };

        if !block.is_empty() || !append.is_empty() {
            Some(Bytes::from([&block[..], &append[..]].concat()))
        } else {
            None
        }
    }

    pub fn get_store<'a>(
        &'a self,
        txn_id: &'a TxnId,
        path: &'a TCPath,
    ) -> BoxFuture<'a, Option<Arc<Store>>> {
        Box::pin(async move {
            if path.is_empty() {
                return None;
            }

            let state = self.state.lock().await;
            if path.len() == 1 {
                if let Some(subdirs) = state.cache.subdirs.get(txn_id) {
                    if let Some(store) = subdirs.get(&path[0]) {
                        return Some(store.clone());
                    }
                }

                state.subdirs.get(&path[0]).cloned()
            } else if let Some(store) = state.subdirs.get(&path[0]) {
                store.get_store(txn_id, &path.slice_from(1)).await
            } else {
                None
            }
        })
    }

    pub async fn new_block(
        &self,
        txn_id: &TxnId,
        block_id: PathSegment,
        initial_value: Bytes,
    ) -> TCResult<()> {
        println!(
            "new block {} in txn {}, initial size {}",
            block_id,
            txn_id,
            initial_value.len()
        );

        let mut state = self.state.lock().await;
        if state.blocks.contains_key(&block_id) {
            Err(error::internal(format!(
                "Attempted to truncate an existing block: {}",
                block_id
            )))
        } else if let Some(blocks) = state.cache.blocks.get_mut(txn_id) {
            if blocks.contains_key(&block_id) {
                Err(error::internal(format!(
                    "Attempted to truncate an existing block: {}",
                    block_id
                )))
            } else {
                blocks.insert(block_id, initial_value[..].into());
                Ok(())
            }
        } else {
            println!("creating block {}", block_id);
            let mut blocks = HashMap::new();
            blocks.insert(block_id, initial_value[..].into());
            state.cache.blocks.insert(txn_id.clone(), blocks);
            Ok(())
        }
    }

    pub fn reserve<'a>(
        &'a self,
        txn_id: &'a TxnId,
        path: TCPath,
    ) -> BoxFuture<'a, TCResult<Arc<Store>>> {
        Box::pin(async move {
            if path.is_empty() {
                return Err(error::internal("Tried to create block store with no name"));
            }

            if path.len() == 1 {
                let mut state = self.state.lock().await;

                if !state.cache.subdirs.contains_key(txn_id) {
                    state.cache.subdirs.insert(txn_id.clone(), HashMap::new());
                }

                let path = &path[0];
                if state.cache.subdirs.get(txn_id).unwrap().contains_key(path) {
                    Err(error::bad_request("This path is already reserved", path))
                } else {
                    let subdir = self.subdir(path.clone());
                    state
                        .cache
                        .subdirs
                        .get_mut(txn_id)
                        .unwrap()
                        .insert(path.clone(), subdir);
                    Ok(state
                        .cache
                        .subdirs
                        .get(txn_id)
                        .unwrap()
                        .get(path)
                        .unwrap()
                        .clone())
                }
            } else {
                let store = self.reserve_or_get(txn_id, &path[0].clone().into()).await?;
                store.reserve(txn_id, path.slice_from(1)).await
            }
        })
    }

    pub async fn reserve_or_get(&self, txn_id: &TxnId, path: &TCPath) -> TCResult<Arc<Store>> {
        if let Some(store) = self.get_store(txn_id, path).await {
            Ok(store)
        } else {
            self.reserve(txn_id, path.clone()).await
        }
    }

    fn fs_path(&self, name: &PathSegment) -> PathBuf {
        let mut path = self.mount_point.clone();
        if let Some(context) = &self.context {
            path.push(context.to_string());
        }
        path.push(name.to_string());
        path
    }
}
