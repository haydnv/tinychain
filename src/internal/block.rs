use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use bytes::{Bytes, BytesMut};
use tokio::fs;

use crate::error;
use crate::internal::cache::Map;
use crate::value::{PathSegment, TCPath, TCResult};

pub type Checksum = [u8; 32];

pub trait Block:
    Clone + Into<Bytes> + Into<Checksum> + TryFrom<Bytes, Error = error::TCError>
{
}

pub struct Store {
    mount_point: PathBuf,
    context: Option<PathSegment>,
    children: Map<PathSegment, Arc<Store>>,
    buffer: RwLock<HashMap<PathSegment, BytesMut>>,
    tmp: bool,
}

impl Store {
    pub fn new(mount_point: PathBuf, context: Option<PathSegment>) -> Arc<Store> {
        Arc::new(Store {
            mount_point,
            context,
            children: Map::new(),
            buffer: RwLock::new(HashMap::new()),
            tmp: false,
        })
    }

    pub fn new_tmp(mount_point: PathBuf, context: Option<PathSegment>) -> Arc<Store> {
        Arc::new(Store {
            mount_point,
            context,
            children: Map::new(),
            buffer: RwLock::new(HashMap::new()),
            tmp: true,
        })
    }

    fn child(&self, context: PathSegment) -> Arc<Store> {
        let child = Arc::new(Store {
            mount_point: self.fs_path(&context),
            context: Some(context.clone()),
            children: Map::new(),
            buffer: RwLock::new(HashMap::new()),
            tmp: self.tmp,
        });

        self.children.insert(context, child.clone());
        child
    }

    pub fn reserve(self: &Arc<Self>, path: TCPath) -> TCResult<Arc<Store>> {
        if path.is_empty() {
            return Err(error::internal("Tried to create block store with no name"));
        }

        if path.len() == 1 {
            let path = &path[0];
            if self.children.contains_key(path) {
                return Err(error::internal(&format!(
                    "Tried to create a block store that already exists! {}",
                    path
                )));
            }

            Ok(self.child(path.clone()))
        } else {
            let store = if let Some(store) = self.children.get(&path[0]) {
                store
            } else {
                self.child(path[0].clone())
            };

            store.reserve(path.slice_from(1))
        }
    }

    pub fn reserve_or_get(self: &Arc<Self>, path: &TCPath) -> TCResult<Arc<Store>> {
        if let Some(store) = self.get_store(path) {
            Ok(store)
        } else {
            self.reserve(path.clone())
        }
    }

    pub async fn exists(&self, path: &PathSegment) -> TCResult<bool> {
        let fs_path = self.fs_path(path);
        if self.children.contains_key(path) || self.buffer.read().unwrap().contains_key(path) {
            return Ok(true);
        }

        match fs::metadata(fs_path).await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    pub fn append(&self, block_id: &PathSegment, data: Bytes) {
        // TODO: check filesystem
        if !self.buffer.read().unwrap().contains_key(block_id) {
            panic!("tried to append to a nonexistent block! {}", block_id);
        }

        self.buffer
            .write()
            .unwrap()
            .get_mut(&block_id)
            .unwrap()
            .extend(BytesMut::from(&data[..]));
    }

    pub async fn flush(self: Arc<Self>, _block_id: PathSegment) {
        // TODO: write block buffer to disk
    }

    pub fn get_store(&self, path: &TCPath) -> Option<Arc<Store>> {
        if path.is_empty() {
            return None;
        }

        if path.len() == 1 {
            self.children.get(&path[0])
        } else if let Some(store) = self.children.get(&path[0]) {
            store.get_store(&path.slice_from(1))
        } else {
            None
        }
    }

    pub async fn get_block<B: Block>(self: Arc<Self>, block_id: PathSegment) -> TCResult<B> {
        // TODO: read from filesystem

        if let Some(buffer) = self.get_bytes(block_id.clone()).await {
            buffer.try_into()
        } else {
            Err(error::not_found(block_id))
        }
    }

    pub async fn get_bytes(self: Arc<Self>, block_id: PathSegment) -> Option<Bytes> {
        if let Some(buffer) = self.buffer.read().unwrap().get(&block_id) {
            Some(Bytes::copy_from_slice(buffer))
        } else {
            None
        }
    }

    pub fn new_block(&self, block_id: PathSegment, initial_value: Bytes) {
        println!("new block, initial size {}", initial_value.len());

        let mut buffer = self.buffer.write().unwrap();
        if buffer.contains_key(&block_id) {
            panic!("Tried to overwrite block {}", block_id);
        }

        buffer.insert(block_id, initial_value[..].into());
    }

    pub fn put_block(&self, block_id: PathSegment, block: Bytes) {
        let mut buffer = self.buffer.write().unwrap();
        if buffer.contains_key(&block_id) {
            panic!("Tried to overwrite block {}", block_id);
        }

        buffer.insert(block_id, BytesMut::from(&block[..]));
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
