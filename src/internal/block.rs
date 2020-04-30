use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use bytes::Bytes;
use futures::Future;
use tokio::fs;

use crate::error;
use crate::internal::cache::Map;
use crate::internal::RECORD_DELIMITER;
use crate::value::{Link, TCResult};

#[derive(Debug)]
pub struct Store {
    mount_point: PathBuf,
    context: Link,
    children: Map<Link, Arc<Store>>,
    buffer: RwLock<HashMap<Link, Vec<u8>>>,
    tmp: bool,
}

impl Store {
    pub fn new(context: Link, mount_point: PathBuf) -> Arc<Store> {
        Arc::new(Store {
            mount_point,
            context,
            children: Map::new(),
            buffer: RwLock::new(HashMap::new()),
            tmp: false,
        })
    }

    pub fn new_tmp(context: Link, mount_point: PathBuf) -> Arc<Store> {
        Arc::new(Store {
            mount_point,
            context,
            children: Map::new(),
            buffer: RwLock::new(HashMap::new()),
            tmp: true,
        })
    }

    fn child(self: &Arc<Self>, context: Link, mount_point: PathBuf) -> Arc<Store> {
        Arc::new(Store {
            mount_point,
            context,
            children: Map::new(),
            buffer: RwLock::new(HashMap::new()),
            tmp: self.tmp,
        })
    }

    pub fn create(self: &Arc<Self>, path: &Link) -> TCResult<Arc<Store>> {
        if path.is_empty() {
            return Err(error::internal("Tried to create block store with no name"));
        }

        if path.len() == 1 {
            if self.children.contains_key(&path) {
                return Err(error::internal(&format!(
                    "Tried to create a block store that already exists! {}",
                    path
                )));
            }

            let store = self.child(path.clone(), self.fs_path(path));
            self.children.insert(path.nth(0), store.clone());
            Ok(store)
        } else {
            let store = if let Some(store) = self.children.get(&path.nth(0)) {
                store
            } else {
                let child_path = path.nth(0);
                let store = Store::new(child_path.clone(), self.fs_path(&child_path));
                self.children.insert(child_path, store.clone());
                store
            };

            store.create(&path.slice_from(1))
        }
    }

    pub fn get(self: &Arc<Self>, path: &Link) -> Option<Arc<Store>> {
        if path.is_empty() {
            return None;
        }

        if path.len() == 1 {
            self.children.get(path)
        } else if let Some(store) = self.children.get(&path.nth(1)) {
            store.get(&path.slice_from(1))
        } else {
            None
        }
    }

    pub fn into_bytes(self: Arc<Self>, path: Link) -> impl Future<Output = Bytes> {
        async move {
            if let Some(buffer) = self.buffer.read().unwrap().get(&path) {
                Bytes::copy_from_slice(buffer)
            } else {
                // TODO
                Bytes::new()
            }
        }
    }

    pub async fn append(self: &Arc<Self>, path: Link, data: Vec<u8>) -> TCResult<()> {
        if data.contains(&(RECORD_DELIMITER as u8)) {
            let msg = "Attempted to write a block containing the ASCII record delimiter (0x30)";
            return Err(error::internal(msg));
        }

        let data = [&data[..], &[RECORD_DELIMITER as u8]].concat();

        let mut buffer = self.buffer.write().unwrap();
        match buffer.get_mut(&path) {
            Some(file_buffer) => file_buffer.extend(data),
            None => {
                buffer.insert(path, data);
            }
        }

        Ok(())
    }

    pub async fn exists(self: &Arc<Self>, path: &Link) -> TCResult<bool> {
        let fs_path = self.fs_path(path);
        if self.children.contains_key(path) {
            return Ok(true);
        }

        match fs::metadata(fs_path).await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    pub fn flush(
        self: Arc<Self>,
        _path: Link,
        _header: &Bytes,
        _data: &[Bytes],
    ) -> impl Future<Output = ()> {
        async move {
            if self.tmp {
                return;
            }

            // TODO
        }
    }

    fn fs_path(&self, name: &Link) -> PathBuf {
        if !name.len() == 1 {
            panic!("Tried to look up the filesystem path of {}", name);
        }

        let mut path = self.mount_point.clone();

        for segment in self.context.clone().into_iter() {
            path.push(&segment.to_string()[1..]);
        }

        for i in 0..name.len() {
            path.push(name.as_str(i));
        }

        path
    }
}
