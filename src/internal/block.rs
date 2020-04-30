use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use bytes::Bytes;
use futures::Future;
use tokio::fs;

use crate::error;
use crate::internal::cache::Map;
use crate::internal::{GROUP_DELIMITER, RECORD_DELIMITER};
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
        println!("read data from {}", path);
        async move {
            if let Some(buffer) = self.buffer.read().unwrap().get(&path) {
                println!("{} bytes", buffer.len());
                Bytes::copy_from_slice(buffer)
            } else {
                // TODO
                Bytes::new()
            }
        }
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
        path: Link,
        header: Bytes,
        data: Vec<Bytes>,
    ) -> impl Future<Output = ()> {
        if data.is_empty() {
            panic!("flush to {} called with no data", path);
        }

        async move {
            if self.tmp {
                println!("flush to {} ignored", path);
                return;
            } else {
                println!("flush to {}", path);
            }

            let group_delimiter = Bytes::from(&[GROUP_DELIMITER as u8][..]);
            let record_delimiter = Bytes::from(&[RECORD_DELIMITER as u8][..]);

            let mut records = Vec::with_capacity(data.len() + 1);
            records.push(header);
            records.push(record_delimiter.clone());
            for record in data {
                records.push(record);
                records.push(record_delimiter.clone());
            }
            records.push(group_delimiter);

            let mut records: Vec<u8> = records.concat();
            let mut buffer = self.buffer.write().unwrap();
            if let Some(block) = buffer.get_mut(&path) {
                block.append(&mut records)
            } else {
                buffer.insert(path, records);
            }

            // TODO: persist data to disk
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
