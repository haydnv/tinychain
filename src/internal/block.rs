use std::collections::HashMap;
use std::convert::TryInto;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use bytes::Bytes;
use futures::Future;
use tokio::fs;

use crate::error;
use crate::internal::cache::Map;
use crate::internal::{GROUP_DELIMITER, RECORD_DELIMITER};
use crate::value::{PathSegment, TCPath, TCResult};

#[derive(Debug)]
pub struct Store {
    mount_point: PathBuf,
    context: Option<PathSegment>,
    children: Map<PathSegment, Arc<Store>>,
    buffer: RwLock<HashMap<PathSegment, Vec<u8>>>,
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

    fn child(self: &Arc<Self>, context: PathSegment, mount_point: PathBuf) -> Arc<Store> {
        Arc::new(Store {
            mount_point,
            context: Some(context),
            children: Map::new(),
            buffer: RwLock::new(HashMap::new()),
            tmp: self.tmp,
        })
    }

    pub fn create<E: Into<error::TCError>, T: TryInto<TCPath, Error = E>>(
        self: &Arc<Self>,
        path: T,
    ) -> TCResult<Arc<Store>> {
        let path: TCPath = path.try_into().map_err(|e| e.into())?;
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

            let store = self.child(path.clone(), self.fs_path(&path));
            self.children.insert(path.clone(), store.clone());
            Ok(store)
        } else {
            let store = if let Some(store) = self.children.get(&path[0]) {
                store
            } else {
                let child_path = &path[0];
                let store = Store::new(self.fs_path(child_path), child_path.clone().into());
                self.children.insert(child_path.clone(), store.clone());
                store
            };

            store.create(path.slice_from(1))
        }
    }

    pub fn get(self: &Arc<Self>, path: &TCPath) -> Option<Arc<Store>> {
        if path.is_empty() {
            return None;
        }

        if path.len() == 1 {
            self.children.get(&path[0])
        } else if let Some(store) = self.children.get(&path[0]) {
            store.get(&path.slice_from(1))
        } else {
            None
        }
    }

    pub fn into_bytes(self: Arc<Self>, block_id: PathSegment) -> impl Future<Output = Bytes> {
        async move {
            if let Some(buffer) = self.buffer.read().unwrap().get(&block_id) {
                Bytes::copy_from_slice(buffer)
            } else {
                // TODO
                Bytes::new()
            }
        }
    }

    pub async fn exists(self: &Arc<Self>, path: &PathSegment) -> TCResult<bool> {
        let fs_path = self.fs_path(path);
        if self.children.contains_key(path) || self.buffer.read().unwrap().contains_key(path) {
            return Ok(true);
        }

        match fs::metadata(fs_path).await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    pub fn flush(
        self: Arc<Self>,
        block_id: PathSegment,
        header: Bytes,
        data: Vec<Bytes>,
    ) -> impl Future<Output = ()> {
        if data.is_empty() {
            panic!("flush to {} called with no data", block_id);
        }

        async move {
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
            if let Some(block) = buffer.get_mut(&block_id) {
                block.append(&mut records)
            } else {
                buffer.insert(block_id, records);
            }

            // TODO: persist data to disk
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
