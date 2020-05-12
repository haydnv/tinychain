use std::collections::HashMap;
use std::path::PathBuf;

use bytes::BytesMut;

use crate::error;
use crate::transaction::TransactionId;
use crate::value::{PathSegment, TCPath, TCResult};

struct TransactionCache {
    subdirs: HashMap<TransactionId, HashMap<PathSegment, Store>>,
    buffer: HashMap<TransactionId, HashMap<PathSegment, BytesMut>>,
}

impl TransactionCache {
    fn new() -> TransactionCache {
        TransactionCache {
            subdirs: HashMap::new(),
            buffer: HashMap::new(),
        }
    }
}

pub struct Store {
    mount_point: PathBuf,
    context: Option<PathSegment>,
    cache: TransactionCache,
    subdirs: HashMap<PathSegment, Store>,
    data: HashMap<PathSegment, BytesMut>,
    tmp: bool,
}

impl Store {
    pub fn new(
        mount_point: PathBuf,
        context: Option<PathSegment>,
    ) -> Store {
        Store {
            mount_point,
            context,
            cache: TransactionCache::new(),
            subdirs: HashMap::new(),
            data: HashMap::new(),
            tmp: false,
        }
    }

    pub fn new_tmp(
        mount_point: PathBuf,
        context: Option<PathSegment>,
    ) -> Store {
        Store {
            mount_point,
            context,
            cache: TransactionCache::new(),
            subdirs: HashMap::new(),
            data: HashMap::new(),
            tmp: true,
        }
    }

    fn subdir(&self, context: PathSegment) -> Store {
        Store {
            mount_point: self.fs_path(&context),
            context: Some(context.clone()),
            cache: TransactionCache::new(),
            subdirs: HashMap::new(),
            data: HashMap::new(),
            tmp: self.tmp,
        }
    }

    pub fn reserve(&mut self, txn_id: &TransactionId, path: TCPath) -> TCResult<&Store> {
        if path.is_empty() {
            return Err(error::internal("Tried to create block store with no name"));
        }

        if path.len() == 1 {
            if !self.cache.subdirs.contains_key(txn_id) {
                self.cache.subdirs.insert(txn_id.clone(), HashMap::new());
            }

            let path = &path[0];
            if self.cache.subdirs.get(txn_id).unwrap().contains_key(path) {
                Err(error::bad_request("The path {} is already reserved", path))
            } else {
                let subdir = self.subdir(path.clone());
                self.cache.subdirs.get_mut(txn_id).unwrap().insert(path.clone(), subdir);
                Ok(self.cache.subdirs.get(txn_id).unwrap().get(path).unwrap())
            }
        } else {
            self.subdirs.get_mut(&path[0]).unwrap().reserve(txn_id, path.slice_from(1))
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
