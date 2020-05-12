use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use std::path::PathBuf;

use bytes::{BufMut, Bytes, BytesMut};

use crate::error;
use crate::transaction::TransactionId;
use crate::value::{PathSegment, TCPath, TCResult};

pub type Checksum = [u8; 32];

pub trait Block:
    Clone + Into<Bytes> + Into<Checksum> + TryFrom<Bytes, Error = error::TCError>
{
}

struct TransactionCache {
    subdirs: HashMap<TransactionId, HashMap<PathSegment, Store>>,
    blocks: HashMap<TransactionId, HashMap<PathSegment, BytesMut>>,
}

impl TransactionCache {
    fn new() -> TransactionCache {
        TransactionCache {
            subdirs: HashMap::new(),
            blocks: HashMap::new(),
        }
    }
}

pub struct Store {
    mount_point: PathBuf,
    context: Option<PathSegment>,
    cache: TransactionCache,
    subdirs: HashMap<PathSegment, Store>,
    blocks: HashMap<PathSegment, BytesMut>,
    tmp: bool,
}

impl Store {
    pub fn new(mount_point: PathBuf, context: Option<PathSegment>) -> Store {
        Store {
            mount_point,
            context,
            cache: TransactionCache::new(),
            subdirs: HashMap::new(),
            blocks: HashMap::new(),
            tmp: false,
        }
    }

    pub fn new_tmp(mount_point: PathBuf, context: Option<PathSegment>) -> Store {
        Store {
            mount_point,
            context,
            cache: TransactionCache::new(),
            subdirs: HashMap::new(),
            blocks: HashMap::new(),
            tmp: true,
        }
    }

    fn subdir(&self, context: PathSegment) -> Store {
        Store {
            mount_point: self.fs_path(&context),
            context: Some(context.clone()),
            cache: TransactionCache::new(),
            subdirs: HashMap::new(),
            blocks: HashMap::new(),
            tmp: self.tmp,
        }
    }

    pub fn append(
        &mut self,
        txn_id: &TransactionId,
        block_id: &PathSegment,
        data: Bytes,
    ) -> TCResult<()> {
        if let Some(blocks) = self.cache.blocks.get_mut(txn_id) {
            if let Some(block) = blocks.get_mut(block_id) {
                block.put(data);
                Ok(())
            } else if self.blocks.contains_key(block_id) {
                blocks.insert(block_id.clone(), data[..].into());
                Ok(())
            } else {
                Err(error::internal(format!(
                    "Tried to append to nonexistent block: {}",
                    block_id
                )))
            }
        } else if self.blocks.contains_key(block_id) {
            let mut blocks = HashMap::new();
            blocks.insert(block_id.clone(), data[..].into());
            self.cache.blocks.insert(txn_id.clone(), blocks);
            Ok(())
        } else {
            Err(error::internal(format!(
                "Tried to append to nonexistent block: {}",
                block_id
            )))
        }
    }

    pub async fn commit(&mut self, txn_id: &TransactionId) {
        if let Some(subdirs) = self.cache.subdirs.remove(txn_id) {
            self.subdirs.extend(subdirs);
        }

        if let Some(mut blocks) = self.cache.blocks.remove(txn_id) {
            for (name, block) in blocks.drain() {
                if let Some(existing_block) = self.blocks.get_mut(&name) {
                    existing_block.put(block);
                } else {
                    self.blocks.insert(name, block);
                }
            }
        }
    }

    pub async fn contains_block(&self, txn_id: &TransactionId, block_id: &PathSegment) -> bool {
        if let Some(blocks) = self.cache.blocks.get(txn_id) {
            if blocks.contains_key(block_id) {
                return true;
            }
        }

        self.blocks.contains_key(block_id)
    }

    pub async fn get_block<B: Block>(
        &self,
        txn_id: &TransactionId,
        block_id: &PathSegment,
    ) -> TCResult<B> {
        // TODO: read from filesystem

        if let Some(buffer) = self.get_bytes(txn_id, block_id).await {
            buffer.try_into()
        } else {
            Err(error::not_found(block_id))
        }
    }

    pub async fn get_bytes(&self, txn_id: &TransactionId, block_id: &PathSegment) -> Option<Bytes> {
        let block = if let Some(block) = self.blocks.get(block_id) {
            Bytes::copy_from_slice(&block[..])
        } else {
            Bytes::from(&[][..])
        };

        let append = if let Some(blocks) = self.cache.blocks.get(txn_id) {
            if let Some(block) = blocks.get(block_id) {
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

    pub fn get_store(&self, txn_id: &TransactionId, path: &TCPath) -> Option<&Store> {
        if path.is_empty() {
            return None;
        }

        if path.len() == 1 {
            if let Some(subdirs) = self.cache.subdirs.get(txn_id) {
                if let Some(store) = subdirs.get(&path[0]) {
                    return Some(store);
                }
            }

            self.subdirs.get(&path[0])
        } else if let Some(store) = self.subdirs.get(&path[0]) {
            store.get_store(txn_id, &path.slice_from(1))
        } else {
            None
        }
    }

    pub fn new_block(
        &mut self,
        txn_id: &TransactionId,
        block_id: PathSegment,
        initial_value: Bytes,
    ) -> TCResult<()> {
        println!("new block, initial size {}", initial_value.len());

        if self.blocks.contains_key(&block_id) {
            Err(error::internal(format!(
                "Attempted to truncate an existing block: {}",
                block_id
            )))
        } else if let Some(blocks) = self.cache.blocks.get_mut(txn_id) {
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
            let mut blocks = HashMap::new();
            blocks.insert(block_id, initial_value[..].into());
            self.cache.blocks.insert(txn_id.clone(), blocks);
            Ok(())
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
                self.cache
                    .subdirs
                    .get_mut(txn_id)
                    .unwrap()
                    .insert(path.clone(), subdir);
                Ok(self.cache.subdirs.get(txn_id).unwrap().get(path).unwrap())
            }
        } else {
            self.subdirs
                .get_mut(&path[0])
                .unwrap()
                .reserve(txn_id, path.slice_from(1))
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
