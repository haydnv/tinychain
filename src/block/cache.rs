use std::collections::HashMap;

use futures::future::join_all;

use crate::class::TCBoxFuture;
use crate::transaction::lock::TxnLock;
use crate::transaction::{Transact, TxnId};

use super::{BlockData, BlockId};

pub struct Cache<T: BlockData> {
    blocks: HashMap<BlockId, TxnLock<T>>,
}

impl<T: BlockData> Cache<T> {
    pub fn new() -> Cache<T> {
        Cache {
            blocks: HashMap::new(),
        }
    }

    pub fn block_ids(&self) -> impl Iterator<Item = &BlockId> {
        self.blocks.keys()
    }

    pub fn get(&self, block_id: &BlockId) -> Option<TxnLock<T>> {
        self.blocks.get(block_id).cloned()
    }

    pub fn insert(&mut self, block_id: BlockId, block: T) -> TxnLock<T> {
        let lock = TxnLock::new(format!("Block {}", &block_id), block);
        self.blocks.insert(block_id, lock.clone());
        lock
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }
}

impl<T: BlockData> Transact for Cache<T> {
    fn commit<'a>(&'a self, txn_id: &'a TxnId) -> TCBoxFuture<'a, ()> {
        Box::pin(async move {
            join_all(self.blocks.values().map(|lock| lock.commit(txn_id))).await;
        })
    }

    fn rollback<'a>(&'a self, txn_id: &'a TxnId) -> TCBoxFuture<'a, ()> {
        Box::pin(async move {
            join_all(self.blocks.values().map(|lock| lock.rollback(txn_id))).await;
        })
    }
}
