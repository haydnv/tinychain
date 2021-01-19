use std::collections::HashMap;

use async_trait::async_trait;
use futures::future::join_all;

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

#[async_trait]
impl<T: BlockData> Transact for Cache<T> {
    async fn commit(&self, txn_id: &TxnId) {
        join_all(self.blocks.values().map(|lock| lock.commit(txn_id))).await;
    }

    async fn rollback(&self, txn_id: &TxnId) {
        join_all(self.blocks.values().map(|lock| lock.rollback(txn_id))).await;
    }

    async fn finalize(&self, txn_id: &TxnId) {
        join_all(self.blocks.values().map(|lock| lock.finalize(txn_id))).await;
    }
}
