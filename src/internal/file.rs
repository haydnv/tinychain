use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use futures::{future, Future};

use crate::error;
use crate::transaction::{Transact, TxnId};
use crate::transaction::lock::{TxnLock, TxnLockReadGuard, TxnLockWriteGuard};
use crate::value::link::PathSegment;
use crate::value::TCResult;

pub type BlockId = PathSegment;

pub struct File {
    blocks: TxnLock<HashMap<BlockId, TxnLock<BytesMut>>>,
}

impl File {
    pub fn new() -> Arc<File> {
        Arc::new(File {
            blocks: TxnLock::new(HashMap::new()),
        })
    }

    pub async fn contains_block(&self, txn_id: &TxnId, block_id: &BlockId) -> bool {
        self.blocks.read(txn_id).await.contains_key(block_id)
    }

    pub async fn get_block(&self, txn_id: &TxnId, block_id: &BlockId) -> Option<TxnLockReadGuard<BytesMut>> {
        match self.blocks.read(txn_id).await.get(block_id) {
            Some(lock) => Some(lock.read(txn_id).await),
            None => None
        }
    }

    pub async fn get_block_mut(&self, txn_id: &TxnId, block_id: &BlockId) -> TCResult<Option<TxnLockWriteGuard<BytesMut>>> {
        match self.blocks.read(txn_id).await.get(block_id) {
            Some(lock) => Ok(Some(lock.write(txn_id).await?)),
            None => Ok(None)
        }
    }

    pub async fn block_ids(&self, txn_id: &TxnId) -> HashSet<BlockId> {
        self.blocks.read(txn_id).await.keys().cloned().into_iter().collect()
    }

    pub async fn is_empty(&self, txn_id: &TxnId) -> bool {
        self.blocks.read(txn_id).await.is_empty()
    }

    pub fn new_block(
        &self,
        txn_id: TxnId,
        block_id: BlockId,
        initial_value: Bytes,
    ) -> impl Future<Output = TCResult<()>> {
        future::ready(Err(error::not_implemented()))
    }
}

#[async_trait]
impl Transact for File {
    async fn commit(&self, txn_id: &TxnId) {
        // TODO
    }

    async fn rollback(&self, txn_id: &TxnId) {
        // TODO
    }
}

impl fmt::Display for File {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(file)")
    }
}
