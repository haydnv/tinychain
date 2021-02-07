use std::collections::HashMap;
use std::sync::Arc;

use futures_locks::{RwLockReadGuard, RwLockWriteGuard};

use error::*;
use transact::TxnId;

use super::block::*;
use super::cache::CacheFile;

struct Inner<B: BlockData> {
    file: CacheFile<B>,
    versions: HashMap<TxnId, CacheFile<B>>,
}

#[derive(Clone)]
pub struct File<B: BlockData> {
    inner: Arc<Inner<B>>,
}

impl<B: BlockData> File<B> {
    pub async fn get_block<'a>(
        &'a self,
        txn_id: &'a TxnId,
        block_id: BlockId,
    ) -> TCResult<Block<'a, B>> {
        let lock = self.lock_block(txn_id, &block_id).await?;
        let block = Block::new(self, txn_id, block_id, lock);
        Ok(block)
    }

    pub async fn get_block_mut<'a>(
        &'a self,
        txn_id: &'a TxnId,
        block_id: BlockId,
    ) -> TCResult<BlockMut<'a, B>> {
        let lock = self.mutate(txn_id, block_id.clone()).await?;
        Ok(BlockMut::new(self, txn_id, block_id, lock))
    }

    pub async fn get_block_owned(
        self,
        txn_id: TxnId,
        block_id: BlockId,
    ) -> TCResult<BlockOwned<B>> {
        let lock = self.lock_block(&txn_id, &block_id).await?;
        Ok(BlockOwned::new(self, txn_id, block_id, lock))
    }

    pub async fn get_block_owned_mut(
        self,
        txn_id: TxnId,
        block_id: BlockId,
    ) -> TCResult<BlockOwnedMut<B>> {
        let lock = self.mutate(&txn_id, block_id.clone()).await?;
        Ok(BlockOwnedMut::new(self, txn_id, block_id, lock))
    }

    async fn lock_block(
        &self,
        _txn_id: &TxnId,
        _block_id: &BlockId,
    ) -> TCResult<RwLockReadGuard<B>> {
        Err(TCError::not_implemented("transact::File::lock_block"))
    }

    async fn mutate(&self, _txn_id: &TxnId, _block_id: BlockId) -> TCResult<RwLockWriteGuard<B>> {
        // TODO: create the new version file if it doesn't exist and copy the block to it
        Err(TCError::not_implemented("transact::File::mutate"))
    }
}
