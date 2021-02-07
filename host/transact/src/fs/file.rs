use std::collections::HashSet;
use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use futures_locks::*;
use uuid::Uuid;

use error::*;
use generic::{label, Id, Label, PathSegment};

use crate::{Transact, TxnId};

use super::hostfs;
use super::{Block, BlockData, BlockId, BlockMut, BlockOwned, BlockOwnedMut};

const TXN_CACHE: Label = label(".pending");

struct Inner<T: BlockData> {
    file: RwLock<hostfs::File>,
    pending: RwLock<hostfs::Dir>,
    phantom: PhantomData<T>,
}

#[derive(Clone)]
pub struct File<T: BlockData> {
    inner: Arc<Inner<T>>,
}

impl<T: BlockData> File<T> {
    pub async fn create(name: Id, cache: &mut hostfs::Dir) -> TCResult<File<T>> {
        let file = cache.create_file(name).await?;
        let pending = cache.create_dir(TXN_CACHE.into()).await?;

        let inner = Inner {
            file,
            pending,
            phantom: PhantomData,
        };

        Ok(File {
            inner: Arc::new(inner),
        })
    }

    pub async fn unique_id(&self, txn_id: &TxnId) -> TCResult<BlockId> {
        let existing_ids = self.block_ids(txn_id).await?;
        loop {
            let id: PathSegment = Uuid::new_v4().to_string().parse()?;
            if !existing_ids.contains(&id) {
                return Ok(id);
            }
        }
    }

    async fn block_ids(&'_ self, txn_id: &'_ TxnId) -> TCResult<HashSet<BlockId>> {
        let pending = self.inner.pending.read().await;
        let file = if let Some(file) = pending.get_file(&txn_id.to_id())? {
            file.read().await
        } else {
            self.inner.file.read().await
        };

        Ok(file.block_ids().clone())
    }

    pub async fn create_block(
        self,
        txn_id: TxnId,
        block_id: BlockId,
        data: T,
    ) -> TCResult<BlockOwned<T>> {
        if &block_id == &TXN_CACHE {
            return Err(TCError::bad_request("This name is reserved", block_id));
        }

        let mut pending = self.inner.pending.write().await;
        let version = pending.create_or_get_file(&txn_id.to_id()).await?;
        let mut version = version.write().await;
        version.create_block(block_id.clone(), data.into()).await?;

        Err(TCError::not_implemented("transact::File::create_block"))
    }

    pub async fn get_block<'a>(
        &'a self,
        txn_id: &'a TxnId,
        block_id: BlockId,
    ) -> TCResult<Block<'a, T>> {
        let lock = self.lock_block(txn_id, &block_id).await?;
        let block = Block::new(self, txn_id, block_id, lock);
        Ok(block)
    }

    pub async fn get_block_mut<'a>(
        &'a self,
        txn_id: &'a TxnId,
        block_id: BlockId,
    ) -> TCResult<BlockMut<'a, T>> {
        let lock = self.mutate(txn_id, block_id.clone()).await?;
        Ok(BlockMut::new(self, txn_id, block_id, lock))
    }

    pub async fn get_block_owned(
        self,
        txn_id: TxnId,
        block_id: BlockId,
    ) -> TCResult<BlockOwned<T>> {
        let lock = self.lock_block(&txn_id, &block_id).await?;
        Ok(BlockOwned::new(self, txn_id, block_id, lock))
    }

    pub async fn get_block_owned_mut(
        self,
        txn_id: TxnId,
        block_id: BlockId,
    ) -> TCResult<BlockOwnedMut<T>> {
        let lock = self.mutate(&txn_id, block_id.clone()).await?;
        Ok(BlockOwnedMut::new(self, txn_id, block_id, lock))
    }

    async fn lock_block(
        &self,
        _txn_id: &TxnId,
        _block_id: &BlockId,
    ) -> TCResult<RwLockReadGuard<T>> {
        Err(TCError::not_implemented("transact::File::lock_block"))
    }

    async fn mutate(&self, _txn_id: &TxnId, _block_id: BlockId) -> TCResult<RwLockWriteGuard<T>> {
        // TODO: create the new version file if it doesn't exist and copy the block to it
        Err(TCError::not_implemented("transact::File::mutate"))
    }

    pub async fn is_empty(&self, txn_id: &TxnId) -> TCResult<bool> {
        let pending = self.inner.pending.read().await;
        let file = if let Some(file) = pending.get_file(&txn_id.to_id())? {
            file.read().await
        } else {
            self.inner.file.read().await
        };

        Ok(file.is_empty())
    }
}

#[async_trait]
impl<T: BlockData> Transact for File<T> {
    async fn commit(&self, _txn_id: &TxnId) {
        todo!()
    }

    async fn finalize(&self, txn_id: &TxnId) {
        let mut pending = self.inner.pending.write().await;
        if pending.contents().contains_key(&txn_id.to_id()) {
            pending.delete(&txn_id.to_id()).await.unwrap();
        }
    }
}
