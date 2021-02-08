use std::sync::Arc;

use async_trait::async_trait;
use futures_locks::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use error::*;
use generic::Id;
use transact::fs;
use transact::TxnId;

use super::cache::CacheFile;

#[derive(Clone)]
pub enum InstanceFile<B> {
    Persistent(RwLock<CacheFile<B>>),
    Temporary(RwLock<FileView<B>>),
}

impl<B: Clone> InstanceFile<B> {
    pub async fn version(&self, txn_id: &TxnId) -> TCResult<RwLock<FileView<B>>> {
        match self {
            Self::Persistent(_file) => Err(TCError::not_implemented("InstanceFile::version")),
            Self::Temporary(view) => {
                let lock = view.read().await;
                if &lock.inner.txn_id == txn_id {
                    Ok(view.clone())
                } else {
                    Err(TCError::forbidden(
                        format!("cannot access transaction {} from", &lock.inner.txn_id),
                        txn_id,
                    ))
                }
            }
        }
    }
}

impl<B> From<RwLock<FileView<B>>> for InstanceFile<B> {
    fn from(view: RwLock<FileView<B>>) -> InstanceFile<B> {
        InstanceFile::Temporary(view)
    }
}

struct Inner<B> {
    txn_id: TxnId,
    version: CacheFile<B>,
    source: RwLockReadGuard<CacheFile<B>>,
}

#[derive(Clone)]
pub struct FileView<B> {
    inner: Arc<Inner<B>>,
}

impl<B: fs::BlockData> FileView<B> {
    async fn lock_block(&self, _block_id: &fs::BlockId) -> TCResult<RwLockReadGuard<B>> {
        Err(TCError::not_implemented("transact::File::lock_block"))
    }

    async fn mutate(&self, _block_id: fs::BlockId) -> TCResult<RwLockWriteGuard<B>> {
        // TODO: create the new version file if it doesn't exist and copy the block to it
        Err(TCError::not_implemented("transact::File::mutate"))
    }
}

#[async_trait]
impl<B: fs::BlockData + 'static> fs::File for FileView<B> {
    type Block = B;

    async fn create_block(
        &mut self,
        _name: Id,
        _initial_value: Self::Block,
    ) -> TCResult<fs::BlockOwned<Self>> {
        Err(TCError::not_implemented("FileView::create_block"))
    }

    async fn get_block<'a>(&'a self, block_id: &'a fs::BlockId) -> TCResult<fs::Block<'a, Self>> {
        let lock = self.lock_block(&block_id).await?;
        let block = fs::Block::new(self, block_id, lock);
        Ok(block)
    }

    async fn get_block_mut<'a>(
        &'a self,
        block_id: &'a fs::BlockId,
    ) -> TCResult<fs::BlockMut<'a, Self>> {
        let lock = self.mutate(block_id.clone()).await?;
        Ok(fs::BlockMut::new(self, block_id, lock))
    }

    async fn get_block_owned(self, block_id: fs::BlockId) -> TCResult<fs::BlockOwned<Self>> {
        let lock = self.lock_block(&block_id).await?;
        Ok(fs::BlockOwned::new(self, block_id, lock))
    }

    async fn get_block_owned_mut(self, block_id: fs::BlockId) -> TCResult<fs::BlockOwnedMut<Self>> {
        let lock = self.mutate(block_id.clone()).await?;
        Ok(fs::BlockOwnedMut::new(self, block_id, lock))
    }
}
