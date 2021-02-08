use std::collections::HashSet;
use std::marker::PhantomData;
use std::path::PathBuf;

use async_trait::async_trait;
use futures_locks::RwLockReadGuard;

use error::*;
use generic::{label, Id, Label};
use transact::fs;
use transact::lock::{Mutable, TxnLock};
use transact::TxnId;

use super::{file_name, Cache, CacheBlock, CacheLock, DirContents};

const VERSION: Label = label(".version");

#[derive(Clone)]
pub struct File<B> {
    cache: Cache,
    path: PathBuf,
    listing: TxnLock<Mutable<HashSet<fs::BlockId>>>,
    phantom: PhantomData<B>,
}

impl<B> File<B> {
    fn _new(cache: Cache, path: PathBuf, listing: HashSet<fs::BlockId>) -> Self {
        let listing = TxnLock::new(format!("file listing at {:?}", &path), listing.into());
        let phantom = PhantomData;

        Self {
            cache,
            path,
            listing,
            phantom,
        }
    }

    pub fn new(cache: Cache, path: PathBuf) -> Self {
        Self::_new(cache, path, HashSet::new())
    }

    pub async fn load(cache: Cache, path: PathBuf, contents: DirContents) -> TCResult<Self> {
        if contents.iter().all(|(_, meta)| meta.is_file()) {
            let listing = contents
                .into_iter()
                .map(|(handle, _)| file_name(&handle))
                .collect::<TCResult<HashSet<fs::BlockId>>>()?;

            Ok(Self::_new(cache, path, listing))
        } else {
            Err(TCError::internal(format!(
                "directory at {:?} contains both blocks and subdirectories",
                path
            )))
        }
    }

    async fn lock_block(
        &self,
        _txn_id: &TxnId,
        _block_id: &fs::BlockId,
    ) -> TCResult<Option<RwLockReadGuard<B>>> {
        Err(TCError::not_implemented("File::lock_block"))
    }
}

#[async_trait]
impl<B: fs::BlockData + 'static> fs::File for File<B>
where
    CacheBlock: From<CacheLock<B>>,
{
    type Block = B;

    async fn create_block(
        &mut self,
        name: fs::BlockId,
        txn_id: TxnId,
        initial_value: Self::Block,
    ) -> TCResult<fs::BlockOwned<Self>> {
        let path = block_path(&self.path, &txn_id, &name);
        let lock = self.cache.write(path, initial_value).await?;
        let read_lock = lock.read().await;
        Ok(fs::BlockOwned::new(self.clone(), txn_id, name, read_lock))
    }

    async fn get_block<'a>(
        &'a self,
        _txn_id: &'a TxnId,
        _name: &'a fs::BlockId,
    ) -> TCResult<fs::Block<'a, Self>> {
        unimplemented!()
    }

    async fn get_block_mut<'a>(
        &'a self,
        _txn_id: &'a TxnId,
        _name: &'a fs::BlockId,
    ) -> TCResult<fs::BlockMut<'a, Self>> {
        unimplemented!()
    }

    async fn get_block_owned(self, _txn_id: TxnId, _name: Id) -> TCResult<fs::BlockOwned<Self>> {
        unimplemented!()
    }

    async fn get_block_owned_mut(
        self,
        _txn_id: TxnId,
        _name: fs::BlockId,
    ) -> TCResult<fs::BlockOwnedMut<Self>> {
        unimplemented!()
    }
}

fn block_path(file_path: &PathBuf, txn_id: &TxnId, block_id: &fs::BlockId) -> PathBuf {
    let mut path = file_path.clone();
    path.push(VERSION.to_string());
    path.push(txn_id.to_string());
    path.push(block_id.to_string());
    path
}
