use std::collections::HashSet;
use std::marker::PhantomData;
use std::path::PathBuf;

use async_trait::async_trait;
use futures_locks::{RwLockReadGuard, RwLockWriteGuard};

use error::*;
use generic::Id;
use transact::fs;
use transact::lock::{Mutable, TxnLock};
use transact::TxnId;

use super::{file_name, Cache, DirContents};

#[derive(Clone)]
pub struct File<B> {
    cache: Cache,
    path: PathBuf,
    listing: TxnLock<Mutable<HashSet<fs::BlockId>>>,
    mutated: TxnLock<Mutable<HashSet<fs::BlockId>>>,
    phantom: PhantomData<B>,
}

impl<B> File<B> {
    pub async fn load(cache: Cache, path: PathBuf, contents: DirContents) -> TCResult<Self> {
        if contents.iter().all(|(_, meta)| meta.is_file()) {
            let listing = contents
                .into_iter()
                .map(|(handle, _)| file_name(&handle))
                .collect::<TCResult<HashSet<fs::BlockId>>>()?;
            let listing = TxnLock::new(format!("file listing at {:?}", &path), listing.into());
            let mutated = TxnLock::new(
                format!("mutation listing at {:?}", &path),
                HashSet::new().into(),
            );
            let phantom = PhantomData;

            Ok(Self {
                cache,
                path,
                listing,
                mutated,
                phantom,
            })
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

    async fn mutate_block(
        &self,
        _txn_id: &TxnId,
        _block_id: &fs::BlockId,
    ) -> TCResult<Option<RwLockWriteGuard<B>>> {
        Err(TCError::not_implemented("File::mutate_block"))
    }
}

#[async_trait]
impl<B: fs::BlockData> fs::File for File<B> {
    type Block = B;

    async fn create_block(
        &mut self,
        _name: Id,
        _initial_value: Self::Block,
    ) -> TCResult<fs::BlockOwned<Self>> {
        unimplemented!()
    }

    async fn get_block<'a>(
        &'a self,
        _txn_id: &'a TxnId,
        _name: &'a Id,
    ) -> TCResult<fs::Block<'a, Self>> {
        unimplemented!()
    }

    async fn get_block_mut<'a>(
        &'a self,
        _txn_id: &'a TxnId,
        _name: &'a Id,
    ) -> TCResult<fs::BlockMut<'a, Self>> {
        unimplemented!()
    }

    async fn get_block_owned(self, _txn_id: TxnId, _name: Id) -> TCResult<fs::BlockOwned<Self>> {
        unimplemented!()
    }

    async fn get_block_owned_mut(
        self,
        _txn_id: TxnId,
        _name: Id,
    ) -> TCResult<fs::BlockOwnedMut<Self>> {
        unimplemented!()
    }
}
