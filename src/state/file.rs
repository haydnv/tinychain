use std::collections::hash_map::HashMap;

use async_trait::async_trait;
use futures::join;

use crate::internal::cache;
use crate::internal::lock::RwLock;
use crate::transaction::lock::{Mutate, TxnLock};
use crate::transaction::TxnId;
use crate::value::link::PathSegment;
use crate::value::TCResult;

const TXN_CACHE: &str = ".txn_cache";

pub type BlockId = PathSegment;

pub struct Block<'a> {
    file: &'a File<'a>,
    name: BlockId,
    cached: RwLock<cache::Block>,
}

#[async_trait]
impl<'a> Mutate for Block<'a> {
    fn diverge(&self, txn_id: &TxnId) -> Self {
        self.file.version(self.name.clone(), txn_id)
    }

    async fn converge(&mut self, other: Block<'a>) {
        let (mut this, mut that) = join!(self.cached.write(), other.cached.write());
        this.swap(&mut *that).await;
    }
}

#[derive(Clone)]
struct FileContents<'a>(HashMap<BlockId, TxnLock<Block<'a>>>);

#[async_trait]
impl<'a> Mutate for FileContents<'a> {
    fn diverge(&self, _txn_id: &TxnId) -> Self {
        self.clone()
    }

    async fn converge(&mut self, _other: FileContents<'a>) {
        // TODO
    }
}

pub struct File<'a> {
    cache: RwLock<cache::Dir>,
    txn_cache: RwLock<cache::Dir>,
    contents: TxnLock<FileContents<'a>>,
}

impl<'a> File<'a> {
    pub async fn new(txn_id: TxnId, cache: RwLock<cache::Dir>) -> TCResult<File<'a>> {
        let txn_cache = cache.write().await.create_dir(TXN_CACHE.parse()?)?;

        Ok(File {
            cache,
            txn_cache,
            contents: TxnLock::new(txn_id, FileContents(HashMap::new())),
        })
    }

    fn version(&'a self, _name: BlockId, _txn_id: &TxnId) -> Block<'a> {
        // TODO
        panic!("NOT IMPLEMENTED")
    }
}
