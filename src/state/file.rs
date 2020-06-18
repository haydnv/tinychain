use std::collections::hash_map::HashMap;
use std::collections::HashSet;

use async_trait::async_trait;
use futures::executor::block_on;
use futures::future::FutureExt;
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
    cached: RwLock<cache::Block>,
}

#[async_trait]
impl<'a> Mutate for Block<'a> {
    fn diverge(&self, txn_id: &TxnId) -> Self {
        let cached = block_on(self.cached.read());
        block_on(self.file.version(cached.name().clone(), txn_id.clone()))
    }

    async fn converge(&mut self, other: Block<'a>) {
        let (mut this, mut that) = join!(self.cached.write(), other.cached.write());
        this.copy_from(&mut *that).await;
    }
}

#[derive(Clone)]
struct FileContents<'a>(HashMap<BlockId, TxnLock<Block<'a>>>);

#[async_trait]
impl<'a> Mutate for FileContents<'a> {
    fn diverge(&self, _txn_id: &TxnId) -> Self {
        self.clone()
    }

    async fn converge(&mut self, mut new_value: FileContents<'a>) {
        let existing: HashSet<BlockId> = self.0.keys().cloned().collect();
        let new: HashSet<BlockId> = new_value.0.keys().cloned().collect();
        let deleted = existing.difference(&new);

        self.0.extend(new_value.0.drain());

        for name in deleted {
            self.0.remove(name);
        }
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

    async fn version(&'a self, name: BlockId, txn_id: TxnId) -> Block<'a> {
        let cache = self.cache.read();
        let txn_cache = self
            .txn_cache
            .write()
            .then(|mut lock| lock.create_or_get_dir(&txn_id.into()).unwrap().write());
        let (cache, mut txn_cache) = join!(cache, txn_cache);

        let block_to_cache = cache.get_block(&name).unwrap().unwrap();
        let cached_block = txn_cache.create_block(name).unwrap();
        let (block_to_cache_reader, mut cached_block_writer) =
            join!(block_to_cache.read(), cached_block.write());
        cached_block_writer.copy_from(&*block_to_cache_reader);
        Block {
            file: self,
            cached: cached_block,
        }
    }
}
