use std::collections::hash_map::{Entry, HashMap};
use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::executor::block_on;
use futures::future::{join_all, FutureExt};
use futures::join;
use futures::lock::Mutex;

use crate::error;
use crate::internal::cache;
use crate::internal::lock::RwLock;
use crate::transaction::lock::{Mutate, TxnLock, TxnLockReadGuard, TxnLockWriteGuard};
use crate::transaction::{Transact, TxnId};
use crate::value::link::PathSegment;
use crate::value::TCResult;

const TXN_CACHE: &str = ".txn_cache";

pub type BlockId = PathSegment;

pub struct Block {
    file: Arc<File>,
    cached: RwLock<cache::Block>,
}

impl Block {
    pub async fn as_bytes(&self) -> Bytes {
        Bytes::copy_from_slice(self.cached.read().await.as_bytes().await)
    }

    pub async fn rewrite(&self, new_contents: Bytes) {
        self.cached.write().await.rewrite(new_contents).await
    }
}

#[async_trait]
impl Mutate for Block {
    fn diverge(&self, txn_id: &TxnId) -> Self {
        let cached = block_on(self.cached.read());
        block_on(
            self.file
                .clone()
                .version(cached.name().clone(), txn_id.clone()),
        )
    }

    async fn converge(&mut self, other: Block) {
        let (mut this, that) = join!(self.cached.write(), other.cached.read());
        this.copy_from(&that).await;
    }
}

#[derive(Clone)]
struct FileContents(HashMap<BlockId, TxnLock<Block>>);

#[async_trait]
impl<'a> Mutate for FileContents {
    fn diverge(&self, _txn_id: &TxnId) -> Self {
        self.clone()
    }

    async fn converge(&mut self, mut new_value: FileContents) {
        let existing: HashSet<BlockId> = self.0.keys().cloned().collect();
        let new: HashSet<BlockId> = new_value.0.keys().cloned().collect();
        let deleted = existing.difference(&new);

        self.0.extend(new_value.0.drain());

        for name in deleted {
            self.0.remove(name);
        }
    }
}

pub struct File {
    cache: RwLock<cache::Dir>,
    txn_cache: RwLock<cache::Dir>,
    contents: TxnLock<FileContents>,
    mutated: Mutex<HashMap<TxnId, HashSet<BlockId>>>,
}

impl File {
    async fn version(self: Arc<Self>, name: BlockId, txn_id: TxnId) -> Block {
        let cache = self.cache.read();
        let txn_cache = self
            .txn_cache
            .write()
            .then(|mut lock| lock.create_or_get_dir(&txn_id.into()).unwrap().write());
        let (cache, mut txn_cache) = join!(cache, txn_cache);

        let block_to_cache = cache.get_block(&name).unwrap().unwrap();
        let cached_block = txn_cache.create_block(name, Bytes::new()).unwrap();
        let (block_to_cache_reader, mut cached_block_writer) =
            join!(block_to_cache.read(), cached_block.write());
        cached_block_writer.copy_from(&*block_to_cache_reader).await;

        Block {
            file: self,
            cached: cached_block,
        }
    }

    pub async fn create(txn_id: TxnId, cache: RwLock<cache::Dir>) -> TCResult<Arc<File>> {
        let mut cache_lock = cache.write().await;
        if !cache_lock.is_empty() {
            return Err(error::bad_request(
                "Tried to create a new File but there is already data in the cache!",
                "(filesystem cache)",
            ));
        }

        let txn_cache = cache_lock.create_dir(TXN_CACHE.parse()?)?;

        Ok(Arc::new(File {
            cache,
            txn_cache,
            contents: TxnLock::new(txn_id, FileContents(HashMap::new())),
            mutated: Mutex::new(HashMap::new()),
        }))
    }

    pub async fn block_ids(&self, txn_id: &TxnId) -> TCResult<HashSet<BlockId>> {
        Ok(self
            .contents
            .read(txn_id)
            .await?
            .0
            .keys()
            .cloned()
            .collect())
    }

    pub async fn contains_block(&self, txn_id: &TxnId, block_id: &BlockId) -> TCResult<bool> {
        Ok(self.contents.read(txn_id).await?.0.contains_key(block_id))
    }

    pub async fn create_block(
        self: Arc<Self>,
        txn_id: TxnId,
        block_id: BlockId,
        initial_value: Bytes,
    ) -> TCResult<()> {
        if block_id.to_string() == TXN_CACHE {
            return Err(error::bad_request("This name is reserved", block_id));
        }

        let contents = &mut self.contents.write(txn_id.clone()).await?.0;
        match contents.entry(block_id) {
            Entry::Occupied(entry) => Err(error::bad_request(
                "This file already has a block at",
                entry.key(),
            )),
            Entry::Vacant(entry) => {
                let block = self
                    .cache
                    .write()
                    .await
                    .create_block(entry.key().clone(), initial_value)?;
                let block = Block {
                    file: self,
                    cached: block,
                };
                entry.insert(TxnLock::new(txn_id, block));
                Ok(())
            }
        }
    }

    pub async fn get_block(
        &self,
        txn_id: &TxnId,
        block_id: &BlockId,
    ) -> TCResult<Option<TxnLockReadGuard<Block>>> {
        let contents = &self.contents.read(txn_id).await?.0;
        match contents.get(block_id) {
            Some(block) => Ok(Some(block.read(txn_id).await?)),
            None => Ok(None),
        }
    }

    pub async fn get_block_mut(
        &self,
        txn_id: TxnId,
        block_id: BlockId,
    ) -> TCResult<Option<TxnLockWriteGuard<Block>>> {
        let contents = &self.contents.read(&txn_id).await?.0;
        match contents.get(&block_id) {
            Some(block) => {
                self.mutated
                    .lock()
                    .await
                    .entry(txn_id.clone())
                    .or_insert_with(HashSet::new)
                    .insert(block_id);
                Ok(Some(block.write(txn_id).await?))
            }
            None => Ok(None),
        }
    }

    pub async fn is_empty(&self, txn_id: &TxnId) -> TCResult<bool> {
        Ok(self.contents.read(txn_id).await?.0.is_empty())
    }
}

#[async_trait]
impl Transact for File {
    async fn commit(&self, txn_id: &TxnId) {
        let contents = &self.contents.read(txn_id).await.unwrap().0;
        if let Some(mut mutated) = self.mutated.lock().await.remove(txn_id) {
            let mut commits = Vec::with_capacity(mutated.len());
            for block_id in mutated.drain() {
                commits.push(contents.get(&block_id).unwrap().commit(txn_id));
            }

            join_all(commits).await;
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        let contents = &self.contents.read(txn_id).await.unwrap().0;
        if let Some(mut mutated) = self.mutated.lock().await.remove(txn_id) {
            let mut rollbacks = Vec::with_capacity(mutated.len());
            for block_id in mutated.drain() {
                rollbacks.push(contents.get(&block_id).unwrap().rollback(txn_id));
            }

            join_all(rollbacks).await;
        }
    }
}

impl fmt::Display for File {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(file)")
    }
}
