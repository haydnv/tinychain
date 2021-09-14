//! A transactional file

use std::collections::hash_map::{Entry, HashMap};
use std::collections::HashSet;
use std::convert::TryFrom;
use std::fmt;
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use destream::{de, en};
use futures::future::{try_join_all, TryFutureExt};
use futures::stream::{FuturesUnordered, StreamExt};
use futures::{try_join, TryStreamExt};
use log::debug;
use tokio::sync::{OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};
use uuid::Uuid;

use tc_error::*;
use tc_transact::fs::{self, BlockData, BlockId, Store};
use tc_transact::lock::TxnLock;
use tc_transact::{Transact, TxnId};

use super::cache::*;
use super::{file_name, fs_path, DirContents, TMP};

#[derive(Clone)]
pub struct Block<B> {
    name: BlockId,
    txn_id: TxnId,
    lock: CacheLock<B>,
}

#[async_trait]
impl<'en, B: BlockData + en::IntoStream<'en> + 'en> fs::Block<B, File<B>> for Block<B>
where
    CacheBlock: From<CacheLock<B>>,
    CacheLock<B>: TryFrom<CacheBlock, Error = TCError>,
{
    type ReadLock = OwnedRwLockReadGuard<B>;
    type WriteLock = OwnedRwLockWriteGuard<B>;

    async fn read(self) -> Self::ReadLock {
        self.lock.read().await
    }

    async fn write(self) -> Self::WriteLock {
        self.lock.write().await
    }
}

/// A transactional file
#[derive(Clone)]
pub struct File<B> {
    path: PathBuf,
    cache: Cache,
    contents: TxnLock<HashSet<BlockId>>,
    mutated: Arc<RwLock<HashMap<TxnId, HashSet<BlockId>>>>,
    phantom: PhantomData<B>,
}

impl<B: BlockData> File<B> {
    fn _new(cache: Cache, path: PathBuf, block_ids: HashSet<BlockId>) -> Self {
        let txn_lock_name = format!("file contents at {:?}", &path);

        File {
            path,
            cache,
            contents: TxnLock::new(txn_lock_name, block_ids),
            mutated: Arc::new(RwLock::new(HashMap::new())),
            phantom: PhantomData,
        }
    }

    async fn mutate(&self, txn_id: TxnId, block_id: BlockId) {
        let mut mutated = self.mutated.write().await;
        match mutated.entry(txn_id) {
            Entry::Vacant(entry) => entry.insert(HashSet::new()).insert(block_id),
            Entry::Occupied(mut entry) => entry.get_mut().insert(block_id),
        };
    }

    pub fn new(cache: Cache, mut path: PathBuf, ext: &str) -> Self {
        path.set_extension(ext);
        Self::_new(cache, path, HashSet::new())
    }

    pub fn load(cache: Cache, path: PathBuf, contents: DirContents) -> TCResult<Self> {
        if contents.iter().all(|(_, meta)| meta.is_file()) {
            let contents = contents
                .into_iter()
                .filter(|(handle, _)| {
                    handle.path().extension().and_then(|e| e.to_str()) != Some(TMP)
                })
                .map(|(handle, _)| file_name(&handle))
                .collect::<TCResult<HashSet<BlockId>>>()?;

            Ok(Self::_new(cache, path, contents))
        } else {
            Err(TCError::internal(format!(
                "directory at {:?} contains both blocks and subdirectories",
                path
            )))
        }
    }

    pub async fn sync_block<'en>(
        &self,
        txn_id: TxnId,
        name: BlockId,
    ) -> TCResult<OwnedRwLockWriteGuard<B>>
    where
        B: en::IntoStream<'en> + 'en,
        CacheBlock: From<CacheLock<B>>,
        CacheLock<B>: TryFrom<CacheBlock, Error = TCError>,
    {
        debug!("File::sync_block");
        self.cache
            .sync(&block_version(&self.path, &txn_id, &name))
            .await?;

        fs::File::write_block(self, txn_id, name).await
    }
}

#[async_trait]
impl<B: BlockData> Store for File<B> {
    async fn is_empty(&self, txn_id: TxnId) -> TCResult<bool> {
        self.contents
            .read(txn_id)
            .map_ok(|contents| contents.is_empty())
            .await
    }
}

#[async_trait]
impl<'en, B: BlockData + en::IntoStream<'en> + 'en> fs::File<B> for File<B>
where
    CacheBlock: From<CacheLock<B>>,
    CacheLock<B>: TryFrom<CacheBlock, Error = TCError>,
{
    type Block = Block<B>;

    async fn block_ids(&self, txn_id: TxnId) -> TCResult<HashSet<BlockId>> {
        let contents = self.contents.read(txn_id).await?;
        Ok((*contents).clone())
    }

    async fn unique_id(&self, txn_id: TxnId) -> TCResult<BlockId> {
        let contents = self.contents.read(txn_id).await?;
        let id = loop {
            let id: BlockId = Uuid::new_v4().into();
            if !contents.contains(&id) {
                break id;
            }
        };

        Ok(id)
    }

    async fn contains_block(&self, txn_id: TxnId, name: &BlockId) -> TCResult<bool> {
        self.contents
            .read(txn_id)
            .map_ok(|contents| contents.contains(name))
            .await
    }

    async fn copy_from(&self, other: &Self, txn_id: TxnId) -> TCResult<()> {
        let (new_block_ids, mut contents) =
            try_join!(other.contents.read(txn_id), self.contents.write(txn_id))?;

        let mut copied_block_ids = HashSet::with_capacity(new_block_ids.len());

        let mut block_copies =
            FuturesUnordered::from_iter(new_block_ids.iter().cloned().map(|block_id| {
                let path = block_version(&self.path, &txn_id, &block_id);

                other
                    .read_block(txn_id, block_id.clone())
                    .and_then(|source| self.cache.write(path, source.clone()))
                    .map_ok(|_lock| block_id)
            }));

        while let Some(block_id) = block_copies.try_next().await? {
            contents.insert(block_id.clone());
            copied_block_ids.insert(block_id);
        }

        let mut mutated = self.mutated.write().await;
        match mutated.entry(txn_id) {
            Entry::Vacant(entry) => {
                entry.insert(copied_block_ids);
            }
            Entry::Occupied(mut entry) => entry.get_mut().extend(copied_block_ids),
        };

        Ok(())
    }

    async fn create_block(
        &self,
        txn_id: TxnId,
        name: BlockId,
        initial_value: B,
    ) -> TCResult<Self::Block> {
        let mut contents = self.contents.write(txn_id).await?;
        if contents.contains(&name) {
            return Err(TCError::bad_request(
                "there is already a block with this ID",
                name,
            ));
        }

        let version = block_version(&self.path, &txn_id, &name);
        contents.insert(name.clone());

        self.mutate(txn_id, name.clone()).await;
        self.cache
            .write(version, initial_value)
            .map_ok(|lock| Block { name, txn_id, lock })
            .await
    }

    async fn delete_block(&self, txn_id: TxnId, name: BlockId) -> TCResult<()> {
        let mut contents = self.contents.write(txn_id).await?;
        if !contents.remove(&name) {
            return Err(TCError::not_found(format!("block named {}", name)));
        }

        let version = block_version(&self.path, &txn_id, &name);
        self.mutate(txn_id, name).await;
        self.cache.delete(&version).await;
        Ok(())
    }

    async fn get_block(&self, txn_id: TxnId, name: BlockId) -> TCResult<Block<B>> {
        debug!("File::get_block {}", name);

        {
            let contents = self.contents.read(txn_id).await?;
            if !contents.contains(&name) {
                return Err(TCError::not_found(format!("block named {}", name)));
            }
        }

        let version = block_version(&self.path, &txn_id, &name);
        if let Some(lock) = self.cache.read(&version).await? {
            Ok(Block { name, txn_id, lock })
        } else {
            let canon = fs_path(&self.path, &name);
            let block = self.cache.read(&canon).await?;
            let block = block.ok_or_else(|| TCError::internal("failed reading block"))?;
            let data = block.read().await;

            self.cache
                .write(version, data.deref().clone())
                .map_ok(|lock| Block { name, txn_id, lock })
                .await
        }
    }

    async fn read_block(&self, txn_id: TxnId, name: BlockId) -> TCResult<OwnedRwLockReadGuard<B>> {
        debug!("File::read_block {}", name);

        let block = self.get_block(txn_id, name).await?;
        Ok(fs::Block::read(block).await)
    }

    async fn read_block_owned(
        self,
        txn_id: TxnId,
        name: BlockId,
    ) -> TCResult<OwnedRwLockReadGuard<B>> {
        debug!("File::read_block_owned {}", name);

        let block = self.get_block(txn_id, name).await?;
        Ok(fs::Block::read(block).await)
    }

    async fn write_block(
        &self,
        txn_id: TxnId,
        name: BlockId,
    ) -> TCResult<OwnedRwLockWriteGuard<B>> {
        debug!("File::write_block");
        let block = self.get_block(txn_id, name.clone()).await?;
        self.mutate(txn_id, name).await;
        Ok(fs::Block::write(block).await)
    }

    async fn truncate(&self, txn_id: TxnId) -> TCResult<()> {
        let mut contents = self.contents.write(txn_id).await?;
        let deletes = FuturesUnordered::from_iter(contents.drain().map(|block_id| async move {
            let path = block_version(&self.path, &txn_id, &block_id);
            self.cache.delete(&path).await;
            block_id
        }));

        let mut mutated = self.mutated.write().await;
        mutated.insert(txn_id, deletes.collect().await);
        Ok(())
    }
}

#[async_trait]
impl<'en, B: BlockData + 'en> Transact for File<B>
where
    B: en::IntoStream<'en>,
    CacheBlock: From<CacheLock<B>>,
    CacheLock<B>: TryFrom<CacheBlock, Error = TCError>,
{
    async fn commit(&self, txn_id: &TxnId) {
        debug!("commit file {:?} at {}", &self.path, txn_id);

        let file_path = &self.path;
        let cache = &self.cache;
        {
            let contents = self.contents.read(*txn_id).await.expect("file block list");
            let mutated = self.mutated.read().await;
            if let Some(blocks) = mutated.get(&txn_id) {
                let commits = blocks
                    .iter()
                    .filter(|block_id| contents.contains(block_id))
                    .map(|block_id| {
                        let version_path = block_version(file_path, &txn_id, block_id);
                        let block_path = fs_path(file_path, block_id);
                        cache.sync_and_copy(version_path, block_path)
                    });

                try_join_all(commits).await.expect("commit file blocks");
            } else {
                debug!("no blocks mutated at {}", txn_id);
            }
        }

        self.contents.commit(&txn_id).await;
        debug!("committed {:?} at {}", &self.path, txn_id);
    }

    async fn finalize(&self, txn_id: &TxnId) {
        debug!("finalize file {:?} at {}", &self.path, txn_id);

        let file_path = &self.path;
        let cache = &self.cache;
        {
            let contents = self.contents.read(*txn_id).await.expect("file block list");
            let mut mutated = self.mutated.write().await;
            if let Some(blocks) = mutated.remove(txn_id) {
                let deletes = blocks
                    .iter()
                    .filter(|block_id| !contents.contains(block_id))
                    .map(|block_id| {
                        let block_path = fs_path(file_path, block_id);
                        cache.delete_and_sync(block_path)
                    });

                try_join_all(deletes).await.expect("delete file blocks");
            }
        }

        let version = file_version(&self.path, txn_id);
        if version.exists() {
            cache
                .delete_dir(version)
                .await
                .expect("delete file version");
        }

        self.contents.finalize(txn_id).await;
        debug!("finalized {:?} at {}", &self.path, txn_id);
    }
}

impl<B> Eq for File<B> {}

impl<B> PartialEq for File<B> {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
    }
}

impl<B> fmt::Display for File<B> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "file at {:?}", &self.path)
    }
}

#[async_trait]
impl<'en, B: BlockData + de::FromStream<Context = ()> + 'en> de::FromStream for File<B>
where
    B: en::IntoStream<'en>,
    CacheBlock: From<CacheLock<B>>,
    CacheLock<B>: TryFrom<CacheBlock, Error = TCError>,
{
    type Context = (TxnId, File<B>);

    async fn from_stream<D: de::Decoder>(
        cxt: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        let visitor = FileVisitor {
            txn_id: cxt.0,
            file: cxt.1,
        };

        decoder.decode_seq(visitor).await
    }
}

struct FileVisitor<B> {
    txn_id: TxnId,
    file: File<B>,
}

#[async_trait]
impl<'en, B: fs::BlockData + de::FromStream<Context = ()> + 'en> de::Visitor for FileVisitor<B>
where
    B: en::IntoStream<'en>,
    CacheBlock: From<CacheLock<B>>,
    CacheLock<B>: TryFrom<CacheBlock, Error = TCError>,
{
    type Value = File<B>;

    fn expecting() -> &'static str {
        "a File"
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let mut i = 0u64;
        while let Some(block) = seq
            .next_element(())
            .map_err(|e| de::Error::custom(format!("invalid block: {}", e)))
            .await?
        {
            debug!("decoded file block {}", i);
            fs::File::create_block(&self.file, self.txn_id, i.into(), block)
                .map_err(de::Error::custom)
                .await?;

            i += 1;
            debug!("checking whether to decode file block {}...", i);
        }

        Ok(self.file)
    }
}

#[inline]
fn file_version(file_path: &PathBuf, txn_id: &TxnId) -> PathBuf {
    let mut path = file_path.clone();
    path.push(super::VERSION.to_string());
    path.push(txn_id.to_string());
    path
}

#[inline]
fn block_version(file_path: &PathBuf, txn_id: &TxnId, block_id: &BlockId) -> PathBuf {
    let mut path = file_version(file_path, txn_id);
    path.push(block_id.to_string());
    path
}
