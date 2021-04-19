//! A transactional file

use std::collections::hash_map::{Entry, HashMap};
use std::collections::HashSet;
use std::convert::TryFrom;
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::path::PathBuf;

use async_trait::async_trait;
use destream::{de, en};
use futures::future::{try_join_all, TryFutureExt};
use log::debug;
use uplock::*;
use uuid::Uuid;

use tc_error::*;
use tc_transact::fs::{self, BlockData, BlockId, Store};
use tc_transact::lock::{Mutable, TxnLock};
use tc_transact::{Transact, TxnId};
use tcgeneric::TCBoxTryFuture;

use super::cache::*;
use super::{file_name, fs_path, DirContents};

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
    type ReadLock = BlockRead<B>;
    type WriteLock = BlockWrite<B>;

    async fn read(self) -> Self::ReadLock {
        let lock = self.lock.read().await;
        BlockRead {
            name: self.name,
            txn_id: self.txn_id,
            lock,
        }
    }

    async fn write(self) -> Self::WriteLock {
        let lock = self.lock.write().await;
        BlockWrite {
            name: self.name,
            txn_id: self.txn_id,
            lock,
        }
    }
}

pub struct BlockRead<B> {
    name: BlockId,
    txn_id: TxnId,
    lock: RwLockReadGuard<B>,
}

impl<B> Deref for BlockRead<B> {
    type Target = B;

    fn deref(&self) -> &B {
        self.lock.deref()
    }
}

impl<'en, B: BlockData + en::IntoStream<'en> + 'en> fs::BlockRead<B, File<B>> for BlockRead<B>
where
    CacheBlock: From<CacheLock<B>>,
    CacheLock<B>: TryFrom<CacheBlock, Error = TCError>,
{
    fn upgrade(self, file: &File<B>) -> TCBoxTryFuture<BlockWrite<B>> {
        Box::pin(fs::File::write_block(file, self.txn_id, self.name))
    }
}

pub struct BlockWrite<B> {
    name: BlockId,
    txn_id: TxnId,
    lock: RwLockWriteGuard<B>,
}

impl<B> Deref for BlockWrite<B> {
    type Target = B;

    fn deref(&self) -> &B {
        self.lock.deref()
    }
}

impl<B> DerefMut for BlockWrite<B> {
    fn deref_mut(&mut self) -> &mut B {
        self.lock.deref_mut()
    }
}

impl<'en, B: BlockData + en::IntoStream<'en> + 'en> fs::BlockWrite<B, File<B>> for BlockWrite<B>
where
    CacheBlock: From<CacheLock<B>>,
    CacheLock<B>: TryFrom<CacheBlock, Error = TCError>,
{
    fn downgrade(self, file: &File<B>) -> TCBoxTryFuture<BlockRead<B>> {
        Box::pin(fs::File::read_block(file, self.txn_id, self.name))
    }
}

/// A transactional file
#[derive(Clone)]
pub struct File<B> {
    path: PathBuf,
    cache: Cache,
    contents: TxnLock<Mutable<HashSet<BlockId>>>,
    mutated: RwLock<HashMap<TxnId, HashSet<BlockId>>>,
    phantom: PhantomData<B>,
}

impl<B: BlockData> File<B> {
    fn _new(cache: Cache, path: PathBuf, block_ids: HashSet<BlockId>) -> Self {
        let lock_name = format!("file contents at {:?}", &path);

        File {
            path,
            cache,
            contents: TxnLock::new(lock_name, block_ids.into()),
            mutated: RwLock::new(HashMap::new()),
            phantom: PhantomData,
        }
    }

    pub fn new(cache: Cache, mut path: PathBuf, ext: &str) -> Self {
        path.set_extension(ext);
        Self::_new(cache, path, HashSet::new())
    }

    pub fn load(cache: Cache, path: PathBuf, contents: DirContents) -> TCResult<Self> {
        if contents.iter().all(|(_, meta)| meta.is_file()) {
            let contents = contents
                .into_iter()
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
}

#[async_trait]
impl<B: BlockData> Store for File<B> {
    async fn is_empty(&self, txn_id: &TxnId) -> TCResult<bool> {
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

    async fn block_ids(&self, txn_id: &TxnId) -> TCResult<HashSet<BlockId>> {
        let contents = self.contents.read(txn_id).await?;
        Ok((*contents).clone())
    }

    async fn unique_id(&self, txn_id: &TxnId) -> TCResult<BlockId> {
        let contents = self.contents.read(txn_id).await?;
        let id = loop {
            let id: BlockId = Uuid::new_v4().into();
            if !contents.contains(&id) {
                break id;
            }
        };

        Ok(id)
    }

    async fn contains_block(&self, txn_id: &TxnId, name: &BlockId) -> TCResult<bool> {
        self.contents
            .read(txn_id)
            .map_ok(|contents| contents.contains(name))
            .await
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

        let path = fs_path(&self.path, &name);
        contents.insert(name.clone());

        self.cache
            .write(path, initial_value)
            .map_ok(|lock| Block { name, txn_id, lock })
            .await
    }

    async fn delete_block(&self, txn_id: TxnId, name: &BlockId) -> TCResult<()> {
        let mut contents = self.contents.write(txn_id).await?;
        if !contents.remove(&name) {
            return Err(TCError::not_found(name));
        }

        let version = block_version(&self.path, &txn_id, name);
        self.cache.delete(&version).await
    }

    async fn get_block(&self, txn_id: TxnId, name: BlockId) -> TCResult<Block<B>> {
        {
            let contents = self.contents.read(&txn_id).await?;
            if !contents.contains(&name) {
                return Err(TCError::not_found(name));
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

    async fn read_block(&self, txn_id: TxnId, name: BlockId) -> TCResult<BlockRead<B>> {
        let block = self.get_block(txn_id, name).await?;
        Ok(fs::Block::read(block).await)
    }

    async fn read_block_owned(self, txn_id: TxnId, name: BlockId) -> TCResult<BlockRead<B>> {
        let block = self.get_block(txn_id, name).await?;
        Ok(fs::Block::read(block).await)
    }

    async fn write_block(&self, txn_id: TxnId, name: BlockId) -> TCResult<BlockWrite<B>> {
        debug!("File::write_block");
        let mut mutated = self.mutated.write().await;
        let block = self.get_block(txn_id, name.clone()).await?;
        debug!("File::write_block got block {}", name);

        match mutated.entry(txn_id) {
            Entry::Vacant(entry) => entry.insert(HashSet::new()).insert(block.name.clone()),
            Entry::Occupied(mut entry) => entry.get_mut().insert(block.name.clone()),
        };

        debug!("File::write_block getting write lock on {}...", name);
        Ok(fs::Block::write(block).await)
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
            let contents = self.contents.read(txn_id).await.expect("file block list");
            let mutated = self.mutated.read().await;
            if let Some(blocks) = mutated.get(txn_id) {
                let commits = blocks
                    .iter()
                    .filter(|block_id| contents.contains(block_id))
                    .map(|block_id| async move {
                        let block_path = fs_path(file_path, block_id);

                        let block = fs::File::get_block(self, *txn_id, block_id.clone())
                            .await
                            .expect("get block");

                        let data = fs::Block::read(block).await;

                        cache.write_and_sync(block_path, data.deref().clone()).await
                    });

                try_join_all(commits).await.expect("commit file blocks");
            }
        }

        self.contents.commit(txn_id).await;
        debug!("committed {:?} at {}", &self.path, txn_id);
    }

    async fn finalize(&self, txn_id: &TxnId) {
        debug!("finalize file {:?} at {}", &self.path, txn_id);

        let file_path = &self.path;
        let cache = &self.cache;
        {
            let contents = self.contents.read(txn_id).await.expect("file block list");
            let mutated = self.mutated.read().await;
            if let Some(blocks) = mutated.get(txn_id) {
                let commits = blocks
                    .iter()
                    .filter(|block_id| !contents.contains(block_id))
                    .map(|block_id| async move {
                        let block_path = fs_path(file_path, block_id);
                        cache.delete_and_sync(&block_path).await
                    });

                try_join_all(commits).await.expect("commit file blocks");
            }
        }

        let version = file_version(&self.path, txn_id);
        if version.exists() {
            tokio::fs::remove_dir_all(version)
                .await
                .expect("delete file version");
        }

        self.contents.finalize(txn_id).await;
        debug!("finalized {:?} at {}", &self.path, txn_id);
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
