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
use futures::future::{try_join_all, FutureExt, TryFutureExt};
use log::debug;
use uplock::*;

use tc_error::*;
use tc_transact::fs::{self, BlockData, BlockId, Store};
use tc_transact::lock::{Mutable, TxnLock};
use tc_transact::{Transact, TxnId};

use super::cache::*;
use super::{file_name, fs_path, DirContents};

pub struct Block<B> {
    lock: CacheLock<B>,
}

#[async_trait]
impl<'en, B: BlockData<'en>> fs::Block<'en, B> for Block<B> {
    type ReadLock = BlockRead<B>;
    type WriteLock = BlockWrite<B>;

    async fn read(&self) -> Self::ReadLock {
        self.lock.read().map(|lock| BlockRead { lock }).await
    }

    async fn write(&self) -> Self::WriteLock {
        self.lock.write().map(|lock| BlockWrite { lock }).await
    }
}

pub struct BlockRead<B> {
    lock: RwLockReadGuard<B>,
}

impl<B> Deref for BlockRead<B> {
    type Target = B;

    fn deref(&self) -> &B {
        self.lock.deref()
    }
}

pub struct BlockWrite<B> {
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

/// A transactional file
#[derive(Clone)]
pub struct File<B> {
    path: PathBuf,
    cache: Cache,
    contents: TxnLock<Mutable<HashSet<BlockId>>>,
    mutated: RwLock<HashMap<TxnId, HashSet<BlockId>>>,
    phantom: PhantomData<B>,
}

impl<'en, B: BlockData<'en> + 'en> File<B> {
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

    async fn get_block(&self, txn_id: &TxnId, name: &BlockId) -> TCResult<Block<B>>
    where
        B: en::IntoStream<'en>,
        CacheBlock: From<CacheLock<B>>,
        CacheLock<B>: TryFrom<CacheBlock, Error = TCError>,
    {
        if !fs::File::contains_block(self, txn_id, &name).await? {
            return Err(TCError::not_found(name));
        }

        let version = fs_path(&version_path(&self.path, txn_id), &name);
        if let Some(lock) = self.cache.read(&version).await? {
            Ok(Block { lock })
        } else {
            let canon = fs_path(&self.path, name);
            let block = self.cache.read(&canon).await?;
            let block = block.ok_or_else(|| TCError::internal("failed reading block"))?;
            let data = block.read().await;
            self.cache
                .write(version, data.deref().clone())
                .map_ok(|lock| Block { lock })
                .await
        }
    }
}

#[async_trait]
impl<'en, B: BlockData<'en> + 'en> Store for File<B> {
    async fn is_empty(&self, txn_id: &TxnId) -> TCResult<bool> {
        self.contents
            .read(txn_id)
            .map_ok(|contents| contents.is_empty())
            .await
    }
}

#[async_trait]
impl<'en, B: BlockData<'en> + 'en> fs::File<'en, B> for File<B>
where
    B: en::IntoStream<'en>,
    CacheBlock: From<CacheLock<B>>,
    CacheLock<B>: TryFrom<CacheBlock, Error = TCError>,
{
    type Block = Block<B>;

    async fn block_ids(&self, txn_id: &TxnId) -> TCResult<HashSet<BlockId>> {
        let block_ids = self.contents.read(txn_id).await?;
        Ok((*block_ids).clone())
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
        contents.insert(name);
        self.cache
            .write(path, initial_value)
            .map_ok(|lock| Block { lock })
            .await
    }

    async fn delete_block(&self, _txn_id: &TxnId, _name: &BlockId) -> TCResult<()> {
        Err(TCError::not_implemented("File::delete_block"))
    }

    async fn read_block(&self, txn_id: &TxnId, name: &BlockId) -> TCResult<BlockRead<B>> {
        let block = self.get_block(txn_id, &name).await?;
        Ok(fs::Block::read(&block).await)
    }

    async fn read_block_owned(self, txn_id: TxnId, name: BlockId) -> TCResult<BlockRead<B>> {
        let block = self.get_block(&txn_id, &name).await?;
        Ok(fs::Block::read(&block).await)
    }

    async fn write_block(&self, txn_id: TxnId, name: BlockId) -> TCResult<BlockWrite<B>> {
        let mut mutated = self.mutated.write().await;
        let block = self.get_block(&txn_id, &name).await?;

        match mutated.entry(txn_id) {
            Entry::Vacant(entry) => entry.insert(HashSet::new()).insert(name.clone()),
            Entry::Occupied(mut entry) => entry.get_mut().insert(name.clone()),
        };

        Ok(fs::Block::write(&block).await)
    }
}

#[async_trait]
impl<'en, B: BlockData<'en> + 'en> Transact for File<B>
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
            let mutated = self.mutated.read().await;
            if let Some(blocks) = mutated.get(txn_id) {
                let commits = blocks.iter().map(|block_id| async move {
                    let block_path = fs_path(file_path, block_id);
                    let block = self.get_block(txn_id, block_id).await.expect("get block");

                    let data = fs::Block::read(&block).await;

                    cache.write_and_sync(block_path, data.deref().clone()).await
                });

                try_join_all(commits).await.expect("commit file blocks");
            }
        }

        self.contents.commit(txn_id).await;
        debug!("committed {:?} at {}", &self.path, txn_id);
    }

    async fn finalize(&self, txn_id: &TxnId) {
        let version = version_path(&self.path, txn_id);
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
impl<'en, B: BlockData<'en> + de::FromStream<Context = ()> + 'en> de::FromStream for File<B>
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
impl<'en, B: fs::BlockData<'en> + de::FromStream<Context = ()> + 'en> de::Visitor for FileVisitor<B>
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
fn version_path(file_path: &PathBuf, txn_id: &TxnId) -> PathBuf {
    let mut path = file_path.clone();
    path.push(super::VERSION.to_string());
    path.push(txn_id.to_string());
    path
}
