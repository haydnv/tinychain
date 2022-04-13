//! A transactional file

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use freqfs::*;
use futures::future::{join_all, try_join_all, TryFutureExt};
use futures::try_join;
use log::{debug, trace};
use safecast::AsType;
use tokio::sync::RwLock;
use uuid::Uuid;

use tc_error::*;
use tc_transact::fs;
use tc_transact::fs::BlockId;
use tc_transact::lock::TxnLock;
use tc_transact::{Transact, TxnId};

use super::{io_err, CacheBlock, VERSION};

type Blocks = HashMap<fs::BlockId, TxnLock<TxnId>>;

/// A transactional file
#[derive(Clone)]
pub struct File<B> {
    canon: DirLock<CacheBlock>,
    versions: DirLock<CacheBlock>,

    // don't try to keep the block contents in a TxnLock, since the block versions
    // may need to be backed up to disk--just keep a lock on the TxnId of the last mutation
    // and guard the block content by acquiring a lock on the mutation ID
    // before allowing access to a block
    blocks: Arc<RwLock<Blocks>>,

    present: TxnLock<HashSet<fs::BlockId>>,
    phantom: PhantomData<B>,
}

impl<B: fs::BlockData> File<B>
where
    CacheBlock: AsType<B>,
{
    pub async fn new(canon: DirLock<CacheBlock>) -> TCResult<Self> {
        let mut fs_dir = canon.write().await;
        if fs_dir.len() > 0 {
            return Err(TCError::internal("new file is not empty"));
        }

        let name = format!("block listing for {:?}", &*fs_dir);
        let versions = fs_dir.create_dir(VERSION.to_string()).map_err(io_err)?;
        Ok(Self {
            canon,
            versions,
            blocks: Arc::new(RwLock::new(HashMap::new())),
            present: TxnLock::new(name, HashSet::new()),
            phantom: PhantomData,
        })
    }

    pub(super) async fn load(canon: DirLock<CacheBlock>, txn_id: TxnId) -> TCResult<Self> {
        let mut fs_dir = canon.write().await;
        let versions = fs_dir
            .get_or_create_dir(VERSION.to_string())
            .map_err(io_err)?;

        let mut blocks = HashMap::new();
        let mut present = HashSet::new();
        let mut version = versions
            .write()
            .await
            .create_dir(txn_id.to_string())
            .map_err(io_err)?
            .write()
            .await;

        for (name, block) in fs_dir.iter() {
            if name.starts_with('.') {
                debug!("File::load skipping hidden filesystem entry");
                continue;
            }

            if name.len() < B::ext().len() + 1 || !name.ends_with(B::ext()) {
                return Err(TCError::internal(format!(
                    "block has invalid extension: {}",
                    name
                )));
            }

            let (size_hint, contents) = match block {
                DirEntry::File(block) => {
                    let size_hint = block.size_hint().await;
                    let contents = block
                        .read()
                        .map_ok(|contents| B::clone(&*contents))
                        .map_err(io_err)
                        .await?;

                    (size_hint, contents)
                }
                DirEntry::Dir(_) => {
                    return Err(TCError::internal(format!(
                        "expected block file but found directory: {}",
                        name
                    )))
                }
            };

            let block_id: fs::BlockId = name[..(name.len() - B::ext().len() - 1)]
                .parse()
                .map_err(TCError::internal)?;

            present.insert(block_id.clone());

            let lock_name = format!("block {}", block_id);
            blocks.insert(block_id, TxnLock::new(lock_name, txn_id));

            version
                .create_file(name.clone(), contents, size_hint)
                .map_err(io_err)?;
        }

        #[cfg(debug_assertions)]
        let lock_name = format!("block list of file {:?}", &*fs_dir);
        #[cfg(not(debug_assertions))]
        let lock_name = "block list of transactional file";

        Ok(Self {
            canon,
            versions,
            blocks: Arc::new(RwLock::new(blocks)),
            present: TxnLock::new(lock_name, present),
            phantom: PhantomData,
        })
    }

    async fn version(&self, txn_id: &TxnId) -> TCResult<DirLock<CacheBlock>> {
        trace!("getting write lock on file dir");

        let mut versions = self.versions.write().await;
        trace!("got write lock on file dir");

        versions
            .get_or_create_dir(txn_id.to_string())
            .map_err(io_err)
    }

    async fn version_read(&self, txn_id: &TxnId) -> TCResult<DirReadGuard<CacheBlock>> {
        trace!("getting read lock on file dir");
        let version = self.version(txn_id).await?;
        trace!("getting read lock on file version {} dir", txn_id);
        Ok(version.read().await)
    }

    async fn version_write(&self, txn_id: &TxnId) -> TCResult<DirWriteGuard<CacheBlock>> {
        trace!("getting write lock on file dir");
        let version = self.version(txn_id).await?;
        trace!("getting write lock on file version {} dir", txn_id);
        Ok(version.write().await)
    }

    async fn with_version_read<F, T>(&self, txn_id: &TxnId, then: F) -> TCResult<T>
    where
        F: FnOnce(DirReadGuard<CacheBlock>) -> T,
    {
        let dir = self.version_read(txn_id).await?;
        trace!("got read lock on file version {} dir", txn_id);
        Ok(then(dir))
    }

    async fn with_version_write<F, T>(&self, txn_id: &TxnId, then: F) -> TCResult<T>
    where
        F: FnOnce(DirWriteGuard<CacheBlock>) -> T,
    {
        let dir = self.version_write(txn_id).await?;
        trace!("got write lock on file version {} dir", txn_id);
        Ok(then(dir))
    }

    fn file_name(block_id: &fs::BlockId) -> String {
        format!("{}.{}", block_id, B::ext())
    }
}

#[async_trait]
impl<B: fs::BlockData> fs::Store for File<B> {
    async fn is_empty(&self, txn_id: TxnId) -> TCResult<bool> {
        self.present
            .read(txn_id)
            .map_ok(|present| present.is_empty())
            .await
    }
}

#[async_trait]
impl<B> fs::File<B> for File<B>
where
    B: fs::BlockData,
    CacheBlock: AsType<B>,
{
    type Read = FileReadGuard<CacheBlock, B>;
    type Write = FileWriteGuard<CacheBlock, B>;

    async fn block_ids(&self, txn_id: TxnId) -> TCResult<HashSet<fs::BlockId>> {
        let present = self.present.read(txn_id).await?;
        Ok(present.clone())
    }

    async fn contains_block(&self, txn_id: TxnId, name: &fs::BlockId) -> TCResult<bool> {
        let present = self.present.read(txn_id).await?;
        Ok(present.contains(name))
    }

    async fn copy_from(&self, other: &Self, txn_id: TxnId) -> TCResult<()> {
        let (mut this_present, that_present) =
            try_join!(self.present.write(txn_id), other.present.read(txn_id))?;

        let (mut this_version, that_version) =
            try_join!(self.version_write(&txn_id), other.version_read(&txn_id))?;

        let mut blocks = self.blocks.write().await;
        let canon = other.canon.read().await;

        for block_id in that_present.iter() {
            let file_name = Self::file_name(block_id);
            let source = if let Some(version) = that_version.get_file(&file_name) {
                version
            } else {
                canon.get_file(&file_name).expect("block version")
            };

            this_present.insert(block_id.clone());

            if let Some(last_mutation) = blocks.get_mut(block_id) {
                *last_mutation.write(txn_id).await? = txn_id;
            } else {
                let lock_name = format!("block {}", block_id);
                blocks.insert(block_id.clone(), TxnLock::new(lock_name, txn_id));
            }

            let block = source.read().map_err(io_err).await?;
            let size_hint = source.size_hint().await;
            this_version.delete(file_name.clone());
            this_version
                .create_file(file_name, block.clone(), size_hint)
                .map_err(io_err)?;
        }

        Ok(())
    }

    async fn create_block(
        &self,
        txn_id: TxnId,
        block_id: fs::BlockId,
        initial_value: B,
        size_hint: usize,
    ) -> TCResult<Self::Write> {
        debug!("File::create_block {}", block_id);

        let mut present = self.present.write(txn_id).await?;
        if present.contains(&block_id) {
            return Err(TCError::bad_request("block already exists", block_id));
        }

        let mut blocks = self.blocks.write().await;
        let version = self.version_write(&txn_id).await?;

        let block = create_block_inner(
            &mut present,
            &mut blocks,
            version,
            txn_id,
            block_id,
            initial_value,
            size_hint,
        )
        .await?;

        block.write().map_err(io_err).await
    }

    async fn create_block_unique(
        &self,
        txn_id: TxnId,
        initial_value: B,
        size_hint: usize,
    ) -> TCResult<(fs::BlockId, Self::Write)> {
        debug!("File::create_block_tmp");

        let mut present = self.present.write(txn_id).await?;

        let block_id: BlockId = loop {
            let name = Uuid::new_v4().into();
            if !present.contains(&name) {
                break name;
            }
        };

        let mut blocks = self.blocks.write().await;
        let version = self.version_write(&txn_id).await?;

        let block = create_block_inner(
            &mut present,
            &mut blocks,
            version,
            txn_id,
            block_id.clone(),
            initial_value,
            size_hint,
        )
        .await?;

        let lock = block.write().map_err(io_err).await?;
        Ok((block_id, lock))
    }

    async fn delete_block(&self, txn_id: TxnId, block_id: fs::BlockId) -> TCResult<()> {
        debug!("File::delete_block {}", block_id);

        let mut present = self.present.write(txn_id).await?;

        let mut blocks = self.blocks.write().await;
        if let Some(last_mutation) = blocks.get_mut(&block_id) {
            *last_mutation.write(txn_id).await? = txn_id;

            // keep the version directory in sync in case create_block is called later
            // with the same block_id
            self.with_version_write(&txn_id, |mut version| {
                version.delete(Self::file_name(&block_id))
            })
            .await?;
        }

        present.remove(&block_id);
        Ok(())
    }

    async fn read_block(
        &self,
        txn_id: TxnId,
        block_id: fs::BlockId,
    ) -> TCResult<FileReadGuard<CacheBlock, B>> {
        debug!("File::read_block {}", block_id);

        let last_mutation = {
            let block = {
                let present = self.present.read(txn_id).await?;
                trace!("File::read_block got read lock on block ID list");

                if !present.contains(&block_id) {
                    return Err(TCError::not_found(block_id));
                }

                let blocks = self.blocks.read().await;
                trace!("File::read_block got read lock on last mutation IDs");

                trace!(
                    "File::read_block getting read lock on last mutation ID of {}",
                    block_id
                );

                blocks
                    .get(&block_id)
                    .expect("block last mutation ID")
                    .clone()
            };

            block.read(txn_id).await?
        };

        trace!("got read lock on block last mutation ID");

        let name = Self::file_name(&block_id);
        let block = self
            .with_version_read(&txn_id, |version| version.get_file(&name))
            .await?;

        if let Some(block) = block {
            let lock = block.read().map_err(io_err).await?;

            trace!(
                "got read lock on existing version of block {} at {}",
                block_id,
                txn_id
            );

            return Ok(lock);
        }

        assert!(*last_mutation < txn_id);
        trace!(
            "last mutation of block {} was at {}",
            block_id,
            &*last_mutation
        );

        let (size_hint, value) = {
            trace!(
                "File::read_block locking prior version dir {}...",
                &*last_mutation
            );

            let block_version = self
                .with_version_read(&last_mutation, |version| version.get_file(&name))
                .await?
                .expect("block prior value");

            let size_hint = block_version.size_hint().await;

            let value = {
                let value = block_version.read().map_err(io_err).await?;

                trace!(
                    "File::read_block locked prior block version {}",
                    &*last_mutation
                );

                B::clone(&*value)
            };

            trace!(
                "File::read_block unlocked prior block version {}",
                &*last_mutation
            );

            (size_hint, value)
        };

        trace!(
            "File::read_block locking version dir {} to create new block {}",
            txn_id,
            block_id
        );

        let block = self
            .with_version_write(&txn_id, |mut version| {
                version.create_file(name, value, size_hint).map_err(io_err)
            })
            .await??;

        trace!(
            "File::read_block getting read lock on new block {}",
            block_id
        );

        let lock = block.read().map_err(io_err).await?;
        trace!("File::read_block got read lock on new block {}", block_id);
        Ok(lock)
    }

    async fn read_block_owned(
        self,
        txn_id: TxnId,
        block_id: fs::BlockId,
    ) -> TCResult<FileReadGuard<CacheBlock, B>> {
        self.read_block(txn_id, block_id).await
    }

    async fn write_block(
        &self,
        txn_id: TxnId,
        block_id: fs::BlockId,
    ) -> TCResult<FileWriteGuard<CacheBlock, B>> {
        debug!("File::write_block {}", block_id);

        let mut last_mutation = {
            {
                let block = {
                    trace!("File::write_block getting read lock on present block IDs...");
                    let present = self.present.read(txn_id).await?;
                    trace!("File::write_block got read lock on present block IDs");

                    if !present.contains(&block_id) {
                        return Err(TCError::not_found(block_id));
                    }

                    trace!("File::write block getting read locks on last mutations IDs...");
                    let blocks = self.blocks.read().await;
                    trace!("File::write block got read lock on last mutation IDs");

                    trace!(
                        "File::write_block getting write lock on last mutation ID of block {}...",
                        block_id
                    );

                    blocks
                        .get(&block_id)
                        .expect("block last mutation ID")
                        .clone()
                };

                block.write(txn_id).await?
            }
        };

        trace!("File::write_block got write lock on block last mutation ID");

        *last_mutation = txn_id;

        let name = Self::file_name(&block_id);

        trace!(
            "File::write_block checking for existing version of {} at {}...",
            block_id,
            txn_id
        );

        let block = self
            .with_version_read(&txn_id, |version| version.get_file(&name))
            .await?;

        if let Some(block) = block {
            trace!(
                "File::write_block read existing version of block {} at {}",
                block_id,
                txn_id
            );

            return block.write().map_err(io_err).await;
        }

        trace!(
            "File::write_block got write lock on block version {}",
            txn_id
        );

        // a write can only happen before a commit
        // therefore the canonical version must be current

        let block_canon = {
            let canon = self.canon.read().await;
            canon.get_file(&name).expect("canonical block")
        };

        let size_hint = block_canon.size_hint().await;
        let value = {
            trace!(
                "File::write block getting read lock on canonical version of {}...",
                block_id
            );
            let value = block_canon.read().map_err(io_err).await?;
            trace!(
                "File::write block got read lock on canonical version of {}",
                block_id
            );
            B::clone(&*value)
        };

        trace!(
            "File::write_block creating new version of {} at {}",
            block_id,
            txn_id
        );
        let block = self
            .with_version_write(&txn_id, |mut version| {
                version.create_file(name, value, size_hint).map_err(io_err)
            })
            .await??;

        trace!(
            "File::write_block getting write lock on new version {} of {}...",
            txn_id,
            block_id
        );

        let lock = block.write().map_err(io_err).await?;

        trace!(
            "File::write_block getting write lock on new version {} of {}",
            txn_id,
            block_id
        );

        Ok(lock)
    }

    async fn truncate(&self, txn_id: TxnId) -> TCResult<()> {
        let mut present = self.present.write(txn_id).await?;
        let mut version = self.version_write(&txn_id).await?;
        for block_id in present.drain() {
            version.delete(Self::file_name(&block_id));
        }

        Ok(())
    }
}

#[async_trait]
impl<B> Transact for File<B>
where
    B: fs::BlockData,
    CacheBlock: AsType<B>,
{
    async fn commit(&self, txn_id: &TxnId) {
        debug!("File::commit");

        let mut blocks = self.blocks.write().await;

        self.present.commit(txn_id).await;
        trace!("File::commit committed block listing");

        let block_commits = blocks
            .values()
            .map(|last_mutation| last_mutation.commit(txn_id));

        join_all(block_commits).await;

        {
            let present = self.present.read(*txn_id).await.expect("file block list");

            let version = {
                let fs_dir = self.versions.read().await;
                if let Some(version) = fs_dir.get_dir(&txn_id.to_string()) {
                    Some(version.read().await)
                } else {
                    None
                }
            };

            let mut canon = self.canon.write().await;
            let mut deleted = Vec::with_capacity(blocks.len());
            let mut synchronize = Vec::with_capacity(present.len());
            for block_id in blocks.keys() {
                let name = Self::file_name(block_id);
                if present.contains(block_id) {
                    if let Some(version) = &version {
                        if let Some(version) = version.get_file(&name) {
                            let block = version.read().await.expect("block version");
                            let canon = if let Some(canon) = canon.get_file(&name) {
                                *canon.write().await.expect("canonical block") = block.clone();
                                canon
                            } else {
                                let size_hint = version.size_hint().await;
                                canon
                                    .create_file(name, block.clone(), size_hint)
                                    .expect("new canonical block")
                            };

                            synchronize.push(async move { canon.sync(true).await });
                        } else {
                            trace!("block {} has no version to commit at {}", block_id, txn_id);
                        }
                    }
                } else {
                    canon.delete(name);
                    deleted.push(block_id.clone());
                }
            }

            for block_id in deleted.into_iter() {
                assert!(blocks.remove(&block_id).is_some());
            }

            try_join_all(synchronize)
                .await
                .expect("sync block contents to disk");
        }

        self.canon
            .sync(false)
            .await
            .expect("sync file content to disk");
    }

    async fn finalize(&self, txn_id: &TxnId) {
        {
            let blocks = self.blocks.read().await;
            join_all(
                blocks
                    .values()
                    .map(|last_commit_id| last_commit_id.finalize(txn_id)),
            )
            .await;
        }

        self.present.finalize(txn_id).await;

        let mut versions = self.versions.write().await;
        versions.delete(txn_id.to_string());
    }
}

impl<B: Send + Sync + 'static> fmt::Display for File<B> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "file of {} blocks", std::any::type_name::<B>())
    }
}

async fn create_block_inner<'a, B: fs::BlockData + 'a>(
    present: &mut HashSet<BlockId>,
    blocks: &mut Blocks,
    mut version: DirWriteGuard<CacheBlock>,
    txn_id: TxnId,
    block_id: fs::BlockId,
    value: B,
    size: usize,
) -> TCResult<FileLock<CacheBlock>>
where
    CacheBlock: AsType<B>,
{
    debug_assert!(!blocks.contains_key(&block_id));

    blocks.insert(
        block_id.clone(),
        TxnLock::new(format!("block {}", block_id), txn_id),
    );

    let name = format!("{}.{}", block_id, B::ext());
    let block = version
        .create_file(name, value, Some(size))
        .map_err(io_err)?;

    present.insert(block_id);

    trace!("dropping file version dir lock at {}", txn_id);

    Ok(block.into())
}
