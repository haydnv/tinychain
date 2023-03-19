use std::fmt;
use std::marker::PhantomData;

use async_trait::async_trait;
use freqfs::{FileLoad, FileSave};
use futures::future::TryFutureExt;
use futures::stream::{self, Stream, StreamExt, TryStreamExt};
use get_size::GetSize;
use safecast::AsType;

use tc_error::*;
use tcgeneric::{Id, ThreadSafe};

use super::{TCResult, Transact, Transaction, TxnId};

pub use txfs::VERSIONS;

/// The underlying filesystem directory type backing a [`Dir`]
pub type Inner<FE> = freqfs::DirLock<FE>;

/// A read lock on a block in a [`File`]
pub type BlockRead<FE, B> = txfs::FileVersionRead<TxnId, FE, B>;

/// A write lock on a block in a [`File`]
pub type BlockWrite<FE, B> = txfs::FileVersionWrite<TxnId, FE, B>;

/// An entry in a [`Dir`]
pub enum DirEntry<FE, N, B> {
    Dir(Dir<FE>),
    File(File<FE, N, B>),
}

/// A transactional directory
pub struct Dir<FE> {
    inner: txfs::Dir<TxnId, FE>,
}

impl<FE> Clone for Dir<FE> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<FE: ThreadSafe> Dir<FE> {
    /// Load a transactional [`Dir`] from the filesystem cache
    pub async fn load(txn_id: TxnId, canon: freqfs::DirLock<FE>) -> TCResult<Self> {
        txfs::Dir::load(txn_id, canon)
            .map_ok(|inner| Self { inner })
            .map_err(TCError::from)
            .await
    }

    /// Destructure this [`Dir`] into its underlying [`freqfs::DirLock`].
    pub fn into_inner(self) -> Inner<FE> {
        self.inner.into_inner()
    }

    /// Check whether this [`Dir`] has an entry with the given `name` at `txn_id`.
    pub async fn contains(&self, txn_id: TxnId, name: &Id) -> TCResult<bool> {
        self.inner
            .contains(txn_id, name.as_str())
            .map_err(TCError::from)
            .await
    }

    /// Create a new sub-directory with the given `name` at `txn_id`.
    pub async fn create_dir(&self, txn_id: TxnId, name: Id) -> TCResult<Self> {
        self.inner
            .create_dir(txn_id, name.into())
            .map_ok(|inner| Self { inner })
            .map_err(TCError::from)
            .await
    }

    /// Create a new [`File`] with the given `name` at `txn_id`.
    pub async fn create_file<N, B>(&self, txn_id: TxnId, name: Id) -> TCResult<File<FE, N, B>>
    where
        B: GetSize + Clone,
        N: fmt::Display,
        FE: AsType<B>,
    {
        self.inner
            .create_dir(txn_id, name.into())
            .map_ok(File::new)
            .map_err(TCError::from)
            .await
    }

    /// Iterate over the names of the entries in this [`Dir`] at `txn_id`.
    pub async fn entry_names(&self, txn_id: TxnId) -> TCResult<impl Iterator<Item = Id>> {
        let names = self.inner.dir_names(txn_id).await?;
        // TODO: avoid this call to parse()
        Ok(names.map(|name| name.parse().expect("dir entry name")))
    }

    /// Iterate over each [`DirEntry`] in this [`Dir`] at `txn_id`, assuming it is a [`File`].
    pub async fn files<N, B>(
        &self,
        txn_id: TxnId,
    ) -> TCResult<impl Iterator<Item = (Id, File<FE, N, B>)>> {
        let entries = self.inner.iter(txn_id).await?;
        Ok(entries.map(|(name, entry)| {
            let file = match &*entry {
                txfs::DirEntry::Dir(blocks_dir) => File::new(blocks_dir.clone()),
                other => panic!("not a block directory: {:?}", other),
            };

            (name.parse().expect("name"), file)
        }))
    }

    /// Get the sub-[`Dir`] with the given `name` at `txn_id`, or return a "not found" error.
    pub async fn get_dir(&self, txn_id: TxnId, name: &Id) -> TCResult<Self> {
        if let Some(dir) = self.inner.get_dir(txn_id, name.as_str()).await? {
            Ok(Self { inner: dir.clone() })
        } else {
            Err(TCError::not_found(name))
        }
    }

    /// Get the sub-[`Dir`] with the given `name` at `txn_id`, or create a new one.
    pub async fn get_or_create_dir(&self, txn_id: TxnId, name: Id) -> TCResult<Self> {
        if let Some(dir) = self.inner.get_dir(txn_id, name.as_str()).await? {
            Ok(Self { inner: dir.clone() })
        } else {
            self.create_dir(txn_id, name).await
        }
    }

    /// Get the [`File`] with the given `name` at `txn_id`, or return a "not found" error.
    pub async fn get_file<N, B>(&self, txn_id: TxnId, name: &Id) -> TCResult<File<FE, N, B>>
    where
        B: GetSize + Clone,
        N: fmt::Display,
        FE: AsType<B>,
    {
        if let Some(blocks) = self.inner.get_dir(txn_id, name.as_str()).await? {
            Ok(File::new(blocks.clone()))
        } else {
            Err(TCError::not_found(name))
        }
    }

    /// Return `true` if this [`Dir`] is empty at `txn_id`.
    pub async fn is_empty(&self, txn_id: TxnId) -> TCResult<bool> {
        self.inner.is_empty(txn_id).map_err(TCError::from).await
    }

    /// Iterate over the [`DirEntry`]s in this [`Dir`] at `txn_id`.
    pub async fn iter<N, B>(
        &self,
        txn_id: TxnId,
    ) -> TCResult<impl Stream<Item = TCResult<(Id, DirEntry<FE, N, B>)>> + Unpin + Send + '_> {
        let entries = self.inner.iter(txn_id).await?;

        let entries = stream::iter(entries).then(move |(name, entry)| async move {
            // TODO: avoid this call to parse()
            let name: Id = name.parse().expect("dir entry name");
            let entry = match &*entry {
                txfs::DirEntry::Dir(dir) => {
                    if dir.is_empty(txn_id).await? {
                        panic!("an empty filesystem directory is ambiguous");
                        // return Err(unexpected!("an empty filesystem directory is ambiguous"));
                    } else if dir.contains_files(txn_id).await? {
                        DirEntry::File(File::new(dir.clone()))
                    } else {
                        DirEntry::Dir(Self { inner: dir.clone() })
                    }
                }
                txfs::DirEntry::File(block) => panic!(
                    "a transactional Dir should never contain blocks: {:?}",
                    block
                ),
            };

            Ok((name, entry))
        });

        Ok(Box::pin(entries))
    }

    /// Delete any empty entries in this [`Dir`] at `txn_id`.
    pub async fn trim(&self, txn_id: TxnId) -> TCResult<()> {
        let mut to_delete = Vec::new();
        let entries = self.inner.iter(txn_id).await?;

        for (name, entry) in entries {
            if let txfs::DirEntry::Dir(dir) = &*entry {
                if dir.is_empty(txn_id).await? {
                    to_delete.push(name);
                }
            }
        }

        for name in to_delete {
            self.inner.delete(txn_id, (&*name).clone()).await?;
        }

        Ok(())
    }
}

impl<FE: ThreadSafe + for<'a> FileSave<'a>> Dir<FE> {
    /// Commit this [`Dir`] at `txn_id`.
    pub async fn commit(&self, txn_id: TxnId, recursive: bool) {
        self.inner.commit(txn_id, recursive).await
    }

    /// Roll back this [`Dir`] at `txn_id`.
    pub async fn rollback(&self, txn_id: &TxnId, recursive: bool) {
        self.inner.rollback(*txn_id, recursive).await
    }

    /// Finalize this [`Dir`] at `txn_id`.
    pub async fn finalize(&self, txn_id: &TxnId) {
        self.inner.finalize(*txn_id).await
    }
}

impl<FE> fmt::Debug for Dir<FE> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.inner.fmt(f)
    }
}

/// A transactional file
pub struct File<FE, N, B> {
    inner: txfs::Dir<TxnId, FE>,
    name: PhantomData<N>,
    block: PhantomData<B>,
}

impl<FE, N, B> Clone for File<FE, N, B> {
    fn clone(&self) -> Self {
        Self::new(self.inner.clone())
    }
}

impl<FE, N, B> File<FE, N, B> {
    fn new(inner: txfs::Dir<TxnId, FE>) -> Self {
        Self {
            inner,
            name: PhantomData,
            block: PhantomData,
        }
    }

    /// Destructure this [`File`] into its underlying [`freqfs::DirLock`].
    pub fn into_inner(self) -> Inner<FE> {
        self.inner.into_inner()
    }
}

// TODO: there should be a way to avoid calling name.to_string() on every lookup
impl<FE, N, B> File<FE, N, B>
where
    FE: for<'a> FileSave<'a> + AsType<B> + Clone + Send + Sync,
    N: fmt::Display,
    B: FileLoad + GetSize + Clone,
{
    /// Construct an iterator over the name of each block in this [`File`] at `txn_id`.
    pub async fn block_ids(&self, txn_id: TxnId) -> TCResult<impl Iterator<Item = Id>> {
        self.inner
            .file_names(txn_id)
            .map_ok(|names| names.map(|name| name.parse().expect("block ID")))
            .map_err(TCError::from)
            .await
    }

    /// Create a new block at `txn_id` with the given `name` and `contents`.
    pub async fn create_block(
        &self,
        txn_id: TxnId,
        name: N,
        contents: B,
    ) -> TCResult<BlockWrite<FE, B>> {
        let block = self
            .inner
            .create_file(txn_id, name.to_string(), contents)
            .await?;

        block.into_write(txn_id).map_err(TCError::from).await
    }

    /// Delete the block with the given `name` at `txn_id` and return `true` if it was present.
    pub async fn delete_block(&self, txn_id: TxnId, name: &N) -> TCResult<bool> {
        self.inner
            .delete(txn_id, name.to_string())
            .map_err(TCError::from)
            .await
    }

    /// Iterate over the blocks in this [`File`].
    pub async fn iter(
        &self,
        txn_id: TxnId,
    ) -> TCResult<impl Stream<Item = TCResult<(Id, BlockRead<FE, B>)>> + Send + Unpin + '_> {
        self.inner
            .files(txn_id)
            // TODO: avoid this call to parse()
            .map_ok(|blocks| {
                blocks
                    .map_ok(|(name, data)| (name.parse().expect("block ID"), data))
                    .map_err(TCError::from)
            })
            .map_err(TCError::from)
            .await
    }

    /// Lock the block at `name` for reading at `txn_id`.
    pub async fn read_block(&self, txn_id: TxnId, name: &N) -> TCResult<BlockRead<FE, B>> {
        self.inner
            .read_file(txn_id, &name.to_string())
            .map_err(TCError::from)
            .await
    }

    /// Lock the block at `name` for writing at `txn_id`.
    pub async fn write_block(&self, txn_id: TxnId, name: &N) -> TCResult<BlockWrite<FE, B>> {
        self.inner
            .write_file(txn_id, &name.to_string())
            .map_err(TCError::from)
            .await
    }
}

#[async_trait]
impl<FE, N, B> Transact for File<FE, N, B>
where
    FE: for<'a> FileSave<'a> + AsType<B> + Clone + Send + Sync,
    B: FileLoad + GetSize + Clone,
    Self: Send + Sync,
{
    type Commit = ();

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        self.inner.commit(txn_id, true).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.inner.rollback(*txn_id, true).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.inner.finalize(*txn_id).await
    }
}

impl<FE, N, B> fmt::Debug for File<FE, N, B> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "a transactional File with blocks of type {}",
            std::any::type_name::<B>()
        )
    }
}

/// Defines how to load a persistent data structure from the filesystem.
#[async_trait]
pub trait Persist<FE: ThreadSafe>: Sized {
    type Txn: Transaction<FE>;
    type Schema: Clone + Send + Sync;

    /// Create a new instance of [`Self`] from an empty `Store`.
    async fn create(txn_id: TxnId, schema: Self::Schema, store: Dir<FE>) -> TCResult<Self>;

    /// Load a saved instance of [`Self`] from persistent storage.
    /// Should only be invoked at startup time.
    async fn load(txn_id: TxnId, schema: Self::Schema, store: Dir<FE>) -> TCResult<Self>;

    /// Load a saved instance of [`Self`] from persistent storage if present, or create a new one.
    async fn load_or_create(txn_id: TxnId, schema: Self::Schema, store: Dir<FE>) -> TCResult<Self> {
        if store.is_empty(txn_id).await? {
            Self::create(txn_id, schema, store).await
        } else {
            Self::load(txn_id, schema, store).await
        }
    }

    /// Access the filesystem directory backing this persistent data structure.
    fn dir(&self) -> Inner<FE>;
}

/// Copy a base state from another instance, possibly a view.
#[async_trait]
pub trait CopyFrom<FE: ThreadSafe, I>: Persist<FE> {
    /// Copy a new instance of `Self` from an existing instance.
    async fn copy_from(
        txn: &<Self as Persist<FE>>::Txn,
        store: Dir<FE>,
        instance: I,
    ) -> TCResult<Self>;
}

/// Restore a persistent state from a backup.
#[async_trait]
pub trait Restore<FE: ThreadSafe>: Persist<FE> {
    /// Restore this persistent state from a backup.
    async fn restore(&self, txn_id: TxnId, backup: &Self) -> TCResult<()>;
}
