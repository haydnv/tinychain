//! A directory of [`Cluster`]s

use std::fmt;
use std::ops::Deref;

use async_trait::async_trait;
use futures::future::{join_all, FutureExt, TryFutureExt};
use futures::join;
use futures::stream::{FuturesUnordered, StreamExt};
use log::*;

use tc_error::*;
use tc_transact::fs;
use tc_transact::lock::{TxnMapLock, TxnMapLockEntry};
use tc_transact::public::Route;
use tc_transact::{Transact, Transaction, TxnId};
use tc_value::Version as VersionNumber;
use tcgeneric::{label, Label, PathSegment, ThreadSafe};

use crate::{CacheBlock, State, Txn};

use super::{Cluster, Schema};

/// The name of the endpoint which lists the names of each entry in a [`Dir`]
pub const ENTRIES: Label = label("entries");

#[async_trait]
pub trait DirCreate: Sized {
    async fn create_dir(
        &self,
        txn: &Txn,
        name: PathSegment,
        schema: Schema,
    ) -> TCResult<Cluster<Self>>;
}

#[async_trait]
pub trait DirCreateItem<T: DirItem> {
    async fn create_item(
        &self,
        txn: &Txn,
        name: PathSegment,
        schema: Schema,
    ) -> TCResult<Cluster<T>>;
}

/// Defines methods common to any item in a [`Dir`].
#[async_trait]
pub trait DirItem:
    fs::Persist<CacheBlock, Txn = Txn, Schema = ()> + Transact + Clone + Send + Sync
{
    type Schema;
    type Version;

    /// Create a new [`Self::Version`] of this [`DirItem`].
    async fn create_version(
        &self,
        txn: &Txn,
        number: VersionNumber,
        schema: <Self as DirItem>::Schema,
    ) -> TCResult<Self::Version>;
}

/// An entry in a [`Dir`] of [`Cluster`]s
#[derive(Clone)]
pub enum DirEntry<T> {
    Dir(Cluster<Dir<T>>),
    Item(Cluster<T>),
}

/// A commit guard for a [`DirEntry`]
pub enum DirEntryCommit<T: Transact + Clone + Send + Sync + fmt::Debug + 'static> {
    Dir(<Cluster<Dir<T>> as Transact>::Commit),
    Item(<Cluster<T> as Transact>::Commit),
}

#[async_trait]
impl<T: Transact + Clone + Send + Sync + fmt::Debug + 'static> Transact for DirEntry<T> {
    type Commit = DirEntryCommit<T>;

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        debug!("commit {:?} at {}", self, txn_id);

        match self {
            Self::Dir(dir) => dir.commit(txn_id).map(DirEntryCommit::Dir).await,
            Self::Item(item) => item.commit(txn_id).map(DirEntryCommit::Item).await,
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        debug!("roll back {:?} at {}", self, txn_id);

        match self {
            Self::Dir(dir) => dir.rollback(txn_id).await,
            Self::Item(item) => item.rollback(txn_id).await,
        };
    }

    async fn finalize(&self, txn_id: &TxnId) {
        debug!("finalize {:?} at {}", self, txn_id);

        match self {
            Self::Dir(dir) => dir.finalize(txn_id).await,
            Self::Item(item) => item.finalize(txn_id).await,
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for DirEntry<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dir(dir) => dir.fmt(f),
            Self::Item(item) => item.fmt(f),
        }
    }
}

#[derive(Clone)]
pub struct Dir<T> {
    dir: fs::Dir<CacheBlock>,
    contents: TxnMapLock<PathSegment, DirEntry<T>>,
}

impl<T: Clone + fmt::Debug> Dir<T> {
    fn new(txn_id: TxnId, dir: fs::Dir<CacheBlock>) -> TCResult<Self> {
        let contents = TxnMapLock::new(txn_id);

        Ok(Self { dir, contents })
    }

    fn with_contents<C: IntoIterator<Item = (PathSegment, DirEntry<T>)>>(
        txn_id: TxnId,
        dir: fs::Dir<CacheBlock>,
        contents: C,
    ) -> TCResult<Self> {
        let contents = TxnMapLock::with_contents(txn_id, contents);

        Ok(Self { dir, contents })
    }

    /// Recursive synchronous [`Dir`] entry lookup
    pub fn lookup<'a>(
        &self,
        txn_id: TxnId,
        path: &'a [PathSegment],
    ) -> TCResult<Option<(&'a [PathSegment], DirEntry<T>)>> {
        if path.is_empty() {
            return Ok(None);
        }

        match self.contents.try_get(txn_id, &path[0])? {
            Some(entry) => match &*entry {
                DirEntry::Item(item) => Ok(Some((&path[1..], DirEntry::Item(item.clone())))),
                DirEntry::Dir(dir) => dir.lookup(txn_id, &path[1..]).map(Some),
            },
            None => Ok(None),
        }
    }
}

impl<T: fmt::Debug> Dir<T>
where
    DirEntry<T>: Clone,
{
    pub async fn entry(
        &self,
        txn_id: TxnId,
        name: &PathSegment,
    ) -> TCResult<Option<impl Deref<Target = DirEntry<T>>>> {
        self.contents.get(txn_id, name).map_err(TCError::from).await
    }
}

/// Defines a method to create a new subdirectory in a [`Dir`].
#[async_trait]
impl<T: Send + Sync + fmt::Debug> DirCreate for Dir<T>
where
    DirEntry<T>: Clone,
    Cluster<Self>: Clone,
    Self: fs::Persist<CacheBlock, Txn = Txn, Schema = ()> + Route<State> + fmt::Debug,
{
    /// Create a new subdirectory in this [`Dir`].
    async fn create_dir(
        &self,
        txn: &Txn,
        name: PathSegment,
        schema: Schema,
    ) -> TCResult<Cluster<Self>> {
        let txn_id = *txn.id();

        match self.contents.entry(txn_id, name.clone()).await? {
            TxnMapLockEntry::Vacant(entry) => {
                let dir = self.dir.create_dir(txn_id, name).await?;
                let cluster: Cluster<Dir<T>> = fs::Persist::create(txn_id, schema, dir).await?;
                entry.insert(DirEntry::Dir(cluster.clone()));
                Ok(cluster)
            }
            TxnMapLockEntry::Occupied(entry) => match entry.get() {
                DirEntry::Dir(dir) => Ok(dir.clone()),
                DirEntry::Item(_) => Err(bad_request!("there is already an entry at {name}")),
            },
        }
    }
}

/// Defines a method to create a new item in this [`Dir`].
#[async_trait]
impl<T> DirCreateItem<T> for Dir<T>
where
    T: DirItem + Route<State> + fmt::Debug,
    DirEntry<T>: Clone,
{
    /// Create a new item in this [`Dir`].
    async fn create_item(
        &self,
        txn: &Txn,
        name: PathSegment,
        schema: Schema,
    ) -> TCResult<Cluster<T>> {
        debug!("cluster::Dir::create_item {name} with schema {schema:?}");

        let txn_id = *txn.id();

        match self.contents.entry(txn_id, name.clone()).await? {
            TxnMapLockEntry::Vacant(entry) => {
                let dir = self.dir.create_dir(txn_id, name).await?;
                let item: Cluster<T> = fs::Persist::create(txn_id, schema, dir).await?;
                entry.insert(DirEntry::Item(item.clone()));
                Ok(item)
            }
            TxnMapLockEntry::Occupied(entry) => match entry.get() {
                DirEntry::Item(item) => Ok(item.clone()),
                DirEntry::Dir(_) => Err(bad_request!("there is already a directory at {name}")),
            },
        }
    }
}

#[async_trait]
impl<T: Transact + ThreadSafe + Clone + fmt::Debug> Transact for Dir<T>
where
    DirEntry<T>: Clone,
{
    type Commit = ();

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        debug!("commit {:?}", self);

        let (_, (contents, _)) = join!(
            self.dir.commit(txn_id, false),
            self.contents.read_and_commit(txn_id),
        );

        let commits = contents
            .iter()
            .map(|(name, entry)| entry.commit(txn_id).map(|guard| (name.clone(), guard)))
            .collect::<FuturesUnordered<_>>();

        commits.fold((), |(), _| futures::future::ready(())).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        let (_, (contents, _)) = join!(
            self.dir.rollback(*txn_id, false),
            self.contents.read_and_rollback(*txn_id)
        );

        join_all(contents.values().map(|entry| entry.rollback(txn_id))).await;
    }

    async fn finalize(&self, txn_id: &TxnId) {
        if let Some(contents) = self.contents.read_and_finalize(*txn_id) {
            join_all(contents.values().map(|entry| entry.finalize(txn_id))).await;
        }

        self.dir.finalize(*txn_id).await;
    }
}

impl<T> fmt::Debug for Dir<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} directory", std::any::type_name::<T>())
    }
}
