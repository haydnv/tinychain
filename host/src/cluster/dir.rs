use std::collections::hash_map::{self, HashMap};
use std::fmt;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::future::{join_all, FutureExt, TryFutureExt};

use tc_error::*;
use tc_transact::fs::{BlockData, Persist};
use tc_transact::lock::map::*;
use tc_transact::{Transact, Transaction, TxnId};
use tc_value::{Link, Version as VersionNumber};
use tcgeneric::PathSegment;

use crate::fs;
use crate::state::State;
use crate::txn::Txn;

use super::library::Version;
use super::{Cluster, Replica};

#[async_trait]
pub trait DirItem:
    Persist<fs::Dir, Txn = Txn, Schema = (), Store = fs::File<VersionNumber, Version>>
    + Transact
    + Send
    + Sync
{
    async fn create_version(
        &self,
        txn_id: TxnId,
        number: VersionNumber,
        version: State,
    ) -> TCResult<()>;
}

#[derive(Clone)]
pub enum DirEntry<T> {
    Dir(Cluster<Dir<T>>),
    Item(Cluster<T>),
}

impl<T> fmt::Display for DirEntry<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dir(dir) => dir.fmt(f),
            Self::Item(item) => item.fmt(f),
        }
    }
}

pub enum DirEntryCommitGuard<T>
where
    T: Clone + Transact + Send + Sync,
{
    Dir(<Cluster<Dir<T>> as Transact>::Commit),
    Item(<Cluster<T> as Transact>::Commit),
}

#[async_trait]
impl<T> Transact for DirEntry<T>
where
    T: Transact + Clone + Send + Sync,
{
    type Commit = DirEntryCommitGuard<T>;

    async fn commit(&self, txn_id: &TxnId) -> Self::Commit {
        match self {
            Self::Dir(dir) => dir.commit(txn_id).map(DirEntryCommitGuard::Dir).await,
            Self::Item(item) => item.commit(txn_id).map(DirEntryCommitGuard::Item).await,
        }
    }

    async fn finalize(&self, txn_id: &TxnId) {
        match self {
            Self::Dir(dir) => dir.finalize(txn_id).await,
            Self::Item(item) => item.finalize(txn_id).await,
        }
    }
}

enum Delta {
    Create,
}

#[derive(Clone)]
pub struct Dir<T> {
    cache: freqfs::DirLock<fs::CacheBlock>,
    contents: TxnMapLock<PathSegment, DirEntry<T>>,
    deltas: Arc<Mutex<HashMap<TxnId, HashMap<PathSegment, Delta>>>>,
}

impl<T> Dir<T>
where
    DirEntry<T>: Clone,
{
    pub async fn entry(&self, txn_id: TxnId, name: &PathSegment) -> TCResult<Option<DirEntry<T>>> {
        self.contents
            .read(txn_id)
            .map_ok(|contents| contents.get(name))
            .await
    }

    pub(super) fn contents(
        &self,
        txn_id: TxnId,
    ) -> TCResult<TxnMapLockReadGuard<PathSegment, DirEntry<T>>> {
        self.contents.try_read(txn_id)
    }

    pub(super) async fn create_dir(
        &self,
        txn: &Txn,
        link: &Link,
        name: PathSegment,
    ) -> TCResult<()> {
        let mut contents = self.contents.write(*txn.id()).await?;
        let mut cache = self.cache.write().await;

        let dir = cache.create_dir(name.to_string()).map_err(fs::io_err)?;
        let dir = fs::Dir::new(dir);

        let dir = Self::create(txn, (), dir).await?;
        let dir = Cluster::with_state(link.clone().append(name.clone()), dir);
        contents.insert(name.clone(), DirEntry::Dir(dir));

        self.record_delta(*txn.id(), name, Delta::Create).await;

        Ok(())
    }

    async fn record_delta(&self, txn_id: TxnId, name: PathSegment, delta: Delta) {
        let mut deltas = self.deltas.lock().expect("dir deltas");
        match deltas.entry(txn_id) {
            hash_map::Entry::Occupied(mut entry) => {
                entry.get_mut().insert(name, delta);
            }
            hash_map::Entry::Vacant(entry) => {
                let mut deltas = HashMap::new();
                deltas.insert(name, delta);
                entry.insert(deltas);
            }
        };
    }
}

impl<T> Dir<T>
where
    T: DirItem,
    DirEntry<T>: Clone,
{
    pub(super) async fn create_item(
        &self,
        txn: &Txn,
        link: &Link,
        name: PathSegment,
        number: VersionNumber,
        state: State,
    ) -> TCResult<()> {
        let mut contents = self.contents.write(*txn.id()).await?;
        let mut cache = self.cache.write().await;

        let file = cache
            .create_dir(format!("{}.{}", name, Version::ext()))
            .map_err(fs::io_err)
            .and_then(fs::File::new)?;

        let lib = T::create(txn, (), file).await?;

        let lib = Cluster::with_state(link.clone().append(name.clone()), lib);
        lib.state().create_version(*txn.id(), number, state).await?;
        contents.insert(name.clone(), DirEntry::Item(lib));

        self.record_delta(*txn.id(), name, Delta::Create).await;

        Ok(())
    }
}

#[async_trait]
impl<T> Replica for Dir<T>
where
    T: Replica + Transact + Clone + Send + Sync,
{
    async fn replicate(&self, _txn: &Txn, _source: &Link) -> TCResult<()> {
        Err(TCError::not_implemented("cluster::Dir::replicate"))
    }
}

#[async_trait]
impl<T> Transact for Dir<T>
where
    T: Transact + Clone + Send + Sync,
    DirEntry<T>: Clone,
{
    type Commit = TxnMapLockCommitGuard<PathSegment, DirEntry<T>>;

    async fn commit(&self, txn_id: &TxnId) -> Self::Commit {
        let guard = self.contents.commit(txn_id).await;

        if let Some(deltas) = {
            let mut deltas = self.deltas.lock().expect("dir commit deltas");
            let txn_deltas = deltas.remove(txn_id);
            txn_deltas
        } {
            let commits = deltas
                .into_iter()
                .map(|(name, _delta)| guard.get(name).expect("dir entry"))
                .map(|entry| async move { entry.commit(txn_id).await });

            join_all(commits).await;
        }

        guard
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.contents.finalize(txn_id).await
    }
}

#[async_trait]
impl<T> Persist<fs::Dir> for Dir<T> {
    type Schema = ();
    type Store = fs::Dir;
    type Txn = Txn;

    async fn create(_txn: &Self::Txn, _schema: Self::Schema, dir: Self::Store) -> TCResult<Self> {
        Ok(Self {
            cache: dir.into_inner(),
            contents: TxnMapLock::new("service directory"),
            deltas: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    async fn load(txn: &Self::Txn, schema: Self::Schema, dir: Self::Store) -> TCResult<Self> {
        // TODO: read existing contents
        Self::create(txn, schema, dir).await
    }
}

impl<T> fmt::Display for Dir<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} directory", std::any::type_name::<T>())
    }
}
