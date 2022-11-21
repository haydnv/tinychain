use std::collections::hash_map::{self, HashMap};
use std::convert::TryFrom;
use std::fmt;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::future::{join_all, FutureExt, TryFutureExt};
use safecast::TryCastFrom;

use tc_error::*;
use tc_transact::fs::{BlockData, Persist};
use tc_transact::lock::map::*;
use tc_transact::{Transact, Transaction, TxnId};
use tc_value::{Link, Version as VersionNumber};
use tcgeneric::PathSegment;

use crate::chain::{BlockChain, ChainInstance};
use crate::fs;
use crate::route::Route;
use crate::state::State;
use crate::txn::Txn;

use super::{Cluster, Library, Replica};

pub type File = fs::File<VersionNumber, super::library::Version>;

#[async_trait]
pub trait DirCreate: Sized {
    async fn create_dir(
        &self,
        txn: &Txn,
        link: Link,
        name: PathSegment,
    ) -> TCResult<Cluster<BlockChain<Self>>>;
}

#[async_trait]
pub trait DirCreateItem<T: DirItem> {
    async fn create_item(
        &self,
        txn: &Txn,
        link: Link,
        name: PathSegment,
    ) -> TCResult<Cluster<BlockChain<T>>>;
}

#[async_trait]
pub trait DirItem:
    Persist<fs::Dir, Txn = Txn, Schema = ()> + Transact + Clone + Send + Sync
{
    type Version: BlockData + TryCastFrom<State>;

    async fn create_version(
        &self,
        txn_id: TxnId,
        number: VersionNumber,
        version: Self::Version,
    ) -> TCResult<()>;
}

#[derive(Clone)]
pub enum DirEntry<T> {
    Dir(Cluster<BlockChain<Dir<T>>>),
    Item(Cluster<BlockChain<T>>),
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
    Dir(<Cluster<BlockChain<Dir<T>>> as Transact>::Commit),
    Item(<Cluster<BlockChain<T>> as Transact>::Commit),
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

impl<T> Dir<T> {
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
    T: Clone,
    BlockChain<Dir<T>>: ChainInstance<Dir<T>>,
{
    pub fn lookup<'a>(
        &self,
        txn_id: TxnId,
        path: &'a [PathSegment],
    ) -> TCResult<(&'a [PathSegment], DirEntry<T>)> {
        assert!(!path.is_empty());

        // IMPORTANT! Only use synchronous lock acquisition!
        // async lock acquisition here could cause deadlocks
        // and make services in this `Dir` impossible to update
        let contents = self.contents.try_read(txn_id)?;
        match contents.get(&path[0]) {
            Some(DirEntry::Item(item)) => Ok((&path[1..], DirEntry::Item(item))),
            Some(DirEntry::Dir(dir)) => dir.lookup(txn_id, &path[1..]),
            None => Err(TCError::not_found(&path[0])),
        }
    }
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
}

#[async_trait]
impl<T> DirCreate for Dir<T>
where
    T: Send + Sync,
    DirEntry<T>: Clone,
    BlockChain<Self>: Persist<fs::Dir, Txn = Txn, Schema = Link>,
    Cluster<BlockChain<Self>>: Clone,
    Self: Route + fmt::Display,
{
    async fn create_dir(
        &self,
        txn: &Txn,
        link: Link,
        name: PathSegment,
    ) -> TCResult<Cluster<BlockChain<Self>>> {
        if link.path().last() != Some(&name) {
            return Err(TCError::unsupported(format!(
                "cluster directory link for {} must end with {} (found {})",
                name, name, link
            )));
        }

        let mut contents = self.contents.write(*txn.id()).await?;
        let mut cache = self.cache.write().await;

        let dir = cache.create_dir(name.to_string()).map_err(fs::io_err)?;
        let dir = fs::Dir::new(dir);

        let self_link = txn.link(link.path().clone());

        let dir = BlockChain::create(txn, link.clone(), fs::Store::from(dir)).await?;
        let dir = Cluster::with_state(self_link, link, dir);

        contents.insert(name.clone(), DirEntry::Dir(dir.clone()));

        self.record_delta(*txn.id(), name, Delta::Create).await;

        Ok(dir)
    }
}

#[async_trait]
impl<T> DirCreateItem<T> for Dir<T>
where
    T: DirItem + Route + fmt::Display,
    DirEntry<T>: Clone,
{
    async fn create_item(
        &self,
        txn: &Txn,
        link: Link,
        name: PathSegment,
    ) -> TCResult<Cluster<BlockChain<T>>> {
        let mut contents = self.contents.write(*txn.id()).await?;
        let mut cache = self.cache.write().await;

        let dir = cache
            .create_dir(format!("{}.{}", name, <T::Version as BlockData>::ext()))
            .map(fs::Dir::new)
            .map_err(fs::io_err)?;

        let cluster = link.clone().append(name.clone());
        let self_link = txn.link(cluster.path().clone());

        let item = BlockChain::create(txn, (), fs::Store::from(dir)).await?;
        let item = Cluster::with_state(self_link, cluster, item);
        contents.insert(name.clone(), DirEntry::Item(item.clone()));

        self.record_delta(*txn.id(), name, Delta::Create).await;

        Ok(item)
    }
}

#[async_trait]
impl<T> Replica for Dir<T>
where
    T: Replica + Transact + Clone + Send + Sync,
{
    async fn state(&self, _txn_id: TxnId) -> TCResult<State> {
        Err(TCError::not_implemented("cluster::Dir::state"))
    }

    async fn replicate(&self, _txn: &Txn, _source: Link) -> TCResult<()> {
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
impl Persist<fs::Dir> for Dir<Library> {
    type Txn = Txn;
    type Schema = Link;

    async fn create(_txn: &Txn, _schema: Link, store: fs::Store) -> TCResult<Self> {
        let dir = fs::Dir::try_from(store)?;

        Ok(Self {
            cache: tc_transact::fs::Dir::into_inner(dir),
            contents: TxnMapLock::new("service directory"),
            deltas: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    async fn load(txn: &Txn, _link: Link, store: fs::Store) -> TCResult<Self> {
        let txn_id = *txn.id();
        let dir = fs::Dir::try_from(store)?;
        let _lock = tc_transact::fs::Dir::read(&dir, txn_id).await?;
        let contents = HashMap::new();

        // TODO: fill in `contents` by iterating over the entries in `lock`

        Ok(Self {
            cache: tc_transact::fs::Dir::into_inner(dir),
            contents: TxnMapLock::with_contents("service directory", contents),
            deltas: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    fn dir(&self) -> <fs::Dir as tc_transact::fs::Dir>::Inner {
        self.cache.clone()
    }
}

impl<T> fmt::Display for Dir<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} directory", std::any::type_name::<T>())
    }
}
