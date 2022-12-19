use std::collections::HashMap;
use std::convert::TryFrom;
use std::fmt;

use async_trait::async_trait;
use futures::future::{FutureExt, TryFutureExt};
use log::debug;
use safecast::CastInto;

use tc_error::*;
use tc_transact::fs::Persist;
use tc_transact::lock::map::*;
use tc_transact::{Transact, Transaction, TxnId};
use tc_value::{Link, Value, Version as VersionNumber};
use tcgeneric::{label, Label, Map, PathSegment};

use crate::chain::BlockChain;
use crate::fs;
use crate::route::Route;
use crate::state::State;
use crate::txn::Txn;

use super::{Class, Cluster, Library, Replica};

/// The type of file stored in a [`library`] directory
pub type File = fs::File<VersionNumber, super::library::Version>;

/// The name of the endpoint which lists the names of each entry in a [`Dir`]
pub const ENTRIES: Label = label("entries");

#[async_trait]
pub trait DirCreate: Sized {
    async fn create_dir(&self, txn: &Txn, name: PathSegment, link: Link)
        -> TCResult<Cluster<Self>>;
}

#[async_trait]
pub trait DirCreateItem<T: DirItem> {
    async fn create_item(
        &self,
        txn: &Txn,
        name: PathSegment,
        link: Link,
    ) -> TCResult<Cluster<BlockChain<T>>>;
}

#[async_trait]
pub trait DirItem:
    Persist<fs::Dir, Txn = Txn, Schema = ()> + Transact + Clone + Send + Sync
{
    type Version;

    async fn create_version(
        &self,
        txn_id: TxnId,
        number: VersionNumber,
        version: Self::Version,
    ) -> TCResult<()>;
}

#[derive(Clone)]
pub enum DirEntry<T> {
    Dir(Cluster<Dir<T>>),
    Item(Cluster<BlockChain<T>>),
}

pub enum DirEntryCommitGuard<T>
where
    T: Clone + Transact + Send + Sync,
{
    Dir(<Cluster<Dir<T>> as Transact>::Commit),
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

impl<T> fmt::Display for DirEntry<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dir(dir) => dir.fmt(f),
            Self::Item(item) => item.fmt(f),
        }
    }
}

#[derive(Clone)]
pub struct Dir<T> {
    cache: freqfs::DirLock<fs::CacheBlock>,
    contents: TxnMapLock<PathSegment, DirEntry<T>>,
}

impl<T: Clone> Dir<T> {
    pub fn lookup<'a>(
        &self,
        txn_id: TxnId,
        path: &'a [PathSegment],
    ) -> TCResult<Option<(&'a [PathSegment], DirEntry<T>)>> {
        if path.is_empty() {
            return Ok(None);
        }

        // IMPORTANT! Only use synchronous lock acquisition!
        // async lock acquisition here could cause deadlocks
        // and make services in this `Dir` impossible to update
        let contents = self.contents.try_read(txn_id)?;
        match contents.get(&path[0]) {
            Some(DirEntry::Item(item)) => Ok(Some((&path[1..], DirEntry::Item(item)))),
            Some(DirEntry::Dir(dir)) => dir.lookup(txn_id, &path[1..]).map(Some),
            None => Ok(None),
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
impl<T: Send + Sync> DirCreate for Dir<T>
where
    DirEntry<T>: Clone,
    Cluster<Self>: Clone,
    Self: Persist<fs::Dir, Txn = Txn, Schema = Link> + Route + fmt::Display,
{
    async fn create_dir(
        &self,
        txn: &Txn,
        name: PathSegment,
        link: Link,
    ) -> TCResult<Cluster<Self>> {
        if link.path().last() != Some(&name) {
            return Err(TCError::unsupported(format!(
                "cluster directory link for {} must end with {} (found {})",
                name, name, link
            )));
        }

        let mut contents = self.contents.write(*txn.id()).await?;

        if contents.contains_key(&name) {
            return Err(TCError::bad_request(
                "there is already a directory at",
                name,
            ));
        }

        let mut cache = self.cache.write().await;

        let dir = cache.create_dir(name.to_string()).map_err(fs::io_err)?;
        let dir = fs::Dir::new(dir);

        let self_link = txn.link(link.path().clone());

        let dir = Self::create(txn, link.clone(), fs::Store::from(dir)).await?;
        let dir = Cluster::with_state(self_link, link, dir);

        contents.insert(name.clone(), DirEntry::Dir(dir.clone()));

        Ok(dir)
    }
}

#[async_trait]
impl<T: DirItem + Route + fmt::Display> DirCreateItem<T> for Dir<T>
where
    DirEntry<T>: Clone,
{
    async fn create_item(
        &self,
        txn: &Txn,
        name: PathSegment,
        link: Link,
    ) -> TCResult<Cluster<BlockChain<T>>> {
        if link.path().last() != Some(&name) {
            return Err(TCError::unsupported(format!(
                "cluster link for {} must end with {} (found {})",
                name, name, link
            )));
        }

        let mut contents = self.contents.write(*txn.id()).await?;

        if contents.contains_key(&name) {
            return Err(TCError::bad_request("there is already a cluster at", name));
        }

        let cluster = link;
        let self_link = txn.link(cluster.path().clone());

        let store = {
            let dir = fs::Dir::new(self.cache.clone());
            let dir = tc_transact::fs::Dir::write(&dir, *txn.id()).await?;
            dir.create_store(name.clone())
        };

        let item = BlockChain::create(txn, (), store).await?;
        let item = Cluster::with_state(self_link, cluster, item);
        contents.insert(name.clone(), DirEntry::Item(item.clone()));

        Ok(item)
    }
}

#[async_trait]
impl<T: DirItem + Route + fmt::Display> Replica for Dir<T>
where
    BlockChain<T>: Replica,
    DirEntry<T>: Clone,
    Cluster<Self>: Clone,
    Self: Persist<fs::Dir, Txn = Txn, Schema = Link> + Route + fmt::Display,
{
    async fn state(&self, txn_id: TxnId) -> TCResult<State> {
        let contents = self.contents.read(txn_id).await?;
        let mut state = Map::<State>::new();
        for (name, entry) in contents.iter() {
            let class = match entry {
                DirEntry::Dir(_) => true.into(),
                DirEntry::Item(_) => false.into(),
            };

            state.insert(name.clone(), class);
        }

        debug!("directory state to replicate is {}", state);

        Ok(State::Map(state.cast_into()))
    }

    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()> {
        let entries = txn
            .get(source.clone().append(ENTRIES), Value::default())
            .await?;

        let entries = entries.try_into_map(|s| {
            TCError::bad_gateway(format!("{} listed invalid directory entries {}", source, s))
        })?;

        debug!("directory entries to replicate are {}", entries);

        let entries = entries
            .into_iter()
            .map(|(name, is_dir)| bool::try_from(is_dir).map(|is_dir| (name, is_dir)))
            .collect::<TCResult<Map<bool>>>()?;

        for (name, is_dir) in entries {
            let link = source.clone().append(name.clone());

            if let Some(entry) = self.entry(*txn.id(), &name).await? {
                match entry {
                    DirEntry::Dir(dir) => dir.lead_and_add_replica(txn.clone()).await?,
                    DirEntry::Item(item) => item.lead_and_add_replica(txn.clone()).await?,
                };
            } else if is_dir {
                let dir = self.create_dir(txn, name, link).await?;
                dir.lead_and_add_replica(txn.clone()).await?;
            } else {
                let item = self.create_item(txn, name, link).await?;
                item.lead_and_add_replica(txn.clone()).await?;
            }
        }

        Ok(())
    }
}

#[async_trait]
impl<T: Transact + Clone + Send + Sync> Transact for Dir<T>
where
    DirEntry<T>: Clone,
{
    type Commit = TxnMapLockCommitGuard<PathSegment, DirEntry<T>>;

    async fn commit(&self, txn_id: &TxnId) -> Self::Commit {
        debug!("commit {}", self);

        self.contents.commit(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.contents.finalize(txn_id).await
    }
}

#[async_trait]
impl Persist<fs::Dir> for Dir<Class> {
    type Txn = Txn;
    type Schema = Link;

    async fn create(_txn: &Txn, _schema: Link, store: fs::Store) -> TCResult<Self> {
        let dir = fs::Dir::try_from(store)?;

        Ok(Self {
            cache: tc_transact::fs::Dir::into_inner(dir),
            contents: TxnMapLock::new("class directory"),
        })
    }

    async fn load(txn: &Txn, _link: Link, store: fs::Store) -> TCResult<Self> {
        let dir = fs::Dir::try_from(store)?;

        let lock = tc_transact::fs::Dir::read(&dir, *txn.id()).await?;
        let contents = HashMap::with_capacity(tc_transact::fs::DirRead::len(&lock));

        for _entry in lock.iter() {
            return Err(TCError::not_implemented(
                "load a cluster::Dir of class sets",
            ));
        }

        Ok(Self {
            cache: tc_transact::fs::Dir::into_inner(dir),
            contents: TxnMapLock::with_contents("class directory", contents),
        })
    }

    fn dir(&self) -> <fs::Dir as tc_transact::fs::Dir>::Inner {
        self.cache.clone()
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
            contents: TxnMapLock::new("library directory"),
        })
    }

    async fn load(txn: &Txn, link: Link, store: fs::Store) -> TCResult<Self> {
        let txn_id = *txn.id();
        let dir = fs::Dir::try_from(store)?;
        let lock = tc_transact::fs::Dir::read(&dir, txn_id).await?;
        let mut contents = HashMap::new();

        for (name, entry) in lock.iter() {
            let link = link.clone().append(name.clone());

            match entry {
                fs::DirEntry::Dir(dir) => {
                    let dir = Cluster::load(txn, link, dir.into()).await?;
                    contents.insert(name.clone(), DirEntry::Dir(dir));
                }
                fs::DirEntry::File(file) => match file {
                    fs::FileEntry::Library(file) => {
                        let lib = Cluster::load(txn, link, file.into()).await?;
                        contents.insert(name.clone(), DirEntry::Item(lib));
                    }
                    file => {
                        return Err(TCError::internal(format!(
                            "{} is in the library directory but {} is not a library",
                            name, file
                        )))
                    }
                },
            }
        }

        Ok(Self {
            cache: tc_transact::fs::Dir::into_inner(dir),
            contents: TxnMapLock::with_contents("service directory", contents),
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
