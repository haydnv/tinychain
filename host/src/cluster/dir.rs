use std::collections::BTreeMap;
use std::convert::TryFrom;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::future::{join_all, try_join_all, FutureExt, TryFutureExt};
use futures::stream::{FuturesUnordered, StreamExt};
use log::{debug, info};
use safecast::CastInto;

use tc_error::*;
use tc_transact::fs::Persist;
use tc_transact::lock::{TxnLock, TxnLockError};
use tc_transact::{Transact, Transaction, TxnId};
use tc_value::{Link, Version as VersionNumber};
use tcgeneric::{label, Label, Map, PathSegment};

use crate::chain::{BlockChain, Recover};
use crate::fs;
use crate::route::Route;
use crate::state::State;
use crate::txn::{Actor, Txn};

use super::{Class, Cluster, Library, Replica, Schema, Service, REPLICAS};

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
    Persist<fs::Dir, Txn = Txn, Schema = Arc<Actor>> + Transact + Clone + Send + Sync
{
    type Schema;
    type Version;

    async fn create_version(
        &self,
        txn: &Txn,
        number: VersionNumber,
        schema: <Self as DirItem>::Schema,
    ) -> TCResult<Self::Version>;
}

#[derive(Clone)]
pub enum DirEntry<T> {
    Dir(Cluster<Dir<T>>),
    Item(Cluster<BlockChain<T>>),
}

/// A commit guard for a [`DirEntry`]
pub enum DirEntryCommit<T: Transact + Clone + Send + Sync + 'static> {
    Dir(<Cluster<Dir<T>> as Transact>::Commit),
    Item(<Cluster<BlockChain<T>> as Transact>::Commit),
}

#[async_trait]
impl<T: Transact + Clone + Send + Sync + 'static> Transact for DirEntry<T> {
    type Commit = DirEntryCommit<T>;

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        debug!("commit {} at {}", self, txn_id);

        match self {
            Self::Dir(dir) => dir.commit(txn_id).map(DirEntryCommit::Dir).await,
            Self::Item(item) => item.commit(txn_id).map(DirEntryCommit::Item).await,
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        debug!("roll back {} at {}", self, txn_id);

        match self {
            Self::Dir(dir) => dir.rollback(txn_id).await,
            Self::Item(item) => item.rollback(txn_id).await,
        };
    }

    async fn finalize(&self, txn_id: &TxnId) {
        debug!("finalize {} at {}", self, txn_id);

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
    schema: Schema,
    cache: freqfs::DirLock<fs::CacheBlock>,
    contents: TxnLock<BTreeMap<PathSegment, DirEntry<T>>>,
}

impl<T> Dir<T> {
    /// Borrow the [`Schema`] of this [`Dir`]
    pub fn schema(&self) -> &Schema {
        &self.schema
    }
}

impl<T: Clone> Dir<T> {
    fn new(txn_id: TxnId, schema: Schema, store: fs::Store) -> TCResult<Self> {
        let dir = fs::Dir::try_from(store)?;

        let lock = tc_transact::fs::Dir::try_read(&dir, txn_id)?;

        if !tc_transact::fs::DirRead::is_empty(&lock) {
            return Err(TCError::unsupported(
                "cannot create a cluster directory from a non-empty filesystem directory",
            ));
        }

        let cache = tc_transact::fs::Dir::into_inner(dir);

        let contents = TxnLock::new(
            format!("dir at {}{}", schema.host, schema.path),
            txn_id,
            BTreeMap::new(),
        );

        Ok(Self {
            schema,
            cache,
            contents,
        })
    }

    fn with_contents(
        txn_id: TxnId,
        schema: Schema,
        dir: fs::Dir,
        contents: BTreeMap<PathSegment, DirEntry<T>>,
    ) -> TCResult<Self> {
        let cache = tc_transact::fs::Dir::into_inner(dir);

        let contents = TxnLock::new(
            format!("dir at {}{}", schema.host, schema.path),
            txn_id,
            contents,
        );

        Ok(Self {
            schema,
            cache,
            contents,
        })
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

        // IMPORTANT! Only use synchronous lock acquisition!
        // async lock acquisition here could cause deadlocks
        // and make services in this `Dir` impossible to update

        const SLEEP: Duration = Duration::from_millis(100);
        const MAX_RETRIES: usize = 3;

        let mut num_retries = 0;
        let contents = loop {
            match self.contents.try_read(txn_id) {
                Err(TxnLockError::WouldBlock) if num_retries < MAX_RETRIES => {
                    futures::executor::block_on(tokio::time::sleep(SLEEP));
                    num_retries += 1;
                }
                result => break result,
            }
        }?;

        match contents.get(&path[0]) {
            Some(DirEntry::Item(item)) => Ok(Some((&path[1..], DirEntry::Item(item.clone())))),
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
            .map_ok(|contents| contents.get(name).cloned())
            .map_err(TCError::from)
            .await
    }
}

#[async_trait]
impl<T: Send + Sync> DirCreate for Dir<T>
where
    DirEntry<T>: Clone,
    Cluster<Self>: Clone,
    Self: Persist<fs::Dir, Txn = Txn, Schema = Schema> + Route + fmt::Display,
{
    async fn create_dir(
        &self,
        txn: &Txn,
        name: PathSegment,
        link: Link,
    ) -> TCResult<Cluster<Self>> {
        let schema = entry_schema(txn, &self.schema.actor, &name, link).await?;

        let txn_id = *txn.id();
        let mut contents = self.contents.write(txn_id).await?;

        if contents.contains_key(&name) {
            return Err(TCError::bad_request(
                "there is already a directory at",
                name,
            ));
        }

        let mut cache = self.cache.write().await;

        let dir = cache.create_dir(name.to_string()).map_err(fs::io_err)?;
        let dir = fs::Dir::new(dir, txn_id);

        let dir = Self::create(txn_id, schema.clone(), fs::Store::from(dir))?;

        let dir = Cluster::with_state(schema, txn_id, dir);

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
        debug!("cluster::Dir::create_item {} at {}", name, link);

        let schema = entry_schema(txn, &self.schema.actor, &name, link).await?;

        let txn_id = *txn.id();
        let mut contents = self.contents.write(txn_id).await?;

        if contents.contains_key(&name) {
            return Err(TCError::bad_request("there is already a cluster at", name));
        }

        let store = {
            let dir = fs::Dir::new(self.cache.clone(), txn_id);
            let dir = tc_transact::fs::Dir::write(&dir, txn_id).await?;
            dir.create_store(name.clone())
        };

        let item = BlockChain::create(txn_id, schema.actor.clone(), store)?;
        let item = Cluster::with_state(schema, txn_id, item);
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
    Self: Persist<fs::Dir, Txn = Txn, Schema = Schema> + Route + fmt::Display,
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
        info!("replicate {} from {}", self, source);

        let params = Map::one(label("add"), txn.host().clone().into());

        let entries = txn
            .post(source.clone().append(REPLICAS), State::Map(params))
            .await?;

        let entries = entries.try_into_map(|s| {
            TCError::bad_gateway(format!("{} listed invalid directory entries {}", source, s))
        })?;

        debug!("directory entries to replicate are {}", entries);

        let entries = entries
            .into_iter()
            .map(|(name, is_dir)| bool::try_from(is_dir).map(|is_dir| (name, is_dir)))
            .collect::<TCResult<Map<bool>>>()?;

        let txn_id = *txn.id();
        for (name, is_dir) in entries {
            let link = source.clone().append(name.clone());

            if let Some(entry) = self.entry(txn_id, &name).await? {
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
impl<T: Transact + Clone + Send + Sync + 'static> Transact for Dir<T>
where
    DirEntry<T>: Clone,
{
    type Commit = Option<Map<DirEntryCommit<T>>>;

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        debug!("commit {}", self);

        if let Some(contents) = self.contents.commit(txn_id).await {
            let commits = contents
                .iter()
                .map(|(name, entry)| entry.commit(txn_id).map(|guard| (name.clone(), guard)))
                .collect::<FuturesUnordered<_>>();

            Some(commits.collect().await)
        } else {
            None
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        if let Some(contents) = self.contents.rollback(txn_id).await {
            join_all(contents.values().map(|entry| entry.rollback(txn_id))).await;
        }
    }

    async fn finalize(&self, txn_id: &TxnId) {
        if let Some(contents) = self.contents.finalize(txn_id).await {
            join_all(contents.values().map(|entry| entry.finalize(txn_id))).await;
        }
    }
}

#[async_trait]
impl<T: Send + Sync + 'static> Recover for Dir<T>
where
    DirEntry<T>: Clone,
    Cluster<BlockChain<T>>: Recover,
{
    async fn recover(&self, txn: &Txn) -> TCResult<()> {
        let contents = self.contents.read(*txn.id()).await?;
        let recovery = contents.values().map(|entry| match entry {
            DirEntry::Dir(dir) => dir.recover(txn),
            DirEntry::Item(item) => item.recover(txn),
        });

        try_join_all(recovery).await?;

        Ok(())
    }
}

impl Persist<fs::Dir> for Dir<Class> {
    type Txn = Txn;
    type Schema = Schema;

    fn create(txn_id: TxnId, schema: Self::Schema, store: fs::Store) -> TCResult<Self> {
        Self::new(txn_id, schema, store)
    }

    fn load(txn_id: TxnId, schema: Self::Schema, store: fs::Store) -> TCResult<Self> {
        let dir = fs::Dir::try_from(store)?;

        let lock = tc_transact::fs::Dir::try_write(&dir, txn_id)?;

        let mut contents = BTreeMap::new();
        for (name, entry) in lock.iter() {
            let schema = schema.extend(name.clone());

            let entry = match entry {
                fs::DirEntry::Dir(dir) => {
                    let is_chain = {
                        let cache = tc_transact::fs::Dir::into_inner(dir.clone());
                        let contents = cache.try_read().expect("cache read");

                        if contents.is_empty() {
                            return Err(TCError::internal(format!(
                                "an empty directory at {} is ambiguous",
                                schema.path
                            )));
                        }

                        contents.contains(&*crate::chain::HISTORY)
                    };

                    if is_chain {
                        Cluster::load(txn_id, schema, dir.clone().into()).map(DirEntry::Item)
                    } else {
                        Cluster::load(txn_id, schema, dir.clone().into()).map(DirEntry::Dir)
                    }
                }
                file => Err(TCError::internal(format!(
                    "invalid Class dir entry: {}",
                    file
                ))),
            }?;

            contents.insert(name.clone(), entry);
        }

        Self::with_contents(txn_id, schema, dir, contents)
    }

    fn dir(&self) -> <fs::Dir as tc_transact::fs::Dir>::Inner {
        self.cache.clone()
    }
}

impl Persist<fs::Dir> for Dir<Library> {
    type Txn = Txn;
    type Schema = Schema;

    fn create(txn_id: TxnId, schema: Self::Schema, store: fs::Store) -> TCResult<Self> {
        Self::new(txn_id, schema, store)
    }

    fn load(txn_id: TxnId, schema: Self::Schema, store: fs::Store) -> TCResult<Self> {
        let dir = fs::Dir::try_from(store)?;
        let lock = tc_transact::fs::Dir::try_read(&dir, txn_id)?;
        let mut contents = BTreeMap::new();

        for (name, entry) in lock.iter() {
            let schema = schema.extend(name.clone());

            match entry {
                fs::DirEntry::Dir(dir) => {
                    let dir = Cluster::load(txn_id, schema, dir.clone().into())?;
                    contents.insert(name.clone(), DirEntry::Dir(dir));
                }
                fs::DirEntry::File(file) => match file {
                    fs::FileEntry::Library(file) => {
                        let lib = Cluster::load(txn_id, schema, file.clone().into())?;
                        contents.insert(name.clone(), DirEntry::Item(lib));
                    }
                    file => {
                        return Err(TCError::internal(format!(
                            "{} is in the library directory but {} is not a library",
                            name, file
                        )))
                    }
                },
            };
        }

        Self::with_contents(txn_id, schema, dir, contents)
    }

    fn dir(&self) -> <fs::Dir as tc_transact::fs::Dir>::Inner {
        self.cache.clone()
    }
}

impl Persist<fs::Dir> for Dir<Service> {
    type Txn = Txn;
    type Schema = Schema;

    fn create(txn_id: TxnId, schema: Self::Schema, store: fs::Store) -> TCResult<Self> {
        Self::new(txn_id, schema, store)
    }

    fn load(txn_id: TxnId, schema: Self::Schema, store: fs::Store) -> TCResult<Self> {
        let dir = fs::Dir::try_from(store)?;

        let lock = tc_transact::fs::Dir::try_read(&dir, txn_id)?;
        let mut contents = BTreeMap::new();

        for (name, entry) in lock.iter() {
            let schema = schema.extend(name.clone());

            let entry = match entry {
                fs::DirEntry::File(file) => Err(TCError::internal(format!(
                    "invalid Service directory entry: {}",
                    file
                ))),
                fs::DirEntry::Dir(dir) => {
                    let is_service = {
                        let lock = tc_transact::fs::Dir::try_read(dir, txn_id)?;
                        tc_transact::fs::DirRead::contains(&lock, &super::service::SCHEMA.into())
                    };

                    let store = dir.clone().into();
                    if is_service {
                        Cluster::load(txn_id, schema, store).map(DirEntry::Item)
                    } else {
                        Cluster::load(txn_id, schema, store).map(DirEntry::Dir)
                    }
                }
            }?;

            contents.insert(name.clone(), entry);
        }

        Self::with_contents(txn_id, schema, dir, contents)
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

async fn entry_schema(
    txn: &Txn,
    actor: &Arc<Actor>,
    name: &PathSegment,
    link: Link,
) -> TCResult<Schema> {
    if link.path().last() != Some(&name) {
        return Err(TCError::unsupported(format!(
            "link for cluster directory entry {} must end with {} (found {})",
            name, name, link
        )));
    }

    let (lead, path) = link.into_inner();

    Ok(Schema {
        path,
        lead,
        host: txn.host().clone(),
        actor: actor.clone(),
    })
}
