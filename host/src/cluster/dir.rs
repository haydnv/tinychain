//! A directory of [`Cluster`]s

use std::collections::BTreeMap;
use std::convert::TryFrom;
use std::fmt;
use std::ops::Deref;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{join_all, try_join_all, FutureExt, TryFutureExt};
use futures::join;
use futures::stream::{FuturesUnordered, StreamExt, TryStreamExt};
use log::*;
use safecast::CastInto;

use tc_chain::Recover;
use tc_error::*;
use tc_scalar::Scalar;
use tc_state::chain::BlockChain;
use tc_state::object::InstanceClass;
use tc_state::State;
use tc_transact::fs::Persist;
use tc_transact::lock::TxnMapLock;
use tc_transact::public::Route;
use tc_transact::{RPCClient, Transact, Transaction, TxnId};
use tc_value::{Link, Version as VersionNumber};
use tcgeneric::{label, Id, Label, Map, PathSegment, ThreadSafe};

use crate::txn::Txn;

use super::{Class, Cluster, Library, Replica, Schema, Service, REPLICAS};

/// The type of file stored in a [`Library`] directory
pub type File = tc_fs::File<Map<Scalar>>;

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

/// Defines methods common to any item in a [`Dir`].
#[async_trait]
pub trait DirItem:
    Persist<tc_fs::CacheBlock, Txn = Txn, Schema = ()> + Transact + Clone + Send + Sync
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

impl<T> fmt::Debug for DirEntry<T> {
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
    dir: tc_fs::Dir,
    contents: TxnMapLock<PathSegment, DirEntry<T>>,
}

impl<T> Dir<T> {
    /// Borrow the [`Schema`] of this [`Dir`]
    pub fn schema(&self) -> &Schema {
        &self.schema
    }
}

impl<T: Clone> Dir<T> {
    fn new(txn_id: TxnId, schema: Schema, dir: tc_fs::Dir) -> TCResult<Self> {
        let contents = TxnMapLock::new(txn_id);

        Ok(Self {
            schema,
            dir,
            contents,
        })
    }

    fn with_contents<C: IntoIterator<Item = (PathSegment, DirEntry<T>)>>(
        txn_id: TxnId,
        schema: Schema,
        dir: tc_fs::Dir,
        contents: C,
    ) -> TCResult<Self> {
        let contents = TxnMapLock::with_contents(txn_id, contents);

        Ok(Self {
            schema,
            dir,
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

        match self.contents.try_get(txn_id, &path[0])? {
            Some(entry) => match &*entry {
                DirEntry::Item(item) => Ok(Some((&path[1..], DirEntry::Item(item.clone())))),
                DirEntry::Dir(dir) => dir.lookup(txn_id, &path[1..]).map(Some),
            },
            None => Ok(None),
        }
    }
}

impl<T> Dir<T>
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
impl<T: Send + Sync> DirCreate for Dir<T>
where
    DirEntry<T>: Clone,
    Cluster<Self>: Clone,
    Self: Persist<tc_fs::CacheBlock, Txn = Txn, Schema = Schema> + Route<State> + fmt::Debug,
{
    /// Create a new subdirectory in this [`Dir`].
    async fn create_dir(
        &self,
        txn: &Txn,
        name: PathSegment,
        link: Link,
    ) -> TCResult<Cluster<Self>> {
        let txn_id = *txn.id();

        if let Some(entry) = self.contents.get(txn_id, &name).await? {
            return match &*entry {
                DirEntry::Dir(dir) => {
                    if dir.link() == link {
                        Ok(dir.clone())
                    } else {
                        Err(bad_request!(
                            "cannot replace lead {} with {}",
                            dir.link(),
                            link
                        ))
                    }
                }
                DirEntry::Item(_) => Err(bad_request!("there is already an entry at {}", name)),
            };
        }

        let schema = entry_schema(txn, &self.schema.actor, &name, link).await?;

        let dir = self.dir.create_dir(txn_id, name.clone()).await?;
        let dir = Self::create(txn_id, schema.clone(), dir).await?;
        let dir = Cluster::with_state(schema, dir);

        self.contents
            .insert(txn_id, name, DirEntry::Dir(dir.clone()))
            .await?;

        Ok(dir)
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
        link: Link,
    ) -> TCResult<Cluster<BlockChain<T>>> {
        debug!("cluster::Dir::create_item {} at {}", name, link);

        let txn_id = *txn.id();

        if let Some(dir) = self.contents.get(txn_id, &name).await? {
            match &*dir {
                DirEntry::Item(item) => {
                    if item.link() == link {
                        Ok(item.clone())
                    } else {
                        Err(bad_request!(
                            "cannot replace cluster lead {} with {}",
                            item.link(),
                            link
                        ))
                    }
                }
                DirEntry::Dir(_) => Err(bad_request!("there is already a directory at {}", name)),
            }
        } else {
            let schema = entry_schema(txn, &self.schema.actor, &name, link).await?;

            let dir = self.dir.create_dir(txn_id, name.clone()).await?;
            let item = BlockChain::create(txn_id, (), dir).await?;
            let item = Cluster::with_state(schema, item);

            self.contents
                .insert(txn_id, name, DirEntry::Item(item.clone()))
                .await?;

            Ok(item)
        }
    }
}

#[async_trait]
impl<T: DirItem + Route<State> + fmt::Debug> Replica for Dir<T>
where
    BlockChain<T>: Replica,
    DirEntry<T>: Clone,
    Cluster<Self>: Clone,
    Self: Persist<tc_fs::CacheBlock, Txn = Txn, Schema = Schema> + Route<State> + fmt::Debug,
{
    async fn state(&self, txn_id: TxnId) -> TCResult<State> {
        let mut state = Map::<State>::new();
        for (name, entry) in self.contents.iter(txn_id).await? {
            let class = match &*entry {
                DirEntry::Dir(_) => true.into(),
                DirEntry::Item(_) => false.into(),
            };

            state.insert(Id::clone(&*name), class);
        }

        debug!("directory state to replicate is {:?}", state);

        Ok(State::Map(state.cast_into()))
    }

    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()> {
        info!("replicate {:?} from {}", self, source);

        let params = Map::one(label("add"), txn.host().clone().into());

        let entries = txn
            .post(source.clone().append(REPLICAS), State::Map(params))
            .await?;

        let entries = entries.try_into_map(|s| {
            bad_gateway!("{} listed invalid directory entries {:?}", source, s)
        })?;

        debug!("directory entries to replicate are {:?}", entries);

        let entries = entries
            .into_iter()
            .map(|(name, is_dir)| bool::try_from(is_dir).map(|is_dir| (name, is_dir)))
            .collect::<TCResult<Map<bool>>>()?;

        let txn_id = *txn.id();
        for (name, is_dir) in entries {
            let link = source.clone().append(name.clone());

            if let Some(entry) = self.entry(txn_id, &name).await? {
                match &*entry {
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
impl<T: Transact + ThreadSafe + Clone> Transact for Dir<T>
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

#[async_trait]
impl<T: Send + Sync + 'static> Recover<tc_fs::CacheBlock> for Dir<T>
where
    DirEntry<T>: Clone,
    Cluster<BlockChain<T>>: Recover<tc_fs::CacheBlock, Txn = Txn>,
{
    type Txn = Txn;

    async fn recover(&self, txn: &Txn) -> TCResult<()> {
        let contents = self.contents.iter(*txn.id()).await?;
        let recovery = contents.map(|(_name, entry)| async move {
            match entry.clone() {
                DirEntry::Dir(dir) => dir.recover(txn).await,
                DirEntry::Item(item) => item.recover(txn).await,
            }
        });

        try_join_all(recovery).await?;

        Ok(())
    }
}

#[async_trait]
impl Persist<tc_fs::CacheBlock> for Dir<Class> {
    type Txn = Txn;
    type Schema = Schema;

    async fn create(txn_id: TxnId, schema: Self::Schema, store: tc_fs::Dir) -> TCResult<Self> {
        Self::new(txn_id, schema, store)
    }

    async fn load(txn_id: TxnId, schema: Self::Schema, dir: tc_fs::Dir) -> TCResult<Self> {
        dir.trim(txn_id).await?;

        let mut loaded = BTreeMap::new();

        let mut contents = dir.iter::<InstanceClass>(txn_id).await?;
        while let Some((name, entry)) = contents.try_next().await? {
            let schema = schema.extend((*name).clone());

            let entry = match entry {
                tc_transact::fs::DirEntry::Dir(dir) => {
                    let is_chain = dir.contains(txn_id, &tc_chain::HISTORY.into()).await?;

                    if is_chain {
                        Cluster::load(txn_id, schema, dir)
                            .map_ok(DirEntry::Item)
                            .await
                    } else {
                        Cluster::load(txn_id, schema, dir)
                            .map_ok(DirEntry::Dir)
                            .await
                    }
                }
                tc_transact::fs::DirEntry::File(file) => {
                    Err(internal!("invalid Class dir entry: {:?}", file))
                }
            }?;

            loaded.insert((*name).clone(), entry);
        }

        std::mem::drop(contents);

        Self::with_contents(txn_id, schema, dir, loaded)
    }

    fn dir(&self) -> tc_transact::fs::Inner<tc_fs::CacheBlock> {
        self.dir.clone().into_inner()
    }
}

#[async_trait]
impl Persist<tc_fs::CacheBlock> for Dir<Library> {
    type Txn = Txn;
    type Schema = Schema;

    async fn create(txn_id: TxnId, schema: Self::Schema, store: tc_fs::Dir) -> TCResult<Self> {
        Self::new(txn_id, schema, store)
    }

    async fn load(txn_id: TxnId, schema: Self::Schema, dir: tc_fs::Dir) -> TCResult<Self> {
        // let mut contents = BTreeMap::new();
        //
        // let mut entries = dir.iter(txn_id).await?;
        //
        // while let Some((name, entry)) = entries.try_next().await? {
        //     let schema = schema.extend(name.clone());
        //
        //     match entry {
        //         fs::DirEntry::Dir(dir) => {
        //             let dir = Cluster::load(txn_id, schema, dir.clone().into()).await?;
        //             contents.insert(name.clone(), DirEntry::Dir(dir));
        //         }
        //         fs::DirEntry::File(file) => {
        //             let lib = Cluster::load(txn_id, schema, file.clone().into()).await?;
        //             contents.insert(name.clone(), DirEntry::Item(lib));
        //         }
        //     };
        // }
        //
        // Self::with_contents(txn_id, schema, dir, contents)
        todo!()
    }

    fn dir(&self) -> tc_transact::fs::Inner<tc_fs::CacheBlock> {
        self.dir.clone().into_inner()
    }
}

#[async_trait]
impl Persist<tc_fs::CacheBlock> for Dir<Service> {
    type Txn = Txn;
    type Schema = Schema;

    async fn create(txn_id: TxnId, schema: Self::Schema, store: tc_fs::Dir) -> TCResult<Self> {
        Self::new(txn_id, schema, store)
    }

    async fn load(txn_id: TxnId, schema: Self::Schema, dir: tc_fs::Dir) -> TCResult<Self> {
        let mut loaded = BTreeMap::new();

        let mut contents = dir.iter::<InstanceClass>(txn_id).await?;

        while let Some((name, entry)) = contents.try_next().await? {
            let schema = schema.extend((*name).clone());

            let entry = match entry {
                tc_transact::fs::DirEntry::File(file) => {
                    Err(internal!("invalid Service directory entry: {:?}", file))
                }
                tc_transact::fs::DirEntry::Dir(dir) => {
                    let is_service = dir.contains(txn_id, &super::service::SCHEMA.into()).await?;

                    if is_service {
                        trace!("load item {} in service dir", name);

                        Cluster::load(txn_id, schema, dir.clone())
                            .map_ok(DirEntry::Item)
                            .await
                    } else {
                        trace!("load sub-dir {} in service dir", name);

                        Cluster::load(txn_id, schema, dir.clone())
                            .map_ok(DirEntry::Dir)
                            .await
                    }
                }
            }?;

            loaded.insert((*name).clone(), entry);
        }

        std::mem::drop(contents);

        Self::with_contents(txn_id, schema, dir, loaded)
    }

    fn dir(&self) -> tc_transact::fs::Inner<tc_fs::CacheBlock> {
        self.dir.clone().into_inner()
    }
}

impl<T> fmt::Debug for Dir<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} directory", std::any::type_name::<T>())
    }
}

async fn entry_schema(
    txn: &Txn,
    actor: &Arc<tc_fs::Actor>,
    name: &PathSegment,
    link: Link,
) -> TCResult<Schema> {
    if link.path().last() != Some(&name) {
        return Err(bad_request!(
            "link for cluster directory entry {} must end with {} (found {})",
            name,
            name,
            link
        ));
    }

    let (lead, path) = link.into_inner();

    Ok(Schema {
        path,
        lead,
        host: txn.host().clone(),
        actor: actor.clone(),
    })
}
