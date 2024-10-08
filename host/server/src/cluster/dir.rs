//! A directory of [`Cluster`]s

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::ops::Deref;

use async_trait::async_trait;
use futures::future::{join_all, FutureExt, TryFutureExt};
use futures::join;
use futures::stream::TryStreamExt;
use futures::stream::{FuturesUnordered, StreamExt};
use log::*;
use safecast::TryCastFrom;

use tc_error::*;
#[cfg(feature = "service")]
use tc_state::chain::Recover;
use tc_state::object::InstanceClass;
use tc_transact::hash::*;
use tc_transact::lock::{TxnLockReadGuard, TxnMapLock, TxnMapLockEntry, TxnMapLockIter};
use tc_transact::public::Route;
use tc_transact::{fs, Gateway};
use tc_transact::{Replicate, Transact, Transaction, TxnId};
use tc_value::{Host, Link, Value, Version as VersionNumber};
use tcgeneric::{Id, Map, PathSegment, TCBoxTryFuture, TCPath, ThreadSafe};

use crate::{CacheBlock, State, Txn, VerifyingKey};

use super::{Class, Cluster, IsDir, Library, Schema};

#[async_trait]
pub trait DirCreate: Sized {
    async fn create_dir(&self, txn: &Txn, name: PathSegment) -> TCResult<Cluster<Self>>;
}

#[async_trait]
pub trait DirCreateItem<T: DirItem> {
    async fn create_item(&self, txn: &Txn, name: PathSegment) -> TCResult<Cluster<T>>;
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

impl<T> DirEntry<T> {
    pub fn is_dir(&self) -> bool {
        match self {
            Self::Dir(_) => true,
            _ => false,
        }
    }
}

/// A commit guard for a [`DirEntry`]
pub enum DirEntryCommit<T: Transact + Clone + Send + Sync + fmt::Debug + 'static> {
    Dir(<Cluster<Dir<T>> as Transact>::Commit),
    Item(<Cluster<T> as Transact>::Commit),
}

#[cfg(feature = "service")]
#[async_trait]
impl<T> Recover<CacheBlock> for DirEntry<T>
where
    T: Recover<CacheBlock, Txn = Txn> + fmt::Debug + Clone + Send + Sync,
{
    type Txn = Txn;

    async fn recover(&self, txn: &Txn) -> TCResult<()> {
        match self {
            Self::Dir(dir) => dir.recover(txn).await,
            Self::Item(cluster) => cluster.recover(txn).await,
        }
    }
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
        trace!("finalize {:?} at {}", self, txn_id);

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
    schema: Schema,
    dir: fs::Dir<CacheBlock>,
    contents: TxnMapLock<PathSegment, DirEntry<T>>,
}

impl<T: fmt::Debug> Dir<T> {
    fn new(txn_id: TxnId, schema: Schema, dir: fs::Dir<CacheBlock>) -> TCResult<Self> {
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
        dir: fs::Dir<CacheBlock>,
        contents: C,
    ) -> TCResult<Self> {
        let contents = TxnMapLock::with_contents(txn_id, contents);

        Ok(Self {
            schema,
            dir,
            contents,
        })
    }
}

impl<T: Clone + Send + Sync + fmt::Debug> Dir<T> {
    /// Recursive [`Dir`] entry lookup
    pub fn lookup<'a>(
        self,
        txn_id: TxnId,
        path: &'a [PathSegment],
    ) -> TCBoxTryFuture<'a, Option<(&'a [PathSegment], DirEntry<T>)>>
    where
        T: 'a,
    {
        Box::pin(async move {
            trace!("look up {} in {:?}", TCPath::from(path), self);

            if path.is_empty() {
                return Ok(None);
            }

            let entry = self.contents.get(txn_id, &path[0]).await?;

            match entry {
                Some(entry) => match &*entry {
                    DirEntry::Item(item) => Ok(Some((&path[1..], DirEntry::Item(item.clone())))),
                    DirEntry::Dir(dir) => dir.clone().lookup(txn_id, &path[1..]).map_ok(Some).await,
                },
                None => Ok(None),
            }
        })
    }
}

impl<T: fmt::Debug> Dir<T>
where
    DirEntry<T>: Clone,
{
    pub(super) async fn entries(
        &self,
        txn_id: TxnId,
    ) -> TCResult<TxnMapLockIter<PathSegment, DirEntry<T>>> {
        self.contents.iter(txn_id).map_err(TCError::from).await
    }

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
    Cluster<T>: fs::Persist<CacheBlock, Txn = Txn, Schema = Schema>,
    Cluster<Self>: fs::Persist<CacheBlock, Txn = Txn, Schema = Schema> + Clone,
    Self: fs::Persist<CacheBlock, Txn = Txn, Schema = Schema> + Route<State> + fmt::Debug,
{
    /// Create a new subdirectory in this [`Dir`].
    async fn create_dir(&self, txn: &Txn, name: PathSegment) -> TCResult<Cluster<Self>> {
        let txn_id = *txn.id();

        match self.contents.entry(txn_id, name.clone()).await? {
            TxnMapLockEntry::Vacant(entry) => {
                let dir = self.dir.create_dir(txn_id, name.clone()).await?;
                let schema = self.schema.clone().append(name);
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
    T: DirItem + fmt::Debug,
    Cluster<T>: fs::Persist<CacheBlock, Txn = Txn, Schema = Schema>,
    DirEntry<T>: Clone,
{
    /// Create a new item in this [`Dir`].
    async fn create_item(&self, txn: &Txn, name: PathSegment) -> TCResult<Cluster<T>> {
        debug!("cluster::Dir::create_item {name}");

        let txn_id = *txn.id();

        match self.contents.entry(txn_id, name.clone()).await? {
            TxnMapLockEntry::Vacant(entry) => {
                let dir = self.dir.create_dir(txn_id, name.clone()).await?;
                let schema = self.schema.clone().append(name);
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
impl<T: AsyncHash + fmt::Debug> IsDir for Dir<T> {
    fn is_dir(&self) -> bool {
        true
    }

    async fn get_dir_item_key(
        &self,
        txn_id: TxnId,
        name: &PathSegment,
    ) -> TCResult<Option<(Output<Sha256>, VerifyingKey)>> {
        if let Some(entry) = self.contents.get(txn_id, name).await? {
            match &*entry {
                DirEntry::Dir(cluster) => {
                    let hash = AsyncHash::hash(cluster.state(), txn_id).await?;
                    Ok(Some((hash, cluster.public_key())))
                }
                DirEntry::Item(cluster) => {
                    let hash = AsyncHash::hash(cluster.state(), txn_id).await?;
                    Ok(Some((hash, cluster.public_key())))
                }
            }
        } else {
            Ok(None)
        }
    }

    async fn get_dir_item_keyring(
        &self,
        txn_id: TxnId,
        name: &PathSegment,
    ) -> TCResult<Option<TxnLockReadGuard<HashMap<Host, VerifyingKey>>>> {
        if let Some(entry) = self.contents.get(txn_id, name).await? {
            match &*entry {
                DirEntry::Dir(cluster) => cluster.keyring(txn_id).map_ok(Some).await,
                DirEntry::Item(cluster) => cluster.keyring(txn_id).map_ok(Some).await,
            }
        } else {
            Ok(None)
        }
    }
}

#[cfg(feature = "service")]
#[async_trait]
impl<T> Recover<CacheBlock> for Dir<T>
where
    T: Recover<CacheBlock, Txn = Txn> + fmt::Debug + Clone + Send + Sync,
{
    type Txn = Txn;

    async fn recover(&self, txn: &Txn) -> TCResult<()> {
        let contents = self.contents.iter(*txn.id()).await?;

        let mut recovered = FuturesUnordered::new();
        for (_name, entry) in contents {
            recovered.push(async move { entry.recover(txn).await });
        }

        while let Some(()) = recovered.try_next().await? {
            // pass
        }

        Ok(())
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

#[async_trait]
impl<T> AsyncHash for Dir<T>
where
    T: fmt::Debug + Send + Sync,
{
    async fn hash(&self, txn_id: TxnId) -> TCResult<Output<Sha256>> {
        let contents = self.contents.iter(txn_id).await?;
        let mut is_empty = true;
        let mut hasher = Sha256::new();

        for (name, entry) in contents {
            hasher.update(Hash::<Sha256>::hash((&*name, entry.is_dir())));
            is_empty = false;
        }

        if is_empty {
            Ok(default_hash::<Sha256>())
        } else {
            Ok(hasher.finalize())
        }
    }
}

#[async_trait]
impl<T> Replicate<Txn> for Dir<T>
where
    T: Send + Sync + fmt::Debug,
    Cluster<T>: fs::Persist<CacheBlock, Txn = Txn, Schema = Schema>,
    Cluster<Dir<T>>: fs::Persist<CacheBlock, Txn = Txn, Schema = Schema>,
{
    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<Output<Sha256>> {
        debug!("replicate {self:?} from {source}...");

        assert_eq!(self.schema.link.path(), source.path());

        let state = txn.get(source, Value::default()).await?;
        let state = state.try_into_map(|dir| bad_request!("invalid dir state: {dir:?}"))?;
        let state = state
            .into_iter()
            .map(|(name, is_dir)| bool::try_from(is_dir).map(|is_dir| (name, is_dir)))
            .collect::<TCResult<Map<bool>>>()?;

        let txn_id = *txn.id();

        let mut deleted = vec![];
        let entries = self.contents.iter(txn_id).await?;
        for (name, _) in entries {
            if !state.contains_key(&*name) {
                deleted.push(name.clone());
            }
        }

        for name in deleted {
            self.contents.remove(txn_id, &*name).await?;
        }

        for (name, is_dir) in state {
            if !self.contents.contains_key(txn_id, &name).await? {
                let schema = self.schema.clone().append(name.clone());
                let store = self.dir.get_or_create_dir(txn_id, name.clone()).await?;

                let entry = if is_dir {
                    fs::Persist::load_or_create(txn_id, schema, store)
                        .map_ok(DirEntry::Dir)
                        .await
                } else {
                    fs::Persist::load_or_create(txn_id, schema, store)
                        .map_ok(DirEntry::Item)
                        .await
                }?;

                self.contents.insert(txn_id, name, entry).await?;
            }
        }

        AsyncHash::hash(self, txn_id).await
    }
}

#[async_trait]
impl fs::Persist<CacheBlock> for Dir<Class> {
    type Txn = Txn;
    type Schema = Schema;

    async fn create(txn_id: TxnId, schema: Schema, store: fs::Dir<CacheBlock>) -> TCResult<Self> {
        if store.is_empty(txn_id).await? {
            Self::new(txn_id, schema, store)
        } else {
            Err(bad_request!(
                "cannot create an empty cluster dir from a non-empty filesystem dir"
            ))
        }
    }

    async fn load(txn_id: TxnId, schema: Schema, dir: fs::Dir<CacheBlock>) -> TCResult<Self> {
        let mut loaded = BTreeMap::new();

        let mut entries = dir.entries::<InstanceClass>(txn_id).await?;
        while let Some((name, entry)) = entries.try_next().await? {
            let entry = match entry {
                fs::DirEntry::Dir(dir) => {
                    let is_class = {
                        let mut names = dir.entry_names(txn_id).await?;
                        names.any(|name| VersionNumber::can_cast_from(&name.as_str()))
                    };

                    let schema = schema.clone().append(Id::clone(&*name));

                    if is_class {
                        Cluster::load(txn_id, schema, dir)
                            .map_ok(DirEntry::Item)
                            .await
                    } else {
                        Cluster::load(txn_id, schema, dir)
                            .map_ok(DirEntry::Dir)
                            .await
                    }
                }
                fs::DirEntry::File(file) => Err(internal!("invalid Class dir entry: {:?}", file)),
            }?;

            loaded.insert((*name).clone(), entry);
        }

        std::mem::drop(entries); // needed because `entries` borrows `dir`

        Self::with_contents(txn_id, schema, dir, loaded)
    }

    fn dir(&self) -> fs::Inner<CacheBlock> {
        self.dir.clone().into_inner()
    }
}

// TODO: dedupe logic with impl Persist for Dir<Class> above
#[async_trait]
impl fs::Persist<CacheBlock> for Dir<Library> {
    type Txn = Txn;
    type Schema = Schema;

    async fn create(txn_id: TxnId, schema: Schema, store: fs::Dir<CacheBlock>) -> TCResult<Self> {
        if store.is_empty(txn_id).await? {
            Self::new(txn_id, schema, store)
        } else {
            Err(bad_request!(
                "cannot create an empty cluster dir from a non-empty filesystem dir"
            ))
        }
    }

    async fn load(txn_id: TxnId, schema: Schema, dir: fs::Dir<CacheBlock>) -> TCResult<Self> {
        let mut contents = BTreeMap::new();

        let mut entries = dir.entries::<Library>(txn_id).await?;

        while let Some((name, entry)) = entries.try_next().await? {
            // first, test if this is a directory or a library entry
            // if it's a library entry, it will contain exactly one file called "lib"
            // otherwise, load it as a directory

            let dir = match entry {
                fs::DirEntry::Dir(dir) => dir,
                fs::DirEntry::File(file) => {
                    return Err(internal!(
                        "tried to load a library file {name}: {file:?} as a cluster directory"
                    ));
                }
            };

            let is_lib = {
                let mut is_lib = !dir.is_empty(txn_id).await?;

                let mut entries = dir.entries::<Library>(txn_id).await?;
                while let Some((name, entry)) = entries.try_next().await? {
                    if &*name == &*super::library::LIB && entry.is_file() {
                        is_lib = true;
                    } else {
                        is_lib = false;
                    }
                }

                is_lib
            };

            let name = Id::clone(&*name);
            let schema = schema.clone().append(name.clone());

            let entry = if is_lib {
                Cluster::load(txn_id, schema, dir)
                    .map_ok(DirEntry::Item)
                    .await
            } else {
                Cluster::load(txn_id, schema, dir)
                    .map_ok(DirEntry::Dir)
                    .await
            }?;

            contents.insert(name, entry);
        }

        std::mem::drop(entries); // needed because `entries` borrows `dir`

        Self::with_contents(txn_id, schema, dir, contents)
    }

    fn dir(&self) -> fs::Inner<CacheBlock> {
        self.dir.clone().into_inner()
    }
}

#[cfg(feature = "service")]
#[async_trait]
impl fs::Persist<CacheBlock> for Dir<super::Service> {
    type Txn = Txn;
    type Schema = Schema;

    async fn create(
        txn_id: TxnId,
        schema: Self::Schema,
        store: fs::Dir<CacheBlock>,
    ) -> TCResult<Self> {
        Self::new(txn_id, schema, store)
    }

    async fn load(txn_id: TxnId, schema: Self::Schema, dir: fs::Dir<CacheBlock>) -> TCResult<Self> {
        let mut loaded = BTreeMap::new();
        let mut contents = dir.entries::<InstanceClass>(txn_id).await?;

        while let Some((name, entry)) = contents.try_next().await? {
            let schema = schema.clone().append((*name).clone());

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

    fn dir(&self) -> fs::Inner<CacheBlock> {
        self.dir.clone().into_inner()
    }
}

impl<T> fmt::Debug for Dir<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} directory", std::any::type_name::<T>())
    }
}
