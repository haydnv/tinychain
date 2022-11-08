use std::collections::hash_map::{self, HashMap};
use std::fmt;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use destream::{de, en};
use futures::future::{join_all, FutureExt, TryFutureExt};
use safecast::TryCastFrom;

use tc_error::*;
use tc_transact::fs::{BlockData, File, FileRead, FileWrite, Persist};
use tc_transact::lock::{
    TxnMapLock, TxnMapLockCommitGuard, TxnMapLockReadGuard, TxnMapRead, TxnMapWrite,
};
use tc_transact::{Transact, Transaction, TxnId};
use tc_value::Version as VersionNumber;
use tcgeneric::{Map, PathSegment};

use crate::fs;
use crate::fs::CacheBlock;
use crate::scalar::value::Link;
use crate::scalar::Scalar;
use crate::state::State;
use crate::txn::Txn;

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
    cache: freqfs::DirLock<CacheBlock>,
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

#[derive(Clone)]
pub struct Version {
    lib: Map<Scalar>,
}

impl Version {
    pub fn attribute(&self, name: &PathSegment) -> Option<&Scalar> {
        self.lib.get(name)
    }
}

#[async_trait]
impl de::FromStream for Version {
    type Context = ();

    async fn from_stream<D: de::Decoder>(
        cxt: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        Map::<Scalar>::from_stream(cxt, decoder)
            .map_ok(|lib| Self { lib })
            .await
    }
}

impl<'en> en::IntoStream<'en> for Version {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        self.lib.into_stream(encoder)
    }
}

impl<'en> en::ToStream<'en> for Version {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        self.lib.to_stream(encoder)
    }
}

impl BlockData for Version {
    fn ext() -> &'static str {
        "lib"
    }
}

impl From<Map<Scalar>> for Version {
    fn from(lib: Map<Scalar>) -> Self {
        Self { lib }
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "library version {}", self.lib)
    }
}

#[derive(Clone)]
pub struct Library {
    file: fs::File<VersionNumber, Version>,
}

impl Library {
    pub async fn get_version(
        &self,
        txn_id: TxnId,
        number: VersionNumber,
    ) -> TCResult<fs::BlockReadGuard<Version>> {
        let file = self.file.read(txn_id).await?;
        file.read_block(&number).await
    }
}

#[async_trait]
impl DirItem for Library {
    async fn create_version(
        &self,
        txn_id: TxnId,
        number: VersionNumber,
        version: State,
    ) -> TCResult<()> {
        let version = Map::<Scalar>::try_cast_from(version, |s| {
            TCError::bad_request("invalid library version", s)
        })?;

        let mut file = self.file.write(txn_id).await?;
        file.create_block(number, version.into(), 0).await?;
        Ok(())
    }
}

#[async_trait]
impl Replica for Library {
    async fn replicate(&self, _txn: &Txn, _source: &Link) -> TCResult<()> {
        Err(TCError::not_implemented("replicate a Library"))
    }
}

#[async_trait]
impl Transact for Library {
    type Commit = <fs::File<VersionNumber, Version> as Transact>::Commit;

    async fn commit(&self, txn_id: &TxnId) -> Self::Commit {
        self.file.commit(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.file.finalize(txn_id).await
    }
}

#[async_trait]
impl Persist<fs::Dir> for Library {
    type Schema = ();
    type Store = fs::File<VersionNumber, Version>;
    type Txn = Txn;

    async fn create(txn: &Self::Txn, _schema: Self::Schema, file: Self::Store) -> TCResult<Self> {
        let versions = file.read(*txn.id()).await?;
        if versions.is_empty() {
            Ok(Self { file })
        } else {
            Err(TCError::unsupported(
                "cannot create a new Library from a non-empty file",
            ))
        }
    }

    async fn load(_txn: &Self::Txn, _schema: Self::Schema, file: Self::Store) -> TCResult<Self> {
        Ok(Self { file })
    }
}

impl From<fs::File<VersionNumber, Version>> for Library {
    fn from(file: fs::File<VersionNumber, Version>) -> Self {
        Self { file }
    }
}

impl fmt::Display for Library {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a Library")
    }
}
