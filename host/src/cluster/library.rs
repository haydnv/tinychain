use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use destream::{de, en};
use futures::future::TryFutureExt;

use tc_error::*;
use tc_transact::fs::{BlockData, File, FileRead, FileWrite, Persist, Store};
use tc_transact::lock::{TxnMapLock, TxnMapLockCommitGuard, TxnMapRead, TxnMapWrite};
use tc_transact::{Transact, Transaction, TxnId};
use tc_value::Version as VersionNumber;
use tcgeneric::{Map, PathSegment};

use crate::fs;
use crate::fs::CacheBlock;
use crate::scalar::value::Link;
use crate::scalar::Scalar;
use crate::txn::Txn;

use super::{Cluster, Replica};

#[derive(Clone)]
pub enum DirEntry {
    Dir(Arc<Cluster<Dir>>),
    Item(Arc<Cluster<Library>>),
}

#[derive(Clone)]
pub struct Dir {
    cache: freqfs::DirLock<CacheBlock>,
    contents: TxnMapLock<PathSegment, DirEntry>,
}

impl Dir {
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
        contents.insert(name, DirEntry::Dir(Arc::new(dir)));

        Ok(())
    }

    pub(super) async fn create_lib(
        &self,
        txn: &Txn,
        link: &Link,
        name: PathSegment,
        number: VersionNumber,
        version: Map<Scalar>,
    ) -> TCResult<()> {
        let mut contents = self.contents.write(*txn.id()).await?;
        let mut cache = self.cache.write().await;

        let file = cache
            .create_dir(format!("{}.{}", name, Version::ext()))
            .map_err(fs::io_err)
            .and_then(fs::File::new)?;

        let lib = Library::create(txn, (), file).await?;
        lib.create_version(*txn.id(), number, version).await?;

        let lib = Cluster::with_state(link.clone().append(name.clone()), lib);
        contents.insert(name, DirEntry::Item(Arc::new(lib)));

        Ok(())
    }

    pub async fn entry(&self, txn_id: TxnId, name: &PathSegment) -> TCResult<Option<DirEntry>> {
        self.contents
            .read(txn_id)
            .map_ok(|contents| contents.get(name))
            .await
    }
}

#[async_trait]
impl Replica for Dir {
    async fn replicate(&self, _txn: &Txn, _source: &Link) -> TCResult<()> {
        Err(TCError::not_implemented("cluster::Dir::replicate"))
    }
}

#[async_trait]
impl Transact for Dir {
    type Commit = TxnMapLockCommitGuard<PathSegment, DirEntry>;

    async fn commit(&self, txn_id: &TxnId) -> Self::Commit {
        self.contents.commit(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.contents.finalize(txn_id).await
    }
}

#[async_trait]
impl Persist<fs::Dir> for Dir {
    type Schema = ();
    type Store = fs::Dir;
    type Txn = Txn;

    async fn create(_txn: &Self::Txn, _schema: Self::Schema, dir: Self::Store) -> TCResult<Self> {
        Ok(Self {
            cache: dir.into_inner(),
            contents: TxnMapLock::new("service directory"),
        })
    }

    async fn load(txn: &Self::Txn, schema: Self::Schema, dir: Self::Store) -> TCResult<Self> {
        assert!(dir.is_empty(*txn.id()).await?);
        Self::create(txn, schema, dir).await
    }
}

impl fmt::Display for Dir {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("library directory")
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
    pub async fn create_version<V>(
        &self,
        txn_id: TxnId,
        number: VersionNumber,
        version: V,
    ) -> TCResult<()>
    where
        Version: From<V>,
    {
        let mut file = self.file.write(txn_id).await?;
        file.create_block(number, version.into(), 0).await?;
        Ok(())
    }

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
