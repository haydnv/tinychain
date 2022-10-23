use std::fmt;

use async_trait::async_trait;
use destream::{de, en};
use futures::future::TryFutureExt;
use safecast::AsType;

use tc_error::*;
use tc_transact::fs::{BlockData, DirCreate, DirCreateFile, File, FileWrite};
use tc_transact::{Transact, TxnId};
use tc_value::Version as VersionNumber;
use tcgeneric::{Map, PathSegment};

use crate::fs;
use crate::scalar::value::Link;
use crate::scalar::Scalar;
use crate::txn::Txn;

use super::Replica;

pub enum DirEntry {
    Dir(Dir),
    Item(Library),
}

#[derive(Clone)]
pub struct Dir {
    dir: fs::Dir,
}

impl Dir {
    pub async fn create_dir(&self, txn_id: TxnId, name: PathSegment) -> TCResult<Self> {
        let mut lock = tc_transact::fs::Dir::write(&self.dir, txn_id).await?;
        lock.create_dir(name).map(|dir| Self { dir })
    }

    pub async fn create_lib(
        &self,
        txn_id: TxnId,
        name: PathSegment,
        library: Map<Scalar>,
    ) -> TCResult<Library> {
        let mut lock = tc_transact::fs::Dir::write(&self.dir, txn_id).await?;

        let lib = lock.create_file(name).map(Library::from)?;
        lib.create_version(txn_id, VersionNumber::default(), library)
            .await?;

        Ok(lib)
    }

    pub async fn get(&self, txn_id: TxnId, name: &PathSegment) -> TCResult<Option<DirEntry>> {
        match self.dir.get(txn_id, name).await? {
            Some(entry) => match entry {
                fs::DirEntry::Dir(dir) => Ok(Some(DirEntry::Dir(Self::from(dir)))),
                fs::DirEntry::File(file) => {
                    match AsType::<fs::File<VersionNumber, Version>>::into_type(file) {
                        Some(lib) => Ok(Some(DirEntry::Item(Library::from(lib)))),
                        None => Err(TCError::internal("wrong file type")),
                    }
                }
            },
            None => Ok(None),
        }
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
    type Commit = <fs::Dir as Transact>::Commit;

    async fn commit(&self, txn_id: &TxnId) -> Self::Commit {
        self.dir.commit(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.dir.finalize(txn_id).await
    }
}

impl From<fs::Dir> for Dir {
    fn from(dir: fs::Dir) -> Self {
        Self { dir }
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
}

impl From<fs::File<VersionNumber, Version>> for Library {
    fn from(file: fs::File<VersionNumber, Version>) -> Self {
        Self { file }
    }
}
