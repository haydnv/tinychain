use std::fmt;

use async_trait::async_trait;
use destream::{de, en};
use futures::future::TryFutureExt;
use safecast::TryCastFrom;

use tc_error::*;
use tc_transact::fs::{BlockData, File, FileRead, FileWrite, Persist};
use tc_transact::{Transact, Transaction, TxnId};
use tc_value::Version as VersionNumber;
use tcgeneric::{Map, PathSegment};

use crate::fs;
use crate::scalar::value::Link;
use crate::scalar::Scalar;
use crate::state::State;
use crate::txn::Txn;

use super::{DirItem, Replica};

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

    async fn schema(&self, _txn_id: TxnId) -> TCResult<Self::Schema> {
        Ok(())
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
