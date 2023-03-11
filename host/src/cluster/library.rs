//! A replicated, versioned, stateless [`Library`]

use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::ops::Deref;

use async_trait::async_trait;
use destream::{de, en};
use futures::future::TryFutureExt;
use futures::TryStreamExt;
use get_size::GetSize;
use get_size_derive::*;
use log::{error, info};
use safecast::{AsType, TryCastFrom};

use tc_error::*;
use tc_transact::fs::{Dir, File, Persist};
use tc_transact::{Transact, Transaction, TxnId};
use tc_value::Version as VersionNumber;
use tcgeneric::{Instance, Map, PathSegment};

use crate::fs;
use crate::object::ObjectType;
use crate::scalar::{OpDef, Scalar};
use crate::state::State;
use crate::txn::Txn;

use super::DirItem;

/// A version of a [`Library`]
#[derive(Clone, GetSize)]
pub struct Version {
    lib: Map<Scalar>,
}

impl Version {
    pub fn get_attribute(&self, name: &PathSegment) -> Option<&Scalar> {
        self.lib.get(name)
    }
}

impl Instance for Version {
    type Class = ObjectType;

    fn class(&self) -> Self::Class {
        ObjectType::Class
    }
}

impl From<Map<Scalar>> for Version {
    fn from(lib: Map<Scalar>) -> Self {
        Self { lib }
    }
}

impl TryCastFrom<Scalar> for Version {
    fn can_cast_from(scalar: &Scalar) -> bool {
        Map::<Scalar>::can_cast_from(scalar)
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        Map::<Scalar>::opt_cast_from(scalar).map(Self::from)
    }
}

impl From<Version> for Map<Scalar> {
    fn from(version: Version) -> Self {
        version.lib
    }
}

impl From<Version> for Scalar {
    fn from(version: Version) -> Self {
        Self::Map(version.into())
    }
}

impl From<Version> for State {
    fn from(version: Version) -> Self {
        Self::Scalar(version.into())
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

impl fmt::Debug for Version {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "library version {:?}", self.lib)
    }
}

/// A versioned collection of [`Scalar`]s
#[derive(Clone)]
pub struct Library {
    file: fs::File<VersionNumber, Version>,
}

impl Library {
    pub async fn latest(&self, txn_id: TxnId) -> TCResult<Option<VersionNumber>> {
        let block_ids = self.file.block_ids(txn_id).await?;

        Ok(block_ids
            .last()
            .map(|id| id.as_str().parse().expect("version number")))
    }

    pub async fn get_version(
        &self,
        txn_id: TxnId,
        number: &VersionNumber,
    ) -> TCResult<impl Deref<Target = Version>> {
        self.file.read_block(txn_id, number).await
    }

    pub async fn to_state(&self, txn_id: TxnId) -> TCResult<State> {
        let mut blocks = self.file.iter(txn_id).await?;

        let mut map = Map::new();
        while let Some((number, block)) = blocks.try_next().await? {
            let number = number.as_str().parse()?;
            map.insert(number, (&*block).clone().into());
        }

        Ok(State::Map(map))
    }
}

#[async_trait]
impl DirItem for Library {
    type Schema = Map<Scalar>;
    type Version = Version;

    async fn create_version(
        &self,
        txn: &Txn,
        number: VersionNumber,
        schema: Map<Scalar>,
    ) -> TCResult<Version> {
        let version = Version::from(validate(schema)?);

        self.file
            .create_block(*txn.id(), number, version.clone())
            .map_ok(|_| ())
            .await?;

        Ok(version)
    }
}

#[async_trait]
impl Transact for Library {
    type Commit = ();

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        self.file.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.file.rollback(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.file.finalize(txn_id).await
    }
}

#[async_trait]
impl Persist<fs::CacheBlock> for Library {
    type Txn = Txn;
    type Schema = ();

    async fn create(txn_id: TxnId, _schema: (), dir: fs::Dir) -> TCResult<Self> {
        // let file = dir.into_inner().into();
        // if file.is_empty(txn_id).await? {
        //     Ok(Self { file })
        // } else {
        //     Err(bad_request!(
        //         "cannot create a new library from a non-empty file"
        //     ))
        // }
        todo!()
    }

    async fn load(_txn_id: TxnId, _schema: (), dir: fs::Dir) -> TCResult<Self> {
        // let file = dir.into_inner().into();
        // Ok(Self { file })
        todo!()
    }

    fn dir(&self) -> tc_transact::fs::Inner<fs::CacheBlock> {
        self.file.clone().into_inner()
    }
}

impl fmt::Debug for Library {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("library")
    }
}

fn validate(proto: Map<Scalar>) -> TCResult<Map<Scalar>> {
    if proto
        .values()
        .filter_map(|member| member.as_type())
        .any(|op: &OpDef| op.is_write())
    {
        Err(bad_request!("a Library may not define write operations"))
    } else {
        Ok(proto)
    }
}
