use std::collections::BTreeMap;
use std::convert::{TryFrom, TryInto};
use std::fmt;

use async_trait::async_trait;
use destream::{de, en};
use futures::future::TryFutureExt;
use log::{error, info};
use safecast::TryCastFrom;

use tc_error::*;
use tc_transact::fs::{BlockData, Dir, File, FileRead, FileWrite, Persist};
use tc_transact::{Transact, Transaction, TxnId};
use tc_value::Version as VersionNumber;
use tcgeneric::{Map, PathSegment};

use crate::fs;
use crate::object::Object;
use crate::scalar::Scalar;
use crate::state::State;
use crate::txn::Txn;

use super::DirItem;

#[derive(Clone)]
pub struct Version {
    lib: Map<Scalar>,
}

impl Version {
    pub fn attribute(&self, name: &PathSegment) -> Option<&Scalar> {
        self.lib.get(name)
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

impl TryCastFrom<State> for Version {
    fn can_cast_from(state: &State) -> bool {
        match state {
            State::Map(map) => map.values().all(Scalar::can_cast_from),
            State::Object(Object::Class(_)) => true,
            State::Scalar(scalar) => Self::can_cast_from(scalar),
            _ => false,
        }
    }

    fn opt_cast_from(state: State) -> Option<Self> {
        match state {
            State::Map(map) => BTreeMap::opt_cast_from(map).map(Map::from).map(Self::from),
            State::Object(Object::Class(class)) => {
                let (_extends, proto) = class.into_inner();
                Some(Self::from(proto))
            }
            State::Scalar(scalar) => Self::opt_cast_from(scalar),
            _ => None,
        }
    }
}

impl From<Version> for Scalar {
    fn from(version: Version) -> Scalar {
        Scalar::Map(version.lib)
    }
}

impl From<Version> for State {
    fn from(version: Version) -> State {
        State::Scalar(version.into())
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
    pub async fn latest(&self, txn_id: TxnId) -> TCResult<Option<VersionNumber>> {
        self.file
            .read(txn_id)
            .map_ok(|file| file.block_ids().last().cloned())
            .await
    }

    pub async fn get_version(
        &self,
        txn_id: TxnId,
        number: VersionNumber,
    ) -> TCResult<fs::BlockReadGuard<Version>> {
        let file = self.file.read(txn_id).await?;
        if file.is_empty() {
            Err(TCError::internal("library has no versions"))
        } else {
            file.read_block(&number).await
        }
    }

    pub async fn to_state(&self, txn_id: TxnId) -> TCResult<State> {
        let file = self.file.read(txn_id).await?;

        let mut map = Map::new();
        for number in file.block_ids() {
            let block = file.read_block(number).await?;
            map.insert(number.clone().into(), (&*block).clone().into());
        }

        Ok(State::Map(map))
    }
}

#[async_trait]
impl DirItem for Library {
    type Version = Version;

    async fn create_version(
        &self,
        txn_id: TxnId,
        number: VersionNumber,
        version: Self::Version,
    ) -> TCResult<()> {
        let mut file = self.file.write(txn_id).await?;
        if file.contains(&number) {
            error!("duplicate library version {}", number);

            Err(TCError::unsupported(format!(
                "{} already has a version {}",
                self, number
            )))
        } else {
            info!("create new library version {}", number);
            file.create_block(number, version, 0).map_ok(|_| ()).await
        }
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
    type Txn = Txn;
    type Schema = ();

    async fn create(txn: &Self::Txn, _schema: Self::Schema, store: fs::Store) -> TCResult<Self> {
        let file = super::dir::File::try_from(store)?;
        let versions = file.read(*txn.id()).await?;
        if versions.is_empty() {
            Ok(Self { file })
        } else {
            Err(TCError::unsupported(
                "cannot create a new library from a non-empty file",
            ))
        }
    }

    async fn load(_txn: &Self::Txn, _schema: Self::Schema, store: fs::Store) -> TCResult<Self> {
        store.try_into().map(|file| Self { file })
    }

    fn dir(&self) -> <fs::Dir as Dir>::Inner {
        self.file.clone().into_inner()
    }
}

impl From<fs::File<VersionNumber, Version>> for Library {
    fn from(file: fs::File<VersionNumber, Version>) -> Self {
        Self { file }
    }
}

impl fmt::Display for Library {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("library")
    }
}
