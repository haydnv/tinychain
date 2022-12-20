use std::collections::HashMap;
use std::convert::TryFrom;
use std::fmt;

use async_trait::async_trait;
use futures::future::{join_all, FutureExt, TryFutureExt};
use futures::try_join;
use safecast::{as_type, AsType, CastFrom};

use tc_error::*;
use tc_transact::fs::*;
use tc_transact::lock::{TxnMapLock, TxnMapLockCommitGuard, TxnMapRead, TxnMapWrite};
use tc_transact::{Transact, Transaction};
use tc_value::Version as VersionNumber;
use tcgeneric::{label, Id, Instance, Label, Map, NativeClass};

use crate::chain::{Chain, ChainInstance};
use crate::cluster::DirItem;
use crate::collection::{CollectionBase, CollectionBaseCommitGuard};
use crate::fs;
use crate::scalar::Scalar;
use crate::state::State;
use crate::transact::TxnId;
use crate::txn::Txn;

const SCHEMA: Label = label("schemata");

#[derive(Clone)]
pub enum Attr {
    Chain(Chain<CollectionBase>),
    Scalar(Scalar),
}

as_type!(Attr, Chain, Chain<CollectionBase>);
as_type!(Attr, Scalar, Scalar);

#[derive(Clone)]
pub struct Version {
    attrs: Map<Attr>,
}

impl Version {
    async fn new(txn: &Txn, dir: fs::Dir, state: Map<State>) -> TCResult<Self> {
        let mut attrs = Map::new();

        for (name, state) in state {
            if state.is_ref() {
                return Err(TCError::unsupported(format!(
                    "invalid Service attribute {}: {}",
                    name, state
                )));
            }

            match state {
                State::Chain(chain) => {
                    let dir = dir.write(*txn.id()).await?;
                    let store = dir.create_store(name.clone());
                    let chain = Chain::copy_from(txn, store, chain).await?;
                    attrs.insert(name, chain.into());
                }
                State::Scalar(scalar) => {
                    attrs.insert(name, scalar.into());
                }
                State::Collection(collection) => {
                    return Err(TCError::unsupported(format!(
                        "{} must be wrapped in a Chain",
                        collection
                    )));
                }
                other => {
                    return Err(TCError::unsupported(format!(
                        "invalid Service attribute {}: {}",
                        name, other
                    )));
                }
            }
        }

        Ok(Self { attrs })
    }

    pub fn get_attribute(&self, name: &Id) -> Option<&Attr> {
        self.attrs.get(name)
    }
}

#[async_trait]
impl Persist<fs::Dir> for Version {
    type Txn = Txn;
    type Schema = Map<Scalar>;

    async fn create(txn: &Self::Txn, schema: Self::Schema, store: fs::Store) -> TCResult<Self> {
        let dir = fs::Dir::try_from(store)?;
        let txn_id = *txn.id();
        let mut attrs = Map::new();

        for (name, scalar) in schema {
            match scalar {
                Scalar::Ref(tc_ref) => {
                    let dir = dir.write(txn_id).await?;
                    let _store = dir.create_store(name);
                    return Err(TCError::not_implemented(format!(
                        "create new chain with schema {}",
                        tc_ref
                    )));
                }
                scalar => {
                    attrs.insert(name, scalar.into());
                }
            }
        }

        Ok(Self { attrs })
    }

    async fn load(txn: &Self::Txn, schema: Self::Schema, store: fs::Store) -> TCResult<Self> {
        let dir = fs::Dir::try_from(store)?;
        let txn_id = *txn.id();
        let mut attrs = Map::new();

        for (name, scalar) in schema {
            match scalar {
                Scalar::Ref(tc_ref) => {
                    let dir = dir.read(txn_id).await?;
                    let _store = dir.get_store(name);
                    return Err(TCError::not_implemented(format!(
                        "load chain with schema {}",
                        tc_ref
                    )));
                }
                scalar => {
                    attrs.insert(name, scalar.into());
                }
            }
        }

        Ok(Self { attrs })
    }

    fn dir(&self) -> <fs::Dir as Dir>::Inner {
        unimplemented!("cluster::service::Version::dir")
    }
}

#[async_trait]
impl Transact for Version {
    type Commit = Map<CollectionBaseCommitGuard>;

    async fn commit(&self, txn_id: &TxnId) -> Self::Commit {
        let guards = join_all(
            self.attrs
                .iter()
                .filter_map(|(name, attr)| attr.as_type().map(|chain: &Chain<_>| (name, chain)))
                .map(|(name, attr)| attr.commit(txn_id).map(move |guard| (name.clone(), guard))),
        )
        .await;

        guards.into_iter().collect()
    }

    async fn finalize(&self, txn_id: &TxnId) {
        join_all(
            self.attrs
                .values()
                .filter_map(|attr| attr.as_type())
                .map(|chain: &Chain<_>| chain.finalize(txn_id)),
        )
        .await;
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a hosted Service")
    }
}

#[derive(Clone)]
pub struct Service {
    dir: fs::Dir,
    schema: fs::File<VersionNumber, super::library::Version>,
    versions: TxnMapLock<VersionNumber, Version>,
}

impl Service {
    pub async fn get_version(&self, txn_id: TxnId, number: &VersionNumber) -> TCResult<Version> {
        let versions = self.versions.read(txn_id).await?;

        versions
            .get(number)
            .ok_or_else(|| TCError::not_found(number))
    }
}

#[async_trait]
impl DirItem for Service {
    type Version = Map<State>;

    async fn create_version(
        &self,
        txn: &Txn,
        number: VersionNumber,
        version: Self::Version,
    ) -> TCResult<()> {
        let mut schema = Map::new();
        for (name, state) in &version {
            match state {
                State::Chain(chain) => {
                    let chain_schema = (
                        Scalar::from(chain.class().path()),
                        Scalar::cast_from(chain.subject().schema()),
                    );

                    schema.insert(name.clone(), Scalar::Tuple(chain_schema.into()));
                }
                State::Scalar(scalar) if !scalar.is_ref() => {
                    schema.insert(name.clone(), scalar.clone());
                }
                State::Collection(_) => {
                    return Err(TCError::unsupported(
                        "a Collection in a Service must be wrapped in a Chain",
                    ))
                }
                other => return Err(TCError::bad_request("invalid Service attribute", other)),
            }
        }

        let txn_id = *txn.id();
        let (mut dir, mut schemata, mut versions) = try_join!(
            self.dir.write(txn_id),
            self.schema.write(txn_id),
            self.versions.write(txn_id)
        )?;

        schemata
            .create_block(number.clone(), schema.into(), 0)
            .await?;

        let dir = dir.create_dir(number.clone().into())?;
        let version = Version::new(txn, dir, version).await?;
        versions.insert(number, version);

        Ok(())
    }
}

#[async_trait]
impl Transact for Service {
    type Commit = TxnMapLockCommitGuard<VersionNumber, Version>;

    async fn commit(&self, txn_id: &TxnId) -> Self::Commit {
        let versions = self.versions.commit(txn_id).await;

        join_all(
            versions
                .iter()
                .map(|(_name, version)| async move { version.commit(txn_id).await }),
        )
        .await;

        versions
    }

    async fn finalize(&self, txn_id: &TxnId) {
        {
            let versions = self.versions.read(*txn_id).await.expect("service versions");

            join_all(
                versions
                    .iter()
                    .map(|(_, version)| async move { version.finalize(txn_id).await }),
            )
            .await
        };

        self.versions.finalize(txn_id).await
    }
}

#[async_trait]
impl Persist<fs::Dir> for Service {
    type Txn = Txn;
    type Schema = ();

    async fn create(txn: &Self::Txn, _schema: Self::Schema, store: fs::Store) -> TCResult<Self> {
        let dir = fs::Dir::try_from(store)?;
        let mut contents = dir.write(*txn.id()).await?;
        if contents.is_empty() {
            let schema = contents.create_file(SCHEMA.into())?;

            Ok(Self {
                dir,
                schema,
                versions: TxnMapLock::new("service versions"),
            })
        } else {
            Err(TCError::unsupported(
                "cannot create a new Service from a non-empty directory",
            ))
        }
    }

    async fn load(txn: &Self::Txn, _schema: Self::Schema, store: fs::Store) -> TCResult<Self> {
        let txn_id = *txn.id();

        let dir = fs::Dir::try_from(store)?;
        let (schema, mut versions) = {
            let dir = dir.read(txn_id).await?;
            let schema: fs::File<VersionNumber, super::library::Version> = dir
                .get_file(&SCHEMA.into())?
                .ok_or_else(|| TCError::internal("service missing schema file"))?;

            (schema, HashMap::with_capacity(dir.len()))
        };

        let version_schema = schema.read(txn_id).await?;
        for number in version_schema.block_ids() {
            let schema = version_schema
                .read_block(number)
                .map_err(TCError::internal)
                .await?;

            let dir = dir.read(txn_id).await?;
            let store = dir.get_store(number.clone().into()).ok_or_else(|| {
                TCError::internal(format!(
                    "missing filesystem entry for service version {}",
                    number
                ))
            })?;

            let schema = super::library::Version::clone(&*schema);
            let version = Version::load(txn, schema.into(), store).await?;
            versions.insert(number.clone(), version);
        }

        Ok(Self {
            dir,
            schema,
            versions: TxnMapLock::with_contents("service versions", versions),
        })
    }

    fn dir(&self) -> <fs::Dir as Dir>::Inner {
        self.dir.clone().into_inner()
    }
}

impl fmt::Display for Service {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a versioned hosted Service")
    }
}
