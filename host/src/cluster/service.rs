use std::collections::BTreeMap;
use std::convert::TryFrom;
use std::fmt;

use async_trait::async_trait;
use futures::future::{self, join_all, FutureExt, TryFutureExt};
use futures::stream::FuturesUnordered;
use futures::{try_join, TryStreamExt};
use safecast::{as_type, AsType};

use tc_error::*;
use tc_transact::fs::*;
use tc_transact::lock::TxnLock;
use tc_transact::{Transact, Transaction};
use tc_value::Version as VersionNumber;
use tcgeneric::{label, Id, Label, Map, NativeClass};

use crate::chain::{Chain, ChainType};
use crate::cluster::{DirItem, Replica};
use crate::collection::{CollectionBase, CollectionSchema};
use crate::fs;
use crate::scalar::value::Link;
use crate::scalar::{OpRef, Scalar, Subject, TCRef};
use crate::state::State;
use crate::transact::TxnId;
use crate::txn::Txn;

pub const CHAINS: Label = label("chains");
pub(super) const SCHEMA: Label = label("schemata");

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
    pub fn get_attribute(&self, name: &Id) -> Option<&Attr> {
        self.attrs.get(name)
    }
}

#[async_trait]
impl Replica for Version {
    async fn state(&self, _txn_id: TxnId) -> TCResult<State> {
        unimplemented!()
    }

    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()> {
        let requests = FuturesUnordered::new();

        for (name, attr) in self.attrs.iter() {
            if let Attr::Chain(chain) = attr {
                let source = source.clone().append(name.clone());
                let txn = txn.subcontext(name.clone()).await?;
                requests.push(async move { chain.replicate(&txn, source).await });
            }
        }

        requests.try_fold((), |(), ()| future::ready(Ok(()))).await
    }
}

#[async_trait]
impl Persist<fs::Dir> for Version {
    type Txn = Txn;
    type Schema = Map<Scalar>;

    async fn create(txn: &Self::Txn, schema: Self::Schema, store: fs::Store) -> TCResult<Self> {
        let txn_id = *txn.id();
        let dir = fs::Dir::try_from(store)?;

        let mut attrs = Map::new();

        for (name, attr) in schema {
            let attr = match attr {
                Scalar::Ref(tc_ref) => match *tc_ref {
                    TCRef::Op(OpRef::Get((chain_type, collection))) => {
                        let chain_type = resolve_type::<ChainType>(chain_type)?;

                        let schema = TCRef::try_from(collection)?;
                        let schema = CollectionSchema::from_scalar(schema)?;

                        let store = dir.create_store(txn_id, name.clone()).await?;
                        let chain = Chain::create(txn, (chain_type, schema), store).await?;

                        Ok(Attr::Chain(chain))
                    }
                    other => Err(TCError::bad_request("invalid Service attribute", other)),
                },
                scalar if scalar.is_ref() => {
                    Err(TCError::bad_request("invalid Service attribute", scalar))
                }
                scalar => Ok(Attr::Scalar(scalar)),
            }?;

            attrs.insert(name, attr);
        }

        Ok(Self { attrs })
    }

    async fn load(txn: &Self::Txn, schema: Self::Schema, store: fs::Store) -> TCResult<Self> {
        let txn_id = *txn.id();
        let dir = fs::Dir::try_from(store)?;

        let mut attrs = Map::new();

        for (name, attr) in schema {
            let attr = match attr {
                Scalar::Ref(tc_ref) => match *tc_ref {
                    TCRef::Op(OpRef::Get((chain_type, collection))) => {
                        let chain_type = resolve_type::<ChainType>(chain_type)?;

                        let schema = TCRef::try_from(collection)?;
                        let schema = CollectionSchema::from_scalar(schema)?;

                        let store = dir.get_or_create_store(txn_id, name.clone()).await?;
                        let chain = Chain::load(txn, (chain_type, schema), store).await?;

                        Ok(Attr::Chain(chain))
                    }
                    other => Err(TCError::bad_request("invalid Service attribute", other)),
                },
                scalar if scalar.is_ref() => {
                    Err(TCError::bad_request("invalid Service attribute", scalar))
                }
                scalar => Ok(Attr::Scalar(scalar)),
            }?;

            attrs.insert(name, attr);
        }

        Ok(Self { attrs })
    }

    fn dir(&self) -> <fs::Dir as Dir>::Inner {
        unimplemented!("cluster::service::Version::dir")
    }
}

#[async_trait]
impl Transact for Version {
    type Commit = ();

    async fn commit(&self, txn_id: &TxnId) -> Self::Commit {
        join_all(
            self.attrs
                .iter()
                .filter_map(|(name, attr)| attr.as_type().map(|chain: &Chain<_>| (name, chain)))
                .map(|(name, attr)| attr.commit(txn_id).map(move |guard| (name.clone(), guard))),
        )
        .await;
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
    versions: TxnLock<BTreeMap<VersionNumber, Version>>,
}

impl Service {
    pub async fn get_version(&self, txn_id: TxnId, number: &VersionNumber) -> TCResult<Version> {
        let versions = self.versions.read(txn_id).await?;

        versions
            .get(number)
            .cloned()
            .ok_or_else(|| TCError::not_found(format!("Service version {}", number)))
    }

    pub async fn latest(&self, txn_id: TxnId) -> TCResult<Option<VersionNumber>> {
        self.schema
            .read(txn_id)
            .map_ok(|file| file.block_ids().iter().last().cloned())
            .await
    }

    pub async fn schemata(&self, txn_id: TxnId) -> TCResult<Map<Map<Scalar>>> {
        let file = self.schema.read(txn_id).await?;
        let mut schemata = Map::new();

        for number in file.block_ids() {
            let version = file.read_block(number).await?;
            schemata.insert(number.clone().into(), (&*version).clone().into());
        }

        Ok(schemata)
    }
}

#[async_trait]
impl DirItem for Service {
    type Schema = Map<Scalar>;
    type Version = Version;

    async fn create_version(
        &self,
        txn: &Txn,
        number: VersionNumber,
        schema: Map<Scalar>,
    ) -> TCResult<Version> {
        let txn_id = *txn.id();
        let (dir, mut schemata, mut versions) = try_join!(
            self.dir.write(txn_id),
            self.schema.write(txn_id),
            self.versions.write(txn_id).map_err(TCError::from),
        )?;

        schemata
            .create_block(number.clone(), schema.clone().into(), 0)
            .await?;

        let store = dir.create_store(number.clone().into());
        let version = Version::create(txn, schema, store).await?;

        versions.insert(number, version.clone());

        Ok(version)
    }
}

#[async_trait]
impl Transact for Service {
    type Commit = ();

    async fn commit(&self, txn_id: &TxnId) -> Self::Commit {
        let versions = self.versions.commit(txn_id).await;

        join_all(
            versions
                .iter()
                .map(|(_name, version)| async move { version.commit(txn_id).await }),
        )
        .await;
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

        self.versions.finalize(txn_id)
    }
}

#[async_trait]
impl Persist<fs::Dir> for Service {
    type Txn = Txn;
    type Schema = ();

    async fn create(txn: &Self::Txn, _schema: Self::Schema, store: fs::Store) -> TCResult<Self> {
        let dir = fs::Dir::try_from(store)?;
        let txn_id = *txn.id();
        let mut contents = dir.write(txn_id).await?;
        if contents.is_empty() {
            let schema = contents.create_file(SCHEMA.into())?;

            Ok(Self {
                dir,
                schema,
                versions: TxnLock::new("service", txn_id, BTreeMap::new()),
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

            (schema, BTreeMap::new())
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
            versions: TxnLock::new("service", txn_id, versions),
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

fn resolve_type<T: NativeClass>(subject: Subject) -> TCResult<T> {
    match subject {
        Subject::Link(link) if link.host().is_none() => {
            T::from_path(link.path()).ok_or_else(|| {
                TCError::unsupported(format!(
                    "{} is not a {}",
                    link.path(),
                    std::any::type_name::<T>()
                ))
            })
        }
        Subject::Link(link) => Err(TCError::not_implemented(format!(
            "support for a user-defined Class of {} in a Service: {}",
            std::any::type_name::<T>(),
            link
        ))),
        subject => Err(TCError::bad_request(
            format!("expected a {} but found", std::any::type_name::<T>()),
            subject,
        )),
    }
}
