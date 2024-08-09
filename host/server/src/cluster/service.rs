use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::ops::Deref;

use async_trait::async_trait;
use futures::future::{join_all, try_join_all, FutureExt, TryFutureExt};
use futures::join;
use futures::stream::{FuturesOrdered, FuturesUnordered, TryStreamExt};
use log::{debug, trace};
use safecast::*;

use tc_error::*;
use tc_scalar::value::{Link, Value, Version as VersionNumber};
use tc_scalar::{OpRef, Refer, Scalar, Subject, TCRef};
use tc_state::chain::{ChainType, Recover};
use tc_state::collection::Schema as CollectionSchema;
use tc_state::object::InstanceClass;
use tc_state::CacheBlock;
use tc_transact::fs::{Dir, File, Persist};
use tc_transact::hash::*;
use tc_transact::lock::TxnMapLock;
use tc_transact::public::ToState;
use tc_transact::{Gateway, Replicate, Transact, Transaction, TxnId};
use tcgeneric::{label, Id, Label, Map, NativeClass, TCPathBuf};

use crate::cluster::IsDir;
use crate::{Chain, Collection, State, Txn};

use super::dir::DirItem;

pub(super) const SCHEMA: Label = label("schemata");

/// An attribute of a [`Version`]
#[derive(Clone)]
pub enum Attr {
    Chain(Chain<Collection>),
    Scalar(Scalar),
}

as_type!(Attr, Chain, Chain<Collection>);
as_type!(Attr, Scalar, Scalar);

#[async_trait]
impl AsyncHash for Attr {
    async fn hash(&self, txn_id: TxnId) -> TCResult<Output<Sha256>> {
        match self {
            Self::Chain(chain) => chain.hash(txn_id).await,
            Self::Scalar(scalar) => Ok(Hash::<Sha256>::hash(scalar)),
        }
    }
}

impl fmt::Debug for Attr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Chain(chain) => fmt::Debug::fmt(chain, f),
            Self::Scalar(scalar) => fmt::Debug::fmt(scalar, f),
        }
    }
}

/// A version of a [`Service`]
#[derive(Clone)]
pub struct Version {
    path: TCPathBuf,
    attrs: Map<Attr>,
}

impl Version {
    pub fn get_attribute(&self, name: &Id) -> Option<&Attr> {
        self.attrs.get(name)
    }
}

#[async_trait]
impl Replicate<Txn> for Version {
    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<Output<Sha256>> {
        debug!("replicate {self:?} from {source}...");

        for (name, attr) in self.attrs.iter() {
            if let Attr::Chain(chain) = attr {
                let source = source.clone().append(name.clone());
                let txn = txn.subcontext(name.clone());
                // TODO: parallelize
                chain.replicate(&txn, source).await?;
            }
        }

        AsyncHash::hash(self, *txn.id()).await
    }
}

impl ToState<State> for Version {
    fn to_state(&self) -> State {
        State::Scalar(Scalar::Cluster(self.path.clone().into()))
    }
}

#[async_trait]
impl Persist<CacheBlock> for Version {
    type Txn = Txn;
    type Schema = (Link, Map<Scalar>);

    async fn create(txn_id: TxnId, schema: Self::Schema, dir: Dir<CacheBlock>) -> TCResult<Self> {
        let (link, proto) = schema;

        let mut attrs = Map::new();

        for (name, attr) in proto {
            let attr = match attr {
                Scalar::Ref(tc_ref) => match *tc_ref {
                    TCRef::Op(OpRef::Get((chain_type, collection))) => {
                        let chain_type = resolve_type::<ChainType>(chain_type)?;
                        let schema = try_cast_into_schema(collection)?;
                        let store = dir.create_dir(txn_id, name.clone()).await?;
                        let chain = Chain::create(txn_id, (chain_type, schema), store).await?;
                        Ok(Attr::Chain(chain))
                    }
                    other => Err(TCError::unexpected(other, "a Service attribute")),
                },
                scalar if Refer::<State>::is_ref(&scalar) => {
                    Err(TCError::unexpected(scalar, "a Service attribute"))
                }
                scalar => Ok(Attr::Scalar(scalar)),
            }?;

            attrs.insert(name, attr);
        }

        Ok(Self {
            attrs,
            path: link.into_path(),
        })
    }

    async fn load(txn_id: TxnId, schema: Self::Schema, dir: Dir<CacheBlock>) -> TCResult<Self> {
        debug!("cluster::service::Version::load {:?}", schema);

        let (link, proto) = schema;

        let mut attrs = Map::new();

        for (name, attr) in proto {
            let attr = match attr {
                Scalar::Ref(tc_ref) => match *tc_ref {
                    TCRef::Op(OpRef::Get((chain_type, collection))) => {
                        let chain_type = resolve_type::<ChainType>(chain_type)?;
                        let schema = try_cast_into_schema(collection)?;
                        let store = dir.get_or_create_dir(txn_id, name.clone()).await?;
                        let chain = Chain::load(txn_id, (chain_type, schema), store).await?;
                        Ok(Attr::Chain(chain))
                    }
                    other => Err(TCError::unexpected(other, "a Service attribute")),
                },
                scalar if Refer::<State>::is_ref(&scalar) => {
                    Err(TCError::unexpected(scalar, "a Service attribute"))
                }
                scalar => Ok(Attr::Scalar(scalar)),
            }?;

            attrs.insert(name, attr);
        }

        Ok(Self {
            attrs,
            path: link.into_path(),
        })
    }

    fn dir(&self) -> tc_transact::fs::Inner<CacheBlock> {
        unimplemented!("cluster::service::Version::dir")
    }
}

#[async_trait]
impl AsyncHash for Version {
    async fn hash(&self, txn_id: TxnId) -> TCResult<Output<Sha256>> {
        let mut attr_hashes: FuturesOrdered<_> = self
            .attrs
            .iter()
            .map(|(name, attr)| {
                attr.hash(txn_id).map_ok(move |attr_hash| {
                    let mut hasher = Sha256::new();
                    hasher.update(Hash::<Sha256>::hash(name));
                    hasher.update(attr_hash);
                    hasher.finalize()
                })
            })
            .collect();

        let mut is_empty = true;
        let mut hasher = Sha256::new();
        while let Some(attr_hash) = attr_hashes.try_next().await? {
            hasher.update(attr_hash);
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
impl Transact for Version {
    type Commit = ();

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        join_all(
            self.attrs
                .values()
                .filter_map(|attr| attr.as_type())
                .map(|attr: &Chain<_>| attr.commit(txn_id)),
        )
        .await;
    }

    async fn rollback(&self, txn_id: &TxnId) {
        join_all(
            self.attrs
                .values()
                .filter_map(|attr| attr.as_type())
                .map(|attr: &Chain<_>| attr.rollback(txn_id)),
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

#[async_trait]
impl Recover<CacheBlock> for Version {
    type Txn = Txn;

    async fn recover(&self, txn: &Txn) -> TCResult<()> {
        let recovery = self
            .attrs
            .values()
            .filter_map(|attr| attr.as_type())
            .map(|chain: &Chain<_>| chain.recover(txn));

        try_join_all(recovery).await?;

        Ok(())
    }
}

impl fmt::Debug for Version {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a hosted Service version")
    }
}

/// A stateful collection of [`Chain`]s and [`Scalar`] methods ([`Attr`]s)
#[derive(Clone)]
pub struct Service {
    dir: Dir<CacheBlock>,
    schema: File<CacheBlock, (Link, Map<Scalar>)>,
    versions: TxnMapLock<VersionNumber, Version>,
}

impl Service {
    pub async fn get_version(
        &self,
        txn_id: TxnId,
        number: &VersionNumber,
    ) -> TCResult<impl Deref<Target = Version>> {
        let version = self.versions.get(txn_id, number).await?;
        version.ok_or_else(|| TCError::not_found(number))
    }
}

#[async_trait]
impl DirItem for Service {
    type Schema = InstanceClass;
    type Version = Version;

    async fn create_version(
        &self,
        txn: &Txn,
        number: VersionNumber,
        class: InstanceClass,
    ) -> TCResult<Version> {
        let version_id = number.into();
        let (link, proto) = class.into_inner();
        let proto = validate(&link, &number, proto)?;

        let txn_id = *txn.id();

        if self.schema.contains_block(txn_id, &version_id).await? {
            let existing = self.schema.read_block(txn_id, &version_id).await?;
            let (existing_link, existing_proto) = &*existing;
            if existing_link == &link && existing_proto == &proto {
                let version = self
                    .versions
                    .get(txn_id, &number)
                    .map_err(TCError::from)
                    .map(|result| {
                        result.and_then(|version| {
                            version.ok_or_else(|| internal!("missing Service version {number}"))
                        })
                    })
                    .await?;

                // dropping the TxnMapReadLock is safe because the Version itself is transactional
                Ok(Version::clone(&*version))
            } else {
                Err(bad_request!(
                    "{self:?} cannot overwrite existing service version {number}"
                ))
            }
        } else {
            self.schema
                .create_block(txn_id, number.into(), (link.clone(), proto.clone()))
                .await?;

            let store = self.dir.create_dir(txn_id, number.clone().into()).await?;
            let version = Version::create(txn_id, (link, proto), store).await?;

            self.versions
                .insert(txn_id, number, version.clone())
                .await?;

            Ok(version)
        }
    }
}

impl IsDir for Service {}

#[async_trait]
impl Replicate<Txn> for Service {
    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<Output<Sha256>> {
        debug!("replicate {self:?} from {source}...");

        let txn_id = *txn.id();
        let version_ids = txn.get(source.clone(), Value::None).await?;
        let version_ids =
            version_ids.try_into_tuple(|s| TCError::unexpected(s, "service version numbers"))?;

        let version_ids = version_ids
            .into_iter()
            .map(|id| {
                let id = Value::try_from(id)?;
                id.try_cast_into(|v| TCError::unexpected(v, "a service version number"))
            })
            .collect::<TCResult<BTreeSet<VersionNumber>>>()?;

        trace!("version IDs to replicate are {version_ids:?}");

        version_ids
            .iter()
            .copied()
            .map(|number| {
                let source = source.clone();
                async move {
                    let version = txn.get(source, number).await?;
                    let version = InstanceClass::try_cast_from(version, |s| {
                        TCError::unexpected(s, "a Service")
                    })?;

                    self.create_version(txn, number, version).await?;

                    TCResult::Ok(())
                }
            })
            .collect::<FuturesUnordered<_>>()
            .try_fold((), |(), ()| futures::future::ready(Ok(())))
            .await?;

        // TODO: delete old versions

        self.hash(txn_id).await
    }
}

#[async_trait]
impl AsyncHash for Service {
    async fn hash(&self, txn_id: TxnId) -> TCResult<Output<Sha256>> {
        let versions = self.versions.iter(txn_id).await?;
        let mut versions: FuturesOrdered<_> = versions
            .map(|(number, version)| async move {
                version
                    .hash(txn_id)
                    .map_ok(move |version_hash| {
                        let mut hasher = Sha256::new();
                        hasher.update(Hash::<Sha256>::hash(*number));
                        hasher.update(version_hash);
                        hasher.finalize()
                    })
                    .await
            })
            .collect();

        let mut is_empty = true;
        let mut hasher = Sha256::new();
        while let Some(hash) = versions.try_next().await? {
            hasher.update(hash);
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
impl Transact for Service {
    type Commit = ();

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        let (versions, _deltas) = self.versions.read_and_commit(txn_id).await;

        join_all(versions.iter().map(|(_name, version)| async move {
            version.commit(txn_id).await;
        }))
        .await;

        join!(self.schema.commit(txn_id), self.dir.commit(txn_id, false));
    }

    async fn rollback(&self, txn_id: &TxnId) {
        let (versions, _deltas) = self.versions.read_and_rollback(*txn_id).await;

        join_all(
            versions
                .iter()
                .map(|(_, version)| async move { version.rollback(txn_id).await }),
        )
        .await;

        join!(
            self.schema.rollback(txn_id),
            self.dir.rollback(*txn_id, false)
        );
    }

    async fn finalize(&self, txn_id: &TxnId) {
        if let Some(versions) = self.versions.read_and_finalize(*txn_id) {
            join_all(
                versions
                    .iter()
                    .map(|(_, version)| async move { version.finalize(txn_id).await }),
            )
            .await;
        }

        join!(self.schema.finalize(txn_id), self.dir.finalize(*txn_id));
    }
}

#[async_trait]
impl Recover<CacheBlock> for Service {
    type Txn = Txn;

    async fn recover(&self, txn: &Txn) -> TCResult<()> {
        let versions = self.versions.iter(*txn.id()).await?;
        let recovery = versions.map(|(_id, version)| async move { version.recover(txn).await });

        try_join_all(recovery).await?;

        Ok(())
    }
}

#[async_trait]
impl Persist<CacheBlock> for Service {
    type Txn = Txn;
    type Schema = ();

    async fn create(txn_id: TxnId, _schema: (), dir: Dir<CacheBlock>) -> TCResult<Self> {
        if dir.is_empty(txn_id).await? {
            let schema = dir.create_file(txn_id, SCHEMA.into()).await?;

            Ok(Self {
                dir,
                schema,
                versions: TxnMapLock::new(txn_id),
            })
        } else {
            Err(bad_request!(
                "cannot create a new Service from a non-empty directory"
            ))
        }
    }

    async fn load(txn_id: TxnId, _schema: (), dir: Dir<CacheBlock>) -> TCResult<Self> {
        debug!("load Service");

        let schema = dir
            .get_file::<(Link, Map<Scalar>)>(txn_id, &SCHEMA.into())
            .await?;

        let mut versions = BTreeMap::new();

        trace!("loading Service versions...");

        let mut blocks = schema.iter(txn_id).await?;
        while let Some((number, schema)) = blocks.try_next().await? {
            let version_number = number.as_str().parse()?;
            trace!("loading Service version {}...", version_number);

            // `get_or_create_dir` here in case of a service with no persistent data
            let store = dir.get_or_create_dir(txn_id, (*number).clone()).await?;

            trace!(
                "got transactional directory for Service version {}",
                version_number
            );

            let schema = <(Link, Map<Scalar>)>::clone(&*schema);
            let version = Version::load(txn_id, schema, store).await?;

            versions.insert(version_number, version);
        }

        std::mem::drop(blocks);

        Ok(Self {
            dir,
            schema,
            versions: TxnMapLock::with_contents(txn_id, versions),
        })
    }

    fn dir(&self) -> tc_transact::fs::Inner<CacheBlock> {
        self.dir.clone().into_inner()
    }
}

impl fmt::Debug for Service {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a versioned hosted Service")
    }
}

fn resolve_type<T: NativeClass>(subject: Subject) -> TCResult<T> {
    match subject {
        Subject::Link(link) if link.host().is_none() => T::from_path(link.path())
            .ok_or_else(|| bad_request!("{} is not a {}", link.path(), std::any::type_name::<T>())),

        Subject::Link(link) => Err(not_implemented!(
            "support for a user-defined Class of {} in a Service: {}",
            std::any::type_name::<T>(),
            link
        )),

        subject => Err(bad_request!(
            "{} is not a {}",
            subject,
            std::any::type_name::<T>()
        )),
    }
}

fn validate(
    cluster_link: &Link,
    number: &VersionNumber,
    proto: Map<Scalar>,
) -> TCResult<Map<Scalar>> {
    let version_link = cluster_link.clone().append(number.clone());

    let mut validated = Map::new();

    for (id, scalar) in proto.into_iter() {
        if let Scalar::Op(op_def) = scalar {
            let op_def = if op_def.is_write() {
                // make sure not to replicate ops internal to this OpDef
                let op_def = op_def.reference_self::<State>(version_link.path());

                for (id, provider) in op_def.form() {
                    // make sure not to duplicate requests to other clusters
                    if Refer::<State>::is_inter_service_write(provider, version_link.path()) {
                        return Err(bad_request!(
                            "replicated op {} may not perform inter-service writes: {:?}",
                            id,
                            provider
                        ));
                    }
                }

                op_def
            } else {
                // make sure to replicate all write ops internal to this OpDef
                // by routing them through the kernel
                op_def.dereference_self::<State>(version_link.path())
            };

            // TODO: check that the Version does not reference any other Version of the same service

            validated.insert(id, Scalar::Op(op_def));
        } else {
            validated.insert(id, scalar);
        }
    }

    Ok(validated)
}

#[inline]
fn try_cast_into_schema(collection: Scalar) -> TCResult<CollectionSchema> {
    let schema = TCRef::try_from(collection)?;
    if let TCRef::Op(OpRef::Get((classpath, schema))) = schema {
        let classpath = TCPathBuf::try_from(classpath)?;
        let schema = Value::try_from(schema)?;
        (classpath, schema).try_into()
    } else {
        Err(bad_request!("invalid collection schema: {:?}", schema))
    }
}
