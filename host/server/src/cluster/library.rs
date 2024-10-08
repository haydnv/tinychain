use std::collections::BTreeSet;
use std::fmt;
use std::ops::Deref;

use async_trait::async_trait;
use futures::stream::FuturesUnordered;
use futures::{TryFutureExt, TryStreamExt};
use log::{debug, trace};
use safecast::{AsType, TryCastFrom, TryCastInto};

use tc_error::*;
use tc_scalar::value::Version as VersionNumber;
use tc_scalar::{OpDef, Scalar};
use tc_state::CacheBlock;
use tc_transact::hash::*;
use tc_transact::{fs, Gateway, Replicate, Transact, Transaction, TxnId};
use tc_value::{Link, Value};
use tcgeneric::{label, Label, Map};

use crate::cluster::IsDir;
use crate::Txn;

use super::dir::DirItem;

pub(super) const LIB: Label = label("lib");

/// A versioned collection of [`Scalar`]s
#[derive(Clone)]
pub struct Library {
    versions: fs::File<CacheBlock, Map<Scalar>>,
}

impl Library {
    pub async fn get_version(
        &self,
        txn_id: TxnId,
        number: &VersionNumber,
    ) -> TCResult<impl Deref<Target = Map<Scalar>>> {
        self.versions.read_block(txn_id, &(*number).into()).await
    }

    pub async fn list_versions(
        &self,
        txn_id: TxnId,
    ) -> TCResult<impl Iterator<Item = VersionNumber>> {
        self.versions
            .block_ids(txn_id)
            .map_ok(|block_ids| {
                block_ids.map(|block_id| block_id.as_str().parse().expect("library version number"))
            })
            .await
    }
}

#[async_trait]
impl DirItem for Library {
    type Schema = Map<Scalar>;
    type Version = Map<Scalar>;

    async fn create_version(
        &self,
        txn: &Txn,
        number: VersionNumber,
        schema: Map<Scalar>,
    ) -> TCResult<Map<Scalar>> {
        let version = validate(schema)?;
        let number = number.into();
        let txn_id = *txn.id();

        // TODO: is there a better/less error-prone way to implement this idempotency check?
        if self.versions.contains_block(txn_id, &number).await? {
            let existing = self.versions.read_block(txn_id, &number).await?;
            if version == *existing {
                Ok(version)
            } else {
                Err(bad_request!(
                    "{self:?} cannot overwrite existing version {number}"
                ))
            }
        } else {
            self.versions
                .create_block(*txn.id(), number.into(), version.clone())
                .map_ok(|_| ())
                .await?;

            Ok(version)
        }
    }
}

impl IsDir for Library {}

#[async_trait]
impl Replicate<Txn> for Library {
    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<Output<Sha256>> {
        debug!("replicate {self:?} from {source}...");

        let txn_id = *txn.id();
        let version_ids = txn.get(source.clone(), Value::None).await?;
        let version_ids =
            version_ids.try_into_tuple(|s| TCError::unexpected(s, "library version numbers"))?;

        let version_ids = version_ids
            .into_iter()
            .map(|id| {
                let id = Value::try_from(id)?;
                id.try_cast_into(|v| TCError::unexpected(v, "a library version number"))
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
                    let version =
                        version.try_into_map(|s| TCError::unexpected(s, "a library version"))?;

                    let version = version
                        .into_iter()
                        .map(move |(name, class)| {
                            Scalar::try_cast_from(class, |s| TCError::unexpected(s, "a Library"))
                                .map(|class| (name, class))
                        })
                        .collect::<TCResult<Map<Scalar>>>()?;

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
impl AsyncHash for Library {
    async fn hash(&self, txn_id: TxnId) -> TCResult<Output<Sha256>> {
        let mut versions = self.versions.iter(txn_id).await?;
        let mut is_empty = true;

        let mut hasher = Sha256::new();
        while let Some((number, library)) = versions.try_next().await? {
            let number: VersionNumber = number.as_str().parse()?;
            hasher.update(Hash::<Sha256>::hash((number, &*library)));
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
impl Transact for Library {
    type Commit = ();

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        self.versions.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.versions.rollback(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.versions.finalize(txn_id).await
    }
}

#[async_trait]
impl fs::Persist<CacheBlock> for Library {
    type Txn = Txn;
    type Schema = ();

    async fn create(txn_id: TxnId, schema: (), dir: fs::Dir<CacheBlock>) -> TCResult<Self> {
        if dir.is_empty(txn_id).await? {
            dir.create_file::<Map<Scalar>>(txn_id, LIB.into()).await?;
            Self::load(txn_id, schema, dir).await
        } else {
            Err(bad_request!(
                "creating a new versioned library requires an empty file"
            ))
        }
    }

    async fn load(txn_id: TxnId, _schema: (), dir: fs::Dir<CacheBlock>) -> TCResult<Self> {
        dir.get_file(txn_id, &LIB.into())
            .map_ok(|versions| Self { versions })
            .await
    }

    fn dir(&self) -> fs::Inner<CacheBlock> {
        self.versions.clone().into_inner()
    }
}

impl fmt::Debug for Library {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a versioned library")
    }
}

fn validate(proto: Map<Scalar>) -> TCResult<Map<Scalar>> {
    if proto
        .values()
        .filter_map(|member| member.as_type())
        .any(|op: &OpDef| op.is_write())
    {
        Err(bad_request!("a library may not define write operations"))
    } else {
        Ok(proto)
    }
}
