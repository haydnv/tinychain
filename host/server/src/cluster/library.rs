use std::fmt;
use std::ops::Deref;

use async_trait::async_trait;
use futures::{TryFutureExt, TryStreamExt};
use safecast::AsType;

use tc_error::*;
use tc_scalar::value::Version as VersionNumber;
use tc_scalar::{OpDef, Scalar};
use tc_state::CacheBlock;
use tc_transact::fs;
use tc_transact::hash::{AsyncHash, Digest, Output, Sha256};
use tc_transact::{Transact, Transaction, TxnId};
use tcgeneric::Map;

use crate::{State, Txn};

use super::dir::DirItem;

/// A versioned collection of [`Scalar`]s
#[derive(Clone)]
pub struct Library {
    versions: fs::File<CacheBlock, Map<Scalar>>,
}

impl Library {
    pub async fn latest(&self, txn_id: TxnId) -> TCResult<Option<VersionNumber>> {
        let block_ids = self.versions.block_ids(txn_id).await?;

        Ok(block_ids
            .last()
            .map(|id| id.as_str().parse().expect("version number")))
    }

    pub async fn get_version(
        &self,
        txn_id: TxnId,
        number: &VersionNumber,
    ) -> TCResult<impl Deref<Target = Map<Scalar>>> {
        self.versions.read_block(txn_id, &(*number).into()).await
    }

    pub async fn to_state(&self, txn_id: TxnId) -> TCResult<State> {
        let mut blocks = self.versions.iter(txn_id).await?;

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
    type Version = Map<Scalar>;

    async fn create_version(
        &self,
        txn: &Txn,
        number: VersionNumber,
        schema: Map<Scalar>,
    ) -> TCResult<Map<Scalar>> {
        let version = validate(schema)?;

        self.versions
            .create_block(*txn.id(), number.into(), version.clone())
            .map_ok(|_| ())
            .await?;

        Ok(version.into())
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
            hasher.update(async_hash::Hash::<Sha256>::hash((number, &*library)));
            is_empty = false;
        }

        if is_empty {
            Ok(async_hash::default_hash::<Sha256>())
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
            Self::load(txn_id, schema, dir).await
        } else {
            Err(bad_request!(
                "creating a new versioned library requires an empty file"
            ))
        }
    }

    async fn load(txn_id: TxnId, _schema: (), dir: fs::Dir<CacheBlock>) -> TCResult<Self> {
        fs::File::load(dir.into_inner(), txn_id)
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
