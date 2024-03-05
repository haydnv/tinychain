/// A replicated, versioned set of [`InstanceClass`]es
use std::fmt;

use async_trait::async_trait;
use futures::stream::FuturesOrdered;
use futures::{TryFutureExt, TryStreamExt};

use tc_error::*;
use tc_scalar::Scalar;
use tc_state::object::InstanceClass;
use tc_transact::fs;
use tc_transact::hash::{AsyncHash, Digest, Hash, Output, Sha256};
use tc_transact::{Transact, Transaction, TxnId};
use tc_value::{Link, Version as VersionNumber};
use tcgeneric::{Id, Map};

use crate::{CacheBlock, State, Txn};

use super::dir::DirItem;

/// A version of a set of [`InstanceClass`]es
#[derive(Clone)]
pub struct Version {
    classes: fs::File<CacheBlock, (Link, Map<Scalar>)>,
}

impl Version {
    fn with_file(classes: fs::File<CacheBlock, (Link, Map<Scalar>)>) -> Self {
        Self { classes }
    }

    pub async fn to_state(&self, txn_id: TxnId) -> TCResult<State> {
        let mut classes = Map::new();

        let mut blocks = self.classes.iter(txn_id).await?;
        while let Some((block_id, class)) = blocks.try_next().await? {
            let (link, proto) = class.clone();

            classes.insert(
                (*block_id).clone(),
                State::Scalar(Scalar::Tuple(vec![link.into(), proto.into()].into())),
            );
        }

        Ok(State::Map(classes))
    }

    // TODO: there should be a way to return a reference to the block, not clone it
    pub async fn get_class(&self, txn_id: TxnId, name: &Id) -> TCResult<InstanceClass> {
        self.classes
            .read_block(txn_id, name)
            .map_ok(|block| InstanceClass::from(block.clone()))
            .await
    }
}

#[async_trait]
impl AsyncHash for Version {
    async fn hash(&self, txn_id: TxnId) -> TCResult<Output<Sha256>> {
        let mut blocks = self.classes.iter(txn_id).await?;
        let mut is_empty = true;
        let mut hasher = Sha256::new();

        while let Some((name, class)) = blocks.try_next().await? {
            hasher.update(Hash::<Sha256>::hash((&*name, &*class)));
            is_empty = false;
        }

        if is_empty {
            Ok(async_hash::default_hash::<Sha256>())
        } else {
            Ok(hasher.finalize())
        }
    }
}

impl fmt::Debug for Version {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a set of classes")
    }
}

/// A versioned set of [`InstanceClass`]es
#[derive(Clone)]
pub struct Class {
    dir: fs::Dir<CacheBlock>,
}

impl Class {
    pub async fn get_version(&self, txn_id: TxnId, number: &VersionNumber) -> TCResult<Version> {
        self.dir
            .get_file(txn_id, &number.clone().into())
            .map_ok(|file| Version::with_file(file))
            .await
    }
}

#[async_trait]
impl DirItem for Class {
    type Schema = Map<InstanceClass>;
    type Version = Map<InstanceClass>;

    async fn create_version(
        &self,
        txn: &Txn,
        number: VersionNumber,
        schema: Map<InstanceClass>,
    ) -> TCResult<Map<InstanceClass>> {
        let txn_id = *txn.id();

        let blocks = self.dir.create_file(txn_id, number.into()).await?;

        for (name, class) in &schema {
            blocks
                .create_block(txn_id, name.clone(), class.clone().into_inner())
                .await?;
        }

        Ok(schema)
    }
}

#[async_trait]
impl Transact for Class {
    type Commit = ();

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        self.dir.commit(txn_id, true).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.dir.rollback(*txn_id, true).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.dir.finalize(*txn_id).await
    }
}

#[async_trait]
impl AsyncHash for Class {
    async fn hash(&self, txn_id: TxnId) -> TCResult<Output<Sha256>> {
        let versions = self.dir.files(txn_id).await?;
        let mut is_empty = true;

        let mut versions: FuturesOrdered<_> = versions
            .map(|(number, file)| async move {
                let number: VersionNumber = number.as_str().parse()?;
                let version_hash = Version::with_file(file).hash(txn_id).await?;

                let mut hasher = Sha256::default();
                hasher.update(Hash::<Sha256>::hash(number));
                hasher.update(version_hash);
                TCResult::Ok(hasher.finalize())
            })
            .collect();

        let mut hasher = Sha256::new();
        while let Some(hash) = versions.try_next().await? {
            hasher.update(hash);
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
impl fs::Persist<CacheBlock> for Class {
    type Txn = Txn;
    type Schema = ();

    async fn create(txn_id: TxnId, _schema: (), dir: fs::Dir<CacheBlock>) -> TCResult<Self> {
        if dir.is_empty(txn_id).await? {
            Ok(Self { dir })
        } else {
            Err(bad_request!(
                "cannot create a new Class cluster with a non-empty directory",
            ))
        }
    }

    async fn load(_txn_id: TxnId, _schema: (), dir: fs::Dir<CacheBlock>) -> TCResult<Self> {
        Ok(Self { dir })
    }

    fn dir(&self) -> fs::Inner<CacheBlock> {
        self.dir.clone().into_inner()
    }
}

impl fmt::Debug for Class {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a versioned set of classes")
    }
}
