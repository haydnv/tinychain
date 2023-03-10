/// A replicated, versioned set of [`InstanceClass`]es
use std::convert::TryFrom;
use std::fmt;
use std::ops::Deref;

use async_trait::async_trait;
use futures::TryFutureExt;

use tc_error::*;
use tc_transact::fs::*;
use tc_transact::{Transact, Transaction, TxnId};
use tc_value::Version as VersionNumber;
use tcgeneric::{Id, Map};

use crate::fs;
use crate::fs::CacheBlock;
use crate::object::InstanceClass;
use crate::state::State;
use crate::txn::Txn;

use super::DirItem;

/// A version of a set of [`InstanceClass`]es
#[derive(Clone)]
pub struct Version {
    classes: fs::File<Id, InstanceClass>,
}

impl Version {
    fn with_file(classes: fs::File<Id, InstanceClass>) -> Self {
        Self { classes }
    }

    async fn to_state(&self, txn_id: TxnId) -> TCResult<State> {
        todo!()

        // let mut classes = Map::new();
        // for (block_id, class) in self.classes.iter(txn_id) {
        //     classes.insert(block_id.into(), class.clone().into());
        // }
        // Ok(State::Map(classes))
    }

    pub async fn get_class(
        &self,
        txn_id: TxnId,
        name: &Id,
    ) -> TCResult<impl Deref<Target = InstanceClass>> {
        self.classes.read_block(txn_id, name).await
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
    dir: fs::Dir,
}

impl Class {
    pub async fn latest(&self, txn_id: TxnId) -> TCResult<Option<VersionNumber>> {
        if self.dir.is_empty(txn_id).await? {
            Ok(None)
        } else {
            // let zero = VersionNumber::default().into();
            // let file_names = self.dir.file_names(txn_id).await?;
            //
            // file_names.fold(&zero, Ord::max).as_str().parse().map(Some)
            todo!()
        }
    }

    pub async fn get_version(&self, txn_id: TxnId, number: &VersionNumber) -> TCResult<Version> {
        self.dir
            .get_file(txn_id, &number.clone().into())
            .map_ok(|file| Version::with_file(file))
            .await
    }

    pub async fn to_state(&self, txn_id: TxnId) -> TCResult<State> {
        todo!()
        // let mut versions = Map::new();
        // for (number, file) in self.dir.iter(txn_id).await? {
        // let file = match file {
        //     fs::DirEntry::File(fs::FileEntry::Class(file)) => Ok(file.clone()),
        //     other => Err(unexpected!(
        //         "class directory contains invalid file: {}",
        //         other
        //     )),
        // }?;
        //
        // let version = Version::with_file(file).to_state(txn_id).await?;
        // versions.insert(number.clone(), version);
        // }

        // Ok(State::Map(versions))
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
                .create_block(txn_id, name.clone(), class.clone())
                .await?;
        }

        Ok(schema)
    }
}

#[async_trait]
impl Transact for Class {
    type Commit = <fs::Dir as Transact>::Commit;

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        self.dir.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.dir.rollback(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.dir.finalize(txn_id).await
    }
}

#[async_trait]
impl Persist<CacheBlock> for Class {
    type Txn = Txn;
    type Schema = ();

    async fn create(txn_id: TxnId, _schema: (), dir: fs::Dir) -> TCResult<Self> {
        if dir.is_empty(txn_id).await? {
            Ok(Self { dir })
        } else {
            Err(bad_request!(
                "cannot create a new Class cluster with a non-empty directory",
            ))
        }
    }

    async fn load(_txn_id: TxnId, _schema: (), dir: fs::Dir) -> TCResult<Self> {
        Ok(Self { dir })
    }

    fn dir(&self) -> tc_transact::fs::Inner<CacheBlock> {
        self.dir.clone().into_inner()
    }
}

impl fmt::Debug for Class {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a versioned set of classes")
    }
}
