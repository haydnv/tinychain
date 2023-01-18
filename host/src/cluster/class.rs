/// A replicated, versioned set of [`InstanceClass`]es
use std::convert::TryFrom;
use std::fmt;
use std::ops::Deref;

use async_trait::async_trait;

use tc_error::*;
use tc_transact::fs::*;
use tc_transact::{Transact, Transaction, TxnId};
use tc_value::Version as VersionNumber;
use tcgeneric::{Id, Map};

use crate::fs;
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
        let file = self.classes.read(txn_id).await?;

        let mut classes = Map::new();
        for block_id in file.block_ids() {
            let class = file.read_block(&block_id).await?;
            classes.insert(block_id.into(), class.clone().into());
        }

        Ok(State::Map(classes))
    }

    pub async fn get_class(
        &self,
        txn_id: TxnId,
        name: &Id,
    ) -> TCResult<impl Deref<Target = InstanceClass>> {
        let file = self.classes.read(txn_id).await?;
        file.read_block(name).await
    }
}

impl fmt::Display for Version {
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
        let dir = self.dir.read(txn_id).await?;
        if dir.is_empty() {
            Ok(None)
        } else {
            let zero = VersionNumber::default().into();
            dir.file_names()
                .fold(&zero, Ord::max)
                .as_str()
                .parse()
                .map(Some)
        }
    }

    pub async fn get_version(&self, txn_id: TxnId, number: &VersionNumber) -> TCResult<Version> {
        let dir = self.dir.read(txn_id).await?;
        if let Some(file) = dir.get_file(&number.clone().into())? {
            Ok(Version::with_file(file))
        } else {
            Err(TCError::not_found(number))
        }
    }

    pub async fn to_state(&self, txn_id: TxnId) -> TCResult<State> {
        let dir = self.dir.read(txn_id).await?;

        let mut versions = Map::new();
        for (number, file) in dir.iter() {
            let file = match file {
                fs::DirEntry::File(fs::FileEntry::Class(file)) => Ok(file.clone()),
                other => Err(unexpected!(
                    "class directory contains invalid file: {}",
                    other
                )),
            }?;

            let version = Version::with_file(file).to_state(txn_id).await?;
            versions.insert(number.clone(), version);
        }

        Ok(State::Map(versions))
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

        let mut dir = self.dir.write(txn_id).await?;
        let file = dir.create_file(number.into())?;
        let mut blocks = file.write(txn_id).await?;

        for (name, class) in &schema {
            blocks.create_block(name.clone(), class.clone(), 0).await?;
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

impl Persist<fs::Dir> for Class {
    type Txn = Txn;
    type Schema = ();

    fn create(txn_id: TxnId, _schema: (), store: fs::Store) -> TCResult<Self> {
        let dir = fs::Dir::try_from(store)?;
        let contents = dir.try_write(txn_id)?;

        if contents.is_empty() {
            Ok(Self { dir })
        } else {
            Err(bad_request!(
                "cannot create a new Class cluster with a non-empty directory",
            ))
        }
    }

    fn load(_txn_id: TxnId, _schema: (), store: fs::Store) -> TCResult<Self> {
        fs::Dir::try_from(store).map(|dir| Self { dir })
    }

    fn dir(&self) -> <fs::Dir as Dir>::Inner {
        self.dir.clone().into_inner()
    }
}

impl fmt::Display for Class {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a versioned set of classes")
    }
}
