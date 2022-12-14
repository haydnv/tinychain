use std::fmt;

use async_trait::async_trait;

use tc_error::*;
use tc_transact::fs::{Dir, Persist};
use tc_transact::{Transact, TxnId};
use tc_value::Version as VersionNumber;
use tcgeneric::{Id, Map};

use crate::fs;
use crate::object::InstanceClass;
use crate::state::State;
use crate::txn::Txn;

use super::DirItem;

#[derive(Clone)]
pub struct Version {
    classes: fs::File<Id, InstanceClass>,
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a set of classes")
    }
}

#[derive(Clone)]
pub struct Class {
    dir: fs::Dir,
}

impl Class {
    pub async fn latest(&self, _txn_id: TxnId) -> TCResult<Option<VersionNumber>> {
        Err(TCError::not_implemented("cluster::Class::latest"))
    }

    pub async fn get_version(
        &self,
        _txn_id: TxnId,
        _number: VersionNumber,
    ) -> TCResult<fs::BlockReadGuard<Version>> {
        Err(TCError::not_implemented("cluster::Class::get_version"))
    }

    pub async fn to_state(&self, _txn_id: TxnId) -> TCResult<State> {
        Err(TCError::not_implemented("cluster::Class::get_version"))
    }
}

#[async_trait]
impl DirItem for Class {
    type Version = Map<InstanceClass>;

    async fn create_version(
        &self,
        _txn_id: TxnId,
        _number: VersionNumber,
        _version: Self::Version,
    ) -> TCResult<()> {
        Err(TCError::not_implemented("Class::create_version"))
    }
}

#[async_trait]
impl Transact for Class {
    type Commit = <fs::Dir as Transact>::Commit;

    async fn commit(&self, txn_id: &TxnId) -> Self::Commit {
        self.dir.commit(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.dir.finalize(txn_id).await
    }
}

#[async_trait]
impl Persist<fs::Dir> for Class {
    type Txn = Txn;
    type Schema = ();

    async fn create(_txn: &Self::Txn, _schema: Self::Schema, _store: fs::Store) -> TCResult<Self> {
        Err(TCError::not_implemented("cluster::Class::create"))
    }

    async fn load(_txn: &Self::Txn, _schema: Self::Schema, _store: fs::Store) -> TCResult<Self> {
        Err(TCError::not_implemented("cluster::Class::load"))
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
