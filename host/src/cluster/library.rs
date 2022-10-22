use std::fmt;

use async_trait::async_trait;

use tc_error::*;
use tc_transact::fs::DirCreate;
use tc_transact::{Transact, TxnId};
use tcgeneric::PathSegment;

use crate::fs;
use crate::scalar::value::Link;
use crate::txn::Txn;

use super::Replica;

#[derive(Clone)]
pub struct Dir {
    dir: fs::Dir,
}

impl Dir {
    pub async fn create_dir(&self, txn_id: TxnId, name: PathSegment) -> TCResult<Self> {
        let mut lock = tc_transact::fs::Dir::write(&self.dir, txn_id).await?;
        lock.create_dir(name).map(|dir| Self { dir })
    }
}

#[async_trait]
impl Replica for Dir {
    async fn replicate(&self, _txn: &Txn, _source: &Link) -> TCResult<()> {
        Err(TCError::not_implemented("cluster::Dir::replicate"))
    }
}

#[async_trait]
impl Transact for Dir {
    type Commit = <fs::Dir as Transact>::Commit;

    async fn commit(&self, txn_id: &TxnId) -> Self::Commit {
        self.dir.commit(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.dir.finalize(txn_id).await
    }
}

impl From<fs::Dir> for Dir {
    fn from(dir: fs::Dir) -> Self {
        Self { dir }
    }
}

impl fmt::Display for Dir {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("library directory")
    }
}
