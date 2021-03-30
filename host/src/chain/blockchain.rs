use async_trait::async_trait;

use tc_error::*;
use tc_transact::fs::Persist;
use tc_transact::Transact;
use tcgeneric::TCPathBuf;

use crate::fs;
use crate::scalar::{Link, Scalar, Value};
use crate::txn::{Txn, TxnId};

use super::{ChainInstance, Schema, Subject};

#[derive(Clone)]
pub struct BlockChain {
    schema: Schema,
    subject: Subject,
}

#[async_trait]
impl ChainInstance for BlockChain {
    async fn append(
        &self,
        _txn_id: TxnId,
        _path: TCPathBuf,
        _key: Value,
        _value: Scalar,
    ) -> TCResult<()> {
        Err(TCError::not_implemented("BlockChain::append"))
    }

    fn subject(&self) -> &Subject {
        &self.subject
    }

    async fn replicate(&self, _txn: &Txn, _source: Link) -> TCResult<()> {
        Err(TCError::not_implemented("BlockChain::replicate"))
    }
}

#[async_trait]
impl Persist for BlockChain {
    type Schema = Schema;
    type Store = fs::Dir;

    fn schema(&self) -> &Schema {
        &self.schema
    }

    async fn load(_schema: Schema, _dir: fs::Dir, _txn_id: TxnId) -> TCResult<Self> {
        Err(TCError::not_implemented("BlockChain::load"))
    }
}

#[async_trait]
impl Transact for BlockChain {
    async fn commit(&self, _txn_id: &TxnId) {
        unimplemented!()
    }

    async fn finalize(&self, _txn_id: &TxnId) {
        unimplemented!()
    }
}
