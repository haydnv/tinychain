use async_trait::async_trait;

use error::*;
use generic::{label, Label};
use transact::fs;
use transact::TxnId;
use value::Value;

use crate::scalar::OpRef;

use super::{ChainBlock, ChainInstance, Subject};

type File = crate::fs::File<ChainBlock>;

const BLOCK_ID: Label = label("0");

#[derive(Clone)]
pub struct SyncChain {
    subject: Subject,
    file: File,
}

#[async_trait]
impl ChainInstance for SyncChain {
    async fn append(&self, txn_id: &TxnId, op_ref: OpRef) -> TCResult<()> {
        let block_id = BLOCK_ID.into();
        let mut block = fs::File::get_block_mut(&self.file, txn_id, &block_id).await?;
        block.append(op_ref);
        Ok(())
    }
}

#[async_trait]
impl fs::Persist for SyncChain {
    type Store = File;
    type Builder = Builder;

    fn builder(_file: File) -> Self::Builder {
        unimplemented!()
    }

    async fn load(_file: File) -> TCResult<Self> {
        unimplemented!()
    }
}

pub struct Builder {
    store: File,
    subject: Option<Value>,
}

impl Builder {
    pub fn subject(mut self, value: Value) -> Self {
        self.subject = Some(value);
        self
    }
}

#[async_trait]
impl fs::Builder for Builder {
    type Store = File;

    async fn build(self, _txn_id: TxnId) -> TCResult<Self::Store> {
        unimplemented!()
    }

    fn store(&self) -> &Self::Store {
        unimplemented!()
    }

    fn into_store(self) -> Self::Store {
        self.store
    }
}
