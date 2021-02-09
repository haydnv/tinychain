use async_trait::async_trait;

use error::*;
use generic::{label, Label};
use transact::fs;
use transact::TxnId;

use crate::scalar::OpRef;

use super::{ChainBlock, ChainInstance, Schema, Subject};

type File = crate::fs::File<ChainBlock>;

const BLOCK_ID: Label = label("0");

#[derive(Clone)]
pub struct SyncChain {
    schema: Schema,
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
    type Schema = Schema;
    type Store = File;

    fn schema(&self) -> &'_ Schema {
        &self.schema
    }

    async fn load(_schema: Self::Schema, _file: File) -> TCResult<Self> {
        unimplemented!()
    }
}
