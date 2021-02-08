use async_trait::async_trait;
use destream::de;

use error::*;
use generic::{label, Label};
use transact::fs::File;
use transact::TxnId;

use crate::fs;
use crate::scalar::{OpRef, Scalar};
use crate::txn::Txn;

use super::{ChainBlock, ChainInstance, Subject};

const BLOCK_ID: Label = label("0");

#[derive(Clone)]
pub struct SyncChain {
    subject: Subject,
    file: fs::File<ChainBlock>,
}

impl SyncChain {
    pub async fn load(_file: fs::File<ChainBlock>, _schema: Scalar) -> TCResult<Self> {
        unimplemented!()
    }
}

#[async_trait]
impl ChainInstance for SyncChain {
    async fn append(&self, txn_id: &TxnId, op_ref: OpRef) -> TCResult<()> {
        let block_id = BLOCK_ID.into();
        let block = self.file.get_block(txn_id, &block_id).await?;
        let mut block = block.upgrade().await?;
        block.append(op_ref);
        Ok(())
    }
}

#[async_trait]
impl de::FromStream for SyncChain {
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(_txn: Txn, _decoder: &mut D) -> Result<Self, D::Error> {
        unimplemented!()
    }
}
