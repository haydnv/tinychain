use std::sync::Arc;

use async_trait::async_trait;

use error::*;
use generic::{label, Label};
use transact::fs::File;
use transact::TxnId;

use crate::state::scalar::OpRef;

use super::{ChainBlock, ChainInstance};

const BLOCK_ID: Label = label("0");

#[derive(Clone)]
pub struct SyncChain {
    file: Arc<File<ChainBlock>>,
}

#[async_trait]
impl ChainInstance for SyncChain {
    async fn append(&self, txn_id: &TxnId, op_ref: OpRef) -> TCResult<()> {
        let block = self.file.get_block(txn_id, BLOCK_ID.into()).await?;
        let mut block = block.upgrade().await?;
        block.append(op_ref);
        Ok(())
    }
}
