use async_trait::async_trait;
use destream::de;
use futures::TryFutureExt;

use error::*;
use generic::{label, Label};
use transact::fs::File;

use crate::state::scalar::OpRef;
use crate::txn::{Transaction, Txn, TxnId};

use super::{ChainBlock, ChainInstance};

const FILE_ID: Label = label("sync_chain");
const BLOCK_ID: Label = label("0");

#[derive(Clone)]
pub struct SyncChain {
    file: File<ChainBlock>,
}

#[async_trait]
impl ChainInstance for SyncChain {
    fn file(&'_ self) -> &'_ File<ChainBlock> {
        &self.file
    }

    fn len(&self) -> u64 {
        1
    }

    async fn append(&self, txn_id: &TxnId, op_ref: OpRef) -> TCResult<()> {
        let block = self.file.get_block(txn_id, BLOCK_ID.into()).await?;
        let mut block = block.upgrade().await?;
        block.append(op_ref);
        Ok(())
    }
}

#[async_trait]
impl de::FromStream for SyncChain {
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        let [block]: [ChainBlock; 1] = de::FromStream::from_stream((), decoder).await?;

        let file: File<ChainBlock> = txn
            .context()
            .await
            .create_file(*txn.id(), FILE_ID.into())
            .map_err(de::Error::custom)
            .await?;

        file.clone()
            .create_block(*txn.id(), BLOCK_ID.into(), block)
            .map_err(de::Error::custom)
            .await?;

        Ok(Self { file })
    }
}
