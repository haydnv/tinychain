use async_trait::async_trait;
use destream::de;
use futures::TryFutureExt;
use futures_locks::RwLock;

use error::*;
use generic::{label, Label};
use transact::fs::{Dir, File};

use crate::fs::{FileView, InstanceFile};
use crate::scalar::OpRef;
use crate::txn::{Transaction, Txn, TxnId};

use super::{ChainBlock, ChainInstance};

const FILE_ID: Label = label("sync_chain");
const BLOCK_ID: Label = label("0");

#[derive(Clone)]
pub struct SyncChain {
    file: InstanceFile<ChainBlock>,
}

impl SyncChain {
    pub async fn load(file: InstanceFile<ChainBlock>) -> TCResult<Self> {
        // TODO: validate file
        Ok(Self { file })
    }
}

#[async_trait]
impl ChainInstance for SyncChain {
    async fn file(&self, txn_id: &TxnId) -> TCResult<RwLock<FileView<ChainBlock>>> {
        self.file.version(txn_id).await
    }

    async fn append(&self, txn_id: &TxnId, op_ref: OpRef) -> TCResult<()> {
        let file = self.file.version(txn_id).await?;
        let lock = file.read().await;
        let block_id = BLOCK_ID.into();
        let block = lock.get_block(&block_id).await?;
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

        let mut dir = txn.context().write().await;
        let file: RwLock<FileView<ChainBlock>> = dir
            .create_file(FILE_ID.into())
            .map_err(de::Error::custom)
            .await?;

        file.write()
            .await
            .create_block(BLOCK_ID.into(), block)
            .map_err(de::Error::custom)
            .await?;

        Ok(Self { file: file.into() })
    }
}
