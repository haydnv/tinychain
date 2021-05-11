use futures::TryFutureExt;

use tc_error::*;
use tc_transact::fs::{BlockData, BlockId, File};
use tc_transact::lock::{Mutable, TxnLock};
use tc_transact::TxnId;
use tcgeneric::{Instance, TCPathBuf};

use crate::fs;
use crate::scalar::Value;
use crate::state::State;

use super::ChainBlock;

pub struct ChainData {
    file: fs::File<ChainBlock>,
    latest: TxnLock<Mutable<u64>>,
}

impl ChainData {
    pub fn new(latest: u64, file: fs::File<ChainBlock>) -> Self {
        Self {
            latest: TxnLock::new("latest BlockChain block ordinal", latest.into()),
            file,
        }
    }

    pub async fn append_delete(&self, txn_id: TxnId, path: TCPathBuf, key: Value) -> TCResult<()> {
        let mut block = self.write_latest(txn_id).await?;
        block.append_delete(txn_id, path, key);
        Ok(())
    }

    pub async fn append_put(
        &self,
        txn_id: TxnId,
        path: TCPathBuf,
        key: Value,
        value: State,
    ) -> TCResult<()> {
        if value.is_ref() {
            return Err(TCError::bad_request(
                "cannot update Chain with reference: {}",
                value,
            ));
        }

        let value_ref = match value {
            State::Scalar(value) => Ok(value),
            other => Err(TCError::not_implemented(format!(
                "Chain <- {}",
                other.class()
            ))),
        }?;

        let mut block = self.write_latest(txn_id).await?;
        block.append_put(txn_id, path, key, value_ref);
        Ok(())
    }

    pub async fn latest_block_id(&self, txn_id: &TxnId) -> TCResult<u64> {
        self.latest.read(txn_id).map_ok(|id| *id).await
    }

    pub async fn create_next_block(&self, txn_id: TxnId) -> TCResult<fs::Block<ChainBlock>> {
        let mut latest = self.latest.write(txn_id).await?;
        let last_block = self.read_block(txn_id, (*latest).into()).await?;
        let hash = last_block.hash().await?;
        let block = ChainBlock::new(hash);

        (*latest) += 1;
        self.file
            .create_block(txn_id, (*latest).into(), block)
            .await
    }

    pub async fn read_block(
        &self,
        txn_id: TxnId,
        block_id: BlockId,
    ) -> TCResult<fs::BlockRead<ChainBlock>> {
        self.file.read_block(txn_id, block_id).await
    }

    pub async fn write_block(
        &self,
        txn_id: TxnId,
        block_id: BlockId,
    ) -> TCResult<fs::BlockWrite<ChainBlock>> {
        self.file.write_block(txn_id, block_id).await
    }

    pub async fn read_latest(&self, txn_id: TxnId) -> TCResult<fs::BlockRead<ChainBlock>> {
        let latest = self.latest.read(&txn_id).await?;
        self.read_block(txn_id, (*latest).into()).await
    }

    pub async fn write_latest(&self, txn_id: TxnId) -> TCResult<fs::BlockWrite<ChainBlock>> {
        let latest = self.latest.read(&txn_id).await?;
        self.write_block(txn_id, (*latest).into()).await
    }

    pub async fn prepare_commit(&self, txn_id: &TxnId) {
        let latest = self.latest.read(txn_id).await.expect("latest block");

        self.file
            .sync_block(*txn_id, (*latest).into())
            .await
            .expect("prepare BlockChain commit");
    }
}
