use async_trait::async_trait;
use futures::join;

use tc_error::*;
use tc_transact::fs::{BlockData, File, Persist};
use tc_transact::lock::{Mutable, TxnLock};
use tc_transact::Transact;
use tcgeneric::TCPathBuf;

use crate::fs;
use crate::scalar::{Link, Scalar, Value};
use crate::txn::{Txn, TxnId};

use super::{ChainBlock, ChainInstance, Schema, Subject};

const BLOCK_SIZE: u64 = 1_000_000;

#[derive(Clone)]
pub struct BlockChain {
    schema: Schema,
    subject: Subject,
    latest: TxnLock<Mutable<u64>>,
    file: fs::File<ChainBlock>,
}

#[async_trait]
impl ChainInstance for BlockChain {
    async fn append(
        &self,
        txn_id: TxnId,
        path: TCPathBuf,
        key: Value,
        value: Scalar,
    ) -> TCResult<()> {
        let latest = self.latest.read(&txn_id).await?;
        let mut block = self.file.write_block(txn_id, (*latest).into()).await?;

        block.append(txn_id, path, key, value);
        Ok(())
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
    async fn commit(&self, txn_id: &TxnId) {
        {
            let latest = self.latest.read(txn_id).await.expect("latest block number");

            let block = self
                .file
                .read_block(txn_id, &(*latest).into())
                .await
                .expect("read latest chain block");

            if block.size().await.expect("block size") >= BLOCK_SIZE {
                let mut latest = latest.upgrade().await.expect("latest block number");
                (*latest) += 1;

                self.file
                    .create_block(*txn_id, (*latest).into(), ChainBlock::new())
                    .await
                    .expect("bump chain block number");
                // TODO: include the hash of the last block in the new latest block
            }
        }

        join!(self.latest.commit(txn_id), self.file.commit(txn_id));
    }

    async fn finalize(&self, txn_id: &TxnId) {
        join!(self.latest.finalize(txn_id), self.file.finalize(txn_id));
    }
}
