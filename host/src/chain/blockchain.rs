use std::convert::TryInto;
use std::str::FromStr;

use async_trait::async_trait;
use futures::join;

use tc_error::*;
use tc_transact::fs::{BlockData, Dir, File, Persist};
use tc_transact::lock::{Mutable, TxnLock};
use tc_transact::Transact;
use tcgeneric::TCPathBuf;

use crate::fs;
use crate::scalar::{Link, Scalar, Value};
use crate::txn::{Txn, TxnId};

use super::{ChainBlock, ChainInstance, ChainType, Schema, Subject, CHAIN, NULL_HASH};

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

    async fn load(schema: Schema, dir: fs::Dir, txn_id: TxnId) -> TCResult<Self> {
        let subject = Subject::load(&schema, &dir, txn_id).await?;
        let mut latest = 0;

        let file = if let Some(file) = dir.get_file(&txn_id, &CHAIN.into()).await? {
            let file: fs::File<ChainBlock> = file.try_into()?;

            for block_id in file.block_ids(&txn_id).await? {
                let block_id = u64::from_str(block_id.as_str()).map_err(|e| {
                    TCError::bad_request("blockchain block ID must be a positive integer", e)
                })?;

                if block_id > latest {
                    latest = block_id;
                }
            }

            file
        } else {
            let file = dir
                .create_file(txn_id, CHAIN.into(), ChainType::Sync.into())
                .await?;

            let file: fs::File<ChainBlock> = file.try_into()?;
            if !file.contains_block(&txn_id, &latest.into()).await? {
                file.create_block(txn_id, latest.into(), ChainBlock::new(NULL_HASH))
                    .await?;
            }

            file
        };

        Ok(BlockChain {
            schema,
            subject,
            file,
            latest: TxnLock::new("blockchain latest block number", latest.into()),
        })
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

                let hash = block.hash().await.expect("block hash");

                self.file
                    .create_block(*txn_id, (*latest).into(), ChainBlock::new(hash))
                    .await
                    .expect("bump chain block number");
            }
        }

        join!(self.latest.commit(txn_id), self.subject.commit(txn_id), self.file.commit(txn_id));
    }

    async fn finalize(&self, txn_id: &TxnId) {
        join!(self.latest.finalize(txn_id), self.subject.commit(txn_id), self.file.finalize(txn_id));
    }
}
