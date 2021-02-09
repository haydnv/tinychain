use std::convert::TryInto;

use async_trait::async_trait;
use bytes::Bytes;

use error::*;
use generic::{label, Instance, Label};
use transact::fs::{Dir, File, Persist};
use transact::TxnId;

use crate::fs;
use crate::scalar::OpRef;

use super::{ChainBlock, ChainInstance, Schema, Subject};
use crate::chain::ChainType;

const CHAIN: Label = label("chain");
const SUBJECT: Label = label("subject");
const BLOCK_ID: Label = label("0");

#[derive(Clone)]
pub struct SyncChain {
    schema: Schema,
    subject: Subject,
    file: fs::File<ChainBlock>,
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
impl Persist for SyncChain {
    type Schema = Schema;
    type Store = fs::Dir;

    fn schema(&self) -> &'_ Schema {
        &self.schema
    }

    async fn load(schema: Self::Schema, dir: fs::Dir, txn_id: TxnId) -> TCResult<Self> {
        let subject = match &schema {
            Schema::Value(value) => {
                let file: fs::File<Bytes> =
                    if let Some(file) = dir.get_file(&txn_id, &SUBJECT.into()).await? {
                        file.try_into()?
                    } else {
                        let file = dir
                            .create_file(txn_id, SUBJECT.into(), value.class().into())
                            .await?;
                        file.try_into()?
                    };

                if !file.block_exists(&txn_id, &SUBJECT.into()).await? {
                    let as_bytes = serde_json::to_vec(value)
                        .map_err(|e| TCError::bad_request("unable to serialize value", e))?;
                    file.create_block(txn_id, SUBJECT.into(), Bytes::from(as_bytes))
                        .await?;
                }

                Subject::Value(file)
            }
        };

        let file = if let Some(file) = dir.get_file(&txn_id, &CHAIN.into()).await? {
            file.try_into()?
        } else {
            let file = dir
                .create_file(txn_id, CHAIN.into(), ChainType::Sync.into())
                .await?;
            file.try_into()?
        };

        Ok(SyncChain {
            schema,
            subject,
            file,
        })
    }
}
