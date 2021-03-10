//! A [`super::Chain`] which keeps only the data needed to recover the state of its subject in the
//! event of a transaction failure.
//!
//! INCOMPLETE AND UNSTABLE.

use std::convert::TryInto;

use async_trait::async_trait;
use futures::join;
use log::debug;

use tc_error::*;
use tc_transact::fs::{Dir, File, Persist};
use tc_transact::{Transact, Transaction, TxnId};
use tcgeneric::{label, Instance, Label, TCPathBuf};

use crate::fs;
use crate::scalar::{Link, Scalar, Value};
use crate::txn::Txn;

use super::{ChainBlock, ChainInstance, ChainType, Schema, Subject, CHAIN, SUBJECT};

const CHAIN_BLOCK: Label = label("sync");

/// A [`super::Chain`] which keeps only the data needed to recover the state of its subject in the
/// event of a transaction failure.
#[derive(Clone)]
pub struct SyncChain {
    schema: Schema,
    subject: Subject,
    file: fs::File<ChainBlock>,
}

#[async_trait]
impl ChainInstance for SyncChain {
    async fn append(
        &self,
        txn_id: TxnId,
        path: TCPathBuf,
        key: Value,
        value: Scalar,
    ) -> TCResult<()> {
        if value.is_ref() {
            return Err(TCError::bad_request(
                "cannot update Chain subject with reference: {}",
                value,
            ));
        }

        let block_id = CHAIN_BLOCK.into();
        let mut block = fs::File::get_block_mut(&self.file, &txn_id, block_id).await?;
        block.append(txn_id, path, key, value);
        Ok(())
    }

    fn subject(&self) -> &Subject {
        &self.subject
    }

    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()> {
        let subject = txn.get(source, Value::None).await?;
        self.subject.put(txn.id(), Value::None, subject).await
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
                let file: fs::File<Value> =
                    if let Some(file) = dir.get_file(&txn_id, &SUBJECT.into()).await? {
                        file.try_into()?
                    } else {
                        let file = dir
                            .create_file(txn_id, SUBJECT.into(), value.class().into())
                            .await?;

                        file.try_into()?
                    };

                if !file.contains_block(&txn_id, &SUBJECT.into()).await? {
                    debug!("sync chain writing new subject...");
                    file.create_block(txn_id, SUBJECT.into(), value.clone())
                        .await?;
                } else {
                    debug!("sync chain found existing subject");
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

            let file: fs::File<ChainBlock> = file.try_into()?;
            if !file.contains_block(&txn_id, &CHAIN_BLOCK.into()).await? {
                file.create_block(txn_id, CHAIN_BLOCK.into(), ChainBlock::new())
                    .await?;
            }

            file
        };

        Ok(SyncChain {
            schema,
            subject,
            file,
        })
    }
}

#[async_trait]
impl Transact for SyncChain {
    async fn commit(&self, txn_id: &TxnId) {
        {
            let mut block = fs::File::get_block_mut(&self.file, &txn_id, CHAIN_BLOCK.into())
                .await
                .unwrap();

            *block = ChainBlock::new();
        }

        join!(self.subject.commit(txn_id), self.file.commit(txn_id));
    }

    async fn finalize(&self, txn_id: &TxnId) {
        join!(self.subject.finalize(txn_id), self.file.finalize(txn_id));
    }
}
