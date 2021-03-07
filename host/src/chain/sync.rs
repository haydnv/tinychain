//! A [`super::Chain`] which keeps only the data needed to recover the state of its subject in the
//! event of a transaction failure.
//! INCOMPLETE AND UNSTABLE.

use std::convert::TryInto;

use async_trait::async_trait;
use futures::join;
use log::debug;

use tc_error::*;
use tc_transact::fs::{Dir, File, Persist};
use tc_transact::{Transact, TxnId};
use tcgeneric::{Instance, TCPathBuf};

use crate::fs;
use crate::scalar::{Scalar, Value};

use super::{ChainBlock, ChainInstance, ChainType, Schema, Subject, CHAIN, SUBJECT};

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

        let block_id = SUBJECT.into();
        let mut block = fs::File::get_block_mut(&self.file, &txn_id, block_id).await?;
        block.append(txn_id, path, key, value);
        Ok(())
    }

    fn subject(&self) -> &Subject {
        &self.subject
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

            file.try_into()?
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
        join!(self.subject.commit(txn_id), self.file.commit(txn_id));
    }

    async fn finalize(&self, txn_id: &TxnId) {
        join!(self.subject.finalize(txn_id), self.file.finalize(txn_id));
    }
}
