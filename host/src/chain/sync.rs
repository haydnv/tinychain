//! A [`super::Chain`] which keeps only the data needed to recover the state of its subject in the
//! event of a transaction failure.

use std::convert::{TryFrom, TryInto};

use async_trait::async_trait;
use destream::de;
use futures::future::TryFutureExt;
use futures::join;

use tc_error::*;
use tc_transact::fs::{Dir, File, Persist};
use tc_transact::{IntoView, Transact, Transaction, TxnId};
use tcgeneric::{label, Label, TCPathBuf};

use crate::fs;
use crate::scalar::{Link, Scalar, Value};
use crate::state::StateView;
use crate::txn::Txn;

use super::{ChainBlock, ChainInstance, ChainType, Schema, Subject, CHAIN, NULL_HASH};

const BLOCK_ID: Label = label("sync");

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
    async fn append_delete(&self, txn_id: TxnId, path: TCPathBuf, key: Value) -> TCResult<()> {
        let mut block = self.file.write_block(txn_id, BLOCK_ID.into()).await?;
        block.append_delete(txn_id, path, key);
        Ok(())
    }

    async fn append_put(
        &self,
        txn_id: TxnId,
        path: TCPathBuf,
        key: Value,
        value: Scalar,
    ) -> TCResult<()> {
        let mut block = self.file.write_block(txn_id, BLOCK_ID.into()).await?;
        block.append_put(txn_id, path, key, value);
        Ok(())
    }

    async fn last_commit(&self, txn_id: TxnId) -> TCResult<Option<TxnId>> {
        let block = self.file.read_block(txn_id, BLOCK_ID.into()).await?;
        Ok(block.mutations().keys().next().cloned())
    }

    fn subject(&self) -> &Subject {
        &self.subject
    }

    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()> {
        let subject = txn.get(source, Value::None).await?;
        self.subject.restore(*txn.id(), subject).await?;

        let mut block = self.file.write_block(*txn.id(), BLOCK_ID.into()).await?;
        *block = ChainBlock::with_txn(NULL_HASH, *txn.id());

        Ok(())
    }

    async fn prepare_commit(&self, txn_id: &TxnId) {
        self.file
            .sync_block(*txn_id, BLOCK_ID.into())
            .await
            .expect("prepare SyncChain commit");
    }
}

#[async_trait]
impl Persist<fs::Dir, Txn> for SyncChain {
    type Schema = Schema;
    type Store = fs::Dir;

    fn schema(&self) -> &Schema {
        &self.schema
    }

    async fn load(txn: &Txn, schema: Self::Schema, dir: fs::Dir) -> TCResult<Self> {
        let subject = Subject::load(txn, schema.clone(), &dir).await?;

        let txn_id = *txn.id();
        let file = if let Some(file) = dir.get_file(&txn_id, &CHAIN.into()).await? {
            let file = fs::File::<ChainBlock>::try_from(file)?;

            let block = file.read_block(txn_id, BLOCK_ID.into()).await?;
            if block.mutations().len() > 1 {
                return Err(TCError::internal(
                    "SyncChain should only store one Transaction record at a time",
                ));
            }

            if let Some((last_txn_id, ops)) = block.mutations().iter().next() {
                for op in ops {
                    subject
                        .apply(txn, op)
                        .map_err(|e| {
                            TCError::internal(format!(
                                "error replaying last transaction {}: {}",
                                last_txn_id, e
                            ))
                        })
                        .await?;
                }
            }

            file
        } else {
            let file = dir
                .create_file(txn_id, CHAIN.into(), ChainType::Sync)
                .await?;

            let file = fs::File::<ChainBlock>::try_from(file)?;
            if !file.contains_block(&txn_id, &BLOCK_ID.into()).await? {
                file.create_block(
                    txn_id,
                    BLOCK_ID.into(),
                    ChainBlock::with_txn(NULL_HASH, txn_id),
                )
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
            let mut block = self
                .file
                .sync_block(*txn_id, BLOCK_ID.into())
                .await
                .unwrap();

            self.subject.commit(txn_id).await;

            *block = ChainBlock::with_txn(NULL_HASH, *txn_id);
        }

        self.file.commit(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        join!(self.subject.finalize(txn_id), self.file.finalize(txn_id));
    }
}

#[async_trait]
impl<'en> IntoView<'en, fs::Dir> for SyncChain {
    type Txn = Txn;
    type View = (Schema, StateView<'en>);

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        Ok((self.schema, self.subject.into_view(txn).await?))
    }
}

struct ChainVisitor {
    txn: Txn,
}

#[async_trait]
impl de::Visitor for ChainVisitor {
    type Value = SyncChain;

    fn expecting() -> &'static str {
        "a SyncChain"
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let file = self
            .txn
            .context()
            .create_file(*self.txn.id(), CHAIN.into(), ChainType::Sync)
            .map_err(de::Error::custom)
            .await?;

        let schema = seq
            .next_element(())
            .await?
            .ok_or_else(|| de::Error::invalid_length(0, "a SyncChain schema"))?;

        let subject = seq
            .next_element(self.txn)
            .await?
            .ok_or_else(|| de::Error::invalid_length(1, "the subject of a SyncChain"))?;

        Ok(SyncChain {
            schema,
            subject,
            file: file.try_into().map_err(de::Error::custom)?,
        })
    }
}

#[async_trait]
impl de::FromStream for SyncChain {
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        let visitor = ChainVisitor { txn };
        decoder.decode_seq(visitor).await
    }
}
