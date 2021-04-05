//! A [`super::Chain`] which keeps only the data needed to recover the state of its subject in the
//! event of a transaction failure.

use std::convert::TryInto;

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
        let mut block = self.file.write_block(txn_id, block_id).await?;
        block.append(txn_id, path, key, value);
        Ok(())
    }

    async fn last_commit(&self, _txn_id: &TxnId) -> TCResult<Option<TxnId>> {
        todo!()
    }

    fn subject(&self) -> &Subject {
        &self.subject
    }

    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()> {
        let subject = txn.get(source, Value::None).await?;
        self.subject
            .put(*txn.id(), TCPathBuf::default(), Value::None, subject)
            .await
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
        let subject = Subject::load(&schema, &dir, txn_id).await?;

        let file = if let Some(file) = dir.get_file(&txn_id, &CHAIN.into()).await? {
            file.try_into()?
        } else {
            let file = dir
                .create_file(txn_id, CHAIN.into(), ChainType::Sync.into())
                .await?;

            let file: fs::File<ChainBlock> = file.try_into()?;
            if !file.contains_block(&txn_id, &CHAIN_BLOCK.into()).await? {
                file.create_block(txn_id, CHAIN_BLOCK.into(), ChainBlock::new(NULL_HASH))
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
                .write_block(*txn_id, CHAIN_BLOCK.into())
                .await
                .unwrap();

            *block = ChainBlock::new(NULL_HASH);
        }

        join!(self.subject.commit(txn_id), self.file.commit(txn_id));
    }

    async fn finalize(&self, txn_id: &TxnId) {
        join!(self.subject.finalize(txn_id), self.file.finalize(txn_id));
    }
}

#[async_trait]
impl<'en> IntoView<'en, fs::Dir> for SyncChain {
    type Txn = Txn;
    type View = (Schema, StateView);

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        let subject = self.subject.at(txn.id()).await?;
        Ok((self.schema, subject.into_view(txn).await?))
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
            .create_file(*self.txn.id(), CHAIN.into(), ChainType::Sync.into())
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
