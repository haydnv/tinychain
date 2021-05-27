//! A [`super::Chain`] which keeps only the data needed to recover the state of its subject in the
//! event of a transaction failure.

use async_trait::async_trait;
use destream::de;
use futures::future::TryFutureExt;
use futures::join;

use tc_error::*;
use tc_transact::fs::{Persist, Store};
use tc_transact::{IntoView, Transact, Transaction, TxnId};
use tcgeneric::TCPathBuf;

use crate::fs;
use crate::scalar::{Link, Value};
use crate::state::{State, StateView};
use crate::txn::Txn;

use super::data::History;
use super::{ChainBlock, ChainInstance, ChainType, Schema, Subject, NULL_HASH};

/// A [`super::Chain`] which keeps only the data needed to recover the state of its subject in the
/// event of a transaction failure.
#[derive(Clone)]
pub struct SyncChain {
    schema: Schema,
    subject: Subject,
    history: History,
}

#[async_trait]
impl ChainInstance for SyncChain {
    async fn append_delete(&self, txn_id: TxnId, path: TCPathBuf, key: Value) -> TCResult<()> {
        let mut block = self.history.write_latest(txn_id).await?;
        block.clear_until(&txn_id);
        block.append_delete(txn_id, path, key);
        Ok(())
    }

    async fn append_put(
        &self,
        txn: Txn,
        path: TCPathBuf,
        key: Value,
        value: State,
    ) -> TCResult<()> {
        {
            let mut block = self.history.write_latest(*txn.id()).await?;
            block.clear_until(txn.id());
        }

        self.history.append_put(txn, path, key, value).await
    }

    async fn last_commit(&self, txn_id: TxnId) -> TCResult<Option<TxnId>> {
        self.history.last_commit(txn_id).await
    }

    fn subject(&self) -> &Subject {
        &self.subject
    }

    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()> {
        let subject = txn.get(source, Value::None).await?;
        self.subject.restore(*txn.id(), subject).await?;

        let mut block = self.history.write_latest(*txn.id()).await?;
        *block = ChainBlock::with_txn(NULL_HASH, *txn.id());

        Ok(())
    }

    async fn write_ahead(&self, txn_id: &TxnId) {
        self.history.commit(txn_id).await
    }
}

#[async_trait]
impl Persist<fs::Dir> for SyncChain {
    type Schema = Schema;
    type Store = fs::Dir;
    type Txn = Txn;

    fn schema(&self) -> &Schema {
        &self.schema
    }

    async fn load(txn: &Txn, schema: Self::Schema, dir: fs::Dir) -> TCResult<Self> {
        let is_new = dir.is_empty(txn.id()).await?;

        let subject = Subject::load(txn, schema.clone(), &dir).await?;

        let history = if is_new {
            History::create(*txn.id(), dir, ChainType::Sync).await?
        } else {
            History::load(txn, (), dir).await?
        };

        let latest = history.latest_block_id(txn.id()).await?;
        if latest > 0 {
            return Err(TCError::internal(format!(
                "a SyncChain can only have one block, found {}",
                latest
            )));
        }

        history.apply_last(txn, &subject).await?;

        Ok(SyncChain {
            schema,
            subject,
            history,
        })
    }
}

#[async_trait]
impl Transact for SyncChain {
    async fn commit(&self, txn_id: &TxnId) {
        self.subject.commit(txn_id).await;
    }

    async fn finalize(&self, txn_id: &TxnId) {
        join!(self.subject.finalize(txn_id), self.history.finalize(txn_id));
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
        let history = History::create(*self.txn.id(), self.txn.context().clone(), ChainType::Sync)
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
            history,
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
