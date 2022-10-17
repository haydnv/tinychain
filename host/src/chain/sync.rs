//! A [`super::Chain`] which keeps only the data needed to recover the state of its subject in the
//! event of a transaction failure.

use async_trait::async_trait;
use destream::de;
use freqfs::{FileLock, FileWriteGuard};
use futures::future::TryFutureExt;
use futures::try_join;
use log::{debug, trace};
use sha2::digest::Output;
use sha2::Sha256;

use tc_error::*;
use tc_transact::fs::{Dir, DirCreate, DirCreateFile, Persist};
use tc_transact::{IntoView, Transact, Transaction, TxnId};
use tc_value::{Link, Value};
use tcgeneric::{label, Id, Label, TCPathBuf};

use crate::fs;
use crate::state::{State, StateView};
use crate::txn::Txn;

use super::{null_hash, ChainBlock, ChainInstance, Schema, Subject};

const BLOCKS: Label = label("blocks");
const COMMITTED: &str = "committed.chain_block";
const PENDING: &str = "pending.chain_block";
const STORE: Label = label("store");

/// A [`super::Chain`] which keeps only the data needed to recover the state of its subject in the
/// event of a transaction failure.
#[derive(Clone)]
pub struct SyncChain {
    schema: Schema,
    subject: Subject,
    pending: FileLock<fs::CacheBlock>,
    committed: FileLock<fs::CacheBlock>,
    store: super::data::Store,
}

#[async_trait]
impl ChainInstance for SyncChain {
    async fn append_delete(&self, txn_id: TxnId, path: TCPathBuf, key: Value) -> TCResult<()> {
        let mut block: FileWriteGuard<_, ChainBlock> =
            self.pending.write().map_err(fs::io_err).await?;

        block.append_delete(txn_id, path, key);
        Ok(())
    }

    async fn append_put(
        &self,
        txn: &Txn,
        path: TCPathBuf,
        key: Value,
        value: State,
    ) -> TCResult<()> {
        debug!("SyncChain::append_put {}: {} <- {}", path, key, value);

        let value = self.store.save_state(txn, value).await?;

        debug!("locking pending transaction log block...");
        let mut block: FileWriteGuard<_, ChainBlock> =
            self.pending.write().map_err(fs::io_err).await?;

        block.append_put(*txn.id(), path, key, value);

        debug!("locked pending transaction log block");
        Ok(())
    }

    async fn hash(self, txn: Txn) -> TCResult<Output<Sha256>> {
        self.subject.hash(txn).await
    }

    fn subject(&self) -> &Subject {
        &self.subject
    }

    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()> {
        let subject = txn.get(source, Value::None).await?;
        self.subject.restore(txn, subject).await?;

        let (mut pending, mut committed): (
            FileWriteGuard<_, ChainBlock>,
            FileWriteGuard<_, ChainBlock>,
        ) = try_join!(
            self.pending.write().map_err(fs::io_err),
            self.committed.write().map_err(fs::io_err)
        )?;

        *pending = ChainBlock::with_txn(null_hash().to_vec(), *txn.id());
        *committed = ChainBlock::new(null_hash().to_vec());

        Ok(())
    }

    async fn write_ahead(&self, txn_id: &TxnId) {
        self.store.commit(txn_id).await;

        {
            let (mut pending, mut committed): (
                FileWriteGuard<_, ChainBlock>,
                FileWriteGuard<_, ChainBlock>,
            ) = try_join!(
                self.pending.write().map_err(fs::io_err),
                self.committed.write().map_err(fs::io_err)
            )
            .expect("SyncChain blocks");

            if let Some(mutations) = pending.mutations.remove(txn_id) {
                committed.mutations.insert(*txn_id, mutations);
            }
        }

        try_join!(self.pending.sync(false), self.committed.sync(false))
            .expect("sync SyncChain blocks");
    }
}

#[async_trait]
impl Persist<fs::Dir> for SyncChain {
    type Schema = Schema;
    type Store = fs::Dir;
    type Txn = Txn;

    async fn load(txn: &Txn, schema: Self::Schema, dir: fs::Dir) -> TCResult<Self> {
        debug!("SyncChain::load");

        let subject = Subject::load(txn, schema.clone(), &dir).await?;

        trace!("SyncChain::load loaded {}", subject);

        let txn_id = *txn.id();
        let store = {
            let mut dir = dir.write(txn_id).await?;
            dir.get_or_create_dir(STORE.into())
                .map(super::data::Store::new)?
        };

        let mut blocks_dir = {
            let mut dir = dir.write(txn_id).await?;
            let file: fs::File<Id, ChainBlock> = dir.get_or_create_file(BLOCKS.into())?;
            file.into_inner().write().await
        };

        let pending = if let Some(file) = blocks_dir.get_file(&PENDING.to_string()) {
            file
        } else {
            let block = ChainBlock::with_txn(null_hash().to_vec(), txn_id);
            blocks_dir
                .create_file(PENDING.to_string(), block, Some(0))
                .map_err(fs::io_err)?
        };

        let committed = if let Some(file) = blocks_dir.get_file(&COMMITTED.to_string()) {
            file
        } else {
            let block = ChainBlock::new(null_hash().to_vec());
            blocks_dir
                .create_file(COMMITTED.to_string(), block, Some(0))
                .map_err(fs::io_err)?
        };

        Ok(SyncChain {
            schema,
            subject,
            pending,
            committed,
            store,
        })
    }
}

#[async_trait]
impl Transact for SyncChain {
    type Commit = ();

    async fn commit(&self, txn_id: &TxnId) {
        debug!("SyncChain::commit");

        let guard = self.subject.commit(txn_id).await;

        // assume the mutations for the transaction have already been moved and sync'd
        // from `self.pending` to `self.committed` by calling the `write_ahead` method
        {
            let mut committed: FileWriteGuard<_, ChainBlock> =
                self.committed.write().await.expect("committed");

            committed.mutations.remove(txn_id);
        }

        self.committed.sync(false).await.expect("sync commit block");

        guard
    }

    async fn finalize(&self, txn_id: &TxnId) {
        {
            let mut pending: FileWriteGuard<_, ChainBlock> =
                self.pending.write().await.expect("pending");

            pending.mutations.remove(txn_id);
        }

        self.pending.sync(false).await.expect("sync pending block");
        self.subject.finalize(txn_id).await
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
        let null_hash = null_hash();
        let txn_id = *self.txn.id();
        let mut dir = {
            let dir = self
                .txn
                .context()
                .create_dir_unique(txn_id)
                .map_err(de::Error::custom)
                .await?;

            dir.write(txn_id).map_err(de::Error::custom).await?
        };

        let schema = seq
            .next_element(())
            .await?
            .ok_or_else(|| de::Error::invalid_length(0, "a SyncChain schema"))?;

        let subject = seq
            .next_element(self.txn)
            .await?
            .ok_or_else(|| de::Error::invalid_length(1, "the subject of a SyncChain"))?;

        let store = dir
            .create_dir(STORE.into())
            .map(super::data::Store::new)
            .map_err(de::Error::custom)?;

        let mut blocks_dir = {
            let file: fs::File<Id, ChainBlock> =
                dir.create_file(BLOCKS.into()).map_err(de::Error::custom)?;

            file.into_inner().write().await
        };

        let committed = blocks_dir
            .create_file(
                COMMITTED.to_string(),
                ChainBlock::new(null_hash.to_vec()),
                Some(null_hash.len()),
            )
            .map_err(de::Error::custom)?;

        let pending = blocks_dir
            .create_file(
                PENDING.to_string(),
                ChainBlock::with_txn(null_hash.to_vec(), txn_id),
                Some(null_hash.len()),
            )
            .map_err(de::Error::custom)?;

        Ok(SyncChain {
            schema,
            subject,
            pending,
            committed,
            store,
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
