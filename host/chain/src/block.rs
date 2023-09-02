//! A `Chain` which stores every mutation of its subject in a series of `ChainBlock`s.
//!
//! Each block in the chain begins with the hash of the previous block.

use std::fmt;
use std::marker::PhantomData;

use async_hash::{Output, Sha256};
use async_trait::async_trait;
use bytes::Bytes;
use destream::de;
use futures::future::TryFutureExt;
use futures::join;
use log::debug;
use safecast::{AsType, TryCastFrom};

use tc_collection::btree::Node as BTreeNode;
use tc_collection::tensor::{DenseCacheFile, Node as TensorNode};
use tc_collection::Collection;
use tc_error::*;
use tc_scalar::Scalar;
use tc_transact::fs;
use tc_transact::public::{Route, StateInstance};
use tc_transact::{AsyncHash, IntoView, Transact, Transaction, TxnId};
use tc_value::Value;
use tcgeneric::{Map, ThreadSafe, Tuple};

use super::data::{ChainBlock, History};
use super::{ChainInstance, Recover, HISTORY};

/// A `Chain` which stores every mutation of its subject in a series of `ChainBlock`s
#[derive(Clone)]
pub struct BlockChain<State: StateInstance, T> {
    history: History<State>,
    subject: T,
}

impl<State, T> BlockChain<State, T>
where
    State: StateInstance,
{
    fn new(subject: T, history: History<State>) -> Self {
        Self { subject, history }
    }

    pub fn history(&self) -> &History<State> {
        &self.history
    }
}

#[async_trait]
impl<State, T> ChainInstance<State, T> for BlockChain<State, T>
where
    State: StateInstance,
    State::FE: DenseCacheFile
        + AsType<BTreeNode>
        + AsType<ChainBlock>
        + AsType<TensorNode>
        + for<'a> fs::FileSave<'a>,
    T: Route<State> + fmt::Debug,
    Collection<State::Txn, State::FE>: TryCastFrom<State>,
    Scalar: TryCastFrom<State>,
{
    async fn append_delete(&self, txn_id: TxnId, key: Value) -> TCResult<()> {
        self.history.append_delete(txn_id, key).await
    }

    async fn append_put(&self, txn: &State::Txn, key: Value, value: State) -> TCResult<()> {
        self.history.append_put(txn, key, value).await
    }

    fn subject(&self) -> &T {
        &self.subject
    }
}

#[async_trait]
impl<State, T> fs::Persist<State::FE> for BlockChain<State, T>
where
    State: StateInstance,
    State::FE: AsType<ChainBlock> + ThreadSafe + for<'a> fs::FileSave<'a>,
    T: Route<State> + fs::Persist<State::FE, Txn = State::Txn> + fmt::Debug,
{
    type Txn = State::Txn;
    type Schema = T::Schema;

    async fn create(
        txn_id: TxnId,
        schema: Self::Schema,
        store: fs::Dir<State::FE>,
    ) -> TCResult<Self> {
        debug!("BlockChain::create");

        let subject = T::create(txn_id, schema.clone(), store).await?;
        let mut dir = subject.dir().try_write_owned()?;

        let history = {
            let dir = dir.get_or_create_dir(HISTORY.to_string())?;
            fs::Dir::load(txn_id, dir).await?
        };

        let history = History::create(txn_id, (), history.into()).await?;

        Ok(BlockChain::new(subject, history))
    }

    async fn load(
        txn_id: TxnId,
        schema: Self::Schema,
        store: fs::Dir<State::FE>,
    ) -> TCResult<Self> {
        debug!("BlockChain::load {}", std::any::type_name::<T>());

        let subject = T::load(txn_id, schema.clone(), store).await?;

        let mut dir = subject.dir().write_owned().await;

        let history = {
            let dir = dir.get_or_create_dir(HISTORY.to_string())?;
            fs::Dir::load(txn_id, dir).await?
        };

        let history = History::load(txn_id, (), history.into()).await?;

        Ok(BlockChain::new(subject, history))
    }

    fn dir(&self) -> tc_transact::fs::Inner<State::FE> {
        self.subject.dir()
    }
}

#[async_trait]
impl<State, T> Recover<State::FE> for BlockChain<State, T>
where
    State: StateInstance + From<Collection<State::Txn, State::FE>> + From<Scalar>,
    State::FE: DenseCacheFile
        + AsType<BTreeNode>
        + AsType<TensorNode>
        + AsType<ChainBlock>
        + for<'a> fs::FileSave<'a>,
    T: Route<State> + fmt::Debug,
    Collection<State::Txn, State::FE>: TryCastFrom<State>,
    Scalar: TryCastFrom<State>,
    BTreeNode: freqfs::FileLoad,
{
    type Txn = State::Txn;

    async fn recover(&self, txn: &State::Txn) -> TCResult<()> {
        let write_ahead_log = self.history.read_log().await?;

        for (past_txn_id, mutations) in &write_ahead_log.mutations {
            super::data::replay_all(
                &self.subject,
                past_txn_id,
                mutations,
                txn,
                self.history.store(),
            )
            .await?;
        }

        Ok(())
    }
}

#[async_trait]
impl<State, T> fs::CopyFrom<State::FE, Self> for BlockChain<State, T>
where
    State: StateInstance,
    State::FE: AsType<ChainBlock> + ThreadSafe + for<'a> fs::FileSave<'a>,
    T: Route<State> + fs::Persist<State::FE, Txn = State::Txn> + fmt::Debug,
{
    async fn copy_from(
        _txn: &State::Txn,
        _store: fs::Dir<State::FE>,
        _instance: Self,
    ) -> TCResult<Self> {
        Err(not_implemented!("BlockChain::copy_from"))
    }
}

#[async_trait]
impl<State, T> AsyncHash for BlockChain<State, T>
where
    State: StateInstance,
    State::FE: AsType<ChainBlock> + for<'a> fs::FileSave<'a>,
    T: Send + Sync,
{
    async fn hash(self, txn_id: TxnId) -> TCResult<Output<Sha256>> {
        self.history.hash(txn_id).await
    }
}

#[async_trait]
impl<State, T> Transact for BlockChain<State, T>
where
    State: StateInstance,
    State::FE: AsType<ChainBlock> + for<'a> fs::FileSave<'a>,
    T: Transact + Send + Sync,
{
    type Commit = T::Commit;

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        debug!("BlockChain::commit");

        self.history.write_ahead(txn_id).await;

        let guard = self.subject.commit(txn_id).await;
        // make sure `self.subject` is committed before moving mutations out of the write-ahead log
        self.history.commit(txn_id).await;
        guard
    }

    async fn rollback(&self, txn_id: &TxnId) {
        join!(self.subject.rollback(txn_id), self.history.rollback(txn_id));
    }

    async fn finalize(&self, txn_id: &TxnId) {
        join!(self.subject.finalize(txn_id), self.history.finalize(txn_id));
    }
}

#[async_trait]
impl<State, T> de::FromStream for BlockChain<State, T>
where
    State: StateInstance
        + de::FromStream<Context = State::Txn>
        + From<Collection<State::Txn, State::FE>>
        + From<Scalar>,
    State::FE: DenseCacheFile
        + AsType<ChainBlock>
        + AsType<BTreeNode>
        + AsType<TensorNode>
        + for<'a> fs::FileSave<'a>,
    T: Route<State> + de::FromStream<Context = State::Txn> + fmt::Debug,
    (Bytes, Map<Tuple<State>>): TryCastFrom<State>,
    Collection<State::Txn, State::FE>: TryCastFrom<State>,
    Scalar: TryCastFrom<State>,
    Value: TryCastFrom<State>,
    (Value,): TryCastFrom<State>,
    (Value, State): TryCastFrom<State>,
{
    type Context = State::Txn;

    async fn from_stream<D: de::Decoder>(
        txn: State::Txn,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        decoder.decode_seq(ChainVisitor::new(txn)).await
    }
}

#[async_trait]
impl<'en, State, T> IntoView<'en, State::FE> for BlockChain<State, T>
where
    State: StateInstance,
    State::FE: DenseCacheFile + AsType<ChainBlock> + AsType<BTreeNode> + AsType<TensorNode>,
    T: IntoView<'en, State::FE, Txn = State::Txn> + Send + Sync,
{
    type Txn = State::Txn;
    type View = (T::View, super::data::HistoryView<'en>);

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        let history = self.history.into_view(txn.clone()).await?;
        let subject = self.subject.into_view(txn).await?;
        Ok((subject, history))
    }
}

impl<State, T> fmt::Debug for BlockChain<State, T>
where
    State: StateInstance,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "BlockChain<{}>", std::any::type_name::<T>())
    }
}

struct ChainVisitor<State: StateInstance, T> {
    txn: State::Txn,
    subject: PhantomData<T>,
}

impl<State, T> ChainVisitor<State, T>
where
    State: StateInstance,
{
    fn new(txn: State::Txn) -> Self {
        Self {
            txn,
            subject: PhantomData,
        }
    }
}

#[async_trait]
impl<State, T> de::Visitor for ChainVisitor<State, T>
where
    State: StateInstance
        + de::FromStream<Context = State::Txn>
        + From<Collection<State::Txn, State::FE>>
        + From<Scalar>,
    State::FE: DenseCacheFile
        + AsType<ChainBlock>
        + AsType<BTreeNode>
        + AsType<TensorNode>
        + for<'a> fs::FileSave<'a>,
    T: Route<State> + de::FromStream<Context = State::Txn> + fmt::Debug,
    (Bytes, Map<Tuple<State>>): TryCastFrom<State>,
    Collection<State::Txn, State::FE>: TryCastFrom<State>,
    Scalar: TryCastFrom<State>,
    Value: TryCastFrom<State>,
    (Value,): TryCastFrom<State>,
    (Value, State): TryCastFrom<State>,
{
    type Value = BlockChain<State, T>;

    fn expecting() -> &'static str {
        "a BlockChain"
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let subject = seq.next_element::<T>(self.txn.clone()).await?;
        let subject = subject.ok_or_else(|| de::Error::invalid_length(0, "a BlockChain schema"))?;

        let txn = self
            .txn
            .subcontext(HISTORY.into())
            .map_err(de::Error::custom)
            .await?;

        let history = seq.next_element::<History<State>>(txn).await?;
        let history =
            history.ok_or_else(|| de::Error::invalid_length(1, "a BlockChain history"))?;

        let write_ahead_log = history.read_log().map_err(de::Error::custom).await?;
        for (past_txn_id, mutations) in &write_ahead_log.mutations {
            super::data::replay_all(&subject, past_txn_id, mutations, &self.txn, history.store())
                .map_err(de::Error::custom)
                .await?;
        }

        Ok(BlockChain::new(subject, history))
    }
}
