//! A [`Chain`] responsible for recovering a [`State`] from a failed transaction.

use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use destream::{de, en};
use freqfs::FileSave;
use futures::future::TryFutureExt;
use safecast::{AsType, TryCastFrom};

use tc_collection::{Collection, CollectionBase, CollectionBlock};
use tc_error::*;
use tc_scalar::Scalar;
use tc_transact::hash::{AsyncHash, GenericArray, Output, Sha256};
use tc_transact::lock::TxnTaskQueue;
use tc_transact::public::{Route, StateInstance};
use tc_transact::{fs, Replicate};
use tc_transact::{IntoView, Transact, Transaction, TxnId};
use tc_value::{Link, Value};
use tcgeneric::*;

use data::{MutationPending, MutationRecord};

pub use block::BlockChain;
pub use data::ChainBlock;
pub use sync::SyncChain;

mod block;
mod data;
mod public;
mod sync;

pub const CHAIN: Label = label("chain");
pub const HISTORY: Label = label(".history");

const BLOCK_SIZE: usize = 1_000_000; // TODO: reduce to 4,096
const PREFIX: PathLabel = path_label(&["state", "chain"]);

/// A block in a file managed by a [`Chain`]
pub trait CacheBlock: AsType<ChainBlock> + CollectionBlock {}

impl<FE> CacheBlock for FE where FE: AsType<ChainBlock> + CollectionBlock {}

/// Defines a method to recover the state of this [`Chain`] from a transaction failure.
#[async_trait]
pub trait Recover<FE> {
    type Txn: Transaction<FE>;

    /// Recover this state after loading, in case the last transaction failed or was interrupted.
    async fn recover(&self, txn: &Self::Txn) -> TCResult<()>;
}

/// Methods common to any type of [`Chain`].
pub trait ChainInstance<State: StateInstance, T> {
    /// Append the given DELETE op to the latest block in this `Chain`.
    fn append_delete(&self, txn_id: TxnId, key: Value) -> TCResult<()>;

    /// Append the given PUT op to the latest block in this `Chain`.
    fn append_put(&self, txn: State::Txn, key: Value, value: State) -> TCResult<()>;

    /// Borrow the subject of this [`Chain`].
    fn subject(&self) -> &T;
}

/// The type of a [`Chain`].
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum ChainType {
    Block,
    Sync,
}

impl Default for ChainType {
    fn default() -> Self {
        Self::Sync
    }
}

impl Class for ChainType {}

impl NativeClass for ChainType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if path.len() == 3 && &path[0..2] == &PREFIX[..] {
            match path[2].as_str() {
                "block" => Some(Self::Block),
                "sync" => Some(Self::Sync),
                _ => None,
            }
        } else {
            None
        }
    }

    fn path(&self) -> TCPathBuf {
        let suffix = match self {
            Self::Block => "block",
            Self::Sync => "sync",
        };

        TCPathBuf::from(PREFIX).append(label(suffix))
    }
}

impl fmt::Debug for ChainType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            Self::Block => "type BlockChain",
            Self::Sync => "type SyncChain",
        })
    }
}

/// A data structure responsible for maintaining the integrity of a mutable subject.
pub enum Chain<State, Txn, FE, T> {
    Block(block::BlockChain<State, Txn, FE, T>),
    Sync(sync::SyncChain<State, Txn, FE, T>),
}

impl<State, Txn, FE, T> Clone for Chain<State, Txn, FE, T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        match self {
            Self::Block(chain) => Self::Block(chain.clone()),
            Self::Sync(chain) => Self::Sync(chain.clone()),
        }
    }
}

impl<State, Txn, FE, T> Instance for Chain<State, Txn, FE, T>
where
    State: Send + Sync,
    Txn: Send + Sync,
    FE: Send + Sync,
    T: Send + Sync,
{
    type Class = ChainType;

    fn class(&self) -> Self::Class {
        match self {
            Self::Block(_) => ChainType::Block,
            Self::Sync(_) => ChainType::Sync,
        }
    }
}

impl<State, T> ChainInstance<State, T> for Chain<State, State::Txn, State::FE, T>
where
    State: StateInstance,
    State::FE: for<'a> FileSave<'a> + CacheBlock,
    T: fs::Persist<State::FE, Txn = State::Txn> + Route<State> + fmt::Debug,
    Collection<State::Txn, State::FE>: TryCastFrom<State>,
    Scalar: TryCastFrom<State>,
{
    fn append_delete(&self, txn_id: TxnId, key: Value) -> TCResult<()> {
        match self {
            Self::Block(chain) => chain.append_delete(txn_id, key),
            Self::Sync(chain) => chain.append_delete(txn_id, key),
        }
    }

    fn append_put(&self, txn: State::Txn, key: Value, value: State) -> TCResult<()> {
        match self {
            Self::Block(chain) => chain.append_put(txn, key, value),
            Self::Sync(chain) => chain.append_put(txn, key, value),
        }
    }

    fn subject(&self) -> &T {
        match self {
            Self::Block(chain) => chain.subject(),
            Self::Sync(chain) => chain.subject(),
        }
    }
}

#[async_trait]
impl<State> Replicate<State::Txn>
    for Chain<State, State::Txn, State::FE, CollectionBase<State::Txn, State::FE>>
where
    State: StateInstance,
    State::FE: CacheBlock,
    State: From<Collection<State::Txn, State::FE>> + From<Scalar>,
    Collection<State::Txn, State::FE>: TryCastFrom<State>,
    CollectionBase<State::Txn, State::FE>: Route<State> + TryCastFrom<State>,
    Scalar: TryCastFrom<State>,
    BlockChain<State, State::Txn, State::FE, CollectionBase<State::Txn, State::FE>>:
        TryCastFrom<State>,
    SyncChain<State, State::Txn, State::FE, CollectionBase<State::Txn, State::FE>>:
        TryCastFrom<State>,
{
    async fn replicate(&self, txn: &State::Txn, source: Link) -> TCResult<Output<Sha256>> {
        match self {
            Self::Block(chain) => chain.replicate(txn, source).await,
            Self::Sync(chain) => chain.replicate(txn, source).await,
        }
    }
}

#[async_trait]
impl<State, T> AsyncHash for Chain<State, State::Txn, State::FE, T>
where
    State: StateInstance,
    State::FE: AsType<ChainBlock> + for<'a> fs::FileSave<'a> + ThreadSafe,
    T: AsyncHash + Send + Sync,
{
    async fn hash(&self, txn_id: TxnId) -> TCResult<Output<Sha256>> {
        match self {
            Self::Block(chain) => chain.hash(txn_id).await,
            Self::Sync(chain) => chain.hash(txn_id).await,
        }
    }
}

#[async_trait]
impl<State, T> Transact for Chain<State, State::Txn, State::FE, T>
where
    State: StateInstance,
    State::FE: CacheBlock + for<'a> fs::FileSave<'a>,
    T: fs::Persist<State::FE, Txn = State::Txn> + Route<State> + Transact + fmt::Debug,
{
    type Commit = T::Commit;

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        match self {
            Self::Block(chain) => chain.commit(txn_id).await,
            Self::Sync(chain) => chain.commit(txn_id).await,
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        match self {
            Self::Block(chain) => chain.rollback(txn_id).await,
            Self::Sync(chain) => chain.rollback(txn_id).await,
        }
    }

    async fn finalize(&self, txn_id: &TxnId) {
        match self {
            Self::Block(chain) => chain.finalize(txn_id).await,
            Self::Sync(chain) => chain.finalize(txn_id).await,
        }
    }
}

#[async_trait]
impl<State, T> Recover<State::FE> for Chain<State, State::Txn, State::FE, T>
where
    State: StateInstance + From<Collection<State::Txn, State::FE>> + From<Scalar>,
    State::FE: CacheBlock + for<'a> fs::FileSave<'a>,
    T: Route<State> + fmt::Debug + Send + Sync,
    Collection<State::Txn, State::FE>: TryCastFrom<State>,
    Scalar: TryCastFrom<State>,
{
    type Txn = State::Txn;

    async fn recover(&self, txn: &State::Txn) -> TCResult<()> {
        match self {
            Self::Block(chain) => chain.recover(txn).await,
            Self::Sync(chain) => chain.recover(txn).await,
        }
    }
}

#[async_trait]
impl<State, T> fs::Persist<State::FE> for Chain<State, State::Txn, State::FE, T>
where
    State: StateInstance,
    State::FE: CacheBlock + for<'a> fs::FileSave<'a>,
    T: fs::Persist<State::FE, Txn = State::Txn> + Route<State> + fmt::Debug,
{
    type Txn = State::Txn;
    type Schema = (ChainType, T::Schema);

    async fn create(
        txn_id: TxnId,
        schema: Self::Schema,
        store: fs::Dir<State::FE>,
    ) -> TCResult<Self> {
        let (class, schema) = schema;

        match class {
            ChainType::Block => {
                BlockChain::create(txn_id, schema, store)
                    .map_ok(Self::Block)
                    .await
            }
            ChainType::Sync => {
                SyncChain::create(txn_id, schema, store)
                    .map_ok(Self::Sync)
                    .await
            }
        }
    }

    async fn load(
        txn_id: TxnId,
        schema: Self::Schema,
        store: fs::Dir<State::FE>,
    ) -> TCResult<Self> {
        let (class, schema) = schema;
        match class {
            ChainType::Block => {
                BlockChain::load(txn_id, schema, store)
                    .map_ok(Self::Block)
                    .await
            }
            ChainType::Sync => {
                SyncChain::load(txn_id, schema, store)
                    .map_ok(Self::Sync)
                    .await
            }
        }
    }

    fn dir(&self) -> tc_transact::fs::Inner<State::FE> {
        match self {
            Self::Block(chain) => chain.dir(),
            Self::Sync(chain) => chain.dir(),
        }
    }
}

#[async_trait]
impl<State, T> fs::CopyFrom<State::FE, Self> for Chain<State, State::Txn, State::FE, T>
where
    State: StateInstance,
    State::FE: CacheBlock + for<'a> fs::FileSave<'a>,
    T: fs::Persist<State::FE, Txn = State::Txn> + Route<State> + fmt::Debug,
{
    async fn copy_from(
        txn: &State::Txn,
        store: fs::Dir<State::FE>,
        instance: Self,
    ) -> TCResult<Self> {
        match instance {
            Chain::Block(chain) => {
                BlockChain::copy_from(txn, store, chain)
                    .map_ok(Chain::Block)
                    .await
            }
            Chain::Sync(chain) => {
                SyncChain::copy_from(txn, store, chain)
                    .map_ok(Chain::Sync)
                    .await
            }
        }
    }
}

impl<State, Txn, FE, T> fmt::Debug for Chain<State, Txn, FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Block(chain) => chain.fmt(f),
            Self::Sync(chain) => chain.fmt(f),
        }
    }
}

impl<State, Txn, FE, T> From<BlockChain<State, Txn, FE, T>> for Chain<State, Txn, FE, T> {
    fn from(chain: BlockChain<State, Txn, FE, T>) -> Self {
        Self::Block(chain)
    }
}

enum ChainViewData<'en, T> {
    Block((T, data::HistoryView<'en>)),
    Sync(T),
}

/// A view of a [`Chain`] within a single `Transaction`, used for serialization.
pub struct ChainView<'en, T> {
    class: ChainType,
    data: ChainViewData<'en, T>,
}

impl<'en, T> en::IntoStream<'en> for ChainView<'en, T>
where
    T: en::IntoStream<'en> + 'en,
{
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        use destream::en::EncodeMap;

        let mut map = encoder.encode_map(Some(1))?;

        map.encode_key(self.class.path().to_string())?;
        match self.data {
            ChainViewData::Block(view) => map.encode_value(view),
            ChainViewData::Sync(view) => map.encode_value(view),
        }?;

        map.end()
    }
}

#[async_trait]
impl<'en, State, T> IntoView<'en, State::FE> for Chain<State, State::Txn, State::FE, T>
where
    State: StateInstance,
    T: IntoView<'en, State::FE, Txn = State::Txn> + Send + Sync + 'en,
    BlockChain<State, State::Txn, State::FE, T>: IntoView<'en, State::FE, View = (T::View, data::HistoryView<'en>), Txn = State::Txn>
        + Send
        + Sync,
    SyncChain<State, State::Txn, State::FE, T>:
        IntoView<'en, State::FE, View = T::View, Txn = State::Txn> + Send + Sync,
{
    type Txn = State::Txn;
    type View = ChainView<'en, T::View>;

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        let class = self.class();

        let data = match self {
            Self::Block(chain) => chain.into_view(txn).map_ok(ChainViewData::Block).await,
            Self::Sync(chain) => chain.into_view(txn).map_ok(ChainViewData::Sync).await,
        }?;

        Ok(ChainView { class, data })
    }
}

#[async_trait]
impl<State, T> de::FromStream for Chain<State, State::Txn, State::FE, T>
where
    State: StateInstance
        + de::FromStream<Context = State::Txn>
        + From<Collection<State::Txn, State::FE>>
        + From<Scalar>,
    State::FE: CacheBlock + for<'a> fs::FileSave<'a>,
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
        decoder.decode_map(ChainVisitor::new(txn)).await
    }
}

/// A [`de::Visitor`] for deserializing a [`Chain`].
pub struct ChainVisitor<State: StateInstance, T> {
    txn: State::Txn,
    phantom: PhantomData<T>,
}

impl<State, T> ChainVisitor<State, T>
where
    State: StateInstance,
{
    pub fn new(txn: State::Txn) -> Self {
        Self {
            txn,
            phantom: PhantomData,
        }
    }
}

impl<State, T> ChainVisitor<State, T>
where
    State: StateInstance
        + de::FromStream<Context = State::Txn>
        + From<Collection<State::Txn, State::FE>>
        + From<Scalar>,
    State::FE: CacheBlock + for<'a> fs::FileSave<'a>,
    T: Route<State> + de::FromStream<Context = State::Txn> + fmt::Debug,
    (Bytes, Map<Tuple<State>>): TryCastFrom<State>,
    Collection<State::Txn, State::FE>: TryCastFrom<State>,
    Scalar: TryCastFrom<State>,
    Value: TryCastFrom<State>,
    (Value,): TryCastFrom<State>,
    (Value, State): TryCastFrom<State>,
{
    pub async fn visit_map_value<A: de::MapAccess>(
        self,
        class: ChainType,
        access: &mut A,
    ) -> Result<Chain<State, State::Txn, State::FE, T>, A::Error> {
        match class {
            ChainType::Block => {
                access
                    .next_value(self.txn)
                    .map_ok(Chain::Block)
                    .map_err(|e| de::Error::custom(format!("invalid BlockChain stream: {}", e)))
                    .await
            }
            ChainType::Sync => access.next_value(self.txn).map_ok(Chain::Sync).await,
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
    State::FE: CacheBlock + for<'a> fs::FileSave<'a>,
    T: Route<State> + de::FromStream<Context = State::Txn> + fmt::Debug,
    (Bytes, Map<Tuple<State>>): TryCastFrom<State>,
    Collection<State::Txn, State::FE>: TryCastFrom<State>,
    Scalar: TryCastFrom<State>,
    Value: TryCastFrom<State>,
    (Value,): TryCastFrom<State>,
    (Value, State): TryCastFrom<State>,
{
    type Value = Chain<State, State::Txn, State::FE, T>;

    fn expecting() -> &'static str {
        "a Chain"
    }

    async fn visit_map<A: de::MapAccess>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let class = if let Some(path) = map.next_key::<TCPathBuf>(()).await? {
            ChainType::from_path(&path)
                .ok_or_else(|| de::Error::invalid_value(path, "a Chain class"))?
        } else {
            return Err(de::Error::custom("expected a Chain class"));
        };

        self.visit_map_value(class, &mut map).await
    }
}

fn new_queue<State>(
    store: data::Store<State::Txn, State::FE>,
) -> TxnTaskQueue<MutationPending<State::Txn, State::FE>, TCResult<MutationRecord>>
where
    State: StateInstance,
    State::FE: for<'a> FileSave<'a> + CacheBlock + Clone,
{
    TxnTaskQueue::new(Arc::pin(move |mutation| {
        let store = store.clone();

        Box::pin(async move {
            match mutation {
                MutationPending::Delete(key) => Ok(MutationRecord::Delete(key)),
                MutationPending::Put(txn, key, state) => {
                    let value = store.save_state(&txn, state).await?;
                    Ok(MutationRecord::Put(key, value))
                }
            }
        })
    }))
}

#[inline]
pub fn null_hash() -> Output<Sha256> {
    GenericArray::default()
}
