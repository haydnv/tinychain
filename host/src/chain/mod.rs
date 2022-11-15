//! A [`Chain`] responsible for recovering a [`State`] from a failed transaction.

use std::convert::TryFrom;
use std::fmt;
use std::marker::PhantomData;

use async_trait::async_trait;
use destream::{de, en};
use futures::future::TryFutureExt;
use sha2::digest::generic_array::GenericArray;
use sha2::digest::Output;
use sha2::Sha256;

use tc_error::*;
use tc_transact::fs::Persist;
use tc_transact::{IntoView, Transact, TxnId};
use tc_value::{Link, Value};
use tcgeneric::*;

use crate::cluster::Replica;
use crate::fs;
use crate::route::{Public, Route};
use crate::state::State;
use crate::txn::Txn;

pub use block::BlockChain;
pub use data::ChainBlock;
pub use sync::SyncChain;

mod block;
mod data;
mod sync;

const BLOCK_SIZE: usize = 1_000_000;
const CHAIN: Label = label("chain");
const SUBJECT: Label = label("subject");
const PREFIX: PathLabel = path_label(&["state", "chain"]);

/// Trait defining methods common to any instance of a [`Chain`], such as a [`SyncChain`].
#[async_trait]
pub trait ChainInstance<T> {
    /// Append the given DELETE op to the latest block in this `Chain`.
    async fn append_delete(&self, txn_id: TxnId, key: Value) -> TCResult<()>;

    /// Append the given PUT op to the latest block in this `Chain`.
    async fn append_put(&self, txn: &Txn, key: Value, value: State) -> TCResult<()>;

    /// Borrow the [`Subject`] of this [`Chain`] immutably.
    fn subject(&self) -> &T;

    /// Write the mutation ops in the current transaction to the write-ahead log.
    async fn write_ahead(&self, txn_id: &TxnId);
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
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for ChainType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            Self::Block => "type BlockChain",
            Self::Sync => "type SyncChain",
        })
    }
}

/// A data structure responsible for maintaining the transactional integrity of its [`Subject`].
#[derive(Clone)]
pub enum Chain<T> {
    Block(block::BlockChain<T>),
    Sync(sync::SyncChain<T>),
}

impl<T> Instance for Chain<T>
where
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

#[async_trait]
impl<T> ChainInstance<T> for Chain<T>
where
    T: Persist<fs::Dir, Txn = Txn> + Route + Public + fmt::Display,
{
    async fn append_delete(&self, txn_id: TxnId, key: Value) -> TCResult<()> {
        match self {
            Self::Block(chain) => chain.append_delete(txn_id, key).await,
            Self::Sync(chain) => chain.append_delete(txn_id, key).await,
        }
    }

    async fn append_put(&self, txn: &Txn, key: Value, value: State) -> TCResult<()> {
        match self {
            Self::Block(chain) => chain.append_put(txn, key, value).await,
            Self::Sync(chain) => chain.append_put(txn, key, value).await,
        }
    }

    fn subject(&self) -> &T {
        match self {
            Self::Block(chain) => chain.subject(),
            Self::Sync(chain) => chain.subject(),
        }
    }

    async fn write_ahead(&self, txn_id: &TxnId) {
        match self {
            Self::Block(chain) => chain.write_ahead(txn_id).await,
            Self::Sync(chain) => chain.write_ahead(txn_id).await,
        }
    }
}

#[async_trait]
impl<T> Replica for Chain<T>
where
    T: Transact + Send + Sync,
    BlockChain<T>: Replica,
    SyncChain<T>: Replica,
{
    async fn state(&self, txn_id: TxnId) -> TCResult<State> {
        match self {
            Self::Block(chain) => chain.state(txn_id).await,
            Self::Sync(chain) => chain.state(txn_id).await,
        }
    }

    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()> {
        match self {
            Self::Block(chain) => chain.replicate(txn, source).await,
            Self::Sync(chain) => chain.replicate(txn, source).await,
        }
    }
}

#[async_trait]
impl<T> Transact for Chain<T>
where
    T: Transact + Send + Sync,
{
    type Commit = T::Commit;

    async fn commit(&self, txn_id: &TxnId) -> Self::Commit {
        match self {
            Self::Block(chain) => chain.commit(txn_id).await,
            Self::Sync(chain) => chain.commit(txn_id).await,
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
impl<T> Persist<fs::Dir> for Chain<T>
where
    T: Persist<fs::Dir, Txn = Txn> + Route + Public,
    <T as Persist<fs::Dir>>::Store: TryFrom<fs::Store>,
    TCError: From<<<T as Persist<fs::Dir>>::Store as TryFrom<fs::Store>>::Error>,
{
    type Schema = (ChainType, T::Schema);
    type Store = fs::Dir;
    type Txn = Txn;

    async fn create(txn: &Self::Txn, schema: Self::Schema, store: Self::Store) -> TCResult<Self> {
        let (class, schema) = schema;
        match class {
            ChainType::Block => {
                BlockChain::create(txn, schema, store)
                    .map_ok(Self::Block)
                    .await
            }
            ChainType::Sync => {
                SyncChain::create(txn, schema, store)
                    .map_ok(Self::Sync)
                    .await
            }
        }
    }

    async fn load(txn: &Self::Txn, schema: Self::Schema, store: Self::Store) -> TCResult<Self> {
        let (class, schema) = schema;
        match class {
            ChainType::Block => {
                BlockChain::load(txn, schema, store)
                    .map_ok(Self::Block)
                    .await
            }
            ChainType::Sync => SyncChain::load(txn, schema, store).map_ok(Self::Sync).await,
        }
    }
}

impl<T> fmt::Debug for Chain<T>
where
    Self: Instance,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "instance of {} with subject type {}",
            self.class(),
            std::any::type_name::<T>()
        )
    }
}

impl<T> fmt::Display for Chain<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Block(chain) => fmt::Display::fmt(chain, f),
            Self::Sync(chain) => fmt::Display::fmt(chain, f),
        }
    }
}

impl<T> From<BlockChain<T>> for Chain<T> {
    fn from(chain: BlockChain<T>) -> Self {
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
impl<'en, T> IntoView<'en, fs::Dir> for Chain<T>
where
    T: IntoView<'en, fs::Dir, Txn = Txn> + 'en,
    Chain<T>: Instance<Class = ChainType>,
    BlockChain<T>:
        IntoView<'en, fs::Dir, View = (T::View, data::HistoryView<'en>), Txn = Txn> + Send + Sync,
    SyncChain<T>: IntoView<'en, fs::Dir, View = T::View, Txn = Txn> + Send + Sync,
{
    type Txn = Txn;
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

/// A [`de::Visitor`] for deserializing a [`Chain`].
pub struct ChainVisitor<T> {
    txn: Txn,
    phantom: PhantomData<T>,
}

impl<T> ChainVisitor<T> {
    pub(crate) fn new(txn: Txn) -> Self {
        Self {
            txn,
            phantom: PhantomData,
        }
    }
}

impl<T> ChainVisitor<T>
where
    T: Route + Public + de::FromStream<Context = Txn>,
{
    pub(crate) async fn visit_map_value<A: de::MapAccess>(
        self,
        class: ChainType,
        access: &mut A,
    ) -> Result<Chain<T>, A::Error> {
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
impl<T> de::Visitor for ChainVisitor<T>
where
    T: Route + Public + de::FromStream<Context = Txn>,
{
    type Value = Chain<T>;

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

#[async_trait]
impl<T> de::FromStream for Chain<T>
where
    T: Route + Public + de::FromStream<Context = Txn>,
{
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        decoder.decode_map(ChainVisitor::new(txn)).await
    }
}

#[inline]
fn null_hash() -> Output<Sha256> {
    GenericArray::default()
}
