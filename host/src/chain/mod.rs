//! A [`Chain`] responsible for recovering a [`State`] from a failed transaction.

use std::fmt;
use std::marker::PhantomData;

use async_hash::generic_array::GenericArray;
use async_hash::{Digest, Hash, Output, Sha256};
use async_trait::async_trait;
use destream::{de, en};
use futures::future::TryFutureExt;

use tc_error::*;
use tc_transact::fs::{CopyFrom, Persist};
use tc_transact::{AsyncHash, IntoView, Transact, TxnId};
use tc_value::{Link, Value};
use tcgeneric::*;

use crate::fs;
use crate::route::Route;
use crate::state::State;
use crate::txn::Txn;

use crate::fs::CacheBlock;
pub use block::BlockChain;
pub(crate) use block::HISTORY;
pub use data::ChainBlock;
pub use sync::SyncChain;

mod block;
mod data;
mod sync;

pub(crate) const CHAIN: Label = label("chain");

const BLOCK_SIZE: usize = 1_000_000; // TODO: reduce to 4,096
const PREFIX: PathLabel = path_label(&["state", "chain"]);

/// Defines a method to recover the state of this [`Chain`] from a transaction failure.
#[async_trait]
pub trait Recover {
    /// Recover this state after loading, in case the last transaction failed or was interrupted.
    async fn recover(&self, txn: &Txn) -> TCResult<()>;
}

/// Methods common to any instance of a [`Chain`], such as a [`SyncChain`].
#[async_trait]
pub trait ChainInstance<T> {
    /// Append the given DELETE op to the latest block in this `Chain`.
    async fn append_delete(&self, txn_id: TxnId, key: Value) -> TCResult<()>;

    /// Append the given PUT op to the latest block in this `Chain`.
    async fn append_put(&self, txn: &Txn, key: Value, value: State) -> TCResult<()>;

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
// TODO: remove the generic type and replace with:
// enum Chain { Block(BlockChain<Box<dyn Public>>, Sync(SyncChain<Box<dyn Restore>>) }
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
    T: Persist<CacheBlock, Txn = Txn> + Route + fmt::Debug,
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
}

#[async_trait]
impl<T: AsyncHash<CacheBlock, Txn = Txn> + Send + Sync> AsyncHash<CacheBlock> for Chain<T> {
    type Txn = Txn;

    async fn hash(self, txn: &Self::Txn) -> TCResult<Output<Sha256>> {
        match self {
            Self::Block(chain) => chain.hash(txn).await,
            Self::Sync(chain) => chain.hash(txn).await,
        }
    }
}

#[async_trait]
impl<T: Transact + Send + Sync> Transact for Chain<T>
where
    BlockChain<T>: ChainInstance<T>,
    SyncChain<T>: ChainInstance<T>,
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
impl<T: Route + fmt::Debug> Recover for Chain<T> {
    async fn recover(&self, txn: &Txn) -> TCResult<()> {
        match self {
            Self::Block(chain) => chain.recover(txn).await,
            Self::Sync(chain) => chain.recover(txn).await,
        }
    }
}

#[async_trait]
impl<T> Persist<CacheBlock> for Chain<T>
where
    T: Persist<CacheBlock, Txn = Txn> + Route + fmt::Debug,
{
    type Txn = Txn;
    type Schema = (ChainType, T::Schema);

    async fn create(txn_id: TxnId, schema: Self::Schema, store: fs::Dir) -> TCResult<Self> {
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

    async fn load(txn_id: TxnId, schema: Self::Schema, store: fs::Dir) -> TCResult<Self> {
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

    fn dir(&self) -> tc_transact::fs::Inner<CacheBlock> {
        match self {
            Self::Block(chain) => chain.dir(),
            Self::Sync(chain) => chain.dir(),
        }
    }
}

// #[async_trait]
// impl<T> CopyFrom<CacheBlock, Chain<T>> for Chain<T>
// where
//     T: Persist<CacheBlock, Txn = Txn> + Route + fmt::Debug,
// {
//     async fn copy_from(
//         txn: &<Self as Persist<CacheBlock>>::Txn,
//         store: fs::Dir,
//         instance: Chain<T>,
//     ) -> TCResult<Self> {
//         match instance {
//             Chain::Block(chain) => {
//                 BlockChain::copy_from(txn, store, chain)
//                     .map_ok(Chain::Block)
//                     .await
//             }
//             Chain::Sync(chain) => {
//                 SyncChain::copy_from(txn, store, chain)
//                     .map_ok(Chain::Sync)
//                     .await
//             }
//         }
//     }
// }

impl<T> fmt::Debug for Chain<T>
where
    Self: Instance,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "instance of {:?} with subject type {}",
            self.class(),
            std::any::type_name::<T>()
        )
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
impl<'en, T> IntoView<'en, CacheBlock> for Chain<T>
where
    T: IntoView<'en, CacheBlock, Txn = Txn> + 'en,
    Chain<T>: Instance<Class = ChainType>,
    BlockChain<T>: IntoView<'en, CacheBlock, View = (T::View, data::HistoryView<'en>), Txn = Txn>
        + Send
        + Sync,
    SyncChain<T>: IntoView<'en, CacheBlock, View = T::View, Txn = Txn> + Send + Sync,
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

// /// A [`de::Visitor`] for deserializing a [`Chain`].
// pub struct ChainVisitor<T> {
//     txn: Txn,
//     phantom: PhantomData<T>,
// }
//
// impl<T> ChainVisitor<T> {
//     pub(crate) fn new(txn: Txn) -> Self {
//         Self {
//             txn,
//             phantom: PhantomData,
//         }
//     }
// }

// impl<T> ChainVisitor<T>
// where
//     T: Route + de::FromStream<Context = Txn> + fmt::Debug,
// {
//     pub(crate) async fn visit_map_value<A: de::MapAccess>(
//         self,
//         class: ChainType,
//         access: &mut A,
//     ) -> Result<Chain<T>, A::Error> {
//         match class {
//             ChainType::Block => {
//                 access
//                     .next_value(self.txn)
//                     .map_ok(Chain::Block)
//                     .map_err(|e| de::Error::custom(format!("invalid BlockChain stream: {}", e)))
//                     .await
//             }
//             ChainType::Sync => access.next_value(self.txn).map_ok(Chain::Sync).await,
//         }
//     }
// }

// #[async_trait]
// impl<T> de::Visitor for ChainVisitor<T>
// where
//     T: Route + de::FromStream<Context = Txn> + fmt::Debug,
// {
//     type Value = Chain<T>;
//
//     fn expecting() -> &'static str {
//         "a Chain"
//     }
//
//     async fn visit_map<A: de::MapAccess>(self, mut map: A) -> Result<Self::Value, A::Error> {
//         let class = if let Some(path) = map.next_key::<TCPathBuf>(()).await? {
//             ChainType::from_path(&path)
//                 .ok_or_else(|| de::Error::invalid_value(path, "a Chain class"))?
//         } else {
//             return Err(de::Error::custom("expected a Chain class"));
//         };
//
//         self.visit_map_value(class, &mut map).await
//     }
// }

// #[async_trait]
// impl<T> de::FromStream for Chain<T>
// where
//     T: Route + de::FromStream<Context = Txn> + fmt::Debug,
// {
//     type Context = Txn;
//
//     async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
//         decoder.decode_map(ChainVisitor::new(txn)).await
//     }
// }

#[inline]
pub(crate) fn null_hash() -> Output<Sha256> {
    GenericArray::default()
}
