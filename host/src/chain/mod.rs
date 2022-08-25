//! A [`Chain`] responsible for recovering a [`State`] from a failed transaction.

use std::convert::TryFrom;
use std::fmt;
use std::iter::FromIterator;

use async_trait::async_trait;
use destream::{de, en};
use futures::future::TryFutureExt;
use safecast::{CastFrom, CastInto, TryCastFrom, TryCastInto};
use sha2::digest::generic_array::GenericArray;
use sha2::digest::Output;
use sha2::Sha256;

use tc_btree::{BTreeType, Column};
use tc_error::*;
use tc_transact::fs::Persist;
use tc_transact::{IntoView, Transact, TxnId};
use tc_value::{Link, Value};
use tcgeneric::*;

#[cfg(feature = "tensor")]
use crate::collection::TensorType;
use crate::collection::{CollectionType, TableType};
use crate::fs;
use crate::scalar::{OpRef, Scalar, TCRef};
use crate::state::{State, StateView};
use crate::txn::Txn;

pub use block::BlockChain;
pub use data::ChainBlock;
pub use subject::{Subject, SubjectCollection, SubjectMap};
pub use sync::SyncChain;

mod block;
mod data;
mod subject;

mod sync;

const BLOCK_SIZE: usize = 1_000_000;
const CHAIN: Label = label("chain");
const PREFIX: PathLabel = path_label(&["state", "chain"]);

/// The schema of a [`Chain`] whose [`Subject`] is a `Collection`.
#[derive(Clone)]
pub enum CollectionSchema {
    BTree(tc_btree::RowSchema),
    Table(tc_table::TableSchema),
    #[cfg(feature = "tensor")]
    Dense(tc_tensor::Schema),
    #[cfg(feature = "tensor")]
    Sparse(tc_tensor::Schema),
}

impl CollectionSchema {
    pub fn from_scalar(tc_ref: TCRef) -> TCResult<Self> {
        match tc_ref {
            TCRef::Op(op_ref) => match op_ref {
                OpRef::Get((class, schema)) => {
                    let class = TCPathBuf::try_from(class)?;
                    let class = CollectionType::from_path(&class)
                        .ok_or_else(|| TCError::bad_request("invalid Collection type", class))?;

                    fn expect_value(scalar: Scalar) -> TCResult<Value> {
                        Value::try_cast_from(scalar, |s| {
                            TCError::bad_request("expected a Value for chain schema, not", s)
                        })
                    }

                    match class {
                        CollectionType::BTree(_) => {
                            let schema = expect_value(schema)?;

                            let schema = schema.try_cast_into(|s| {
                                TCError::bad_request("invalid BTree schema", s)
                            })?;

                            Ok(Self::BTree(schema))
                        }
                        CollectionType::Table(_) => {
                            let schema = expect_value(schema)?;

                            let schema = schema.try_cast_into(|s| {
                                TCError::bad_request("invalid Table schema", s)
                            })?;

                            Ok(Self::Table(schema))
                        }

                        #[cfg(feature = "tensor")]
                        CollectionType::Tensor(tt) => {
                            let schema = expect_value(schema)?;
                            let schema = schema.try_cast_into(|v| {
                                TCError::bad_request("invalid Tensor schema", v)
                            })?;

                            match tt {
                                TensorType::Dense => Ok(Self::Dense(schema)),
                                TensorType::Sparse => Ok(Self::Sparse(schema)),
                            }
                        }
                    }
                }
                other => Err(TCError::bad_request("invalid Collection schema", other)),
            },
            other => Err(TCError::bad_request("invalid Collection schema", other)),
        }
    }
}

impl<'en> en::IntoStream<'en> for CollectionSchema {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        use destream::en::EncodeMap;

        match self {
            Self::BTree(schema) => {
                let mut map = encoder.encode_map(Some(1))?;
                map.encode_entry(BTreeType::default().path(), (schema,))?;
                map.end()
            }

            Self::Table(schema) => {
                let mut map = encoder.encode_map(Some(1))?;
                map.encode_entry(TableType::default().path(), (schema,))?;
                map.end()
            }

            #[cfg(feature = "tensor")]
            Self::Dense(schema) | Self::Sparse(schema) => {
                let mut map = encoder.encode_map(Some(1))?;
                map.encode_entry(TensorType::Dense.path(), (schema,))?;
                map.end()
            }
        }
    }
}

impl CastFrom<CollectionSchema> for Scalar {
    fn cast_from(schema: CollectionSchema) -> Scalar {
        let class: CollectionType = match schema {
            CollectionSchema::BTree(_) => BTreeType::default().into(),
            CollectionSchema::Table(_) => TableType::default().into(),
            #[cfg(feature = "tensor")]
            CollectionSchema::Dense(_) => TensorType::Dense.into(),
            #[cfg(feature = "tensor")]
            CollectionSchema::Sparse(_) => TensorType::Sparse.into(),
        };

        let schema = match schema {
            CollectionSchema::BTree(schema) => {
                Value::Tuple(schema.into_iter().map(Value::from).collect())
            }
            CollectionSchema::Table(schema) => schema.cast_into(),
            #[cfg(feature = "tensor")]
            CollectionSchema::Dense(schema) => schema.cast_into(),
            #[cfg(feature = "tensor")]
            CollectionSchema::Sparse(schema) => schema.cast_into(),
        };

        Scalar::Ref(Box::new(TCRef::Op(OpRef::Get((
            class.path().into(),
            schema.into(),
        )))))
    }
}

impl fmt::Display for CollectionSchema {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BTree(schema) => write!(f, "{}", Tuple::<&Column>::from_iter(schema)),
            Self::Table(schema) => fmt::Display::fmt(schema, f),
            #[cfg(feature = "tensor")]
            Self::Dense(schema) => fmt::Display::fmt(schema, f),
            #[cfg(feature = "tensor")]
            Self::Sparse(schema) => fmt::Display::fmt(schema, f),
        }
    }
}

/// The schema of a [`Chain`], used when constructing a new `Chain` or loading a `Chain` from disk.
#[derive(Clone)]
pub enum Schema {
    Collection(CollectionSchema),
    Map(Map<Schema>),
    Tuple(Tuple<Schema>),
}

impl Schema {
    pub fn from_scalar(scalar: Scalar) -> TCResult<Self> {
        match scalar {
            Scalar::Ref(tc_ref) => CollectionSchema::from_scalar(*tc_ref).map(Self::Collection),
            Scalar::Map(map) => map
                .into_iter()
                .map(|(name, scalar)| Schema::from_scalar(scalar).map(|schema| (name, schema)))
                .collect::<TCResult<Map<Schema>>>()
                .map(Schema::Map),

            Scalar::Tuple(tuple) => tuple
                .into_iter()
                .map(|scalar| Schema::from_scalar(scalar))
                .collect::<TCResult<Tuple<Schema>>>()
                .map(Schema::Tuple),

            other => Err(TCError::bad_request("invalid Chain schema", other)),
        }
    }
}

#[async_trait]
impl de::FromStream for Schema {
    type Context = ();

    async fn from_stream<D: de::Decoder>(cxt: (), decoder: &mut D) -> Result<Self, D::Error> {
        let scalar = Scalar::from_stream(cxt, decoder).await?;
        Self::from_scalar(scalar).map_err(de::Error::custom)
    }
}

impl<'en> en::IntoStream<'en> for Schema {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::Collection(schema) => schema.into_stream(encoder),
            Self::Map(map) => map.into_stream(encoder),
            Self::Tuple(tuple) => tuple.into_stream(encoder),
        }
    }
}

impl fmt::Display for Schema {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Collection(collection) => fmt::Display::fmt(collection, f),
            Self::Map(schema) => fmt::Display::fmt(schema, f),
            Self::Tuple(schema) => fmt::Display::fmt(schema, f),
        }
    }
}

/// Trait defining methods common to any instance of a [`Chain`], such as a [`SyncChain`].
#[async_trait]
pub trait ChainInstance {
    /// Append the given DELETE op to the latest block in this `Chain`.
    async fn append_delete(&self, txn_id: TxnId, path: TCPathBuf, key: Value) -> TCResult<()>;

    /// Append the given PUT op to the latest block in this `Chain`.
    async fn append_put(
        &self,
        txn: &Txn,
        path: TCPathBuf,
        key: Value,
        value: State,
    ) -> TCResult<()>;

    /// Return the latest hash of this `Chain`.
    async fn hash(self, txn: Txn) -> TCResult<Output<Sha256>>;

    /// Return the `TxnId` of the last commit to this `Chain`, if there is one.
    // async fn last_commit(&self, txn_id: TxnId) -> TCResult<Option<TxnId>>;

    /// Borrow the [`Subject`] of this [`Chain`] immutably.
    fn subject(&self) -> &Subject;

    /// Replicate this [`Chain`] from the [`Chain`] at the given [`Link`].
    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()>;

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
pub enum Chain {
    Block(block::BlockChain),
    Sync(sync::SyncChain),
}

impl Instance for Chain {
    type Class = ChainType;

    fn class(&self) -> Self::Class {
        match self {
            Self::Block(_) => ChainType::Block,
            Self::Sync(_) => ChainType::Sync,
        }
    }
}

#[async_trait]
impl ChainInstance for Chain {
    async fn append_delete(&self, txn_id: TxnId, path: TCPathBuf, key: Value) -> TCResult<()> {
        match self {
            Self::Block(chain) => chain.append_delete(txn_id, path, key).await,
            Self::Sync(chain) => chain.append_delete(txn_id, path, key).await,
        }
    }

    async fn append_put(
        &self,
        txn: &Txn,
        path: TCPathBuf,
        key: Value,
        value: State,
    ) -> TCResult<()> {
        match self {
            Self::Block(chain) => chain.append_put(txn, path, key, value).await,
            Self::Sync(chain) => chain.append_put(txn, path, key, value).await,
        }
    }

    async fn hash(self, txn: Txn) -> TCResult<Output<Sha256>> {
        match self {
            Self::Block(chain) => chain.hash(txn).await,
            Self::Sync(chain) => chain.hash(txn).await,
        }
    }

    fn subject(&self) -> &Subject {
        match self {
            Self::Block(chain) => chain.subject(),
            Self::Sync(chain) => chain.subject(),
        }
    }

    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()> {
        match self {
            Self::Block(chain) => chain.replicate(txn, source).await,
            Self::Sync(chain) => chain.replicate(txn, source).await,
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
impl Transact for Chain {
    async fn commit(&self, txn_id: &TxnId) {
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
impl de::FromStream for Chain {
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        decoder.decode_map(ChainVisitor::new(txn)).await
    }
}

#[async_trait]
impl<'en> IntoView<'en, fs::Dir> for Chain {
    type Txn = Txn;
    type View = ChainView<'en>;

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        let class = self.class();

        let data = match self {
            Self::Block(chain) => chain.into_view(txn).map_ok(ChainViewData::Block).await,
            Self::Sync(chain) => {
                chain
                    .into_view(txn)
                    .map_ok(Box::new)
                    .map_ok(ChainViewData::Sync)
                    .await
            }
        }?;

        Ok(ChainView { class, data })
    }
}

impl fmt::Debug for Chain {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "instance of {}", self.class())
    }
}

impl fmt::Display for Chain {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "instance of {}", self.class())
    }
}

/// A helper struct for [`ChainView`]
pub enum ChainViewData<'en> {
    Block((Schema, data::HistoryView<'en>)),
    Sync(Box<(Schema, StateView<'en>)>),
}

/// A view of a [`Chain`] within a single `Transaction`, used for serialization.
pub struct ChainView<'en> {
    class: ChainType,
    data: ChainViewData<'en>,
}

impl<'en> en::IntoStream<'en> for ChainView<'en> {
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

/// Load a [`Chain`] from disk.
pub async fn load(txn: &Txn, class: ChainType, schema: Schema, dir: fs::Dir) -> TCResult<Chain> {
    match class {
        ChainType::Block => {
            BlockChain::load(txn, schema, dir)
                .map_ok(Chain::Block)
                .await
        }
        ChainType::Sync => SyncChain::load(txn, schema, dir).map_ok(Chain::Sync).await,
    }
}

/// A [`de::Visitor`] for deserializing a [`Chain`].
pub struct ChainVisitor {
    txn: Txn,
}

impl ChainVisitor {
    pub fn new(txn: Txn) -> Self {
        Self { txn }
    }

    pub async fn visit_map_value<A: de::MapAccess>(
        self,
        class: ChainType,
        access: &mut A,
    ) -> Result<Chain, A::Error> {
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
impl de::Visitor for ChainVisitor {
    type Value = Chain;

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

#[inline]
fn null_hash() -> Output<Sha256> {
    GenericArray::default()
}
