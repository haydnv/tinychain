//! A [`Chain`] responsible for recovering a [`State`] from a failed transaction.

use std::convert::{TryFrom, TryInto};
use std::fmt;

use async_trait::async_trait;
use destream::{de, en};
use futures::future::TryFutureExt;
use safecast::{TryCastFrom, TryCastInto};

use tc_btree::{BTreeType, Node};
use tc_error::*;
use tc_transact::fs::{Dir, File, Persist, Store};
use tc_transact::{IntoView, Transact, Transaction, TxnId};
use tcgeneric::*;

use crate::collection::{BTree, BTreeFile, CollectionType, Table, TableIndex, TableType};
use crate::fs;
use crate::scalar::{Link, OpRef, Scalar, TCRef, Value, ValueType};
use crate::state::{State, StateView};
use crate::txn::Txn;

pub use block::BlockChain;
pub use data::ChainBlock;
pub use sync::SyncChain;

mod block;
mod data;
mod sync;

const BLOCK_SIZE: u64 = 1_000_000;
const CHAIN: Label = label("chain");
const NULL_HASH: Vec<u8> = vec![];
const PREFIX: PathLabel = path_label(&["state", "chain"]);

/// The name of the file containing a [`Chain`]'s [`Subject`]'s data.
pub const SUBJECT: Label = label("subject");

/// The file extension of a directory of [`ChainBlock`]s on disk.
pub const EXT: &str = "chain";

/// The schema of a [`Chain`], used when constructing a new `Chain` or loading a `Chain` from disk.
#[derive(Clone)]
pub enum Schema {
    BTree(tc_btree::RowSchema),
    Table(tc_table::TableSchema),
    Value(Value),
}

impl Schema {
    pub fn from_scalar(scalar: Scalar) -> TCResult<Self> {
        match scalar {
            Scalar::Ref(tc_ref) => match *tc_ref {
                TCRef::Op(op_ref) => match op_ref {
                    OpRef::Get((class, schema)) => {
                        let class = TCPathBuf::try_from(class)?;
                        let class = CollectionType::from_path(&class).ok_or_else(|| {
                            TCError::bad_request("invalid Collection type", class)
                        })?;

                        match class {
                            CollectionType::BTree(_) => {
                                let schema = Value::try_cast_from(schema, |s| {
                                    TCError::bad_request("invalid BTree schema", s)
                                })?;

                                let schema = schema.try_cast_into(|s| {
                                    TCError::bad_request("invalid BTree schema", s)
                                })?;

                                Ok(Self::BTree(schema))
                            }
                            CollectionType::Table(_) => {
                                let schema = Value::try_cast_from(schema, |s| {
                                    TCError::bad_request("invalid Table schema", s)
                                })?;
                                let schema = schema.try_cast_into(|s| {
                                    TCError::bad_request("invalid Table schema", s)
                                })?;
                                Ok(Self::Table(schema))
                            }
                        }
                    }
                    other => Err(TCError::bad_request("invalid Chain schema", other)),
                },
                other => Err(TCError::bad_request("invalid Chain schema", other)),
            },
            Scalar::Value(value) => Ok(Self::Value(value)),
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
            Self::Value(value) => value.into_stream(encoder),
        }
    }
}

/// The state whose transactional integrity is protected by a [`Chain`].
#[derive(Clone)]
pub enum Subject {
    BTree(BTreeFile),
    Table(TableIndex),
    Value(fs::File<Value>),
}

impl Subject {
    /// Create a new `Subject` with the given `Schema`.
    pub async fn create(schema: Schema, dir: &fs::Dir, txn_id: TxnId) -> TCResult<Self> {
        match schema {
            Schema::BTree(schema) => {
                let file = dir
                    .create_file(txn_id, SUBJECT.into(), BTreeType::default())
                    .await?;

                let file = fs::File::<Node>::try_from(file)?;

                BTreeFile::create(file, schema, txn_id)
                    .map_ok(Self::BTree)
                    .await
            }
            Schema::Table(schema) => {
                TableIndex::create(schema, dir, txn_id)
                    .map_ok(Self::Table)
                    .await
            }
            Schema::Value(value) => {
                let file = dir
                    .create_file(txn_id, SUBJECT.into(), value.class())
                    .await?;

                let file = fs::File::<Value>::try_from(file)?;

                file.create_block(txn_id, SUBJECT.into(), value.clone())
                    .await?;

                Ok(Self::Value(file))
            }
        }
    }

    async fn load(schema: Schema, dir: &fs::Dir, txn_id: TxnId) -> TCResult<Self> {
        match schema {
            Schema::BTree(schema) => {
                if let Some(file) = dir.get_file(&txn_id, &SUBJECT.into()).await? {
                    BTreeFile::load(schema, file.try_into()?, txn_id)
                        .map_ok(Self::BTree)
                        .await
                } else {
                    Self::create(Schema::BTree(schema), dir, txn_id).await
                }
            }
            Schema::Table(schema) => {
                if dir.is_empty(&txn_id).await? {
                    Self::create(Schema::Table(schema), dir, txn_id).await
                } else {
                    TableIndex::load(schema, dir.clone(), txn_id)
                        .map_ok(Self::Table)
                        .await
                }
            }
            Schema::Value(value) => {
                if let Some(file) = dir.get_file(&txn_id, &SUBJECT.into()).await? {
                    file.try_into().map(Self::Value)
                } else {
                    Self::create(Schema::Value(value), dir, txn_id).await
                }
            }
        }
    }
}

#[async_trait]
impl Transact for Subject {
    async fn commit(&self, txn_id: &TxnId) {
        match self {
            Self::BTree(btree) => btree.commit(txn_id).await,
            Self::Table(table) => table.commit(txn_id).await,
            Self::Value(file) => file.commit(txn_id).await,
        }
    }

    async fn finalize(&self, txn_id: &TxnId) {
        match self {
            Self::BTree(btree) => btree.finalize(txn_id).await,
            Self::Table(table) => table.finalize(txn_id).await,
            Self::Value(file) => file.finalize(txn_id).await,
        }
    }
}

#[async_trait]
impl de::FromStream for Subject {
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        let value = Value::from_stream((), decoder).await?;

        let file: fs::FileEntry = txn
            .context()
            .create_file(*txn.id(), SUBJECT.into(), value.class())
            .map_err(de::Error::custom)
            .await?;

        let file: fs::File<Value> = file.try_into().map_err(de::Error::custom)?;
        file.create_block(*txn.id(), SUBJECT.into(), value)
            .map_err(de::Error::custom)
            .await?;

        Ok(Self::Value(file))
    }
}

#[async_trait]
impl<'en> IntoView<'en, fs::Dir> for Subject {
    type Txn = Txn;
    type View = StateView<'en>;

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        match self {
            Self::Value(file) => {
                let value = file.read_block(*txn.id(), SUBJECT.into()).await?;
                State::from(value.clone()).into_view(txn).await
            }
            Self::Table(table) => State::from(Table::Table(table)).into_view(txn).await,
            Self::BTree(btree) => State::from(BTree::File(btree)).into_view(txn).await,
        }
    }
}

impl fmt::Display for Subject {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BTree(btree) => write!(f, "chain Subject, {}", btree.class()),
            Self::Table(table) => write!(f, "chain Subject, {}", table.class()),
            Self::Value(_) => write!(f, "chain Subject, {}", ValueType::Value),
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
        txn_id: TxnId,
        path: TCPathBuf,
        key: Value,
        value: Scalar,
    ) -> TCResult<()>;

    async fn last_commit(&self, txn_id: TxnId) -> TCResult<Option<TxnId>>;

    /// Borrow the [`Subject`] of this [`Chain`] immutably.
    fn subject(&self) -> &Subject;

    /// Replicate this [`Chain`] from the [`Chain`] at the given [`Link`].
    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()>;
}

/// The type of a [`Chain`].
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum ChainType {
    Block,
    Sync,
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
        txn_id: TxnId,
        path: TCPathBuf,
        key: Value,
        value: Scalar,
    ) -> TCResult<()> {
        match self {
            Self::Block(chain) => chain.append_put(txn_id, path, key, value).await,
            Self::Sync(chain) => chain.append_put(txn_id, path, key, value).await,
        }
    }

    async fn last_commit(&self, txn_id: TxnId) -> TCResult<Option<TxnId>> {
        match self {
            Self::Block(chain) => chain.last_commit(txn_id).await,
            Self::Sync(chain) => chain.last_commit(txn_id).await,
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

impl fmt::Display for Chain {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "instance of {}", self.class())
    }
}

pub enum ChainViewData<'en> {
    Block((Schema, block::BlockSeq)),
    Sync(Box<(Schema, StateView<'en>)>),
}

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

pub async fn load(
    class: ChainType,
    schema: Schema,
    dir: fs::Dir,
    txn_id: TxnId,
) -> TCResult<Chain> {
    match class {
        ChainType::Block => {
            BlockChain::load(schema, dir, txn_id)
                .map_ok(Chain::Block)
                .await
        }
        ChainType::Sync => {
            SyncChain::load(schema, dir, txn_id)
                .map_ok(Chain::Sync)
                .await
        }
    }
}

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
