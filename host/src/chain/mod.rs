//! A [`Chain`] responsible for recovering a [`State`] from a failed transaction.

use std::convert::TryFrom;
use std::fmt;
use std::iter::FromIterator;

use async_trait::async_trait;
use destream::{de, en};
use futures::future::{join_all, try_join_all, TryFutureExt};
use log::debug;
use safecast::{TryCastFrom, TryCastInto};

use tc_btree::{BTreeType, Column};
use tc_error::*;
use tc_transact::fs::{Dir, Persist, Restore, Store};
use tc_transact::{IntoView, Transact, Transaction, TxnId};
use tc_value::{Link, Value};
use tcgeneric::*;

use crate::collection::{
    BTree, BTreeFile, Collection, CollectionType, Table, TableIndex, TableType,
};
#[cfg(feature = "tensor")]
use crate::collection::{
    DenseTensor, DenseTensorFile, SparseTable, SparseTensor, Tensor, TensorType,
};
use crate::fs;
use crate::scalar::{OpRef, Scalar, TCRef};
use crate::state::{State, StateType, StateView};
use crate::txn::Txn;

pub use block::BlockChain;
pub use data::ChainBlock;
pub use sync::SyncChain;

mod block;
mod data;

mod sync;

const BLOCK_SIZE: usize = 1_000_000;
const CHAIN: Label = label("chain");
const NULL_HASH: Vec<u8> = vec![];
const PREFIX: PathLabel = path_label(&["state", "chain"]);

const SUBJECT: Label = label("subject");

/// The schema of a [`Chain`], used when constructing a new `Chain` or loading a `Chain` from disk.
#[derive(Clone)]
pub enum Schema {
    BTree(tc_btree::RowSchema),
    Table(tc_table::TableSchema),
    #[cfg(feature = "tensor")]
    Dense(tc_tensor::Schema),
    #[cfg(feature = "tensor")]
    Sparse(tc_tensor::Schema),
    Tuple(Tuple<Schema>),
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

                        let schema = Value::try_cast_from(schema, |s| {
                            TCError::bad_request("expected a Value for chain schema, not", s)
                        })?;

                        match class {
                            CollectionType::BTree(_) => {
                                let schema = schema.try_cast_into(|s| {
                                    TCError::bad_request("invalid BTree schema", s)
                                })?;

                                Ok(Self::BTree(schema))
                            }
                            CollectionType::Table(_) => {
                                let schema = schema.try_cast_into(|s| {
                                    TCError::bad_request("invalid Table schema", s)
                                })?;

                                Ok(Self::Table(schema))
                            }

                            #[cfg(feature = "tensor")]
                            CollectionType::Tensor(tt) => {
                                let schema: Value = schema.try_cast_into(|s| {
                                    TCError::bad_request("invalid Tensor schema", s)
                                })?;
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
                    other => Err(TCError::bad_request("invalid Chain schema", other)),
                },
                other => Err(TCError::bad_request("invalid Chain schema", other)),
            },
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
            Self::Tuple(tuple) => tuple.into_stream(encoder),
        }
    }
}

impl fmt::Display for Schema {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BTree(schema) => write!(f, "{}", Tuple::<&Column>::from_iter(schema)),
            #[cfg(feature = "tensor")]
            Self::Dense(schema) => fmt::Display::fmt(schema, f),
            #[cfg(feature = "tensor")]
            Self::Sparse(schema) => fmt::Display::fmt(schema, f),
            Self::Table(schema) => fmt::Display::fmt(schema, f),
            Self::Tuple(tuple) => fmt::Display::fmt(tuple, f),
        }
    }
}

/// The state whose transactional integrity is protected by a [`Chain`].
#[derive(Clone)]
pub enum Subject {
    BTree(BTreeFile),
    Table(TableIndex),
    #[cfg(feature = "tensor")]
    Dense(DenseTensor<DenseTensorFile>),
    #[cfg(feature = "tensor")]
    Sparse(SparseTensor<SparseTable>),
    Tuple(Tuple<Subject>),
}

impl Subject {
    /// Create a new `Subject` with the given `Schema`.
    pub fn create<'a>(schema: Schema, dir: &'a fs::Dir, txn_id: TxnId) -> TCBoxTryFuture<'a, Self> {
        Box::pin(async move {
            match schema {
                #[cfg(feature = "tensor")]
                Schema::Dense(schema) => {
                    let file = dir
                        .create_file(txn_id, SUBJECT.into(), TensorType::Dense)
                        .await?;

                    DenseTensor::create(file, schema, txn_id)
                        .map_ok(Self::Dense)
                        .await
                }
                #[cfg(feature = "tensor")]
                Schema::Sparse(schema) => {
                    let dir = dir.create_dir(txn_id, SUBJECT.into()).await?;
                    let tensor = SparseTensor::create(&dir, schema, txn_id)
                        .map_ok(Self::Sparse)
                        .await?;

                    Ok(tensor)
                }
                Schema::Table(schema) => {
                    TableIndex::create(dir, schema, txn_id)
                        .map_ok(Self::Table)
                        .await
                }
                Schema::Tuple(schema) => {
                    try_join_all(
                        schema
                            .into_iter()
                            .enumerate()
                            .map(|(i, schema)| async move {
                                let dir = dir.create_dir(txn_id, i.into()).await?;
                                Self::create(schema, &dir, txn_id).await
                            }),
                    )
                    .map_ok(Tuple::from)
                    .map_ok(Self::Tuple)
                    .await
                }
                Schema::BTree(schema) => {
                    let file = dir
                        .create_file(txn_id, SUBJECT.into(), BTreeType::default())
                        .await?;

                    BTreeFile::create(file, schema, txn_id)
                        .map_ok(Self::BTree)
                        .await
                }
            }
        })
    }

    fn load<'a>(txn: &'a Txn, schema: Schema, dir: &'a fs::Dir) -> TCBoxTryFuture<'a, Self> {
        Box::pin(async move {
            match schema {
                Schema::BTree(schema) => {
                    if let Some(file) = dir.get_file(*txn.id(), &SUBJECT.into()).await? {
                        BTreeFile::load(txn, schema, file).map_ok(Self::BTree).await
                    } else {
                        Self::create(Schema::BTree(schema), dir, *txn.id()).await
                    }
                }
                Schema::Table(schema) => {
                    if dir.is_empty(*txn.id()).await? {
                        Self::create(Schema::Table(schema), dir, *txn.id()).await
                    } else {
                        TableIndex::load(txn, schema, dir.clone())
                            .map_ok(Self::Table)
                            .await
                    }
                }
                #[cfg(feature = "tensor")]
                Schema::Dense(schema) => {
                    if let Some(file) = dir.get_file(*txn.id(), &SUBJECT.into()).await? {
                        DenseTensor::load(txn, schema, file)
                            .map_ok(Self::Dense)
                            .await
                    } else {
                        Self::create(Schema::Dense(schema), dir, *txn.id()).await
                    }
                }
                #[cfg(feature = "tensor")]
                Schema::Sparse(schema) => {
                    if let Some(dir) = dir.get_dir(*txn.id(), &SUBJECT.into()).await? {
                        SparseTensor::load(txn, schema, dir)
                            .map_ok(Self::Sparse)
                            .await
                    } else {
                        Self::create(Schema::Sparse(schema), dir, *txn.id()).await
                    }
                }
                Schema::Tuple(schema) => {
                    let txn_id = *txn.id();
                    try_join_all(
                        schema
                            .into_iter()
                            .enumerate()
                            .map(|(i, schema)| async move {
                                if let Some(dir) = dir.get_dir(txn_id, &i.into()).await? {
                                    Self::load(txn, schema, &dir).await
                                } else {
                                    let dir = dir.create_dir(txn_id, i.into()).await?;
                                    Self::create(schema, &dir, txn_id).await
                                }
                            }),
                    )
                    .map_ok(Tuple::from)
                    .map_ok(Self::Tuple)
                    .await
                }
            }
        })
    }

    fn restore<'a>(&'a self, txn: &'a Txn, backup: State) -> TCBoxTryFuture<()> {
        Box::pin(async move {
            let txn_id = *txn.id();
            match self {
                Self::BTree(btree) => match backup {
                    State::Collection(Collection::BTree(BTree::File(backup))) => {
                        btree.restore(&backup, txn_id).await
                    }
                    other => Err(TCError::bad_request("cannot restore a BTree from", other)),
                },
                #[cfg(feature = "tensor")]
                Self::Dense(tensor) => match backup {
                    State::Collection(Collection::Tensor(Tensor::Dense(backup))) => {
                        let file = txn
                            .context()
                            .create_file_unique(txn_id, TensorType::Dense)
                            .await?;

                        let backup =
                            tc_transact::fs::CopyFrom::copy_from(backup, file, txn).await?;

                        tensor.restore(&backup, txn_id).await
                    }
                    other => Err(TCError::bad_request(
                        "cannot restore a dense Tensor from",
                        other,
                    )),
                },
                #[cfg(feature = "tensor")]
                Self::Sparse(tensor) => match backup {
                    State::Collection(Collection::Tensor(Tensor::Sparse(backup))) => {
                        let dir = txn.context().create_dir_unique(txn_id).await?;
                        let backup = tc_transact::fs::CopyFrom::copy_from(backup, dir, txn).await?;
                        tensor.restore(&backup, txn_id).await
                    }
                    other => Err(TCError::bad_request(
                        "cannot restore a sparse Tensor from",
                        other,
                    )),
                },
                Self::Table(table) => match backup {
                    State::Collection(Collection::Table(Table::Table(backup))) => {
                        table.restore(&backup, txn_id).await
                    }
                    other => Err(TCError::bad_request("cannot restore a Table from", other)),
                },
                Self::Tuple(tuple) => match backup {
                    State::Tuple(backup) if backup.len() == tuple.len() => {
                        let restores =
                            tuple
                                .iter()
                                .zip(backup)
                                .map(|(subject, backup)| async move {
                                    subject.restore(txn, backup).await
                                });

                        try_join_all(restores).await?;
                        Ok(())
                    }
                    State::Tuple(_) => Err(TCError::bad_request(
                        "backup has the wrong number of subjects for schema",
                        tuple,
                    )),
                    backup => Err(TCError::unsupported(format!(
                        "invalid backup for schema {}: {}",
                        tuple, backup
                    ))),
                },
            }
        })
    }
}

impl Instance for Subject {
    type Class = StateType;

    fn class(&self) -> Self::Class {
        match self {
            Self::BTree(btree) => CollectionType::BTree(btree.class()).into(),
            #[cfg(feature = "tensor")]
            Self::Dense(dense) => CollectionType::Tensor(dense.class()).into(),
            #[cfg(feature = "tensor")]
            Self::Sparse(sparse) => CollectionType::Tensor(sparse.class()).into(),
            Self::Table(table) => CollectionType::Table(table.class()).into(),
            Self::Tuple(_) => StateType::Tuple,
        }
    }
}

#[async_trait]
impl Transact for Subject {
    async fn commit(&self, txn_id: &TxnId) {
        debug!("commit chain subject");

        match self {
            Self::BTree(btree) => btree.commit(txn_id).await,
            Self::Table(table) => table.commit(txn_id).await,
            #[cfg(feature = "tensor")]
            Self::Dense(tensor) => tensor.commit(txn_id).await,
            #[cfg(feature = "tensor")]
            Self::Sparse(tensor) => tensor.commit(txn_id).await,
            Self::Tuple(tuple) => {
                join_all(
                    tuple
                        .iter()
                        .map(|subject| async move { subject.commit(txn_id).await }),
                )
                .await;
            }
        }
    }

    async fn finalize(&self, txn_id: &TxnId) {
        debug!("finalize chain subject");

        match self {
            Self::BTree(btree) => btree.finalize(txn_id).await,
            Self::Table(table) => table.finalize(txn_id).await,
            #[cfg(feature = "tensor")]
            Self::Dense(tensor) => tensor.finalize(txn_id).await,
            #[cfg(feature = "tensor")]
            Self::Sparse(tensor) => tensor.finalize(txn_id).await,
            Self::Tuple(tuple) => {
                join_all(tuple.iter().map(|subject| async move {
                    subject.finalize(txn_id).await;
                }))
                .await;
            }
        }
    }
}

#[async_trait]
impl de::FromStream for Subject {
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        Err(de::Error::custom("not implemented: decode a Chain subject"))
    }
}

#[async_trait]
impl<'en> IntoView<'en, fs::Dir> for Subject {
    type Txn = Txn;
    type View = StateView<'en>;

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        match self {
            Self::BTree(btree) => State::from(BTree::File(btree)).into_view(txn).await,
            Self::Table(table) => State::from(Table::Table(table)).into_view(txn).await,
            #[cfg(feature = "tensor")]
            Self::Dense(tensor) => State::from(Tensor::from(tensor)).into_view(txn).await,
            #[cfg(feature = "tensor")]
            Self::Sparse(tensor) => State::from(Tensor::from(tensor)).into_view(txn).await,
            Self::Tuple(tuple) => {
                try_join_all(
                    tuple
                        .into_iter()
                        .map(|subject| subject.into_view(txn.clone())),
                )
                .map_ok(StateView::Tuple)
                .await
            }
        }
    }
}

impl From<Subject> for State {
    fn from(subject: Subject) -> Self {
        match subject {
            Subject::BTree(btree) => State::Collection(btree.into()),
            #[cfg(feature = "tensor")]
            Subject::Dense(dense) => State::Collection(dense.into()),
            #[cfg(feature = "tensor")]
            Subject::Sparse(sparse) => State::Collection(sparse.into()),
            Subject::Table(table) => State::Collection(table.into()),
            Subject::Tuple(tuple) => State::Tuple(tuple.into_iter().map(State::from).collect()),
        }
    }
}

impl fmt::Display for Subject {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BTree(btree) => write!(f, "chain Subject, {}", btree.class()),
            Self::Table(table) => write!(f, "chain Subject, {}", table.class()),
            #[cfg(feature = "tensor")]
            Self::Dense(_) => write!(f, "chain Subject, {}", TensorType::Dense),
            #[cfg(feature = "tensor")]
            Self::Sparse(_) => write!(f, "chain Subject, {}", TensorType::Sparse),
            Self::Tuple(tuple) => fmt::Display::fmt(tuple, f),
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

    async fn last_commit(&self, txn_id: TxnId) -> TCResult<Option<TxnId>>;

    /// Borrow the [`Subject`] of this [`Chain`] immutably.
    fn subject(&self) -> &Subject;

    /// Replicate this [`Chain`] from the [`Chain`] at the given [`Link`].
    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()>;

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

/// A view of a [`Chain`] within a single [`Transaction`], used for serialization.
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
