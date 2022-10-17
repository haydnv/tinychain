//! A [`BTree`], an ordered transaction-aware collection of [`Key`]s

use std::fmt;
use std::marker::PhantomData;
use std::ops::Bound;

use async_trait::async_trait;
use destream::{de, en};
use futures::{future, Stream, TryFutureExt, TryStreamExt};
use log::debug;
use safecast::*;
use uuid::Uuid;

use tc_error::*;
use tc_transact::fs::{Dir, File};
use tc_transact::{IntoView, Transaction, TxnId};
use tc_value::{NumberType, Value, ValueCollator, ValueType};
use tcgeneric::*;

pub use file::{BTreeFile, Node};
pub use slice::BTreeSlice;

mod file;
mod slice;

const ERR_VIEW_WRITE: &str = "BTree view does not support write operations";
const PREFIX: PathLabel = path_label(&["state", "collection", "btree"]);

pub type NodeId = Uuid;

/// The file extension of a [`BTree`] as stored on-disk
pub const EXT: &str = "node";

/// A [`BTree`] key.
pub type Key = Vec<Value>;

/// A [`BTree`] selector.
pub type Range = collate::Range<Value, Key>;

/// Common [`BTree`] methods.
#[async_trait]
pub trait BTreeInstance: Clone + Instance {
    type Slice: BTreeInstance;

    /// Borrow this `BTree`'s collator.
    fn collator(&self) -> &ValueCollator;

    /// Borrow to this `BTree`'s schema.
    fn schema(&self) -> &RowSchema;

    /// Return a slice of this `BTree` with the given range.
    fn slice(self, range: Range, reverse: bool) -> TCResult<Self::Slice>;

    /// Return the number of [`Key`]s in this `BTree`.
    async fn count(&self, txn_id: TxnId) -> TCResult<u64> {
        // TODO: reimplement this more efficiently
        let keys = self.clone().keys(txn_id).await?;
        keys.try_fold(0u64, |count, _| future::ready(Ok(count + 1)))
            .await
    }

    /// Return `true` if this `BTree` has no [`Key`]s.
    async fn is_empty(&self, txn_id: TxnId) -> TCResult<bool>;

    /// Return a `Stream` of this `BTree`'s [`Key`]s.
    async fn keys<'a>(self, txn_id: TxnId) -> TCResult<TCBoxTryStream<'a, Key>>
    where
        Self: 'a;

    /// Return an error if the given key does not match this `BTree`'s schema
    ///
    /// If the key is valid, this will return a copy with the data types correctly casted.
    fn validate_key(&self, key: Key) -> TCResult<Key>;
}

/// [`BTree`] write methods.
#[async_trait]
pub trait BTreeWrite: BTreeInstance {
    /// Delete all the [`Key`]s in this `BTree`.
    async fn delete(&self, txn_id: TxnId, range: Range) -> TCResult<()>;

    /// Insert the given [`Key`] into this `BTree`.
    ///
    /// If the [`Key`] is already present, this is a no-op.
    async fn insert(&self, txn_id: TxnId, key: Key) -> TCResult<()>;

    /// Insert all the keys from the given `Stream` into this `BTree`.
    ///
    /// This will stop and return an error if it encounters an invalid [`Key`].
    async fn try_insert_from<S: Stream<Item = TCResult<Key>> + Send + Unpin>(
        &self,
        txn_id: TxnId,
        keys: S,
    ) -> TCResult<()> {
        keys.map_ok(|key| self.insert(txn_id, key))
            .try_buffer_unordered(num_cpus::get())
            .try_fold((), |(), ()| future::ready(Ok(())))
            .await
    }
}

/// A `Column` used in the schema of a [`BTree`].
#[derive(Clone, Eq, PartialEq)]
pub struct Column {
    pub name: Id,
    pub dtype: ValueType,
    pub max_len: Option<usize>,
}

impl Column {
    /// Get the name of this column.
    #[inline]
    pub fn name(&'_ self) -> &'_ Id {
        &self.name
    }

    /// Get the [`Class`] of this column.
    #[inline]
    pub fn dtype(&self) -> ValueType {
        self.dtype
    }

    /// Get the maximum size (in bytes) of this column.
    #[inline]
    pub fn max_len(&'_ self) -> &'_ Option<usize> {
        &self.max_len
    }
}

impl<I: Into<Id>> From<(I, NumberType)> for Column {
    fn from(column: (I, NumberType)) -> Column {
        let (name, dtype) = column;
        let name: Id = name.into();
        let dtype: ValueType = dtype.into();
        let max_len = None;

        Column {
            name,
            dtype,
            max_len,
        }
    }
}

impl From<(Id, ValueType)> for Column {
    fn from(column: (Id, ValueType)) -> Column {
        let (name, dtype) = column;
        let max_len = None;

        Column {
            name,
            dtype,
            max_len,
        }
    }
}

impl From<(Id, ValueType, usize)> for Column {
    fn from(column: (Id, ValueType, usize)) -> Column {
        let (name, dtype, size) = column;
        let max_len = Some(size);

        Column {
            name,
            dtype,
            max_len,
        }
    }
}

impl TryCastFrom<Value> for Column {
    fn can_cast_from(value: &Value) -> bool {
        debug!("Column::can_cast_from {}?", value);

        value.matches::<(Id, ValueType)>() || value.matches::<(Id, ValueType, u64)>()
    }

    fn opt_cast_from(value: Value) -> Option<Column> {
        if value.matches::<(Id, ValueType)>() {
            let (name, dtype) = value.opt_cast_into().unwrap();

            Some(Column {
                name,
                dtype,
                max_len: None,
            })
        } else if value.matches::<(Id, ValueType, u64)>() {
            let (name, dtype, max_len) = value.opt_cast_into().unwrap();

            Some(Column {
                name,
                dtype,
                max_len: Some(max_len),
            })
        } else {
            None
        }
    }
}

impl From<Column> for Value {
    fn from(column: Column) -> Self {
        Value::Tuple(
            vec![
                column.name.into(),
                column.dtype.path().into(),
                column.max_len.map(Value::from).into(),
            ]
            .into(),
        )
    }
}

struct ColumnVisitor;

#[async_trait]
impl de::Visitor for ColumnVisitor {
    type Value = Column;

    fn expecting() -> &'static str {
        "a Column definition"
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let name = seq
            .next_element(())
            .await?
            .ok_or_else(|| de::Error::invalid_length(0, "a Column name"))?;

        let dtype = seq
            .next_element(())
            .await?
            .ok_or_else(|| de::Error::invalid_length(1, "a Column data type"))?;

        let max_len = seq.next_element(()).await?;

        Ok(Column {
            name,
            dtype,
            max_len,
        })
    }
}

#[async_trait]
impl de::FromStream for Column {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        decoder.decode_seq(ColumnVisitor).await
    }
}

impl<'en> en::IntoStream<'en> for Column {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        if let Some(max_len) = self.max_len {
            (self.name, self.dtype, max_len).into_stream(encoder)
        } else {
            (self.name, self.dtype).into_stream(encoder)
        }
    }
}

impl<'a> From<&'a Column> for (&'a Id, ValueType) {
    fn from(col: &'a Column) -> (&'a Id, ValueType) {
        (&col.name, col.dtype)
    }
}

impl fmt::Debug for Column {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.max_len {
            Some(max_len) => write!(f, "{}: {}({})", self.name, self.dtype, max_len),
            None => write!(f, "{}: {}", self.name, self.dtype),
        }
    }
}

impl fmt::Display for Column {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.max_len {
            Some(max_len) => write!(f, "{}: {}({})", self.name, self.dtype, max_len),
            None => write!(f, "{}: {}", self.name, self.dtype),
        }
    }
}

/// The schema of a [`BTree`].
pub type RowSchema = Vec<Column>;

/// The [`Class`] of a [`BTree`].
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum BTreeType {
    File,
    Slice,
}

impl Class for BTreeType {}

impl NativeClass for BTreeType {
    // These functions are only used for serialization,
    // and there's no way to transmit a BTreeSlice.

    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if &path[..] == &PREFIX[..] {
            Some(Self::File)
        } else {
            None
        }
    }

    fn path(&self) -> TCPathBuf {
        PREFIX.into()
    }
}

impl Default for BTreeType {
    fn default() -> Self {
        Self::File
    }
}

impl fmt::Display for BTreeType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::File => f.write_str("type BTree"),
            Self::Slice => f.write_str("type BTreeSlice"),
        }
    }
}

/// A stateful, transaction-aware, ordered collection of [`Key`]s with O(log n) inserts and slicing
#[derive(Clone)]
pub enum BTree<F, D, T> {
    File(BTreeFile<F, D, T>),
    Slice(BTreeSlice<F, D, T>),
}

impl<F: Send + Sync, D: Send + Sync, T: Send + Sync> Instance for BTree<F, D, T> {
    type Class = BTreeType;

    fn class(&self) -> BTreeType {
        match self {
            Self::File(file) => file.class(),
            Self::Slice(slice) => slice.class(),
        }
    }
}

#[async_trait]
impl<F, D, T> BTreeInstance for BTree<F, D, T>
where
    F: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
{
    type Slice = Self;

    fn collator(&self) -> &ValueCollator {
        match self {
            Self::File(file) => file.collator(),
            Self::Slice(slice) => slice.collator(),
        }
    }

    fn schema(&self) -> &RowSchema {
        match self {
            Self::File(file) => file.schema(),
            Self::Slice(slice) => slice.schema(),
        }
    }

    fn slice(self, range: Range, reverse: bool) -> TCResult<Self> {
        if range == Range::default() && !reverse {
            return Ok(self);
        }

        match self {
            Self::File(file) => file.slice(range, reverse).map(BTree::Slice),
            Self::Slice(slice) => slice.slice(range, reverse).map(BTree::Slice),
        }
    }

    async fn is_empty(&self, txn_id: TxnId) -> TCResult<bool> {
        match self {
            Self::File(file) => file.is_empty(txn_id).await,
            Self::Slice(slice) => slice.is_empty(txn_id).await,
        }
    }

    async fn keys<'a>(self, txn_id: TxnId) -> TCResult<TCBoxTryStream<'a, Key>>
    where
        Self: 'a,
    {
        match self {
            Self::File(file) => file.keys(txn_id).await,
            Self::Slice(slice) => slice.keys(txn_id).await,
        }
    }

    fn validate_key(&self, key: Key) -> TCResult<Key> {
        match self {
            Self::File(file) => file.validate_key(key),
            Self::Slice(slice) => slice.validate_key(key),
        }
    }
}

#[async_trait]
impl<F: File<Key = NodeId, Block = Node>, D: Dir, T: Transaction<D>> BTreeWrite for BTree<F, D, T>
where
    Self: 'static,
{
    async fn delete(&self, txn_id: TxnId, range: Range) -> TCResult<()> {
        match self {
            Self::File(file) => file.delete(txn_id, range).await,
            _ => Err(TCError::unsupported(ERR_VIEW_WRITE)),
        }
    }

    async fn insert(&self, txn_id: TxnId, key: Key) -> TCResult<()> {
        match self {
            Self::File(file) => file.insert(txn_id, key).await,
            _ => Err(TCError::unsupported(ERR_VIEW_WRITE)),
        }
    }
}

impl<F, D, T> From<BTreeFile<F, D, T>> for BTree<F, D, T> {
    fn from(btree: BTreeFile<F, D, T>) -> Self {
        Self::File(btree)
    }
}

impl<F, D, T> From<BTreeSlice<F, D, T>> for BTree<F, D, T> {
    fn from(btree: BTreeSlice<F, D, T>) -> Self {
        Self::Slice(btree)
    }
}

struct KeyListVisitor<F, D, T> {
    txn_id: TxnId,
    btree: BTreeFile<F, D, T>,
}

#[async_trait]
impl<F, D, T> de::Visitor for KeyListVisitor<F, D, T>
where
    F: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    Self: Send + Sync + 'static,
{
    type Value = Self;

    fn expecting() -> &'static str {
        "a sequence of BTree rows"
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        while let Some(row) = seq.next_element(()).await? {
            self.btree
                .insert(self.txn_id, row)
                .map_err(de::Error::custom)
                .await?;
        }

        Ok(self)
    }
}

#[async_trait]
impl<F, D, T> de::FromStream for KeyListVisitor<F, D, T>
where
    F: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    Self: Send + Sync + 'static,
{
    type Context = (TxnId, BTreeFile<F, D, T>);

    async fn from_stream<De: de::Decoder>(
        cxt: (TxnId, BTreeFile<F, D, T>),
        decoder: &mut De,
    ) -> Result<Self, De::Error> {
        let (txn_id, btree) = cxt;
        decoder.decode_seq(Self { txn_id, btree }).await
    }
}

struct BTreeVisitor<F, D, T> {
    txn: T,
    file: F,
    dir: PhantomData<D>,
}

#[async_trait]
impl<F, D, T> de::Visitor for BTreeVisitor<F, D, T>
where
    F: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    Self: Send + Sync + 'static,
{
    type Value = BTree<F, D, T>;

    fn expecting() -> &'static str {
        "a BTree"
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let schema = seq
            .next_element(())
            .await?
            .ok_or_else(|| de::Error::custom("expected BTree schema"))?;

        let btree = BTreeFile::create(self.file, schema, *self.txn.id())
            .map_err(de::Error::custom)
            .await?;

        if let Some(visitor) = seq
            .next_element::<KeyListVisitor<F, D, T>>((*self.txn.id(), btree.clone()))
            .await?
        {
            Ok(BTree::File(visitor.btree))
        } else {
            Ok(BTree::File(btree))
        }
    }
}

#[async_trait]
impl<F, D, T> de::FromStream for BTree<F, D, T>
where
    F: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    Self: Send + Sync + 'static,
{
    type Context = (T, F);

    async fn from_stream<De: de::Decoder>(
        cxt: (T, F),
        decoder: &mut De,
    ) -> Result<Self, De::Error> {
        let (txn, file) = cxt;
        let visitor = BTreeVisitor {
            txn,
            file,
            dir: PhantomData,
        };

        decoder.decode_seq(visitor).await
    }
}

impl<F, D, T> fmt::Debug for BTree<F, D, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl<F, D, T> fmt::Display for BTree<F, D, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            Self::File(_) => "a BTree",
            Self::Slice(_) => "a BTree slice",
        })
    }
}

#[async_trait]
impl<'en, F, D, T> IntoView<'en, D> for BTree<F, D, T>
where
    F: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    Self: 'static,
{
    type Txn = T;
    type View = BTreeView<'en>;

    async fn into_view(self, txn: T) -> TCResult<BTreeView<'en>> {
        let schema = self.schema().to_vec();
        let keys = self.keys(*txn.id()).await?;
        Ok(BTreeView { schema, keys })
    }
}

/// A view of a [`BTree`] within a single [`Transaction`], used in serialization.
pub struct BTreeView<'en> {
    schema: Vec<Column>,
    keys: TCBoxTryStream<'en, Key>,
}

impl<'en> en::IntoStream<'en> for BTreeView<'en> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        (self.schema, en::SeqStream::from(self.keys)).into_stream(encoder)
    }
}

#[inline]
fn validate_range(range: Range, schema: &[Column]) -> TCResult<Range> {
    if range.len() > schema.len() {
        return Err(TCError::bad_request(
            "too many columns in range",
            range.len(),
        ));
    }

    let (input_prefix, start, end) = range.into_inner();

    let mut prefix = Vec::with_capacity(input_prefix.len());
    for (value, column) in input_prefix.into_iter().zip(schema) {
        let value = column.dtype.try_cast(value)?;
        prefix.push(value);
    }

    if prefix.len() < schema.len() {
        let dtype = schema.get(prefix.len()).unwrap().dtype;
        let validate_bound = |bound| match bound {
            Bound::Unbounded => Ok(Bound::Unbounded),
            Bound::Included(value) => dtype.try_cast(value).map(Bound::Included),
            Bound::Excluded(value) => dtype.try_cast(value).map(Bound::Excluded),
        };

        let start = validate_bound(start)?;
        let end = validate_bound(end)?;

        Ok((prefix, start, end).into())
    } else {
        Ok(Range::with_prefix(prefix))
    }
}
