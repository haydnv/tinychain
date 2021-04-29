use std::fmt;
use std::marker::PhantomData;
use std::ops::Bound;

use async_trait::async_trait;
use destream::{de, en};
use futures::{future, Stream, TryFutureExt, TryStreamExt};
use log::debug;
use safecast::{Match, TryCastFrom, TryCastInto};

use tc_error::*;
use tc_transact::fs::{Dir, File};
use tc_transact::{IntoView, Transaction, TxnId};
use tc_value::{Link, NumberType, Value, ValueCollator, ValueType};
use tcgeneric::*;

#[allow(dead_code)]
mod file;
mod slice;

const PREFIX: PathLabel = path_label(&["state", "collection", "btree"]);

pub const EXT: &str = "node";

pub use file::{BTreeFile, Node};
pub use slice::BTreeSlice;

pub type Key = Vec<Value>;
pub type Range = collate::Range<Value, Key>;

#[async_trait]
pub trait BTreeInstance: Clone + Instance {
    type Slice: BTreeInstance;

    fn collator(&self) -> &ValueCollator;

    fn schema(&self) -> &RowSchema;

    fn slice(self, range: Range, reverse: bool) -> TCResult<Self::Slice>;

    async fn count(&self, txn_id: TxnId) -> TCResult<u64> {
        // TODO: reimplement this more efficiently
        let keys = self.clone().keys(txn_id).await?;
        keys.try_fold(0u64, |count, _| future::ready(Ok(count + 1)))
            .await
    }

    async fn is_empty(&self, txn_id: TxnId) -> TCResult<bool>;

    async fn delete(&self, txn_id: TxnId) -> TCResult<()>;

    async fn insert(&self, txn_id: TxnId, key: Key) -> TCResult<()>;

    async fn keys<'a>(self, txn_id: TxnId) -> TCResult<TCTryStream<'a, Key>>
    where
        Self: 'a;

    async fn try_insert_from<S: Stream<Item = TCResult<Key>> + Send + Unpin>(
        &self,
        txn_id: TxnId,
        mut keys: S,
    ) -> TCResult<()> {
        while let Some(key) = keys.try_next().await? {
            self.insert(txn_id, key).await?;
        }

        Ok(())
    }
}

#[derive(Clone, PartialEq)]
pub struct Column {
    pub name: Id,
    pub dtype: ValueType,
    pub max_len: Option<usize>,
}

impl Column {
    #[inline]
    pub fn name(&'_ self) -> &'_ Id {
        &self.name
    }

    #[inline]
    pub fn dtype(&self) -> ValueType {
        self.dtype
    }

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
        let dtype = Value::Link(Link::from(self.dtype.path()));

        if let Some(max_len) = self.max_len {
            (self.name, dtype.to_string(), max_len).into_stream(encoder)
        } else {
            (self.name, dtype.to_string()).into_stream(encoder)
        }
    }
}

impl<'a> From<&'a Column> for (&'a Id, ValueType) {
    fn from(col: &'a Column) -> (&'a Id, ValueType) {
        (&col.name, col.dtype)
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

pub type RowSchema = Vec<Column>;

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
impl<F: File<Node>, D: Dir, T: Transaction<D>> BTreeInstance for BTree<F, D, T>
where
    Self: 'static,
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

    async fn delete(&self, txn_id: TxnId) -> TCResult<()> {
        match self {
            Self::File(file) => file.delete(txn_id).await,
            Self::Slice(slice) => slice.delete(txn_id).await,
        }
    }

    async fn insert(&self, txn_id: TxnId, key: Key) -> TCResult<()> {
        match self {
            Self::File(file) => file.insert(txn_id, key).await,
            Self::Slice(slice) => slice.insert(txn_id, key).await,
        }
    }

    async fn keys<'a>(self, txn_id: TxnId) -> TCResult<TCTryStream<'a, Key>>
    where
        Self: 'a,
    {
        match self {
            Self::File(file) => file.keys(txn_id).await,
            Self::Slice(slice) => slice.keys(txn_id).await,
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
impl<F: File<Node>, D: Dir, T: Transaction<D>> de::Visitor for KeyListVisitor<F, D, T>
where
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
impl<F: File<Node>, D: Dir, T: Transaction<D>> de::FromStream for KeyListVisitor<F, D, T>
where
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
impl<F: File<Node>, D: Dir, T: Transaction<D>> de::Visitor for BTreeVisitor<F, D, T>
where
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
impl<F: File<Node>, D: Dir, T: Transaction<D>> de::FromStream for BTree<F, D, T>
where
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

impl<F, D, T> fmt::Display for BTree<F, D, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            Self::File(_) => "a BTree",
            Self::Slice(_) => "a BTree slice",
        })
    }
}

#[async_trait]
impl<'en, F: File<Node>, D: Dir, T: Transaction<D>> IntoView<'en, D> for BTree<F, D, T>
where
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

pub struct BTreeView<'en> {
    schema: Vec<Column>,
    keys: TCTryStream<'en, Key>,
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
