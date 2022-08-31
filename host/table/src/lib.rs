//! A [`Table`], an ordered collection of [`Row`]s which supports `BTree`-based indexing

// TODO: regularize use of inline vs "where" trait bounds

use std::fmt;
use std::marker::PhantomData;

use async_trait::async_trait;
use destream::{de, en};
use futures::future::{self, TryFutureExt};
use futures::stream::TryStreamExt;
use safecast::AsType;

use tc_error::*;
use tc_transact::fs::{Dir, DirRead, File};
use tc_transact::{IntoView, Transaction, TxnId};
use tc_value::Value;
use tcgeneric::{
    path_label, Class, Id, Instance, NativeClass, PathLabel, PathSegment, TCBoxTryStream, TCPathBuf,
};

use index::*;
use view::*;

pub use tc_btree::Node;

pub use bounds::*;
pub use index::TableIndex;
pub use schema::*;
pub use view::Merged;

mod bounds;
mod index;
mod schema;
mod view;

/// The key of a [`Table`] row.
pub type Key = Vec<Value>;

/// The values of a [`Table`] row.
pub type Values = Vec<Value>;

const PATH: PathLabel = path_label(&["state", "collection", "table"]);

/// Methods common to every [`Table`] view.
pub trait TableInstance: Instance<Class = TableType> {
    /// Return the schema of this `Table`'s key.
    fn key(&self) -> &[Column];

    /// Return the schema of this `Table`'s values.
    fn values(&self) -> &[Column];

    /// Return the schema of this `Table`.
    fn schema(&self) -> TableSchema;
}

/// [`Table`] sort methods
pub trait TableOrder: TableInstance {
    /// The type of `Table` returned by this instance's `order_by` method.
    type OrderBy: TableInstance;

    /// The type of `Table` returned by this instance's `reversed` method.
    type Reverse: TableInstance;

    /// Return an ordered view of this table.
    fn order_by(self, columns: Vec<Id>, reverse: bool) -> TCResult<Self::OrderBy>;

    /// Reverse the order returned by `rows`.
    fn reverse(self) -> TCResult<Self::Reverse>;

    /// Return an error if this table does not support ordering by the given columns.
    fn validate_order(&self, order: &[Id]) -> TCResult<()>;
}

/// Method to read a single row from a [`Table`]
#[async_trait]
pub trait TableRead: TableInstance {
    async fn read(&self, txn_id: &TxnId, key: &Key) -> TCResult<Option<Vec<Value>>>;
}

/// Methods for slicing a [`Table`]
pub trait TableSlice: TableStream {
    /// The type of `Table` returned by this instance's `slice` method.
    type Slice: TableInstance;

    /// Limit the returned `rows` to the given [`Bounds`].
    fn slice(self, _bounds: Bounds) -> TCResult<Self::Slice>;

    /// Return an error if this table does not support the given [`Bounds`].
    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()>;
}

/// [`Table`] read methods
#[async_trait]
pub trait TableStream: TableInstance + Sized {
    type Limit: TableInstance;
    type Selection: TableInstance;

    /// Return the number of rows in this `Table`.
    async fn count(self, txn_id: TxnId) -> TCResult<u64> {
        let rows = self.rows(txn_id).await?;
        rows.try_fold(0, |count, _| future::ready(Ok(count + 1)))
            .await
    }

    /// Limit the number of rows returned by `rows`.
    fn limit(self, limit: u64) -> Self::Limit;

    /// Limit the columns returned by `rows`.
    fn select(self, columns: Vec<Id>) -> TCResult<Self::Selection>;

    /// Return a stream of the rows in this `Table`.
    async fn rows<'a>(self, txn_id: TxnId) -> TCResult<TCBoxTryStream<'a, Vec<Value>>>;
}

/// [`Table`] write methods
#[async_trait]
pub trait TableWrite: TableInstance {
    /// Delete the given [`Row`] from this table, if present.
    async fn delete(&self, txn_id: TxnId, key: Key) -> TCResult<()>;

    /// Update one row of this table.
    async fn update(&self, txn_id: TxnId, key: Key, values: Row) -> TCResult<()>;

    /// Insert or update the given row.
    async fn upsert(&self, txn_id: TxnId, key: Key, values: Values) -> TCResult<()>;
}

/// The [`Class`] of a [`Table`].
#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub enum TableType {
    Table,
    Index,
    IndexSlice,
    Limit,
    Merge,
    Selection,
    TableSlice,
}

impl Class for TableType {}

impl NativeClass for TableType {
    // these functions are only used for serialization, and only a base table can be deserialized

    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if path.len() == 3 && &path[..] == &PATH[..] {
            Some(Self::Table)
        } else {
            None
        }
    }

    fn path(&self) -> TCPathBuf {
        PATH.into()
    }
}

impl Default for TableType {
    fn default() -> Self {
        Self::Table
    }
}

impl fmt::Display for TableType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Table => write!(f, "type Table"),
            Self::Index => write!(f, "type Index"),
            Self::IndexSlice => write!(f, "type Index slice"),
            Self::Limit => write!(f, "type Limit selection"),
            Self::Merge => write!(f, "type Merge selection"),
            Self::Selection => write!(f, "type Column selection"),
            Self::TableSlice => write!(f, "type Table slice"),
        }
    }
}

/// An ordered collection of [`Row`]s which supports `BTree`-based indexing
#[derive(Clone)]
pub enum Table<F, D, Txn> {
    Table(TableIndex<F, D, Txn>),
    Index(Index<F, D, Txn>),
    IndexSlice(IndexSlice<F, D, Txn>),
    Limit(Box<Limited<F, D, Txn>>),
    Merge(Merged<F, D, Txn>),
    Selection(Box<Selection<F, D, Txn, Table<F, D, Txn>>>),
    TableSlice(view::TableSlice<F, D, Txn>),
}

impl<F, D, Txn> Instance for Table<F, D, Txn>
where
    Self: Send + Sync,
{
    type Class = TableType;

    fn class(&self) -> Self::Class {
        match self {
            Self::Table(_) => TableType::Table,
            Self::Index(_) => TableType::Index,
            Self::IndexSlice(_) => TableType::IndexSlice,
            Self::Limit(_) => TableType::Limit,
            Self::Merge(_) => TableType::Merge,
            Self::Selection(_) => TableType::Selection,
            Self::TableSlice(_) => TableType::TableSlice,
        }
    }
}

impl<F: File<Node>, D: Dir, Txn: Transaction<D>> TableInstance for Table<F, D, Txn>
where
    Self: Send + Sync,
    <D::Read as DirRead>::FileEntry: AsType<F>,
{
    fn key(&self) -> &[Column] {
        match self {
            Self::Table(table) => table.key(),
            Self::Index(index) => index.key(),
            Self::IndexSlice(slice) => slice.key(),
            Self::Limit(limit) => limit.key(),
            Self::Merge(merge) => merge.key(),
            Self::Selection(selection) => selection.key(),
            Self::TableSlice(slice) => slice.key(),
        }
    }

    fn values(&self) -> &[Column] {
        match self {
            Self::Table(table) => table.values(),
            Self::Index(slice) => slice.values(),
            Self::IndexSlice(slice) => slice.values(),
            Self::Limit(limit) => limit.values(),
            Self::Merge(merge) => merge.values(),
            Self::Selection(selection) => selection.values(),
            Self::TableSlice(slice) => slice.values(),
        }
    }

    fn schema(&self) -> TableSchema {
        match self {
            Self::Table(table) => table.schema(),
            Self::Index(slice) => TableInstance::schema(slice),
            Self::IndexSlice(slice) => TableInstance::schema(slice),
            Self::Limit(limit) => limit.schema(),
            Self::Merge(merge) => merge.schema(),
            Self::Selection(selection) => selection.schema(),
            Self::TableSlice(slice) => slice.schema(),
        }
    }
}

#[async_trait]
impl<F: File<Node>, D: Dir, Txn: Transaction<D>> TableOrder for Table<F, D, Txn>
where
    Self: Send + Sync,
    <D::Read as DirRead>::FileEntry: AsType<F>,
{
    type OrderBy = Self;
    type Reverse = Self;

    fn order_by(self, order: Vec<Id>, reverse: bool) -> TCResult<Self::OrderBy> {
        match self {
            Self::Table(table) => table.order_by(order, reverse).map(Self::from),
            Self::Index(index) => index.order_by(order, reverse).map(Self::from),
            Self::IndexSlice(slice) => slice.order_by(order, reverse).map(Self::from),
            Self::Merge(merge) => merge.order_by(order, reverse).map(Self::from),
            Self::Selection(selection) => selection.order_by(order, reverse).map(Self::from),
            Self::TableSlice(slice) => slice.order_by(order, reverse).map(Self::from),
            other => Err(TCError::unsupported(format!(
                "instance of {} does not support ordering",
                other.class()
            ))),
        }
    }

    fn reverse(self) -> TCResult<Self::Reverse> {
        match self {
            Self::Table(table) => table.reverse().map(Self::from),
            Self::Index(index) => index.reverse().map(Self::from),
            Self::IndexSlice(slice) => slice.reverse().map(Self::from),
            Self::Merge(merge) => merge.reverse().map(Self::from),
            Self::Selection(selection) => selection.reverse().map(Self::from),
            Self::TableSlice(slice) => slice.reverse().map(Self::from),
            other => Err(TCError::unsupported(format!(
                "instance of {} does not support ordering",
                other.class()
            ))),
        }
    }

    fn validate_order(&self, order: &[Id]) -> TCResult<()> {
        match self {
            Self::Table(table) => table.validate_order(order),
            Self::Index(index) => index.validate_order(order),
            Self::IndexSlice(slice) => slice.validate_order(order),
            Self::Merge(merge) => merge.validate_order(order),
            Self::Selection(selection) => selection.validate_order(order),
            Self::TableSlice(slice) => slice.validate_order(order),
            other => Err(TCError::unsupported(format!(
                "instance of {} does not support ordering",
                other.class()
            ))),
        }
    }
}
#[async_trait]
impl<F: File<Node>, D: Dir, Txn: Transaction<D>> TableRead for Table<F, D, Txn>
where
    Self: Send + Sync,
    <D::Read as DirRead>::FileEntry: AsType<F>,
{
    async fn read(&self, txn_id: &TxnId, key: &Key) -> TCResult<Option<Vec<Value>>> {
        match self {
            Self::Table(table) => table.read(txn_id, key).await,
            other => Err(TCError::unsupported(format!(
                "{} does not support GET by key",
                other
            ))),
        }
    }
}

#[async_trait]
impl<F: File<Node>, D: Dir, Txn: Transaction<D>> TableStream for Table<F, D, Txn>
where
    Self: Send + Sync,
    <D::Read as DirRead>::FileEntry: AsType<F>,
{
    type Limit = Self;
    type Selection = Self;

    async fn count(self, txn_id: TxnId) -> TCResult<u64> {
        match self {
            Self::Table(table) => table.count(txn_id).await,
            Self::Index(index) => index.count(txn_id).await,
            Self::IndexSlice(slice) => slice.count(txn_id).await,
            Self::Limit(limit) => limit.count(txn_id).await,
            Self::Merge(merge) => merge.count(txn_id).await,
            Self::Selection(selection) => selection.count(txn_id).await,
            Self::TableSlice(slice) => slice.count(txn_id).await,
        }
    }

    fn limit(self, limit: u64) -> <Self as TableStream>::Limit {
        match self {
            Self::Table(table) => table.limit(limit).into(),
            Self::Index(index) => index.limit(limit).into(),
            Self::IndexSlice(slice) => slice.limit(limit).into(),
            Self::Limit(limited) => limited.limit(limit).into(),
            Self::Merge(merge) => merge.limit(limit).into(),
            Self::Selection(selection) => selection.limit(limit).into(),
            Self::TableSlice(slice) => slice.limit(limit).into(),
        }
    }

    fn select(self, columns: Vec<Id>) -> TCResult<<Self as TableStream>::Selection> {
        match self {
            Self::Table(table) => table.select(columns).map(Self::from),
            Self::Index(index) => index.select(columns).map(Self::from),
            Self::IndexSlice(slice) => slice.select(columns).map(Self::from),
            Self::Limit(limited) => limited.select(columns).map(Self::from),
            Self::Merge(merge) => merge.select(columns).map(Self::from),
            Self::Selection(selection) => selection.select(columns).map(Self::from),
            Self::TableSlice(slice) => slice.select(columns).map(Self::from),
        }
    }

    async fn rows<'a>(self, txn_id: TxnId) -> TCResult<TCBoxTryStream<'a, Vec<Value>>> {
        match self {
            Self::Table(table) => table.rows(txn_id).await,
            Self::Index(index) => index.rows(txn_id).await,
            Self::IndexSlice(slice) => slice.rows(txn_id).await,
            Self::Limit(limited) => limited.rows(txn_id).await,
            Self::Merge(merge) => merge.rows(txn_id).await,
            Self::Selection(selection) => selection.rows(txn_id).await,
            Self::TableSlice(slice) => slice.rows(txn_id).await,
        }
    }
}

#[async_trait]
impl<F: File<Node>, D: Dir, Txn: Transaction<D>> TableSlice for Table<F, D, Txn>
where
    Self: Send + Sync,
    <D::Read as DirRead>::FileEntry: AsType<F>,
{
    type Slice = Self;

    fn slice(self, bounds: Bounds) -> TCResult<Table<F, D, Txn>> {
        match self {
            Self::Table(table) => table.slice(bounds).map(Self::from),
            Self::Merge(merge) => merge.slice(bounds).map(Self::from),
            Self::TableSlice(slice) => slice.slice(bounds).map(Self::from),
            other => Err(TCError::unsupported(format!(
                "instance of {} does not support slicing",
                other.class()
            ))),
        }
    }

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()> {
        match self {
            Self::Table(table) => table.validate_bounds(bounds),
            Self::Merge(merge) => merge.validate_bounds(bounds),
            Self::TableSlice(slice) => slice.validate_bounds(bounds),
            other => Err(TCError::unsupported(format!(
                "instance of {} does not support slicing",
                other.class()
            ))),
        }
    }
}

#[async_trait]
impl<F: File<Node>, D: Dir, Txn: Transaction<D>> TableWrite for Table<F, D, Txn>
where
    Self: Send + Sync,
    <D::Read as DirRead>::FileEntry: AsType<F>,
{
    async fn delete(&self, txn_id: TxnId, key: Key) -> TCResult<()> {
        if let Self::Table(table) = self {
            table.delete(txn_id, key).await
        } else {
            Err(TCError::unsupported(format!(
                "instance of {} does not support delete",
                self.class()
            )))
        }
    }

    async fn update(&self, txn_id: TxnId, key: Key, values: Row) -> TCResult<()> {
        if let Self::Table(table) = self {
            table.update(txn_id, key, values).await
        } else {
            Err(TCError::unsupported(format!(
                "instance of {} does not support delete",
                self.class()
            )))
        }
    }

    async fn upsert(&self, txn_id: TxnId, key: Key, values: Values) -> TCResult<()> {
        if let Self::Table(table) = self {
            table.upsert(txn_id, key, values).await
        } else {
            Err(TCError::unsupported(format!(
                "instance of {} does not support delete",
                self.class()
            )))
        }
    }
}

#[async_trait]
impl<F: File<Node>, D: Dir, Txn: Transaction<D>> de::FromStream for Table<F, D, Txn>
where
    <D::Read as DirRead>::FileEntry: AsType<F>,
{
    type Context = Txn;

    async fn from_stream<De: de::Decoder>(txn: Txn, decoder: &mut De) -> Result<Self, De::Error> {
        decoder
            .decode_seq(TableVisitor {
                txn,
                phantom_dir: PhantomData,
                phantom_file: PhantomData,
            })
            .await
    }
}

#[async_trait]
impl<'en, F: File<Node>, D: Dir, Txn: Transaction<D>> IntoView<'en, D> for Table<F, D, Txn>
where
    <D::Read as DirRead>::FileEntry: AsType<F>,
{
    type Txn = Txn;
    type View = TableView<'en>;

    async fn into_view(self, txn: Txn) -> TCResult<TableView<'en>> {
        let schema = self.schema().clone();
        let rows = self.rows(*txn.id()).await?;
        Ok(TableView { schema, rows })
    }
}

impl<F, D, Txn> fmt::Debug for Table<F, D, Txn>
where
    Self: Send + Sync,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl<F, D, Txn> fmt::Display for Table<F, D, Txn>
where
    Self: Send + Sync,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "an instance of {}", self.class())
    }
}

struct TableVisitor<F: File<Node>, D: Dir, Txn: Transaction<D>> {
    txn: Txn,
    phantom_file: PhantomData<F>,
    phantom_dir: PhantomData<D>,
}

#[async_trait]
impl<F: File<Node>, D: Dir, Txn: Transaction<D>> de::Visitor for TableVisitor<F, D, Txn>
where
    <D::Read as DirRead>::FileEntry: AsType<F>,
{
    type Value = Table<F, D, Txn>;

    fn expecting() -> &'static str {
        "a Table"
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let txn_id = *self.txn.id();
        let schema = seq
            .next_element(())
            .await?
            .ok_or_else(|| de::Error::invalid_length(0, "a Table schema"))?;

        let table = TableIndex::create(self.txn.context(), schema, *self.txn.id())
            .map_err(de::Error::custom)
            .await?;

        if let Some(visitor) = seq
            .next_element::<RowVisitor<F, D, Txn>>((txn_id, table.clone()))
            .await?
        {
            Ok(visitor.table.into())
        } else {
            Ok(table.into())
        }
    }
}

struct RowVisitor<F: File<Node>, D: Dir, Txn: Transaction<D>> {
    table: TableIndex<F, D, Txn>,
    txn_id: TxnId,
}

#[async_trait]
impl<F: File<Node>, D: Dir, Txn: Transaction<D>> de::Visitor for RowVisitor<F, D, Txn>
where
    <D::Read as DirRead>::FileEntry: AsType<F>,
{
    type Value = Self;

    fn expecting() -> &'static str {
        "a sequence of table rows"
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let schema = self.table.primary().schema();

        while let Some(row) = seq.next_element(()).await? {
            let row = schema.row_from_values(row).map_err(de::Error::custom)?;
            let (key, values) = schema
                .key_values_from_row(row, true)
                .map_err(de::Error::custom)?;

            self.table
                .upsert(self.txn_id, key, values)
                .map_err(de::Error::custom)
                .await?;
        }

        Ok(self)
    }
}

#[async_trait]
impl<F: File<Node>, D: Dir, Txn: Transaction<D>> de::FromStream for RowVisitor<F, D, Txn>
where
    <D::Read as DirRead>::FileEntry: AsType<F>,
{
    type Context = (TxnId, TableIndex<F, D, Txn>);

    async fn from_stream<De: de::Decoder>(
        cxt: Self::Context,
        decoder: &mut De,
    ) -> Result<Self, De::Error> {
        let (txn_id, table) = cxt;
        decoder.decode_seq(Self { txn_id, table }).await
    }
}

/// A view of a [`Table`] within a single [`Transaction`], used for serialization.
pub struct TableView<'en> {
    schema: TableSchema,
    rows: TCBoxTryStream<'en, Vec<Value>>,
}

impl<'en> en::IntoStream<'en> for TableView<'en> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        (self.schema, en::SeqStream::from(self.rows)).into_stream(encoder)
    }
}
