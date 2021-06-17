//! A [`Table`], an ordered collection of [`Row`]s which supports `BTree`-based indexing

use std::convert::TryFrom;
use std::fmt;
use std::marker::PhantomData;

use async_trait::async_trait;
use destream::{de, en};
use futures::future::{self, TryFutureExt};
use futures::stream::TryStreamExt;

use tc_btree::{BTreeType, Node};
use tc_error::*;
use tc_transact::fs::{Dir, File, Hash};
use tc_transact::{IntoView, Transaction, TxnId};
use tc_value::Value;
use tcgeneric::{
    path_label, Class, Id, Instance, NativeClass, PathLabel, PathSegment, TCPathBuf, TCTryStream,
};

use index::*;
use view::*;

pub use bounds::*;
pub use index::TableIndex;
pub use schema::*;
pub use view::Merged;

mod bounds;
mod index;
mod schema;
mod view;

const PATH: PathLabel = path_label(&["state", "collection", "table"]);

const ERR_DELETE: &str = "Deletion is not supported by instance of";
const ERR_INSERT: &str = "Insertion is not supported by instance of";
const ERR_SLICE: &str = "Slicing is not supported by instance of";
const ERR_UPDATE: &str = "Update is not supported by instance of";

/// Common [`Table`] methods.
#[async_trait]
pub trait TableInstance<F: File<Node>, D: Dir, Txn: Transaction<D>>:
    Instance<Class = TableType> + Clone + Sized + Into<Table<F, D, Txn>>
{
    /// The type of `Table` returned by this instance's `order_by` method.
    type OrderBy: TableInstance<F, D, Txn>;

    /// The type of `Table` returned by this instance's `reversed` method.
    type Reverse: TableInstance<F, D, Txn>;

    /// The type of `Table` returned by this instance's `slice` method.
    type Slice: TableInstance<F, D, Txn>;

    /// Return the number of rows in this `Table`.
    async fn count(self, txn_id: TxnId) -> TCResult<u64> {
        let rows = self.rows(txn_id).await?;
        rows.try_fold(0, |count, _| future::ready(Ok(count + 1)))
            .await
    }

    /// Delete all rows in this `Table`.
    async fn delete(&self, _txn_id: TxnId) -> TCResult<()> {
        Err(TCError::bad_request(ERR_DELETE, self.class()))
    }

    /// Delete the given [`Row`] from this table, if present.
    async fn delete_row(&self, _txn_id: TxnId, _row: Row) -> TCResult<()> {
        Err(TCError::bad_request(ERR_DELETE, self.class()))
    }

    /// Group this `Table` by the given columns.
    fn group_by(self, columns: Vec<Id>) -> TCResult<view::Aggregate<F, D, Txn, Self::OrderBy>> {
        group_by(self, columns)
    }

    /// Construct and return a temporary index of the given columns.
    async fn index(self, txn: Txn, columns: Option<Vec<Id>>) -> TCResult<index::ReadOnly<F, D, Txn>>
    where
        F: TryFrom<D::File, Error = TCError>,
        D::FileClass: From<BTreeType>,
    {
        index::ReadOnly::copy_from(self, txn, columns).await
    }

    /// Return the schema of this `Table`'s key.
    fn key(&self) -> &[Column];

    /// Return the schema of this `Table`'s values.
    fn values(&self) -> &[Column];

    /// Return the schema of this `Table`.
    fn schema(&self) -> TableSchema;

    /// Limit the number of rows returned by `rows`.
    fn limit(self, limit: u64) -> view::Limited<F, D, Txn> {
        view::Limited::new(self.into(), limit)
    }

    /// Set the order returned by `rows`.
    fn order_by(self, columns: Vec<Id>, reverse: bool) -> TCResult<Self::OrderBy>;

    /// Reverse the order returned by `rows`.
    fn reversed(self) -> TCResult<Self::Reverse>;

    /// Limit the columns returned by `rows`.
    fn select(self, columns: Vec<Id>) -> TCResult<view::Selection<F, D, Txn, Self>> {
        let selection = view::Selection::new(self, columns)?;
        Ok(selection)
    }

    /// Limit the returned `rows` to the given [`Bounds`].
    fn slice(self, _bounds: Bounds) -> TCResult<Self::Slice> {
        Err(TCError::bad_request(ERR_SLICE, self.class()))
    }

    /// Return a stream of the rows in this `Table`.
    async fn rows<'a>(self, txn_id: TxnId) -> TCResult<TCTryStream<'a, Vec<Value>>>;

    /// Return an error if this table does not support the given [`Bounds`].
    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()>;

    /// Return an error if this table does not support ordering by the given columns.
    fn validate_order(&self, order: &[Id]) -> TCResult<()>;

    /// Update the values of the columns in this `Table` to match the given [`Row`].
    async fn update(&self, _txn: &Txn, _value: Row) -> TCResult<()>
    where
        F: TryFrom<D::File, Error = TCError>,
        D::FileClass: From<BTreeType>,
    {
        Err(TCError::bad_request(ERR_UPDATE, self.class()))
    }

    /// Update one row of this `Table`.
    async fn update_row(&self, _txn_id: TxnId, _row: Row, _value: Row) -> TCResult<()> {
        Err(TCError::bad_request(ERR_UPDATE, self.class()))
    }

    /// Insert or update the given row.
    async fn upsert(&self, _txn_id: TxnId, _key: Vec<Value>, _value: Vec<Value>) -> TCResult<()> {
        Err(TCError::bad_request(ERR_INSERT, self.class()))
    }
}

/// The [`Class`] of a [`Table`].
#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub enum TableType {
    Index,
    ReadOnly,
    Table,
    Aggregate,
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
            Self::Index => write!(f, "type Index"),
            Self::ReadOnly => write!(f, "type Index (read-only)"),
            Self::Table => write!(f, "type Table"),
            Self::Aggregate => write!(f, "type Aggregate"),
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
    Index(Index<F, D, Txn>),
    ROIndex(ReadOnly<F, D, Txn>),
    Table(TableIndex<F, D, Txn>),
    Aggregate(Box<Aggregate<F, D, Txn, Table<F, D, Txn>>>),
    IndexSlice(IndexSlice<F, D, Txn>),
    Limit(Box<Limited<F, D, Txn>>),
    Merge(Merged<F, D, Txn>),
    Selection(Box<Selection<F, D, Txn, Table<F, D, Txn>>>),
    TableSlice(TableSlice<F, D, Txn>),
}

impl<F, D, Txn> Instance for Table<F, D, Txn>
where
    Self: Send + Sync,
{
    type Class = TableType;

    fn class(&self) -> Self::Class {
        match self {
            Self::Index(_) => TableType::Index,
            Self::ROIndex(_) => TableType::ReadOnly,
            Self::Table(_) => TableType::Table,
            Self::Aggregate(_) => TableType::Aggregate,
            Self::IndexSlice(_) => TableType::IndexSlice,
            Self::Limit(_) => TableType::Limit,
            Self::Merge(_) => TableType::Merge,
            Self::Selection(_) => TableType::Selection,
            Self::TableSlice(_) => TableType::TableSlice,
        }
    }
}

#[async_trait]
impl<F: File<Node>, D: Dir, Txn: Transaction<D>> TableInstance<F, D, Txn> for Table<F, D, Txn>
where
    Self: Send + Sync,
{
    type OrderBy = Self;
    type Reverse = Self;
    type Slice = Self;

    async fn count(self, txn_id: TxnId) -> TCResult<u64> {
        match self {
            Self::Index(index) => index.count(txn_id).await,
            Self::ROIndex(index) => index.count(txn_id).await,
            Self::Table(table) => table.count(txn_id).await,
            Self::Aggregate(aggregate) => aggregate.count(txn_id).await,
            Self::IndexSlice(slice) => slice.count(txn_id).await,
            Self::Limit(limit) => limit.count(txn_id).await,
            Self::Merge(merge) => merge.count(txn_id).await,
            Self::Selection(selection) => selection.count(txn_id).await,
            Self::TableSlice(slice) => slice.count(txn_id).await,
        }
    }

    async fn delete(&self, txn_id: TxnId) -> TCResult<()> {
        match self {
            Self::Index(index) => index.delete(txn_id).await,
            Self::ROIndex(index) => index.delete(txn_id).await,
            Self::Table(table) => table.delete(txn_id).await,
            Self::Aggregate(aggregate) => aggregate.delete(txn_id).await,
            Self::IndexSlice(slice) => slice.delete(txn_id).await,
            Self::Limit(limit) => limit.delete(txn_id).await,
            Self::Merge(merge) => merge.delete(txn_id).await,
            Self::Selection(selection) => selection.delete(txn_id).await,
            Self::TableSlice(slice) => slice.delete(txn_id).await,
        }
    }

    async fn delete_row(&self, txn_id: TxnId, row: Row) -> TCResult<()> {
        match self {
            Self::Index(index) => index.delete_row(txn_id, row).await,
            Self::ROIndex(index) => index.delete_row(txn_id, row).await,
            Self::Table(table) => table.delete_row(txn_id, row).await,
            Self::Aggregate(aggregate) => aggregate.delete_row(txn_id, row).await,
            Self::IndexSlice(slice) => slice.delete_row(txn_id, row).await,
            Self::Limit(limit) => limit.delete_row(txn_id, row).await,
            Self::Merge(merge) => merge.delete_row(txn_id, row).await,
            Self::Selection(selection) => selection.delete_row(txn_id, row).await,
            Self::TableSlice(slice) => slice.delete_row(txn_id, row).await,
        }
    }

    async fn index(self, txn: Txn, columns: Option<Vec<Id>>) -> TCResult<index::ReadOnly<F, D, Txn>>
    where
        F: TryFrom<D::File, Error = TCError>,
        D::FileClass: From<BTreeType>,
    {
        match self {
            Self::Index(index) => index.index(txn, columns).await,
            Self::ROIndex(index) => index.index(txn, columns).await,
            Self::Table(table) => table.index(txn, columns).await,
            Self::Aggregate(aggregate) => aggregate.index(txn, columns).await,
            Self::IndexSlice(slice) => slice.index(txn, columns).await,
            Self::Limit(limit) => limit.index(txn, columns).await,
            Self::Merge(merge) => merge.index(txn, columns).await,
            Self::Selection(selection) => selection.index(txn, columns).await,
            Self::TableSlice(slice) => slice.index(txn, columns).await,
        }
    }

    fn key(&self) -> &[Column] {
        match self {
            Self::Index(index) => index.key(),
            Self::ROIndex(index) => index.key(),
            Self::Table(table) => table.key(),
            Self::Aggregate(aggregate) => aggregate.key(),
            Self::IndexSlice(slice) => slice.key(),
            Self::Limit(limit) => limit.key(),
            Self::Merge(merge) => merge.key(),
            Self::Selection(selection) => selection.key(),
            Self::TableSlice(slice) => slice.key(),
        }
    }

    fn values(&self) -> &[Column] {
        match self {
            Self::Index(index) => index.values(),
            Self::ROIndex(index) => index.values(),
            Self::Table(table) => table.values(),
            Self::Aggregate(aggregate) => aggregate.values(),
            Self::IndexSlice(slice) => slice.values(),
            Self::Limit(limit) => limit.values(),
            Self::Merge(merge) => merge.values(),
            Self::Selection(selection) => selection.values(),
            Self::TableSlice(slice) => slice.values(),
        }
    }

    fn schema(&self) -> TableSchema {
        match self {
            Self::Index(index) => TableInstance::schema(index),
            Self::ROIndex(index) => index.schema(),
            Self::Table(table) => table.schema(),
            Self::Aggregate(aggregate) => aggregate.schema(),
            Self::IndexSlice(slice) => TableInstance::schema(slice),
            Self::Limit(limit) => limit.schema(),
            Self::Merge(merge) => merge.schema(),
            Self::Selection(selection) => selection.schema(),
            Self::TableSlice(slice) => slice.schema(),
        }
    }

    fn limit(self, limit: u64) -> view::Limited<F, D, Txn> {
        match self {
            Self::Index(index) => index.limit(limit),
            Self::ROIndex(index) => index.limit(limit),
            Self::Table(table) => table.limit(limit),
            Self::Aggregate(aggregate) => aggregate.limit(limit),
            Self::IndexSlice(slice) => slice.limit(limit),
            Self::Limit(limited) => limited.limit(limit),
            Self::Merge(merge) => merge.limit(limit),
            Self::Selection(selection) => selection.limit(limit),
            Self::TableSlice(slice) => slice.limit(limit),
        }
    }

    fn order_by(self, order: Vec<Id>, reverse: bool) -> TCResult<Self::OrderBy> {
        match self {
            Self::Index(index) => index.order_by(order, reverse).map(Self::from),
            Self::ROIndex(index) => index.order_by(order, reverse).map(Self::from),
            Self::Table(table) => table.order_by(order, reverse).map(Self::from),
            Self::Aggregate(aggregate) => aggregate.order_by(order, reverse).map(Self::from),
            Self::IndexSlice(slice) => slice.order_by(order, reverse).map(Self::from),
            Self::Limit(limited) => limited.order_by(order, reverse).map(Self::from),
            Self::Merge(merge) => merge.order_by(order, reverse).map(Self::from),
            Self::Selection(selection) => selection.order_by(order, reverse).map(Self::from),
            Self::TableSlice(slice) => slice.order_by(order, reverse).map(Self::from),
        }
    }

    fn reversed(self) -> TCResult<Self::Reverse> {
        match self {
            Self::Index(index) => index.reversed().map(Self::from),
            Self::ROIndex(index) => index.reversed().map(Self::from),
            Self::Table(table) => table.reversed().map(Self::from),
            Self::Aggregate(aggregate) => aggregate.reversed().map(Self::from),
            Self::IndexSlice(slice) => slice.reversed().map(Self::from),
            Self::Limit(limited) => limited.reversed().map(Self::from),
            Self::Merge(merge) => merge.reversed().map(Self::from),
            Self::Selection(selection) => selection.reversed().map(Self::from),
            Self::TableSlice(slice) => slice.reversed().map(Self::from),
        }
    }

    fn slice(self, bounds: Bounds) -> TCResult<Table<F, D, Txn>> {
        match self {
            Self::Index(index) => index.slice(bounds).map(Self::from),
            Self::ROIndex(index) => index.slice(bounds).map(Self::from),
            Self::Table(table) => table.slice(bounds).map(Self::from),
            Self::Aggregate(aggregate) => aggregate.slice(bounds).map(Self::from),
            Self::IndexSlice(slice) => slice.slice(bounds).map(Self::from),
            Self::Limit(limited) => limited.slice(bounds).map(Self::from),
            Self::Merge(merge) => merge.slice(bounds).map(Self::from),
            Self::Selection(selection) => selection.slice(bounds).map(Self::from),
            Self::TableSlice(slice) => slice.slice(bounds).map(Self::from),
        }
    }

    async fn rows<'a>(self, txn_id: TxnId) -> TCResult<TCTryStream<'a, Vec<Value>>> {
        match self {
            Self::Index(index) => index.rows(txn_id).await,
            Self::ROIndex(index) => index.rows(txn_id).await,
            Self::Table(table) => table.rows(txn_id).await,
            Self::Aggregate(aggregate) => aggregate.rows(txn_id).await,
            Self::IndexSlice(slice) => slice.rows(txn_id).await,
            Self::Limit(limited) => limited.rows(txn_id).await,
            Self::Merge(merge) => merge.rows(txn_id).await,
            Self::Selection(selection) => selection.rows(txn_id).await,
            Self::TableSlice(slice) => slice.rows(txn_id).await,
        }
    }

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()> {
        match self {
            Self::Index(index) => index.validate_bounds(bounds),
            Self::ROIndex(index) => index.validate_bounds(bounds),
            Self::Table(table) => table.validate_bounds(bounds),
            Self::Aggregate(aggregate) => aggregate.validate_bounds(bounds),
            Self::IndexSlice(slice) => slice.validate_bounds(bounds),
            Self::Limit(limited) => limited.validate_bounds(bounds),
            Self::Merge(merge) => merge.validate_bounds(bounds),
            Self::Selection(selection) => selection.validate_bounds(bounds),
            Self::TableSlice(slice) => slice.validate_bounds(bounds),
        }
    }

    fn validate_order(&self, order: &[Id]) -> TCResult<()> {
        match self {
            Self::Index(index) => index.validate_order(order),
            Self::ROIndex(index) => index.validate_order(order),
            Self::Table(table) => table.validate_order(order),
            Self::Aggregate(aggregate) => aggregate.validate_order(order),
            Self::IndexSlice(slice) => slice.validate_order(order),
            Self::Limit(limited) => limited.validate_order(order),
            Self::Merge(merge) => merge.validate_order(order),
            Self::Selection(selection) => selection.validate_order(order),
            Self::TableSlice(slice) => slice.validate_order(order),
        }
    }

    async fn update(&self, txn: &Txn, value: Row) -> TCResult<()>
    where
        F: TryFrom<D::File, Error = TCError>,
        D::FileClass: From<BTreeType>,
    {
        match self {
            Self::Index(index) => index.update(txn, value).await,
            Self::ROIndex(index) => index.update(txn, value).await,
            Self::Table(table) => table.update(txn, value).await,
            Self::Aggregate(aggregate) => aggregate.update(txn, value).await,
            Self::IndexSlice(slice) => slice.update(txn, value).await,
            Self::Limit(limited) => limited.update(txn, value).await,
            Self::Merge(merge) => merge.update(txn, value).await,
            Self::Selection(selection) => selection.update(txn, value).await,
            Self::TableSlice(slice) => slice.update(txn, value).await,
        }
    }

    async fn update_row(&self, txn_id: TxnId, row: Row, value: Row) -> TCResult<()> {
        match self {
            Self::Index(index) => index.update_row(txn_id, row, value).await,
            Self::ROIndex(index) => index.update_row(txn_id, row, value).await,
            Self::Table(table) => table.update_row(txn_id, row, value).await,
            Self::Aggregate(aggregate) => aggregate.update_row(txn_id, row, value).await,
            Self::IndexSlice(slice) => slice.update_row(txn_id, row, value).await,
            Self::Limit(limited) => limited.update_row(txn_id, row, value).await,
            Self::Merge(merge) => merge.update_row(txn_id, row, value).await,
            Self::Selection(selection) => selection.update_row(txn_id, row, value).await,
            Self::TableSlice(slice) => slice.update_row(txn_id, row, value).await,
        }
    }

    async fn upsert(&self, txn_id: TxnId, key: Vec<Value>, values: Vec<Value>) -> TCResult<()> {
        match self {
            Self::Index(index) => index.upsert(txn_id, key, values).await,
            Self::ROIndex(index) => index.upsert(txn_id, key, values).await,
            Self::Table(table) => table.upsert(txn_id, key, values).await,
            Self::Aggregate(aggregate) => aggregate.upsert(txn_id, key, values).await,
            Self::IndexSlice(slice) => slice.upsert(txn_id, key, values).await,
            Self::Limit(limited) => limited.upsert(txn_id, key, values).await,
            Self::Merge(merge) => merge.upsert(txn_id, key, values).await,
            Self::Selection(selection) => selection.upsert(txn_id, key, values).await,
            Self::TableSlice(slice) => slice.upsert(txn_id, key, values).await,
        }
    }
}

#[async_trait]
impl<'en, F: File<Node>, D: Dir, Txn: Transaction<D>> Hash<'en, D> for Table<F, D, Txn> {
    type Item = Vec<Value>;
    type Txn = Txn;

    async fn hashable(&'en self, txn: &'en Txn) -> TCResult<TCTryStream<'en, Self::Item>> {
        self.clone().rows(*txn.id()).await
    }
}

#[async_trait]
impl<F: File<Node>, D: Dir, Txn: Transaction<D>> de::FromStream for Table<F, D, Txn>
where
    F: TryFrom<D::File, Error = TCError>,
    D::FileClass: From<BTreeType>,
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
impl<'en, F: File<Node>, D: Dir, Txn: Transaction<D>> IntoView<'en, D> for Table<F, D, Txn> {
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
    F: TryFrom<D::File, Error = TCError>,
    D::FileClass: From<BTreeType>,
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
        let table = TableIndex::create(schema, self.txn.context(), *self.txn.id())
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
impl<F: File<Node>, D: Dir, Txn: Transaction<D>> de::Visitor for RowVisitor<F, D, Txn> {
    type Value = Self;

    fn expecting() -> &'static str {
        "a sequence of table rows"
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let schema = self.table.primary().schema();

        while let Some(row) = seq.next_element(()).await? {
            let row = schema.row_from_values(row).map_err(de::Error::custom)?;
            let (key, values) = schema.key_values_from_row(row).map_err(de::Error::custom)?;
            self.table
                .upsert(self.txn_id, key, values)
                .map_err(de::Error::custom)
                .await?;
        }

        Ok(self)
    }
}

#[async_trait]
impl<F: File<Node>, D: Dir, Txn: Transaction<D>> de::FromStream for RowVisitor<F, D, Txn> {
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
    rows: TCTryStream<'en, Vec<Value>>,
}

impl<'en> en::IntoStream<'en> for TableView<'en> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        (self.schema, en::SeqStream::from(self.rows)).into_stream(encoder)
    }
}
