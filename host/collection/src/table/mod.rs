//! A [`Table`], an ordered collection of [`Row`]s which supports `BTree`-based indexing

use std::fmt;

use async_trait::async_trait;
use futures::{TryFutureExt, TryStreamExt};
use safecast::{as_type, AsType};

use tc_error::*;
use tc_transact::{Transaction, TxnId};
use tc_value::{Value, ValueCollator};
use tcgeneric::{
    path_label, Class, Id, Instance, Map, NativeClass, PathLabel, PathSegment, TCPathBuf,
    ThreadSafe,
};

use super::btree::Node;

pub use b_table::{IndexSchema, Schema};

use crate::btree::BTreeSchema;
pub use file::TableFile;
pub use schema::TableSchema;
pub use stream::Rows;
pub(crate) use stream::TableView;

pub mod public;

mod file;
mod schema;
mod stream;
mod view;

/// The key of a row in a table
pub type Key = b_tree::Key<Value>;

/// The values of a row in a table
pub type Values = b_table::Row<Value>;

/// A range of an individual column
pub type ColumnRange = b_table::ColumnRange<Value>;

/// A range used to select a slice of a table
pub type Range = b_table::Range<Id, Value>;

/// A row in a table
pub type Row = b_table::Row<Value>;

const PATH: PathLabel = path_label(&["state", "collection", "table"]);

/// The [`Class`] of a [`Table`]
#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub enum TableType {
    Limit,
    Table,
    Selection,
    Slice,
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

impl fmt::Debug for TableType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            Self::Limit => "type Table limited result set",
            Self::Selection => "type Table limited column set",
            Self::Slice => "type Table slice",
            Self::Table => "type Table",
        })
    }
}

/// Methods common to every table view
pub trait TableInstance: Instance<Class = TableType> {
    /// Borrow the schema of this `Table`.
    fn schema(&self) -> &TableSchema;
}

/// Table ordering methods
pub trait TableOrder: TableInstance {
    /// The type of `Table` returned by this instance's `order_by` method.
    type OrderBy: TableInstance;

    /// The type of `Table` returned by this instance's `reversed` method.
    type Reverse: TableInstance;

    /// Return an ordered view of this table.
    fn order_by(self, columns: Vec<Id>, reverse: bool) -> TCResult<Self::OrderBy>;

    /// Reverse the order returned by `rows`.
    fn reverse(self) -> TCResult<Self::Reverse>;
}

/// A method to read a single row
#[async_trait]
pub trait TableRead: TableInstance {
    /// Read the row with the given `key` from this table, if present.
    async fn read(&self, txn_id: TxnId, key: &[Value]) -> TCResult<Option<Row>>;
}

/// Methods for slicing a table
pub trait TableSlice: TableInstance {
    /// The type of `Table` returned by this instance's `slice` method.
    type Slice: TableInstance;

    /// Limit the returned `rows` to the given [`Range`].
    fn slice(self, range: Range) -> TCResult<Self::Slice>;
}

/// Table read methods
#[async_trait]
pub trait TableStream: TableInstance + Sized {
    type Limit: TableInstance;
    type Selection: TableInstance;

    /// Return the number of rows in this table.
    async fn count(self, txn_id: TxnId) -> TCResult<u64>;

    /// Return `true` if this table contains zero rows.
    async fn is_empty(self, txn_id: TxnId) -> TCResult<bool> {
        let mut rows = self.rows(txn_id).await?;
        rows.try_next()
            .map_ok(|maybe_row| maybe_row.is_some())
            .await
    }

    /// Limit the number of rows returned by `rows`.
    fn limit(self, limit: u64) -> TCResult<Self::Limit>;

    /// Limit the columns returned by `rows`.
    fn select(self, columns: Vec<Id>) -> TCResult<Self::Selection>;

    /// Return a stream of the rows in this `Table`.
    async fn rows<'a>(self, txn_id: TxnId) -> TCResult<Rows<'a>>;
}

/// The [`Table`] update method
#[async_trait]
pub trait TableUpdate<FE>: TableInstance
where
    FE: AsType<Node> + ThreadSafe,
{
    /// Delete all rows in the given `range` from this table.
    async fn truncate(
        &self,
        txn_id: TxnId,
        range: Range,
        tmp: b_tree::BTreeLock<BTreeSchema, ValueCollator, FE>,
    ) -> TCResult<()>;

    /// Update all rows in `range` with the given `values`.
    async fn update(
        &self,
        txn_id: TxnId,
        range: Range,
        values: Map<Value>,
        tmp: b_tree::BTreeLock<BTreeSchema, ValueCollator, FE>,
    ) -> TCResult<()>;
}

/// [`Table`] write methods
#[async_trait]
pub trait TableWrite: TableInstance {
    /// Delete the given row from this table, if present.
    async fn delete(&self, txn_id: TxnId, key: Vec<Value>) -> TCResult<()>;

    /// Insert or update the given row.
    async fn upsert(&self, txn_id: TxnId, key: Vec<Value>, values: Vec<Value>) -> TCResult<()>;
}

/// A relational database table, or a view of one
pub enum Table<Txn, FE> {
    Limited(Box<view::Limited<Self>>),
    Selection(Box<view::Selection<Self>>),
    Slice(view::TableSlice<Txn, FE>),
    Table(TableFile<Txn, FE>),
}

as_type!(Table<Txn, FE>, Limited, Box<view::Limited<Self>>);
as_type!(Table<Txn, FE>, Selection, Box<view::Selection<Self>>);
as_type!(Table<Txn, FE>, Slice, view::TableSlice<Txn, FE>);
as_type!(Table<Txn, FE>, Table, TableFile<Txn, FE>);

impl<Txn, FE> Clone for Table<Txn, FE> {
    fn clone(&self) -> Self {
        match self {
            Self::Limited(limited) => Self::Limited(limited.clone()),
            Self::Selection(selection) => Self::Selection(selection.clone()),
            Self::Slice(slice) => Self::Slice(slice.clone()),
            Self::Table(table) => Self::Table(table.clone()),
        }
    }
}

impl<Txn, FE> Instance for Table<Txn, FE>
where
    Txn: Send + Sync,
    FE: Send + Sync,
{
    type Class = TableType;

    fn class(&self) -> Self::Class {
        match self {
            Self::Limited(limited) => limited.class(),
            Self::Selection(selection) => selection.class(),
            Self::Slice(slice) => slice.class(),
            Self::Table(table) => table.class(),
        }
    }
}

impl<Txn, FE> TableInstance for Table<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    fn schema(&self) -> &TableSchema {
        match self {
            Self::Limited(limited) => limited.schema(),
            Self::Selection(selection) => selection.schema(),
            Self::Slice(slice) => slice.schema(),
            Self::Table(table) => table.schema(),
        }
    }
}

impl<Txn, FE> TableOrder for Table<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    type OrderBy = Self;
    type Reverse = Self;

    fn order_by(self, columns: Vec<Id>, reverse: bool) -> TCResult<Self> {
        match self {
            Self::Selection(selection) => selection.order_by(columns, reverse).map(Self::from),
            Self::Slice(slice) => slice.order_by(columns, reverse).map(Self::from),
            Self::Table(table) => table.order_by(columns, reverse).map(Self::from),
            other => Err(bad_request!("{:?} does not support ordering", other)),
        }
    }

    fn reverse(self) -> TCResult<Self::Reverse> {
        match self {
            Self::Selection(selection) => selection.reverse().map(Self::from),
            Self::Slice(slice) => slice.reverse().map(Self::from),
            Self::Table(table) => table.reverse().map(Self::from),
            other => Err(bad_request!("{:?} does not support ordering", other)),
        }
    }
}

#[async_trait]
impl<Txn, FE> TableRead for Table<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    async fn read(&self, txn_id: TxnId, key: &[Value]) -> TCResult<Option<Row>> {
        match self {
            Self::Selection(selection) => selection.read(txn_id, key).await,
            Self::Slice(slice) => slice.read(txn_id, key).await,
            Self::Table(table) => table.read(txn_id, key).await,
            other => Err(bad_request!(
                "{:?} does not support reading an individual row",
                other.class()
            )),
        }
    }
}

impl<Txn, FE> TableSlice for Table<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    type Slice = Self;

    fn slice(self, range: Range) -> TCResult<Self> {
        match self {
            Self::Selection(selection) => selection.slice(range).map(Self::from),
            Self::Slice(slice) => slice.slice(range).map(Self::from),
            Self::Table(table) => table.slice(range).map(Self::from),
            other => Err(bad_request!("{:?} does not support slicing", other)),
        }
    }
}

#[async_trait]
impl<Txn, FE> TableStream for Table<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    type Limit = Self;
    type Selection = Self;

    async fn count(self, txn_id: TxnId) -> TCResult<u64> {
        match self {
            Self::Limited(limited) => limited.count(txn_id).await,
            Self::Selection(selection) => selection.count(txn_id).await,
            Self::Slice(slice) => slice.count(txn_id).await,
            Self::Table(table) => table.count(txn_id).await,
        }
    }

    fn limit(self, limit: u64) -> TCResult<Self> {
        match self {
            Self::Limited(limited) => limited.limit(limit).map(Self::from),
            Self::Selection(selection) => selection.limit(limit).map(Self::from),
            Self::Slice(slice) => slice.limit(limit).map(Self::from),
            Self::Table(table) => table.limit(limit).map(Self::from),
        }
    }

    fn select(self, columns: Vec<Id>) -> TCResult<Self> {
        match self {
            Self::Limited(limited) => limited.select(columns).map(Self::from),
            Self::Selection(selection) => selection.select(columns).map(Self::from),
            Self::Slice(slice) => slice.select(columns).map(Self::from),
            Self::Table(table) => table.select(columns).map(Self::from),
        }
    }

    async fn rows<'a>(self, txn_id: TxnId) -> TCResult<Rows<'a>> {
        match self {
            Self::Limited(limited) => limited.rows(txn_id).await,
            Self::Selection(selection) => selection.rows(txn_id).await,
            Self::Slice(slice) => slice.rows(txn_id).await,
            Self::Table(table) => table.rows(txn_id).await,
        }
    }
}

#[async_trait]
impl<Txn, FE> TableUpdate<FE> for Table<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    async fn truncate(
        &self,
        txn_id: TxnId,
        range: Range,
        tmp: b_tree::BTreeLock<BTreeSchema, ValueCollator, FE>,
    ) -> TCResult<()> {
        if let Self::Table(table) = self {
            table.truncate(txn_id, range, tmp).await
        } else {
            Err(bad_request!("{:?} does not support write operations", self))
        }
    }

    async fn update(
        &self,
        txn_id: TxnId,
        range: Range,
        values: Map<Value>,
        tmp: b_tree::BTreeLock<BTreeSchema, ValueCollator, FE>,
    ) -> TCResult<()> {
        if let Self::Table(table) = self {
            table.update(txn_id, range, values, tmp).await
        } else {
            Err(bad_request!("{:?} does not support write operations", self))
        }
    }
}

#[async_trait]
impl<Txn, FE> TableWrite for Table<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    async fn delete(&self, txn_id: TxnId, key: Vec<Value>) -> TCResult<()> {
        if let Self::Table(table) = self {
            table.delete(txn_id, key).await
        } else {
            Err(bad_request!("{:?} does not support write operations", self))
        }
    }

    async fn upsert(&self, txn_id: TxnId, key: Vec<Value>, values: Vec<Value>) -> TCResult<()> {
        if let Self::Table(table) = self {
            table.upsert(txn_id, key, values).await
        } else {
            Err(bad_request!("{:?} does not support write operations", self))
        }
    }
}

impl<Txn, FE> fmt::Debug for Table<Txn, FE> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Limited(limited) => limited.fmt(f),
            Self::Selection(selection) => selection.fmt(f),
            Self::Slice(slice) => slice.fmt(f),
            Self::Table(table) => table.fmt(f),
        }
    }
}
