//! A [`Table`], an ordered collection of [`Row`]s which supports `BTree`-based indexing

use std::fmt;

use async_trait::async_trait;
use safecast::{as_type, AsType};

use tc_error::*;
use tc_transact::{Transaction, TxnId};
use tc_value::Value;
use tcgeneric::{
    path_label, Class, Id, Instance, NativeClass, PathLabel, PathSegment, TCPathBuf, ThreadSafe,
};

use super::Node;

pub use file::TableFile;
pub use schema::Schema;
pub use stream::Rows;
pub(crate) use stream::TableView;

mod file;
mod schema;
mod stream;
mod view;

/// A range of a table
pub type Range = b_table::Range<Id, Value>;

/// The key of a row in a table
pub type Key = Vec<Value>;

/// The values of a row in a table
pub type Values = Vec<Value>;

/// A row in a table
pub type Row = Vec<Value>;

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
    fn schema(&self) -> &Schema;
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

    /// Return an error if this table does not support ordering by the given columns.
    fn validate_order(&self, order: &[Id]) -> TCResult<()>;
}

/// A method to read a single row
#[async_trait]
pub trait TableRead: TableInstance {
    /// Read the row with the given `key` from this table, if present.
    async fn read(&self, txn_id: &TxnId, key: &Key) -> TCResult<Option<Vec<Value>>>;
}

/// Methods for slicing a table
pub trait TableSlice: TableStream {
    /// The type of `Table` returned by this instance's `slice` method.
    type Slice: TableInstance;

    /// Limit the returned `rows` to the given [`Range`].
    fn slice(self, range: Range) -> TCResult<Self::Slice>;

    /// Return an error if this table does not support the given [`Range`].
    fn validate_range(&self, range: &Range) -> TCResult<()>;
}

/// Table read methods
#[async_trait]
pub trait TableStream: TableInstance + Sized {
    type Limit: TableInstance;
    type Selection: TableInstance;

    /// Return the number of rows in this table.
    async fn count(self, txn_id: TxnId) -> TCResult<u64>;

    /// Limit the number of rows returned by `rows`.
    fn limit(self, limit: u64) -> Self::Limit;

    /// Limit the columns returned by `rows`.
    fn select(self, columns: Vec<Id>) -> TCResult<Self::Selection>;

    /// Return a stream of the rows in this `Table`.
    async fn rows<'a>(self, txn_id: TxnId) -> TCResult<Rows<'a>>;
}

/// [`Table`] write methods
#[async_trait]
pub trait TableWrite: TableInstance {
    /// Delete the given row from this table, if present.
    async fn delete(&self, txn_id: TxnId, key: Key) -> TCResult<()>;

    /// Update one row of this table.
    async fn update(&self, txn_id: TxnId, key: Key, values: Row) -> TCResult<()>;

    /// Insert or update the given row.
    async fn upsert(&self, txn_id: TxnId, key: Key, values: Values) -> TCResult<()>;
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
    fn schema(&self) -> &Schema {
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

    fn validate_order(&self, order: &[Id]) -> TCResult<()> {
        match self {
            Self::Selection(selection) => selection.validate_order(order),
            Self::Slice(slice) => slice.validate_order(order),
            Self::Table(table) => table.validate_order(order),
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
    async fn read(&self, txn_id: &TxnId, key: &Key) -> TCResult<Option<Vec<Value>>> {
        match self {
            Self::Slice(slice) => slice.read(txn_id, key).await,
            Self::Table(table) => table.read(txn_id, key).await,
            other => Err(bad_request!(
                "{:?} does not support reading a single row",
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

    fn validate_range(&self, range: &Range) -> TCResult<()> {
        match self {
            Self::Selection(selection) => selection.validate_range(range),
            Self::Slice(slice) => slice.validate_range(range),
            Self::Table(table) => table.validate_range(range),
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

    fn limit(self, limit: u64) -> Self {
        match self {
            Self::Limited(limited) => limited.limit(limit).into(),
            Self::Selection(selection) => selection.limit(limit).into(),
            Self::Slice(slice) => slice.limit(limit).into(),
            Self::Table(table) => table.limit(limit).into(),
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

impl<Txn, FE> fmt::Debug for Table<Txn, FE>
where
    Txn: Send + Sync,
    FE: Send + Sync,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "an instance of {:?}", self.class())
    }
}
