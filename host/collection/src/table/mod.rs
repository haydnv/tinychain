//! A [`Table`], an ordered collection of [`Row`]s which supports `BTree`-based indexing

use std::fmt;

use async_trait::async_trait;

use tc_error::*;
use tc_transact::TxnId;
use tc_value::Value;
use tcgeneric::{
    path_label, Class, Id, Instance, NativeClass, PathLabel, PathSegment, TCBoxTryStream, TCPathBuf,
};

pub use schema::Schema;

mod schema;

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
    Table,
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
        match self {
            Self::Table => write!(f, "type Table"),
        }
    }
}

/// Methods common to every table view
pub trait TableInstance: Instance<Class = TableType> {
    /// Return the schema of this `Table`.
    fn schema(&self) -> Schema;
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
    async fn rows<'a>(self, txn_id: TxnId) -> TCResult<TCBoxTryStream<'a, Row>>;
}
