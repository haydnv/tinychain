use std::fmt;

use async_trait::async_trait;
use futures::future;
use futures::TryStreamExt;

use tc_btree::Node;
use tc_error::*;
use tc_transact::fs::{Dir, File};
use tc_transact::{Transaction, TxnId};
use tc_value::Value;
use tcgeneric::{
    path_label, Class, Id, Instance, NativeClass, PathLabel, PathSegment, TCPathBuf, TCTryStream,
};

mod bounds;
mod index;
mod schema;
mod view;

use index::*;
use view::*;

pub use bounds::*;
pub use index::TableIndex;
pub use schema::*;

const PATH: PathLabel = path_label(&["state", "collection", "table"]);

const ERR_DELETE: &str = "Deletion is not supported by instance of";
const ERR_INSERT: &str = "Insertion is not supported by instance of";
const ERR_SLICE: &str = "Slicing is not supported by instance of";
const ERR_UPDATE: &str = "Update is not supported by instance of";

#[async_trait]
pub trait TableInstance<F: File<Node>, D: Dir, Txn: Transaction<D>>:
    Instance<Class = TableType> + Clone + Sized + Into<Table<F, D, Txn>>
{
    type OrderBy: TableInstance<F, D, Txn>;
    type Reverse: TableInstance<F, D, Txn>;
    type Slice: TableInstance<F, D, Txn>;

    async fn count(&self, txn_id: &TxnId) -> TCResult<u64> {
        let rows = self.stream(&txn_id).await?;
        rows.try_fold(0, |count, _| future::ready(Ok(count + 1)))
            .await
    }

    async fn delete(&self, _txn_id: &TxnId) -> TCResult<()> {
        Err(TCError::bad_request(ERR_DELETE, self.class()))
    }

    async fn delete_row(&self, _txn_id: &TxnId, _row: Row) -> TCResult<()> {
        Err(TCError::bad_request(ERR_DELETE, self.class()))
    }

    fn group_by(self, columns: Vec<Id>) -> TCResult<view::Aggregate<F, D, Txn, Self::OrderBy>> {
        group_by(self, columns)
    }

    async fn index(self, txn: Txn, columns: Option<Vec<Id>>) -> TCResult<index::ReadOnly> {
        index::ReadOnly::copy_from(self, txn, columns).await
    }

    async fn insert(&self, _txn_id: &TxnId, _key: Vec<Value>, _value: Vec<Value>) -> TCResult<()> {
        Err(TCError::bad_request(ERR_INSERT, self.class()))
    }

    fn key(&'_ self) -> &'_ [Column];

    fn values(&'_ self) -> &'_ [Column];

    fn limit(self, limit: u64) -> view::Limited<F, D, Txn> {
        view::Limited::new(self.into(), limit)
    }

    fn order_by(self, columns: Vec<Id>, reverse: bool) -> TCResult<Self::OrderBy>;

    fn reversed(self) -> TCResult<Self::Reverse>;

    fn select(self, columns: Vec<Id>) -> TCResult<view::Selection<F, D, Txn, Self>> {
        let selection = view::Selection::new(self, columns)?;
        Ok(selection)
    }

    fn slice(self, _bounds: Bounds) -> TCResult<Self::Slice> {
        Err(TCError::bad_request(ERR_SLICE, self.class()))
    }

    async fn stream<'a>(&'a self, txn_id: &'a TxnId) -> TCResult<TCTryStream<'a, Vec<Value>>>;

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()>;

    fn validate_order(&self, order: &[Id]) -> TCResult<()>;

    async fn update(&self, _txn: &Txn, _value: Row) -> TCResult<()> {
        Err(TCError::bad_request(ERR_UPDATE, self.class()))
    }

    async fn update_row(&self, _txn_id: &TxnId, _row: Row, _value: Row) -> TCResult<()> {
        Err(TCError::bad_request(ERR_UPDATE, self.class()))
    }

    async fn upsert(&self, _txn_id: &TxnId, _key: Vec<Value>, _value: Vec<Value>) -> TCResult<()> {
        Err(TCError::bad_request(ERR_INSERT, self.class()))
    }
}

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

#[derive(Clone)]
pub enum Table<F: File<Node>, D: Dir, Txn: Transaction<D>> {
    Index(Index),
    ROIndex(ReadOnly),
    Table(TableIndex<F, D, Txn>),
    Aggregate(Box<Aggregate<F, D, Txn, Table<F, D, Txn>>>),
    IndexSlice(IndexSlice),
    Limit(Box<Limited<F, D, Txn>>),
    Merge(Merged),
    Selection(Box<Selection<F, D, Txn, Table<F, D, Txn>>>),
    TableSlice(TableSlice),
}

impl<F: File<Node>, D: Dir, Txn: Transaction<D>> Instance for Table<F, D, Txn> {
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
impl<F: File<Node>, D: Dir, Txn: Transaction<D>> TableInstance<F, D, Txn> for Table<F, D, Txn> {
    type OrderBy = Self;
    type Reverse = Self;
    type Slice = Self;

    async fn count(&self, _txn_id: &TxnId) -> TCResult<u64> {
        todo!()
    }

    async fn delete(&self, _txn_id: &TxnId) -> TCResult<()> {
        todo!()
    }

    async fn delete_row(&self, _txn_id: &TxnId, _row: Row) -> TCResult<()> {
        todo!()
    }

    async fn index(self, _txn: Txn, _columns: Option<Vec<Id>>) -> TCResult<index::ReadOnly> {
        todo!()
    }

    async fn insert(&self, _txn_id: &TxnId, _key: Vec<Value>, _values: Vec<Value>) -> TCResult<()> {
        todo!()
    }

    fn key(&'_ self) -> &'_ [Column] {
        todo!()
    }

    fn values(&'_ self) -> &'_ [Column] {
        todo!()
    }

    fn limit(self, _limit: u64) -> view::Limited<F, D, Txn> {
        todo!()
    }

    fn order_by(self, _order: Vec<Id>, _reverse: bool) -> TCResult<Self::OrderBy> {
        todo!()
    }

    fn reversed(self) -> TCResult<Self::Reverse> {
        todo!()
    }

    fn slice(self, _bounds: Bounds) -> TCResult<Table<F, D, Txn>> {
        todo!()
    }

    async fn stream<'a>(&'a self, _txn_id: &'a TxnId) -> TCResult<TCTryStream<'a, Vec<Value>>> {
        todo!()
    }

    fn validate_bounds(&self, _bounds: &Bounds) -> TCResult<()> {
        todo!()
    }

    fn validate_order(&self, _order: &[Id]) -> TCResult<()> {
        todo!()
    }

    async fn update(&self, _txn: &Txn, _value: Row) -> TCResult<()> {
        todo!()
    }

    async fn update_row(&self, _txn_id: &TxnId, _row: Row, _value: Row) -> TCResult<()> {
        todo!()
    }

    async fn upsert(&self, _txn_id: &TxnId, _key: Vec<Value>, _values: Vec<Value>) -> TCResult<()> {
        todo!()
    }
}

impl<F: File<Node>, D: Dir, Txn: Transaction<D>> fmt::Display for Table<F, D, Txn> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "an instance of {}", self.class())
    }
}
