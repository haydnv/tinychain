use std::convert::TryFrom;
use std::fmt;

use async_trait::async_trait;
use destream::en;
use futures::future;
use futures::TryStreamExt;

use tc_btree::{BTreeType, Node};
use tc_error::*;
use tc_transact::fs::{Dir, File};
use tc_transact::{IntoView, Transaction, TxnId};
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

    async fn count(self, txn_id: TxnId) -> TCResult<u64> {
        let rows = self.rows(txn_id).await?;
        rows.try_fold(0, |count, _| future::ready(Ok(count + 1)))
            .await
    }

    async fn delete(&self, _txn_id: TxnId) -> TCResult<()> {
        Err(TCError::bad_request(ERR_DELETE, self.class()))
    }

    async fn delete_row(&self, _txn_id: TxnId, _row: Row) -> TCResult<()> {
        Err(TCError::bad_request(ERR_DELETE, self.class()))
    }

    fn group_by(self, columns: Vec<Id>) -> TCResult<view::Aggregate<F, D, Txn, Self::OrderBy>> {
        group_by(self, columns)
    }

    async fn index(self, txn: Txn, columns: Option<Vec<Id>>) -> TCResult<index::ReadOnly<F, D, Txn>>
    where
        F: TryFrom<D::File, Error = TCError>,
        D::FileClass: From<BTreeType>,
    {
        index::ReadOnly::copy_from(self, txn, columns).await
    }

    async fn insert(&self, _txn_id: TxnId, _key: Vec<Value>, _value: Vec<Value>) -> TCResult<()> {
        Err(TCError::bad_request(ERR_INSERT, self.class()))
    }

    fn key(&self) -> &[Column];

    fn values(&self) -> &[Column];

    fn schema(&self) -> TableSchema;

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

    async fn rows<'a>(self, txn_id: TxnId) -> TCResult<TCTryStream<'a, Vec<Value>>>;

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()>;

    fn validate_order(&self, order: &[Id]) -> TCResult<()>;

    async fn update(&self, _txn: &Txn, _value: Row) -> TCResult<()>
    where
        F: TryFrom<D::File, Error = TCError>,
        D::FileClass: From<BTreeType>,
    {
        Err(TCError::bad_request(ERR_UPDATE, self.class()))
    }

    async fn update_row(&self, _txn_id: TxnId, _row: Row, _value: Row) -> TCResult<()> {
        Err(TCError::bad_request(ERR_UPDATE, self.class()))
    }

    async fn upsert(&self, _txn_id: TxnId, _key: Vec<Value>, _value: Vec<Value>) -> TCResult<()> {
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

#[derive(Clone)]
pub enum Table<F: File<Node>, D: Dir, Txn: Transaction<D>> {
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

    async fn insert(&self, txn_id: TxnId, key: Vec<Value>, values: Vec<Value>) -> TCResult<()> {
        match self {
            Self::Index(index) => index.insert(txn_id, key, values).await,
            Self::ROIndex(index) => index.insert(txn_id, key, values).await,
            Self::Table(table) => table.insert(txn_id, key, values).await,
            Self::Aggregate(aggregate) => aggregate.insert(txn_id, key, values).await,
            Self::IndexSlice(slice) => slice.insert(txn_id, key, values).await,
            Self::Limit(limit) => limit.insert(txn_id, key, values).await,
            Self::Merge(merge) => merge.insert(txn_id, key, values).await,
            Self::Selection(selection) => selection.insert(txn_id, key, values).await,
            Self::TableSlice(slice) => slice.insert(txn_id, key, values).await,
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
impl<'en, F: File<Node>, D: Dir, Txn: Transaction<D>> IntoView<'en, D> for Table<F, D, Txn> {
    type Txn = Txn;
    type View = TableView<'en>;

    async fn into_view(self, txn: Txn) -> TCResult<TableView<'en>> {
        let schema = self.schema().clone();
        let rows = self.rows(*txn.id()).await?;
        Ok(TableView { schema, rows })
    }
}

impl<F: File<Node>, D: Dir, Txn: Transaction<D>> fmt::Display for Table<F, D, Txn> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "an instance of {}", self.class())
    }
}

pub struct TableView<'en> {
    schema: TableSchema,
    rows: TCTryStream<'en, Vec<Value>>,
}

impl<'en> en::IntoStream<'en> for TableView<'en> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        (self.schema, en::SeqStream::from(self.rows)).into_stream(encoder)
    }
}
