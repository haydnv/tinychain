use std::fmt;
use std::ops::Deref;

use async_trait::async_trait;
use futures::future::{self, TryFutureExt};
use futures::{Stream, StreamExt};

use crate::class::*;
use crate::error;
use crate::general::{TCResult, TCStreamOld, TryCastInto};
use crate::handler::*;
use crate::scalar::{label, Id, Link, MethodType, PathSegment, Scalar, TCPathBuf, Value};
use crate::transaction::{Transact, Txn, TxnId};

use super::class::{CollectionClass, CollectionInstance};
use super::schema::{Column, Row, TableSchema};
use super::{Collection, CollectionType};

mod bounds;
mod handlers;
mod index;
mod view;

const ERR_DELETE: &str = "Deletion is not supported by instance of";
const ERR_INSERT: &str = "Insertion is not supported by instance of";
const ERR_SLICE: &str = "Slicing is not supported by instance of";
const ERR_UPDATE: &str = "Update is not supported by instance of";

pub use bounds::*;
pub use handlers::TableImpl;
pub use index::*;
pub use view::*;

#[derive(Clone, Eq, PartialEq)]
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

impl Class for TableType {
    type Instance = Table;
}

impl NativeClass for TableType {
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        let suffix = Self::prefix().try_suffix(path)?;

        if suffix.is_empty() {
            Ok(TableType::Table)
        } else if suffix.len() == 1 && suffix[0].as_str() == "index" {
            Ok(TableType::Index)
        } else {
            Err(error::path_not_found(path))
        }
    }

    fn prefix() -> TCPathBuf {
        CollectionType::prefix().append(label("table"))
    }
}

#[async_trait]
impl CollectionClass for TableType {
    type Instance = Table;

    async fn get(&self, txn: &Txn, schema: Value) -> TCResult<Table> {
        let schema =
            schema.try_cast_into(|v| error::bad_request("Expected TableSchema but found", v))?;

        TableIndex::create(txn, schema)
            .map_ok(TableImpl::from)
            .map_ok(Table::Table)
            .await
    }
}

impl From<TableType> for Link {
    fn from(tt: TableType) -> Link {
        let prefix = TableType::prefix();

        use TableType::*;
        match tt {
            Index => prefix.append(label("index")).into(),
            ReadOnly => prefix.append(label("ro_index")).into(),
            _ => prefix.into(),
        }
    }
}

impl From<TableType> for TCType {
    fn from(tt: TableType) -> TCType {
        TCType::Collection(CollectionType::Table(tt))
    }
}

impl fmt::Display for TableType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Index => write!(f, "Index"),
            Self::ReadOnly => write!(f, "Index (read-only)"),
            Self::Table => write!(f, "Table"),
            Self::Aggregate => write!(f, "Table or Index Aggregate"),
            Self::IndexSlice => write!(f, "Index Slice"),
            Self::Limit => write!(f, "Table or Index Limit Selection"),
            Self::Merge => write!(f, "Table Merge Selection"),
            Self::Selection => write!(f, "Table or Index Column Selection"),
            Self::TableSlice => write!(f, "Table Slice"),
        }
    }
}

#[async_trait]
pub trait TableInstance: Instance<Class = TableType> + Into<Table> + Sized + 'static {
    type Stream: Stream<Item = Vec<Value>> + Send + Unpin;

    async fn count(&self, txn_id: TxnId) -> TCResult<u64> {
        let count = self
            .clone()
            .stream(txn_id)
            .await?
            .fold(0, |count, _| future::ready(count + 1))
            .await;

        Ok(count)
    }

    async fn delete(self, _txn_id: TxnId) -> TCResult<()> {
        Err(error::bad_request(ERR_DELETE, self.class()))
    }

    async fn delete_row(&self, _txn_id: &TxnId, _row: Row) -> TCResult<()> {
        Err(error::bad_request(ERR_DELETE, self.class()))
    }

    fn group_by(&self, columns: Vec<Id>) -> TCResult<view::Aggregate> {
        view::Aggregate::new(self.clone().into(), columns)
    }

    async fn index(&self, txn: Txn, columns: Option<Vec<Id>>) -> TCResult<index::ReadOnly> {
        index::ReadOnly::copy_from(self.clone().into(), txn, columns).await
    }

    async fn insert(&self, _txn_id: TxnId, _key: Vec<Value>, _value: Vec<Value>) -> TCResult<()> {
        Err(error::bad_request(ERR_INSERT, self.class()))
    }

    fn key(&'_ self) -> &'_ [Column];

    fn values(&'_ self) -> &'_ [Column];

    fn limit(&self, limit: u64) -> view::Limited {
        view::Limited::new(self.clone().into(), limit)
    }

    fn order_by(&self, columns: Vec<Id>, reverse: bool) -> TCResult<Table>;

    fn reversed(&self) -> TCResult<Table>;

    fn select(&self, columns: Vec<Id>) -> TCResult<view::Selection> {
        let selection = view::Selection::new(self.clone().into(), columns)?;
        Ok(selection)
    }

    fn slice(&self, _bounds: Bounds) -> TCResult<Table> {
        Err(error::bad_request(ERR_SLICE, self.class()))
    }

    async fn stream(self, txn_id: TxnId) -> TCResult<Self::Stream>;

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()>;

    fn validate_order(&self, order: &[Id]) -> TCResult<()>;

    async fn update(self, _txn: Txn, _value: Row) -> TCResult<()> {
        Err(error::bad_request(ERR_UPDATE, self.class()))
    }

    async fn update_row(&self, _txn_id: TxnId, _row: Row, _value: Row) -> TCResult<()> {
        Err(error::bad_request(ERR_UPDATE, self.class()))
    }

    async fn upsert(&self, _txn_id: &TxnId, _key: Vec<Value>, _value: Vec<Value>) -> TCResult<()> {
        Err(error::bad_request(ERR_INSERT, self.class()))
    }
}

#[derive(Clone)]
pub enum Table {
    Index(TableImpl<Index>),
    ROIndex(TableImpl<ReadOnly>),
    Table(TableImpl<TableIndex>),
    Aggregate(TableImpl<Aggregate>),
    IndexSlice(TableImpl<IndexSlice>),
    Limit(TableImpl<Limited>),
    Merge(TableImpl<Merged>),
    Selection(TableImpl<Selection>),
    TableSlice(TableImpl<TableSlice>),
}

impl Table {
    pub async fn create(txn: &Txn, schema: TableSchema) -> TCResult<TableIndex> {
        index::TableIndex::create(txn, schema).await
    }
}

impl Instance for Table {
    type Class = TableType;

    fn class(&self) -> Self::Class {
        match self {
            Self::Index(_) => TableType::Index,
            Self::ROIndex(_) => TableType::ReadOnly,
            Self::Table(_) => TableType::Table,
            Self::Aggregate(aggregate) => aggregate.class(),
            Self::IndexSlice(index_slice) => index_slice.class(),
            Self::Limit(limit) => limit.class(),
            Self::Merge(merge) => merge.class(),
            Self::Selection(selection) => selection.class(),
            Self::TableSlice(table_slice) => table_slice.class(),
        }
    }
}

#[async_trait]
impl CollectionInstance for Table {
    type Item = Vec<Value>;

    async fn is_empty(&self, txn: &Txn) -> TCResult<bool> {
        let mut rows = self.clone().stream(*txn.id()).await?;
        Ok(rows.next().await.is_none())
    }

    async fn to_stream(&self, txn: Txn) -> TCResult<TCStreamOld<Scalar>> {
        let stream = self.clone().stream(*txn.id()).await?;
        Ok(Box::pin(stream.map(Scalar::from)))
    }
}

impl Route for Table {
    fn route(
        &'_ self,
        method: MethodType,
        path: &'_ [PathSegment],
    ) -> Option<Box<dyn Handler + '_>> {
        match self {
            Self::Index(index) => index.route(method, path),
            Self::ROIndex(index) => index.route(method, path),
            Self::Table(table) => table.route(method, path),
            Self::Aggregate(aggregate) => aggregate.route(method, path),
            Self::IndexSlice(index) => index.route(method, path),
            Self::Limit(limit) => limit.route(method, path),
            Self::Merge(merged) => merged.route(method, path),
            Self::Selection(selection) => selection.route(method, path),
            Self::TableSlice(slice) => slice.route(method, path),
        }
    }
}

#[async_trait]
impl TableInstance for Table {
    type Stream = TCStreamOld<Vec<Value>>;

    async fn count(&self, txn_id: TxnId) -> TCResult<u64> {
        match self {
            Self::Index(index) => index.count(txn_id).await,
            Self::ROIndex(index) => index.count(txn_id).await,
            Self::Table(table) => table.count(txn_id).await,
            Self::Aggregate(aggregate) => aggregate.count(txn_id).await,
            Self::IndexSlice(index_slice) => index_slice.count(txn_id).await,
            Self::Limit(limited) => limited.count(txn_id).await,
            Self::Merge(merged) => merged.count(txn_id).await,
            Self::Selection(columns) => columns.count(txn_id).await,
            Self::TableSlice(table_slice) => table_slice.count(txn_id).await,
        }
    }

    async fn delete(self, txn_id: TxnId) -> TCResult<()> {
        match self {
            Self::Index(index) => index.into_inner().clone().delete(txn_id).await,
            Self::ROIndex(index) => index.into_inner().clone().delete(txn_id).await,
            Self::Table(table) => table.into_inner().clone().delete(txn_id).await,
            Self::Aggregate(aggregate) => aggregate.into_inner().clone().delete(txn_id).await,
            Self::IndexSlice(index_slice) => index_slice.into_inner().clone().delete(txn_id).await,
            Self::Limit(limited) => limited.into_inner().clone().delete(txn_id).await,
            Self::Merge(merged) => merged.into_inner().clone().delete(txn_id).await,
            Self::Selection(columns) => columns.into_inner().clone().delete(txn_id).await,
            Self::TableSlice(table_slice) => table_slice.into_inner().clone().delete(txn_id).await,
        }
    }

    async fn delete_row(&self, txn_id: &TxnId, row: Row) -> TCResult<()> {
        match self {
            Self::Index(index) => index.delete_row(txn_id, row).await,
            Self::ROIndex(index) => index.delete_row(txn_id, row).await,
            Self::Table(table) => table.delete_row(txn_id, row).await,
            Self::Aggregate(aggregate) => aggregate.delete_row(txn_id, row).await,
            Self::IndexSlice(index_slice) => index_slice.delete_row(txn_id, row).await,
            Self::Limit(limited) => limited.delete_row(txn_id, row).await,
            Self::Merge(merged) => merged.delete_row(txn_id, row).await,
            Self::Selection(columns) => columns.delete_row(txn_id, row).await,
            Self::TableSlice(table_slice) => table_slice.delete_row(txn_id, row).await,
        }
    }

    fn group_by(&self, columns: Vec<Id>) -> TCResult<view::Aggregate> {
        match self {
            Self::Index(index) => index.group_by(columns),
            Self::ROIndex(index) => index.group_by(columns),
            Self::Table(table) => table.group_by(columns),
            Self::Aggregate(aggregate) => aggregate.group_by(columns),
            Self::IndexSlice(index_slice) => index_slice.group_by(columns),
            Self::Limit(limited) => limited.group_by(columns),
            Self::Merge(merged) => merged.group_by(columns),
            Self::Selection(selection) => selection.group_by(columns),
            Self::TableSlice(table_slice) => table_slice.group_by(columns),
        }
    }

    async fn index(&self, txn: Txn, columns: Option<Vec<Id>>) -> TCResult<index::ReadOnly> {
        match self {
            Self::Index(index) => index.index(txn, columns).await,
            Self::ROIndex(index) => index.index(txn, columns).await,
            Self::Table(table) => table.index(txn, columns).await,
            Self::Aggregate(aggregate) => aggregate.index(txn, columns).await,
            Self::IndexSlice(index_slice) => index_slice.index(txn, columns).await,
            Self::Limit(limited) => limited.index(txn, columns).await,
            Self::Merge(merged) => merged.index(txn, columns).await,
            Self::Selection(selection) => selection.index(txn, columns).await,
            Self::TableSlice(table_slice) => table_slice.index(txn, columns).await,
        }
    }

    async fn insert(&self, txn_id: TxnId, key: Vec<Value>, values: Vec<Value>) -> TCResult<()> {
        match self {
            Self::Index(index) => TableInstance::insert(index.deref(), txn_id, key, values).await,
            Self::ROIndex(index) => TableInstance::insert(index.deref(), txn_id, key, values).await,
            Self::Table(table) => TableInstance::insert(table.deref(), txn_id, key, values).await,
            other => Err(error::bad_request(
                "TableView does not support insert",
                other,
            )),
        }
    }

    fn key(&'_ self) -> &'_ [Column] {
        match self {
            Self::Index(index) => index.key(),
            Self::ROIndex(index) => index.key(),
            Self::Table(table) => table.key(),
            Self::Aggregate(aggregate) => aggregate.key(),
            Self::IndexSlice(index_slice) => index_slice.key(),
            Self::Limit(limited) => limited.key(),
            Self::Merge(merged) => merged.key(),
            Self::Selection(columns) => columns.key(),
            Self::TableSlice(table_slice) => table_slice.key(),
        }
    }

    fn values(&'_ self) -> &'_ [Column] {
        match self {
            Self::Index(index) => index.values(),
            Self::ROIndex(index) => index.values(),
            Self::Table(table) => table.values(),
            Self::Aggregate(aggregate) => aggregate.values(),
            Self::IndexSlice(index_slice) => index_slice.values(),
            Self::Limit(limited) => limited.values(),
            Self::Merge(merged) => merged.values(),
            Self::Selection(columns) => columns.values(),
            Self::TableSlice(table_slice) => table_slice.values(),
        }
    }

    fn limit(&self, limit: u64) -> view::Limited {
        match self {
            Self::Index(index) => index.limit(limit),
            Self::ROIndex(index) => index.limit(limit),
            Self::Table(table) => table.limit(limit),
            Self::Aggregate(aggregate) => aggregate.limit(limit),
            Self::IndexSlice(index_slice) => index_slice.limit(limit),
            Self::Limit(limited) => limited.limit(limit),
            Self::Merge(merged) => merged.limit(limit),
            Self::Selection(columns) => columns.limit(limit),
            Self::TableSlice(table_slice) => table_slice.limit(limit),
        }
    }

    fn order_by(&self, order: Vec<Id>, reverse: bool) -> TCResult<Table> {
        match self {
            Self::Index(index) => index.order_by(order, reverse),
            Self::ROIndex(index) => index.order_by(order, reverse),
            Self::Table(table) => table.order_by(order, reverse),
            Self::Aggregate(aggregate) => aggregate.order_by(order, reverse),
            Self::IndexSlice(index_slice) => index_slice.order_by(order, reverse),
            Self::Limit(limited) => limited.order_by(order, reverse),
            Self::Merge(merged) => merged.order_by(order, reverse),
            Self::Selection(columns) => columns.order_by(order, reverse),
            Self::TableSlice(table_slice) => table_slice.order_by(order, reverse),
        }
    }

    fn reversed(&self) -> TCResult<Table> {
        match self {
            Self::Index(index) => index.reversed(),
            Self::ROIndex(index) => index.reversed(),
            Self::Table(table) => table.reversed(),
            Self::Aggregate(aggregate) => aggregate.reversed(),
            Self::IndexSlice(index_slice) => index_slice.reversed(),
            Self::Limit(limited) => limited.reversed(),
            Self::Merge(merged) => merged.reversed(),
            Self::Selection(columns) => columns.reversed(),
            Self::TableSlice(table_slice) => table_slice.reversed(),
        }
    }

    fn select(&self, columns: Vec<Id>) -> TCResult<view::Selection> {
        match self {
            Self::Index(index) => index.select(columns),
            Self::ROIndex(index) => index.select(columns),
            Self::Table(table) => table.select(columns),
            Self::Aggregate(aggregate) => aggregate.select(columns),
            Self::Limit(limited) => limited.select(columns),
            Self::IndexSlice(index_slice) => index_slice.select(columns),
            Self::Merge(merged) => merged.select(columns),
            Self::Selection(selection) => selection.select(columns),
            Self::TableSlice(table_slice) => table_slice.select(columns),
        }
    }

    fn slice(&self, bounds: Bounds) -> TCResult<Table> {
        match self {
            Self::Index(index) => index.slice(bounds),
            Self::ROIndex(index) => index.slice(bounds),
            Self::Table(table) => table.slice(bounds),
            Self::Aggregate(aggregate) => aggregate.slice(bounds),
            Self::Limit(limited) => limited.slice(bounds),
            Self::IndexSlice(index_slice) => index_slice.slice(bounds),
            Self::Merge(merged) => merged.slice(bounds),
            Self::Selection(columns) => columns.slice(bounds),
            Self::TableSlice(table_slice) => table_slice.slice(bounds),
        }
    }

    async fn stream(self, txn_id: TxnId) -> TCResult<Self::Stream> {
        match self {
            Self::Index(index) => index.into_inner().stream(txn_id).await,
            Self::ROIndex(index) => index.into_inner().stream(txn_id).await,
            Self::Table(table) => table.into_inner().stream(txn_id).await,
            Self::Aggregate(aggregate) => aggregate.into_inner().stream(txn_id).await,
            Self::IndexSlice(index_slice) => index_slice.into_inner().stream(txn_id).await,
            Self::Limit(limited) => limited.into_inner().stream(txn_id).await,
            Self::Merge(merged) => merged.into_inner().stream(txn_id).await,
            Self::Selection(columns) => columns.into_inner().stream(txn_id).await,
            Self::TableSlice(table_slice) => table_slice.into_inner().stream(txn_id).await,
        }
    }

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()> {
        match self {
            Self::Index(index) => index.validate_bounds(bounds),
            Self::ROIndex(index) => index.validate_bounds(bounds),
            Self::Table(table) => table.validate_bounds(bounds),
            Self::Aggregate(aggregate) => aggregate.validate_bounds(bounds),
            Self::IndexSlice(index_slice) => index_slice.validate_bounds(bounds),
            Self::Limit(limited) => limited.validate_bounds(bounds),
            Self::Merge(merged) => merged.validate_bounds(bounds),
            Self::Selection(columns) => columns.validate_bounds(bounds),
            Self::TableSlice(table_slice) => table_slice.validate_bounds(bounds),
        }
    }

    fn validate_order(&self, order: &[Id]) -> TCResult<()> {
        match self {
            Self::Index(index) => index.validate_order(order),
            Self::ROIndex(index) => index.validate_order(order),
            Self::Table(table) => table.validate_order(order),
            Self::Aggregate(aggregate) => aggregate.validate_order(order),
            Self::IndexSlice(index_slice) => index_slice.validate_order(order),
            Self::Limit(limited) => limited.validate_order(order),
            Self::Merge(merged) => merged.validate_order(order),
            Self::Selection(columns) => columns.validate_order(order),
            Self::TableSlice(table_slice) => table_slice.validate_order(order),
        }
    }

    async fn update(self, txn: Txn, value: Row) -> TCResult<()> {
        match self {
            Self::Index(index) => index.into_inner().update(txn, value).await,
            Self::ROIndex(index) => index.into_inner().update(txn, value).await,
            Self::Table(table) => table.into_inner().update(txn, value).await,
            Self::Aggregate(aggregate) => aggregate.into_inner().update(txn, value).await,
            Self::IndexSlice(index_slice) => index_slice.into_inner().update(txn, value).await,
            Self::Limit(limited) => limited.into_inner().update(txn, value).await,
            Self::Merge(merged) => merged.into_inner().update(txn, value).await,
            Self::Selection(columns) => columns.into_inner().update(txn, value).await,
            Self::TableSlice(table_slice) => table_slice.into_inner().update(txn, value).await,
        }
    }

    async fn update_row(&self, txn_id: TxnId, row: Row, value: Row) -> TCResult<()> {
        match self {
            Self::Index(index) => index.update_row(txn_id, row, value).await,
            Self::ROIndex(index) => index.update_row(txn_id, row, value).await,
            Self::Table(table) => table.update_row(txn_id, row, value).await,
            Self::Aggregate(aggregate) => aggregate.update_row(txn_id, row, value).await,
            Self::IndexSlice(index_slice) => index_slice.update_row(txn_id, row, value).await,
            Self::Limit(limited) => limited.update_row(txn_id, row, value).await,
            Self::Merge(merged) => merged.update_row(txn_id, row, value).await,
            Self::Selection(columns) => columns.update_row(txn_id, row, value).await,
            Self::TableSlice(table_slice) => table_slice.update_row(txn_id, row, value).await,
        }
    }

    async fn upsert(&self, txn_id: &TxnId, key: Vec<Value>, values: Vec<Value>) -> TCResult<()> {
        match self {
            Self::Index(index) => TableInstance::upsert(index.deref(), txn_id, key, values).await,
            Self::ROIndex(index) => TableInstance::upsert(index.deref(), txn_id, key, values).await,
            Self::Table(table) => TableInstance::upsert(table.deref(), txn_id, key, values).await,
            other => Err(error::bad_request(
                "TableView does not support upsert",
                other,
            )),
        }
    }
}

#[async_trait]
impl Transact for Table {
    async fn commit(&self, txn_id: &TxnId) {
        match self {
            Self::Index(index) => index.commit(txn_id).await,
            Self::ROIndex(_) => (), // no-op
            Self::Table(table) => table.commit(txn_id).await,
            Self::Aggregate(_) => (), // no-op
            Self::IndexSlice(index_slice) => index_slice.commit(txn_id).await,
            Self::Limit(limited) => limited.commit(txn_id).await,
            Self::Merge(merged) => merged.commit(txn_id).await,
            Self::Selection(_) => (), // no-op
            Self::TableSlice(table_slice) => table_slice.commit(txn_id).await,
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        match self {
            Self::Index(index) => index.rollback(txn_id).await,
            Self::ROIndex(_) => (), // no-op
            Self::Table(table) => table.rollback(txn_id).await,
            Self::Aggregate(_) => (), // no-op
            Self::IndexSlice(index_slice) => index_slice.rollback(txn_id).await,
            Self::Limit(limited) => limited.rollback(txn_id).await,
            Self::Merge(merged) => merged.rollback(txn_id).await,
            Self::Selection(_) => (), // no-op
            Self::TableSlice(table_slice) => table_slice.rollback(txn_id).await,
        }
    }

    async fn finalize(&self, txn_id: &TxnId) {
        match self {
            Self::Index(index) => index.finalize(txn_id).await,
            Self::ROIndex(_) => (), // no-op
            Self::Table(table) => table.finalize(txn_id).await,
            Self::Aggregate(_) => (), // no-op
            Self::IndexSlice(index_slice) => index_slice.finalize(txn_id).await,
            Self::Limit(limited) => limited.finalize(txn_id).await,
            Self::Merge(merged) => merged.finalize(txn_id).await,
            Self::Selection(_) => (), // no-op
            Self::TableSlice(table_slice) => table_slice.finalize(txn_id).await,
        }
    }
}

impl From<Aggregate> for Table {
    fn from(aggregate: Aggregate) -> Table {
        Table::Aggregate(aggregate.into())
    }
}

impl From<Index> for Table {
    fn from(index: Index) -> Table {
        Table::Index(index.into())
    }
}

impl From<IndexSlice> for Table {
    fn from(index: IndexSlice) -> Table {
        Table::IndexSlice(index.into())
    }
}

impl From<Limited> for Table {
    fn from(limit: Limited) -> Table {
        Table::Limit(limit.into())
    }
}

impl From<Merged> for Table {
    fn from(merge: Merged) -> Table {
        Table::Merge(merge.into())
    }
}

impl From<ReadOnly> for Table {
    fn from(index: ReadOnly) -> Table {
        Table::ROIndex(index.into())
    }
}

impl From<Selection> for Table {
    fn from(selection: Selection) -> Table {
        Table::Selection(selection.into())
    }
}

impl From<TableIndex> for Table {
    fn from(table: TableIndex) -> Table {
        Table::Table(table.into())
    }
}

impl From<TableSlice> for Table {
    fn from(slice: TableSlice) -> Table {
        Table::TableSlice(slice.into())
    }
}

impl From<Table> for State {
    fn from(table: Table) -> State {
        State::Collection(Collection::Table(table))
    }
}

impl fmt::Display for Table {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Index(_) => write!(f, "Index"),
            Self::ROIndex(_) => write!(f, "Index (read-only)"),
            Self::Table(_) => write!(f, "Table"),
            Self::Aggregate(_) => write!(f, "Table or Index Aggregate"),
            Self::IndexSlice(_) => write!(f, "Index Slice"),
            Self::Limit(_) => write!(f, "Table or Index Limit Selection"),
            Self::Merge(_) => write!(f, "Table Merge Selection"),
            Self::Selection(_) => write!(f, "Table or Index Column Selection"),
            Self::TableSlice(_) => write!(f, "Table Slice"),
        }
    }
}
