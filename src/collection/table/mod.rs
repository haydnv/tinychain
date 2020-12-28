use std::fmt;
use std::ops::Deref;

use async_trait::async_trait;
use futures::future::{self, TryFutureExt};
use futures::TryStreamExt;

use crate::class::*;
use crate::error;
use crate::general::{TCResult, TCTryStream, TryCastInto};
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
pub trait TableInstance: Instance<Class = TableType> + Sized + 'static {
    type OrderBy: TableInstance;
    type Reverse: TableInstance;
    type Slice: TableInstance;

    fn into_table(self) -> Table;

    async fn count(&self, txn_id: &TxnId) -> TCResult<u64> {
        let rows = self.stream(&txn_id).await?;
        rows.try_fold(0, |count, _| future::ready(Ok(count + 1)))
            .await
    }

    async fn delete(&self, _txn_id: &TxnId) -> TCResult<()> {
        Err(error::bad_request(ERR_DELETE, self.class()))
    }

    async fn delete_row(&self, _txn_id: &TxnId, _row: Row) -> TCResult<()> {
        Err(error::bad_request(ERR_DELETE, self.class()))
    }

    fn group_by(self, columns: Vec<Id>) -> TCResult<view::Aggregate<Self::OrderBy>> {
        group_by(self, columns)
    }

    async fn index(self, txn: Txn, columns: Option<Vec<Id>>) -> TCResult<index::ReadOnly> {
        index::ReadOnly::copy_from(self, txn, columns).await
    }

    async fn insert(&self, _txn_id: &TxnId, _key: Vec<Value>, _value: Vec<Value>) -> TCResult<()> {
        Err(error::bad_request(ERR_INSERT, self.class()))
    }

    fn key(&'_ self) -> &'_ [Column];

    fn values(&'_ self) -> &'_ [Column];

    fn limit(self, limit: u64) -> view::Limited {
        view::Limited::new(self, limit)
    }

    fn order_by(self, columns: Vec<Id>, reverse: bool) -> TCResult<Self::OrderBy>;

    fn reversed(self) -> TCResult<Self::Reverse>;

    fn select(self, columns: Vec<Id>) -> TCResult<view::Selection<Self>> {
        let selection = view::Selection::new(self, columns)?;
        Ok(selection)
    }

    fn slice(self, _bounds: Bounds) -> TCResult<Self::Slice> {
        Err(error::bad_request(ERR_SLICE, self.class()))
    }

    async fn stream<'a>(&'a self, txn_id: &'a TxnId) -> TCResult<TCTryStream<'a, Vec<Value>>>;

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()>;

    fn validate_order(&self, order: &[Id]) -> TCResult<()>;

    async fn update(&self, _txn: &Txn, _value: Row) -> TCResult<()> {
        Err(error::bad_request(ERR_UPDATE, self.class()))
    }

    async fn update_row(&self, _txn_id: &TxnId, _row: Row, _value: Row) -> TCResult<()> {
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
    Aggregate(Box<TableImpl<Aggregate<Table>>>),
    IndexSlice(TableImpl<IndexSlice>),
    Limit(TableImpl<Limited>),
    Merge(TableImpl<Merged>),
    Selection(Box<TableImpl<Selection<Table>>>),
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
        let mut rows = self.stream(txn.id()).await?;
        let next = rows.try_next().await?;
        Ok(next.is_none())
    }

    async fn to_stream<'a>(&'a self, txn: &'a Txn) -> TCResult<TCTryStream<'a, Scalar>> {
        let stream = self.stream(txn.id()).await?;
        Ok(Box::pin(stream.map_ok(Scalar::from)))
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
    type OrderBy = Self;
    type Reverse = Self;
    type Slice = Self;

    fn into_table(self) -> Self {
        self
    }

    async fn count(&self, txn_id: &TxnId) -> TCResult<u64> {
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

    async fn delete(&self, txn_id: &TxnId) -> TCResult<()> {
        match self {
            Self::Index(index) => index.deref().delete(txn_id).await,
            Self::ROIndex(index) => index.deref().delete(txn_id).await,
            Self::Table(table) => table.deref().delete(txn_id).await,
            Self::Aggregate(aggregate) => aggregate.deref().deref().delete(txn_id).await,
            Self::IndexSlice(index_slice) => index_slice.deref().delete(txn_id).await,
            Self::Limit(limited) => limited.deref().delete(txn_id).await,
            Self::Merge(merged) => merged.deref().delete(txn_id).await,
            Self::Selection(columns) => columns.deref().deref().delete(txn_id).await,
            Self::TableSlice(table_slice) => table_slice.deref().delete(txn_id).await,
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

    fn group_by(self, columns: Vec<Id>) -> TCResult<view::Aggregate<Table>> {
        match self {
            Self::Index(index) => index.into_table().group_by(columns),
            Self::ROIndex(index) => index.into_table().group_by(columns),
            Self::Table(table) => table.into_table().group_by(columns),
            Self::Aggregate(aggregate) => aggregate.into_table().group_by(columns),
            Self::IndexSlice(index_slice) => index_slice.into_table().group_by(columns),
            Self::Limit(limited) => limited.into_table().group_by(columns),
            Self::Merge(merged) => merged.into_table().group_by(columns),
            Self::Selection(selection) => selection.into_table().group_by(columns),
            Self::TableSlice(table_slice) => table_slice.into_table().group_by(columns),
        }
    }

    async fn index(self, txn: Txn, columns: Option<Vec<Id>>) -> TCResult<index::ReadOnly> {
        match self {
            Self::Index(index) => index.into_inner().index(txn, columns).await,
            Self::ROIndex(index) => index.into_inner().index(txn, columns).await,
            Self::Table(table) => table.into_inner().index(txn, columns).await,
            Self::Aggregate(aggregate) => aggregate.into_inner().index(txn, columns).await,
            Self::IndexSlice(index_slice) => index_slice.into_inner().index(txn, columns).await,
            Self::Limit(limited) => limited.into_inner().index(txn, columns).await,
            Self::Merge(merged) => merged.into_inner().index(txn, columns).await,
            Self::Selection(selection) => selection.into_inner().index(txn, columns).await,
            Self::TableSlice(table_slice) => table_slice.into_inner().index(txn, columns).await,
        }
    }

    async fn insert(&self, txn_id: &TxnId, key: Vec<Value>, values: Vec<Value>) -> TCResult<()> {
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

    fn limit(self, limit: u64) -> view::Limited {
        match self {
            Self::Index(index) => index.into_inner().limit(limit),
            Self::ROIndex(index) => index.into_inner().limit(limit),
            Self::Table(table) => table.into_inner().limit(limit),
            Self::Aggregate(aggregate) => aggregate.into_inner().limit(limit),
            Self::IndexSlice(index_slice) => index_slice.into_inner().limit(limit),
            Self::Limit(limited) => limited.into_inner().limit(limit),
            Self::Merge(merged) => merged.into_inner().limit(limit),
            Self::Selection(columns) => columns.into_inner().limit(limit),
            Self::TableSlice(table_slice) => table_slice.into_inner().limit(limit),
        }
    }

    fn order_by(self, order: Vec<Id>, reverse: bool) -> TCResult<Self::OrderBy> {
        match self {
            Self::Index(index) => index
                .into_inner()
                .order_by(order, reverse)
                .map(TableInstance::into_table),
            Self::ROIndex(index) => index
                .into_inner()
                .order_by(order, reverse)
                .map(TableInstance::into_table),
            Self::Table(table) => table
                .into_inner()
                .order_by(order, reverse)
                .map(TableInstance::into_table),
            Self::Aggregate(aggregate) => aggregate
                .into_inner()
                .order_by(order, reverse)
                .map(TableInstance::into_table),
            Self::IndexSlice(index_slice) => index_slice
                .into_inner()
                .order_by(order, reverse)
                .map(TableInstance::into_table),
            Self::Limit(limited) => limited
                .into_inner()
                .order_by(order, reverse)
                .map(TableInstance::into_table),
            Self::Merge(merged) => merged
                .into_inner()
                .order_by(order, reverse)
                .map(TableInstance::into_table),
            Self::Selection(columns) => columns
                .into_inner()
                .order_by(order, reverse)
                .map(TableInstance::into_table),
            Self::TableSlice(table_slice) => table_slice
                .into_inner()
                .order_by(order, reverse)
                .map(TableInstance::into_table),
        }
    }

    fn reversed(self) -> TCResult<Self::Reverse> {
        match self {
            Self::Index(index) => index.into_inner().reversed().map(TableInstance::into_table),
            Self::ROIndex(index) => index.into_inner().reversed().map(TableInstance::into_table),
            Self::Table(table) => table.into_inner().reversed().map(TableInstance::into_table),
            Self::Aggregate(aggregate) => aggregate
                .into_inner()
                .reversed()
                .map(TableInstance::into_table),
            Self::IndexSlice(index_slice) => index_slice
                .into_inner()
                .reversed()
                .map(TableInstance::into_table),
            Self::Limit(limited) => limited
                .into_inner()
                .reversed()
                .map(TableInstance::into_table),
            Self::Merge(merged) => merged
                .into_inner()
                .reversed()
                .map(TableInstance::into_table),
            Self::Selection(columns) => columns
                .into_inner()
                .reversed()
                .map(TableInstance::into_table),
            Self::TableSlice(table_slice) => table_slice
                .into_inner()
                .reversed()
                .map(TableInstance::into_table),
        }
    }

    fn select(self, columns: Vec<Id>) -> TCResult<view::Selection<Self>> {
        match self {
            Self::Index(index) => index.into_table().select(columns),
            Self::ROIndex(index) => index.into_table().select(columns),
            Self::Table(table) => table.into_table().select(columns),
            Self::Aggregate(aggregate) => aggregate.into_table().select(columns),
            Self::Limit(limited) => limited.into_table().select(columns),
            Self::IndexSlice(index_slice) => index_slice.into_table().select(columns),
            Self::Merge(merged) => merged.into_table().select(columns),
            Self::Selection(selection) => selection.into_table().select(columns),
            Self::TableSlice(table_slice) => table_slice.into_table().select(columns),
        }
    }

    fn slice(self, bounds: Bounds) -> TCResult<Table> {
        match self {
            Self::Index(index) => index
                .into_inner()
                .slice(bounds)
                .map(TableInstance::into_table),
            Self::ROIndex(index) => index
                .into_inner()
                .slice(bounds)
                .map(TableInstance::into_table),
            Self::Table(table) => table
                .into_inner()
                .slice(bounds)
                .map(TableInstance::into_table),
            Self::Aggregate(aggregate) => aggregate
                .into_inner()
                .slice(bounds)
                .map(TableInstance::into_table),
            Self::Limit(limited) => limited
                .into_inner()
                .slice(bounds)
                .map(TableInstance::into_table),
            Self::IndexSlice(index_slice) => index_slice
                .into_inner()
                .slice(bounds)
                .map(TableInstance::into_table),
            Self::Merge(merged) => merged
                .into_inner()
                .slice(bounds)
                .map(TableInstance::into_table),
            Self::Selection(columns) => columns
                .into_inner()
                .slice(bounds)
                .map(TableInstance::into_table),
            Self::TableSlice(table_slice) => table_slice
                .into_inner()
                .slice(bounds)
                .map(TableInstance::into_table),
        }
    }

    async fn stream<'a>(&'a self, txn_id: &'a TxnId) -> TCResult<TCTryStream<'a, Vec<Value>>> {
        match self {
            Self::Index(index) => index.stream(txn_id).await,
            Self::ROIndex(index) => index.stream(txn_id).await,
            Self::Table(table) => table.stream(txn_id).await,
            Self::Aggregate(aggregate) => aggregate.stream(txn_id).await,
            Self::IndexSlice(index_slice) => index_slice.stream(txn_id).await,
            Self::Limit(limited) => limited.stream(txn_id).await,
            Self::Merge(merged) => merged.stream(txn_id).await,
            Self::Selection(columns) => columns.stream(txn_id).await,
            Self::TableSlice(table_slice) => table_slice.stream(txn_id).await,
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

    async fn update(&self, txn: &Txn, value: Row) -> TCResult<()> {
        match self {
            Self::Index(index) => index.update(txn, value).await,
            Self::ROIndex(index) => index.update(txn, value).await,
            Self::Table(table) => table.update(txn, value).await,
            Self::Aggregate(aggregate) => aggregate.update(txn, value).await,
            Self::IndexSlice(index_slice) => index_slice.update(txn, value).await,
            Self::Limit(limited) => limited.update(txn, value).await,
            Self::Merge(merged) => merged.update(txn, value).await,
            Self::Selection(columns) => columns.update(txn, value).await,
            Self::TableSlice(table_slice) => table_slice.update(txn, value).await,
        }
    }

    async fn update_row(&self, txn_id: &TxnId, row: Row, value: Row) -> TCResult<()> {
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

impl From<Selection<Table>> for Table {
    fn from(selection: Selection<Table>) -> Table {
        Table::Selection(Box::new(selection.into()))
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
