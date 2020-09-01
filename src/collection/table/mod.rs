use std::convert::{TryFrom, TryInto};
use std::sync::Arc;

use async_trait::async_trait;
use futures::future;
use futures::{Stream, StreamExt};

use crate::class::{TCBoxTryFuture, TCResult, TCStream};
use crate::error;
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::{Value, ValueId};

use super::schema::{Column, Row, TableSchema};

mod bounds;
mod index;
mod view;

const ERR_DELETE: &str =
    "This table view does not support deletion (try deleting a slice of the source table)";
const ERR_SLICE: &str =
    "This table view does not support slicing (consider slicing the source table directly)";
const ERR_UPDATE: &str =
    "This table view does not support updates (consider updating a slice of the source table)";

pub type ColumnBound = bounds::ColumnBound;
pub type TableIndex = index::TableIndex;

pub trait Selection: Clone + Into<Table> + Sized + Send + Sync + 'static {
    type Stream: Stream<Item = Vec<Value>> + Send + Sync + Unpin;

    fn count(&self, txn_id: TxnId) -> TCBoxTryFuture<u64> {
        Box::pin(async move {
            let count = self
                .clone()
                .stream(txn_id)
                .await?
                .fold(0, |count, _| future::ready(count + 1))
                .await;

            Ok(count)
        })
    }

    fn delete<'a>(self, _txn_id: TxnId) -> TCBoxTryFuture<'a, ()> {
        Box::pin(future::ready(Err(error::unsupported(ERR_DELETE))))
    }

    fn delete_row<'a>(&'a self, _txn_id: &'a TxnId, _row: Row) -> TCBoxTryFuture<'a, ()> {
        Box::pin(future::ready(Err(error::unsupported(ERR_DELETE))))
    }

    fn group_by(&self, columns: Vec<ValueId>) -> TCResult<view::Aggregate> {
        view::Aggregate::new(self.clone().into(), columns)
    }

    fn index<'a>(
        &'a self,
        txn: Arc<Txn>,
        columns: Option<Vec<ValueId>>,
    ) -> TCBoxTryFuture<'a, index::ReadOnly> {
        Box::pin(index::ReadOnly::copy_from(
            self.clone().into(),
            txn,
            columns,
        ))
    }

    fn key(&'_ self) -> &'_ [Column];

    fn values(&'_ self) -> &'_ [Column];

    fn limit(&self, limit: u64) -> TCResult<Arc<view::Limited>> {
        let limited = view::Limited::try_from((self.clone().into(), limit))?;
        Ok(Arc::new(limited))
    }

    fn order_by(&self, columns: Vec<ValueId>, reverse: bool) -> TCResult<Table>;

    fn reversed(&self) -> TCResult<Table>;

    fn select(&self, columns: Vec<ValueId>) -> TCResult<view::ColumnSelection> {
        let selection = (self.clone().into(), columns).try_into()?;
        Ok(selection)
    }

    fn slice(&self, _bounds: bounds::Bounds) -> TCResult<Table> {
        Err(error::unsupported(ERR_SLICE))
    }

    fn stream<'a>(self, txn_id: TxnId) -> TCBoxTryFuture<'a, Self::Stream>;

    fn validate_bounds(&self, bounds: &bounds::Bounds) -> TCResult<()>;

    fn validate_order(&self, order: &[ValueId]) -> TCResult<()>;

    fn update<'a>(self, _txn: Arc<Txn>, _value: Row) -> TCBoxTryFuture<'a, ()> {
        Box::pin(future::ready(Err(error::unsupported(ERR_UPDATE))))
    }

    fn update_row(&self, _txn_id: TxnId, _row: Row, _value: Row) -> TCBoxTryFuture<()> {
        Box::pin(future::ready(Err(error::unsupported(ERR_UPDATE))))
    }
}

#[derive(Clone)]
pub enum Table {
    Aggregate(view::Aggregate),
    Columns(view::ColumnSelection),
    Limit(view::Limited),
    Index(index::Index),
    IndexSlice(view::IndexSlice),
    Merge(view::Merged),
    ROIndex(index::ReadOnly),
    Table(index::TableIndex),
    TableSlice(view::TableSlice),
}

impl Table {
    pub async fn create(txn: Arc<Txn>, schema: TableSchema) -> TCResult<TableIndex> {
        index::TableIndex::create(txn, schema).await
    }
}

impl Selection for Table {
    type Stream = TCStream<Vec<Value>>;

    fn count(&self, txn_id: TxnId) -> TCBoxTryFuture<u64> {
        match self {
            Self::Aggregate(aggregate) => aggregate.count(txn_id),
            Self::Columns(columns) => columns.count(txn_id),
            Self::Limit(limited) => limited.count(txn_id),
            Self::Index(index) => index.count(txn_id),
            Self::IndexSlice(index_slice) => index_slice.count(txn_id),
            Self::Merge(merged) => merged.count(txn_id),
            Self::ROIndex(ro_index) => ro_index.count(txn_id),
            Self::Table(table) => table.count(txn_id),
            Self::TableSlice(table_slice) => table_slice.count(txn_id),
        }
    }

    fn delete<'a>(self, txn_id: TxnId) -> TCBoxTryFuture<'a, ()> {
        match self {
            Self::Aggregate(aggregate) => aggregate.delete(txn_id),
            Self::Columns(columns) => columns.delete(txn_id),
            Self::Limit(limited) => limited.delete(txn_id),
            Self::Index(index) => index.delete(txn_id),
            Self::IndexSlice(index_slice) => index_slice.delete(txn_id),
            Self::Merge(merged) => merged.delete(txn_id),
            Self::ROIndex(ro_index) => ro_index.delete(txn_id),
            Self::Table(table) => table.delete(txn_id),
            Self::TableSlice(table_slice) => table_slice.delete(txn_id),
        }
    }

    fn delete_row<'a>(&'a self, txn_id: &'a TxnId, row: Row) -> TCBoxTryFuture<'a, ()> {
        match self {
            Self::Aggregate(aggregate) => aggregate.delete_row(txn_id, row),
            Self::Columns(columns) => columns.delete_row(txn_id, row),
            Self::Limit(limited) => limited.delete_row(txn_id, row),
            Self::Index(index) => index.delete_row(txn_id, row),
            Self::IndexSlice(index_slice) => index_slice.delete_row(txn_id, row),
            Self::Merge(merged) => merged.delete_row(txn_id, row),
            Self::ROIndex(ro_index) => ro_index.delete_row(txn_id, row),
            Self::Table(table) => table.delete_row(txn_id, row),
            Self::TableSlice(table_slice) => table_slice.delete_row(txn_id, row),
        }
    }

    fn order_by(&self, order: Vec<ValueId>, reverse: bool) -> TCResult<Table> {
        match self {
            Self::Aggregate(aggregate) => aggregate.order_by(order, reverse),
            Self::Columns(columns) => columns.order_by(order, reverse),
            Self::Limit(limited) => limited.order_by(order, reverse),
            Self::Index(index) => index.order_by(order, reverse),
            Self::IndexSlice(index_slice) => index_slice.order_by(order, reverse),
            Self::Merge(merged) => merged.order_by(order, reverse),
            Self::ROIndex(ro_index) => ro_index.order_by(order, reverse),
            Self::Table(table) => table.order_by(order, reverse),
            Self::TableSlice(table_slice) => table_slice.order_by(order, reverse),
        }
    }

    fn reversed(&self) -> TCResult<Table> {
        match self {
            Self::Aggregate(aggregate) => aggregate.reversed(),
            Self::Columns(columns) => columns.reversed(),
            Self::Limit(limited) => limited.reversed(),
            Self::Index(index) => index.reversed(),
            Self::IndexSlice(index_slice) => index_slice.reversed(),
            Self::Merge(merged) => merged.reversed(),
            Self::ROIndex(ro_index) => ro_index.reversed(),
            Self::Table(table) => table.reversed(),
            Self::TableSlice(table_slice) => table_slice.reversed(),
        }
    }

    fn key(&'_ self) -> &'_ [Column] {
        match self {
            Self::Aggregate(aggregate) => aggregate.key(),
            Self::Columns(columns) => columns.key(),
            Self::Limit(limited) => limited.key(),
            Self::Index(index) => index.key(),
            Self::IndexSlice(index_slice) => index_slice.key(),
            Self::Merge(merged) => merged.key(),
            Self::ROIndex(ro_index) => ro_index.key(),
            Self::Table(table) => table.key(),
            Self::TableSlice(table_slice) => table_slice.key(),
        }
    }

    fn values(&'_ self) -> &'_ [Column] {
        match self {
            Self::Aggregate(aggregate) => aggregate.values(),
            Self::Columns(columns) => columns.values(),
            Self::Limit(limited) => limited.values(),
            Self::Index(index) => index.values(),
            Self::IndexSlice(index_slice) => index_slice.values(),
            Self::Merge(merged) => merged.values(),
            Self::ROIndex(ro_index) => ro_index.values(),
            Self::Table(table) => table.values(),
            Self::TableSlice(table_slice) => table_slice.values(),
        }
    }

    fn slice(&self, bounds: bounds::Bounds) -> TCResult<Table> {
        match self {
            Self::Aggregate(aggregate) => aggregate.slice(bounds),
            Self::Columns(columns) => columns.slice(bounds),
            Self::Limit(limited) => limited.slice(bounds),
            Self::Index(index) => index.slice(bounds),
            Self::IndexSlice(index_slice) => index_slice.slice(bounds),
            Self::Merge(merged) => merged.slice(bounds),
            Self::ROIndex(ro_index) => ro_index.slice(bounds),
            Self::Table(table) => table.slice(bounds),
            Self::TableSlice(table_slice) => table_slice.slice(bounds),
        }
    }

    fn stream<'a>(self, txn_id: TxnId) -> TCBoxTryFuture<'a, Self::Stream> {
        match self {
            Self::Aggregate(aggregate) => aggregate.stream(txn_id),
            Self::Columns(columns) => columns.stream(txn_id),
            Self::Limit(limited) => limited.stream(txn_id),
            Self::Index(index) => index.stream(txn_id),
            Self::IndexSlice(index_slice) => index_slice.stream(txn_id),
            Self::Merge(merged) => merged.stream(txn_id),
            Self::ROIndex(ro_index) => ro_index.stream(txn_id),
            Self::Table(table) => table.stream(txn_id),
            Self::TableSlice(table_slice) => table_slice.stream(txn_id),
        }
    }

    fn update<'a>(self, txn: Arc<Txn>, value: Row) -> TCBoxTryFuture<'a, ()> {
        match self {
            Self::Aggregate(aggregate) => aggregate.update(txn, value),
            Self::Columns(columns) => columns.update(txn, value),
            Self::Limit(limited) => limited.update(txn, value),
            Self::Index(index) => index.update(txn, value),
            Self::IndexSlice(index_slice) => index_slice.update(txn, value),
            Self::Merge(merged) => merged.update(txn, value),
            Self::ROIndex(ro_index) => ro_index.update(txn, value),
            Self::Table(table) => table.update(txn, value),
            Self::TableSlice(table_slice) => table_slice.update(txn, value),
        }
    }

    fn update_row(&self, txn_id: TxnId, row: Row, value: Row) -> TCBoxTryFuture<()> {
        match self {
            Self::Aggregate(aggregate) => aggregate.update_row(txn_id, row, value),
            Self::Columns(columns) => columns.update_row(txn_id, row, value),
            Self::Limit(limited) => limited.update_row(txn_id, row, value),
            Self::Index(index) => index.update_row(txn_id, row, value),
            Self::IndexSlice(index_slice) => index_slice.update_row(txn_id, row, value),
            Self::Merge(merged) => merged.update_row(txn_id, row, value),
            Self::ROIndex(ro_index) => ro_index.update_row(txn_id, row, value),
            Self::Table(table) => table.update_row(txn_id, row, value),
            Self::TableSlice(table_slice) => table_slice.update_row(txn_id, row, value),
        }
    }

    fn validate_bounds(&self, bounds: &bounds::Bounds) -> TCResult<()> {
        match self {
            Self::Aggregate(aggregate) => aggregate.validate_bounds(bounds),
            Self::Columns(columns) => columns.validate_bounds(bounds),
            Self::Limit(limited) => limited.validate_bounds(bounds),
            Self::Index(index) => index.validate_bounds(bounds),
            Self::IndexSlice(index_slice) => index_slice.validate_bounds(bounds),
            Self::Merge(merged) => merged.validate_bounds(bounds),
            Self::ROIndex(ro_index) => ro_index.validate_bounds(bounds),
            Self::Table(table) => table.validate_bounds(bounds),
            Self::TableSlice(table_slice) => table_slice.validate_bounds(bounds),
        }
    }

    fn validate_order(&self, order: &[ValueId]) -> TCResult<()> {
        match self {
            Self::Aggregate(aggregate) => aggregate.validate_order(order),
            Self::Columns(columns) => columns.validate_order(order),
            Self::Limit(limited) => limited.validate_order(order),
            Self::Index(index) => index.validate_order(order),
            Self::IndexSlice(index_slice) => index_slice.validate_order(order),
            Self::Merge(merged) => merged.validate_order(order),
            Self::ROIndex(ro_index) => ro_index.validate_order(order),
            Self::Table(table) => table.validate_order(order),
            Self::TableSlice(table_slice) => table_slice.validate_order(order),
        }
    }
}

#[async_trait]
impl Transact for Table {
    async fn commit(&self, txn_id: &TxnId) {
        let no_op = ();

        match self {
            Self::Aggregate(_) => no_op,
            Self::Columns(_) => no_op,
            Self::Limit(limited) => limited.commit(txn_id).await,
            Self::Index(index) => index.commit(txn_id).await,
            Self::IndexSlice(index_slice) => index_slice.commit(txn_id).await,
            Self::Merge(merged) => merged.commit(txn_id).await,
            Self::ROIndex(_) => no_op,
            Self::Table(table) => table.commit(txn_id).await,
            Self::TableSlice(table_slice) => table_slice.commit(txn_id).await,
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        let no_op = ();

        match self {
            Self::Aggregate(_) => no_op,
            Self::Columns(_) => no_op,
            Self::Limit(limited) => limited.rollback(txn_id).await,
            Self::Index(index) => index.rollback(txn_id).await,
            Self::IndexSlice(index_slice) => index_slice.rollback(txn_id).await,
            Self::Merge(merged) => merged.rollback(txn_id).await,
            Self::ROIndex(_) => no_op,
            Self::Table(table) => table.rollback(txn_id).await,
            Self::TableSlice(table_slice) => table_slice.rollback(txn_id).await,
        }
    }
}

impl From<view::Aggregate> for Table {
    fn from(aggregate: view::Aggregate) -> Table {
        Table::Aggregate(aggregate)
    }
}

impl From<view::ColumnSelection> for Table {
    fn from(columns: view::ColumnSelection) -> Table {
        Table::Columns(columns)
    }
}

impl From<view::Limited> for Table {
    fn from(limited: view::Limited) -> Table {
        Table::Limit(limited)
    }
}

impl From<index::Index> for Table {
    fn from(index: index::Index) -> Table {
        Table::Index(index)
    }
}

impl From<view::IndexSlice> for Table {
    fn from(index_slice: view::IndexSlice) -> Table {
        Table::IndexSlice(index_slice)
    }
}

impl From<view::Merged> for Table {
    fn from(merged: view::Merged) -> Table {
        Table::Merge(merged)
    }
}

impl From<index::TableIndex> for Table {
    fn from(table: index::TableIndex) -> Table {
        Table::Table(table)
    }
}

impl From<index::ReadOnly> for Table {
    fn from(ro_index: index::ReadOnly) -> Table {
        Table::ROIndex(ro_index)
    }
}

impl From<view::TableSlice> for Table {
    fn from(table_slice: view::TableSlice) -> Table {
        Table::TableSlice(table_slice)
    }
}
