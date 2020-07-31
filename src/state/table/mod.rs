use std::convert::{TryFrom, TryInto};
use std::sync::Arc;

use async_trait::async_trait;
use futures::future;
use futures::{Stream, StreamExt};

use crate::error;
use crate::transaction::{Txn, TxnId};
use crate::value::{TCBoxTryFuture, TCResult, TCStream, Value, ValueId};

mod index;
pub mod schema;
mod view;

pub type TableBase = index::TableBase;

#[async_trait]
pub trait Selection: Clone + Into<Table> + Sized + Send + Sync + 'static {
    type Stream: Stream<Item = Vec<Value>> + Send + Sync + Unpin;

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
        Err(error::unsupported(
            "This table view does not support deletion (try deleting a slice of the source table)",
        ))
    }

    fn delete_row<'a>(&'a self, _txn_id: &'a TxnId, _row: schema::Row) -> TCBoxTryFuture<'a, ()> {
        let err_msg = "This table view does not support row deletion (try deleting from the source table directly)";
        Box::pin(future::ready(Err(error::unsupported(err_msg))))
    }

    async fn group_by(&self, txn_id: TxnId, columns: Vec<ValueId>) -> TCResult<view::Aggregate> {
        view::Aggregate::new(self.clone().into(), txn_id, columns).await
    }

    async fn index(
        &self,
        txn: Arc<Txn>,
        columns: Option<Vec<ValueId>>,
    ) -> TCResult<index::ReadOnly> {
        index::ReadOnly::copy_from(self.clone().into(), txn, columns).await
    }

    fn limit(&self, limit: u64) -> TCResult<Arc<view::Limited>> {
        let limited = view::Limited::try_from((self.clone().into(), limit))?;
        Ok(Arc::new(limited))
    }

    fn order_by<'a>(
        &'a self,
        txn_id: &'a TxnId,
        columns: Vec<ValueId>,
        reverse: bool,
    ) -> TCBoxTryFuture<'a, Table>;

    fn reversed(&self) -> TCResult<Table>;

    fn select(&self, columns: Vec<ValueId>) -> TCResult<view::ColumnSelection> {
        let selection = (self.clone().into(), columns).try_into()?;
        Ok(selection)
    }

    fn schema(&'_ self) -> &'_ schema::Schema;

    fn slice<'a>(
        &'a self,
        _txn_id: &'a TxnId,
        _bounds: schema::Bounds,
    ) -> TCBoxTryFuture<'a, Table> {
        Box::pin(async move {
            Err(error::unsupported(
                "This table view does not support slicing (consider slicing the source table directly)",
            ))
        })
    }

    fn stream<'a>(self, txn_id: TxnId) -> TCBoxTryFuture<'a, Self::Stream>;

    fn validate_bounds<'a>(
        &'a self,
        txn_id: &'a TxnId,
        bounds: &'a schema::Bounds,
    ) -> TCBoxTryFuture<'a, ()>;

    fn validate_order<'a>(&'a self, txn_id: &'a TxnId, order: &'a [ValueId]) -> TCBoxTryFuture<'a, ()>;

    async fn update(self, _txn: Arc<Txn>, _value: schema::Row) -> TCResult<()> {
        Err(error::unsupported(
            "This table view does not support updates (consider updating a slice of the source table)",
        ))
    }

    async fn update_row(
        &self,
        _txn_id: TxnId,
        _row: schema::Row,
        _value: schema::Row,
    ) -> TCResult<()> {
        Err(error::unsupported("This table view does not support updates (consider updating a row in the source table directly)"))
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
    Table(index::TableBase),
    TableSlice(view::TableSlice),
}

#[async_trait]
impl Selection for Table {
    type Stream = TCStream<Vec<Value>>;

    async fn count(&self, txn_id: TxnId) -> TCResult<u64> {
        match self {
            Self::Aggregate(aggregate) => aggregate.count(txn_id).await,
            Self::Columns(columns) => columns.count(txn_id).await,
            Self::Limit(limited) => limited.count(txn_id).await,
            Self::Index(index) => index.count(txn_id).await,
            Self::IndexSlice(index_slice) => index_slice.count(txn_id).await,
            Self::Merge(merged) => merged.count(txn_id).await,
            Self::ROIndex(ro_index) => ro_index.count(txn_id).await,
            Self::Table(table) => table.count(txn_id).await,
            Self::TableSlice(table_slice) => table_slice.count(txn_id).await,
        }
    }

    async fn delete(self, txn_id: TxnId) -> TCResult<()> {
        match self {
            Self::Aggregate(aggregate) => aggregate.clone().delete(txn_id).await,
            Self::Columns(columns) => columns.clone().delete(txn_id).await,
            Self::Limit(limited) => limited.clone().delete(txn_id).await,
            Self::Index(index) => index.clone().delete(txn_id).await,
            Self::IndexSlice(index_slice) => index_slice.clone().delete(txn_id).await,
            Self::Merge(merged) => merged.clone().delete(txn_id).await,
            Self::ROIndex(ro_index) => ro_index.clone().delete(txn_id).await,
            Self::Table(table) => table.clone().delete(txn_id).await,
            Self::TableSlice(table_slice) => table_slice.clone().delete(txn_id).await,
        }
    }

    fn delete_row<'a>(&'a self, txn_id: &'a TxnId, row: schema::Row) -> TCBoxTryFuture<'a, ()> {
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

    fn order_by<'a>(
        &'a self,
        txn_id: &'a TxnId,
        order: Vec<ValueId>,
        reverse: bool,
    ) -> TCBoxTryFuture<'a, Table> {
        match self {
            Self::Aggregate(aggregate) => aggregate.order_by(txn_id, order, reverse),
            Self::Columns(columns) => columns.order_by(txn_id, order, reverse),
            Self::Limit(limited) => limited.order_by(txn_id, order, reverse),
            Self::Index(index) => index.order_by(txn_id, order, reverse),
            Self::IndexSlice(index_slice) => index_slice.order_by(txn_id, order, reverse),
            Self::Merge(merged) => merged.order_by(txn_id, order, reverse),
            Self::ROIndex(ro_index) => ro_index.order_by(txn_id, order, reverse),
            Self::Table(table) => table.order_by(txn_id, order, reverse),
            Self::TableSlice(table_slice) => table_slice.order_by(txn_id, order, reverse),
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

    fn schema(&'_ self) -> &'_ schema::Schema {
        match self {
            Self::Aggregate(aggregate) => aggregate.schema(),
            Self::Columns(columns) => columns.schema(),
            Self::Limit(limited) => limited.schema(),
            Self::Index(index) => index.schema(),
            Self::IndexSlice(index_slice) => index_slice.schema(),
            Self::Merge(merged) => merged.schema(),
            Self::ROIndex(ro_index) => ro_index.schema(),
            Self::Table(table) => table.schema(),
            Self::TableSlice(table_slice) => table_slice.schema(),
        }
    }

    fn slice<'a>(&'a self, txn_id: &'a TxnId, bounds: schema::Bounds) -> TCBoxTryFuture<'a, Table> {
        match self {
            Self::Aggregate(aggregate) => aggregate.slice(txn_id, bounds),
            Self::Columns(columns) => columns.slice(txn_id, bounds),
            Self::Limit(limited) => limited.slice(txn_id, bounds),
            Self::Index(index) => index.slice(txn_id, bounds),
            Self::IndexSlice(index_slice) => index_slice.slice(txn_id, bounds),
            Self::Merge(merged) => merged.slice(txn_id, bounds),
            Self::ROIndex(ro_index) => ro_index.slice(txn_id, bounds),
            Self::Table(table) => table.slice(txn_id, bounds),
            Self::TableSlice(table_slice) => table_slice.slice(txn_id, bounds),
        }
    }

    fn stream<'a>(self, txn_id: TxnId) -> TCBoxTryFuture<'a, Self::Stream> {
        match self {
            Self::Aggregate(aggregate) => aggregate.clone().stream(txn_id),
            Self::Columns(columns) => columns.clone().stream(txn_id),
            Self::Limit(limited) => limited.clone().stream(txn_id),
            Self::Index(index) => index.clone().stream(txn_id),
            Self::IndexSlice(index_slice) => index_slice.clone().stream(txn_id),
            Self::Merge(merged) => merged.clone().stream(txn_id),
            Self::ROIndex(ro_index) => ro_index.clone().stream(txn_id),
            Self::Table(table) => table.clone().stream(txn_id),
            Self::TableSlice(table_slice) => table_slice.clone().stream(txn_id),
        }
    }

    async fn update(self, txn: Arc<Txn>, value: schema::Row) -> TCResult<()> {
        match self {
            Self::Aggregate(aggregate) => aggregate.clone().update(txn, value).await,
            Self::Columns(columns) => columns.clone().update(txn, value).await,
            Self::Limit(limited) => limited.clone().update(txn, value).await,
            Self::Index(index) => index.clone().update(txn, value).await,
            Self::IndexSlice(index_slice) => index_slice.clone().update(txn, value).await,
            Self::Merge(merged) => merged.clone().update(txn, value).await,
            Self::ROIndex(ro_index) => ro_index.update(txn, value).await,
            Self::Table(table) => table.clone().update(txn, value).await,
            Self::TableSlice(table_slice) => table_slice.update(txn, value).await,
        }
    }

    async fn update_row(
        &self,
        txn_id: TxnId,
        row: schema::Row,
        value: schema::Row,
    ) -> TCResult<()> {
        match self {
            Self::Aggregate(aggregate) => aggregate.update_row(txn_id, row, value).await,
            Self::Columns(columns) => columns.update_row(txn_id, row, value).await,
            Self::Limit(limited) => limited.update_row(txn_id, row, value).await,
            Self::Index(index) => index.update_row(txn_id, row, value).await,
            Self::IndexSlice(index_slice) => index_slice.update_row(txn_id, row, value).await,
            Self::Merge(merged) => merged.update_row(txn_id, row, value).await,
            Self::ROIndex(ro_index) => ro_index.update_row(txn_id, row, value).await,
            Self::Table(table) => table.update_row(txn_id, row, value).await,
            Self::TableSlice(table_slice) => table_slice.update_row(txn_id, row, value).await,
        }
    }

    fn validate_bounds<'a>(
        &'a self,
        txn_id: &'a TxnId,
        bounds: &'a schema::Bounds,
    ) -> TCBoxTryFuture<'a, ()> {
        match self {
            Self::Aggregate(aggregate) => aggregate.validate_bounds(txn_id, bounds),
            Self::Columns(columns) => columns.validate_bounds(txn_id, bounds),
            Self::Limit(limited) => limited.validate_bounds(txn_id, bounds),
            Self::Index(index) => index.validate_bounds(txn_id, bounds),
            Self::IndexSlice(index_slice) => index_slice.validate_bounds(txn_id, bounds),
            Self::Merge(merged) => merged.validate_bounds(txn_id, bounds),
            Self::ROIndex(ro_index) => ro_index.validate_bounds(txn_id, bounds),
            Self::Table(table) => table.validate_bounds(txn_id, bounds),
            Self::TableSlice(table_slice) => table_slice.validate_bounds(txn_id, bounds),
        }
    }

    fn validate_order<'a>(&'a self, txn_id: &'a TxnId, order: &'a [ValueId]) -> TCBoxTryFuture<'a, ()> {
        match self {
            Self::Aggregate(aggregate) => aggregate.validate_order(txn_id, order),
            Self::Columns(columns) => columns.validate_order(txn_id, order),
            Self::Limit(limited) => limited.validate_order(txn_id, order),
            Self::Index(index) => index.validate_order(txn_id, order),
            Self::IndexSlice(index_slice) => index_slice.validate_order(txn_id, order),
            Self::Merge(merged) => merged.validate_order(txn_id, order),
            Self::ROIndex(ro_index) => ro_index.validate_order(txn_id, order),
            Self::Table(table) => table.validate_order(txn_id, order),
            Self::TableSlice(table_slice) => table_slice.validate_order(txn_id, order),
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

impl From<index::TableBase> for Table {
    fn from(table: index::TableBase) -> Table {
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
