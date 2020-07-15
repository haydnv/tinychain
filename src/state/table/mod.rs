use std::convert::{TryFrom, TryInto};
use std::sync::Arc;

use async_trait::async_trait;
use futures::future;
use futures::{Stream, StreamExt};

use crate::error;
use crate::transaction::{Txn, TxnId};
use crate::value::{TCResult, TCStream, Value, ValueId};

mod base;
mod index;
mod view;

pub type Bounds = base::Bounds;
pub type Column = base::Column;
pub type Row = base::Row;
pub type Schema = base::Schema;

#[async_trait]
pub trait Selection: Clone + Into<Table> + Sized + Send + Sync + 'static {
    type Stream: Stream<Item = Vec<Value>> + Send + Sync + Unpin;

    async fn count(&self, txn_id: TxnId) -> TCResult<u64> {
        let count = self
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

    async fn delete_row(&self, _txn_id: &TxnId, _row: Row) -> TCResult<()> {
        Err(error::unsupported("This table view does not support row deletion (try deleting from the source table directly)"))
    }

    async fn index(
        &self,
        txn: Arc<Txn>,
        columns: Option<Vec<ValueId>>,
    ) -> TCResult<Arc<index::ReadOnly>> {
        index::ReadOnly::copy_from(self.clone().into(), txn, columns)
            .await
            .map(Arc::new)
    }

    fn limit(&self, limit: u64) -> TCResult<Arc<view::Limited>> {
        let limited = view::Limited::try_from((self.clone().into(), limit))?;
        Ok(Arc::new(limited))
    }

    fn order_by(&self, _columns: Vec<ValueId>, _reverse: bool) -> TCResult<Arc<Table>> {
        Err(error::not_implemented())
    }

    fn select(&self, columns: Vec<ValueId>) -> TCResult<Arc<view::ColumnSelection>> {
        let selection = (self.clone().into(), columns).try_into()?;
        Ok(Arc::new(selection))
    }

    fn schema(&'_ self) -> &'_ Schema;

    async fn slice(&self, _txn_id: TxnId, _bounds: Bounds) -> TCResult<Table> {
        Err(error::unsupported(
            "This table view does not support slicing (consider slicing the source table directly)",
        ))
    }

    async fn stream(&self, txn_id: TxnId) -> TCResult<Self::Stream>;

    async fn validate(&self, txn_id: &TxnId, bounds: &Bounds) -> TCResult<()>;

    async fn update(self, _txn: Arc<Txn>, _value: Row) -> TCResult<()> {
        Err(error::unsupported(
            "This table view does not support updates (consider updating a slice of the source table)",
        ))
    }

    async fn update_row(&self, _txn_id: TxnId, _row: Row, _value: Row) -> TCResult<()> {
        Err(error::unsupported("This table view does not support updates (consider updating a row in the source table directly)"))
    }
}

#[derive(Clone)]
pub enum Table {
    Columns(view::ColumnSelection),
    Limit(view::Limited),
    Table(index::IndexTable),
    Index(index::Index),
    Merge(view::Merged),
    ROIndex(index::ReadOnly),
}

#[async_trait]
impl Selection for Table {
    type Stream = TCStream<Vec<Value>>;

    async fn count(&self, txn_id: TxnId) -> TCResult<u64> {
        match self {
            Self::Columns(columns) => columns.count(txn_id).await,
            Self::Limit(limited) => limited.count(txn_id).await,
            Self::Table(table) => table.count(txn_id).await,
            Self::Index(index) => index.count(txn_id).await,
            Self::Merge(merged) => merged.count(txn_id).await,
            Self::ROIndex(ro_index) => ro_index.count(txn_id).await,
        }
    }

    async fn delete(self, txn_id: TxnId) -> TCResult<()> {
        match self {
            Self::Columns(columns) => columns.clone().delete(txn_id).await,
            Self::Limit(limited) => limited.clone().delete(txn_id).await,
            Self::Table(table) => table.clone().delete(txn_id).await,
            Self::Index(index) => index.clone().delete(txn_id).await,
            Self::Merge(merged) => merged.clone().delete(txn_id).await,
            Self::ROIndex(ro_index) => ro_index.clone().delete(txn_id).await,
        }
    }

    async fn delete_row(&self, txn_id: &TxnId, row: Row) -> TCResult<()> {
        match self {
            Self::Columns(columns) => columns.delete_row(txn_id, row).await,
            Self::Limit(limited) => limited.delete_row(txn_id, row).await,
            Self::Table(table) => table.delete_row(txn_id, row).await,
            Self::Index(index) => index.delete_row(txn_id, row).await,
            Self::Merge(merged) => merged.delete_row(txn_id, row).await,
            Self::ROIndex(ro_index) => ro_index.delete_row(txn_id, row).await,
        }
    }

    fn schema(&'_ self) -> &'_ Schema {
        match self {
            Self::Columns(columns) => columns.schema(),
            Self::Limit(limited) => limited.schema(),
            Self::Table(table) => table.schema(),
            Self::Index(index) => index.schema(),
            Self::Merge(merged) => merged.schema(),
            Self::ROIndex(ro_index) => ro_index.schema(),
        }
    }

    async fn slice(&self, txn_id: TxnId, bounds: Bounds) -> TCResult<Table> {
        match self {
            Self::Columns(columns) => columns.slice(txn_id, bounds).await,
            Self::Limit(limited) => limited.slice(txn_id, bounds).await,
            Self::Table(table) => table.slice(txn_id, bounds).await,
            Self::Index(index) => index.slice(txn_id, bounds).await,
            Self::Merge(merged) => merged.slice(txn_id, bounds).await,
            Self::ROIndex(ro_index) => ro_index.slice(txn_id, bounds).await,
        }
    }

    async fn stream(&self, txn_id: TxnId) -> TCResult<Self::Stream> {
        match self {
            Self::Columns(columns) => columns.stream(txn_id).await,
            Self::Limit(limited) => limited.stream(txn_id).await,
            Self::Table(table) => table.stream(txn_id).await,
            Self::Index(index) => index.stream(txn_id).await,
            Self::Merge(merged) => merged.stream(txn_id).await,
            Self::ROIndex(ro_index) => ro_index.stream(txn_id).await,
        }
    }

    async fn update(self, txn: Arc<Txn>, value: Row) -> TCResult<()> {
        match self {
            Self::Columns(columns) => columns.clone().update(txn, value).await,
            Self::Limit(limited) => limited.clone().update(txn, value).await,
            Self::Table(table) => table.clone().update(txn, value).await,
            Self::Index(index) => index.clone().update(txn, value).await,
            Self::Merge(merged) => merged.clone().update(txn, value).await,
            Self::ROIndex(ro_index) => ro_index.update(txn, value).await,
        }
    }

    async fn update_row(&self, txn_id: TxnId, row: Row, value: Row) -> TCResult<()> {
        match self {
            Self::Columns(columns) => columns.update_row(txn_id, row, value).await,
            Self::Limit(limited) => limited.update_row(txn_id, row, value).await,
            Self::Table(table) => table.update_row(txn_id, row, value).await,
            Self::Index(index) => index.update_row(txn_id, row, value).await,
            Self::Merge(merged) => merged.update_row(txn_id, row, value).await,
            Self::ROIndex(ro_index) => ro_index.update_row(txn_id, row, value).await,
        }
    }

    async fn validate(&self, txn_id: &TxnId, bounds: &Bounds) -> TCResult<()> {
        match self {
            Self::Columns(columns) => columns.validate(txn_id, bounds).await,
            Self::Limit(limited) => limited.validate(txn_id, bounds).await,
            Self::Table(table) => table.validate(txn_id, bounds).await,
            Self::Index(index) => index.validate(txn_id, bounds).await,
            Self::Merge(merged) => merged.validate(txn_id, bounds).await,
            Self::ROIndex(ro_index) => ro_index.validate(txn_id, bounds).await,
        }
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

impl From<index::IndexTable> for Table {
    fn from(table: index::IndexTable) -> Table {
        Table::Table(table)
    }
}

impl From<view::Merged> for Table {
    fn from(merged: view::Merged) -> Table {
        Table::Merge(merged)
    }
}

impl From<index::ReadOnly> for Table {
    fn from(ro_index: index::ReadOnly) -> Table {
        Table::ROIndex(ro_index)
    }
}
