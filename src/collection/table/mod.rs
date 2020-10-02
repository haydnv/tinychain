use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future;
use futures::{Stream, StreamExt};

use crate::class::{Class, Instance, TCBoxFuture, TCBoxTryFuture, TCResult, TCStream};
use crate::collection::class::CollectionInstance;
use crate::collection::{Collection, CollectionBase, CollectionItem, CollectionView};
use crate::error;
use crate::scalar::{Link, Scalar, TCPath, Value, ValueId};
use crate::transaction::{Transact, Txn, TxnId};

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
pub type TableBase = index::TableBase;
pub type TableBaseType = index::TableBaseType;
pub type TableIndex = index::TableIndex;
pub type TableView = view::TableView;
pub type TableViewType = view::TableViewType;

#[derive(Clone, Eq, PartialEq)]
pub enum TableType {
    Base(TableBaseType),
    View(TableViewType),
}

impl Class for TableType {
    type Instance = Table;

    fn from_path(path: &TCPath) -> TCResult<Self> {
        TableBaseType::from_path(path).map(TableType::Base)
    }

    fn prefix() -> TCPath {
        TableBaseType::prefix()
    }
}

impl From<TableBaseType> for TableType {
    fn from(tbt: TableBaseType) -> TableType {
        TableType::Base(tbt)
    }
}

impl From<TableViewType> for TableType {
    fn from(tvt: TableViewType) -> TableType {
        TableType::View(tvt)
    }
}

impl From<TableType> for Link {
    fn from(tt: TableType) -> Link {
        use TableType::*;
        match tt {
            Base(base) => base.into(),
            View(view) => view.into(),
        }
    }
}

impl fmt::Display for TableType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Base(base) => write!(f, "{}", base),
            Self::View(view) => write!(f, "{}", view),
        }
    }
}

#[async_trait]
pub trait TableInstance: Clone + Into<Table> + Sized + Send + 'static {
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

    fn select(&self, columns: Vec<ValueId>) -> TCResult<view::Selection> {
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
    Base(TableBase),
    View(TableView),
}

impl Table {
    pub async fn create(txn: Arc<Txn>, schema: TableSchema) -> TCResult<TableIndex> {
        index::TableIndex::create(txn, schema).await
    }
}

impl Instance for Table {
    type Class = TableType;

    fn class(&self) -> Self::Class {
        match self {
            Self::Base(base) => base.class().into(),
            Self::View(view) => view.class().into(),
        }
    }
}

impl CollectionInstance for Table {
    type Item = Vec<Value>;
    type Slice = TableView;

    fn get<'a>(
        &'a self,
        txn: Arc<Txn>,
        path: TCPath,
        selector: Value,
    ) -> TCBoxTryFuture<'a, CollectionItem<Self::Item, Self::Slice>> {
        match self {
            Self::Base(base) => base.get(txn, path, selector),
            Self::View(view) => view.get(txn, path, selector),
        }
    }

    fn is_empty<'a>(&'a self, txn: Arc<Txn>) -> TCBoxTryFuture<'a, bool> {
        match self {
            Self::Base(base) => base.is_empty(txn),
            Self::View(view) => view.is_empty(txn),
        }
    }

    fn put<'a>(
        &'a self,
        txn: Arc<Txn>,
        path: TCPath,
        selector: Value,
        value: CollectionItem<Self::Item, Self::Slice>,
    ) -> TCBoxTryFuture<'a, ()> {
        match self {
            Self::Base(base) => base.put(txn, path, selector, value),
            Self::View(view) => view.put(txn, path, selector, value),
        }
    }

    fn to_stream<'a>(&'a self, txn: Arc<Txn>) -> TCBoxTryFuture<'a, TCStream<Scalar>> {
        match self {
            Self::Base(base) => base.to_stream(txn),
            Self::View(view) => view.to_stream(txn),
        }
    }
}

#[async_trait]
impl TableInstance for Table {
    type Stream = TCStream<Vec<Value>>;

    async fn count(&self, txn_id: TxnId) -> TCResult<u64> {
        match self {
            Self::Base(base) => base.count(txn_id).await,
            Self::View(view) => view.count(txn_id).await,
        }
    }

    fn delete<'a>(self, txn_id: TxnId) -> TCBoxTryFuture<'a, ()> {
        match self {
            Self::Base(base) => base.delete(txn_id),
            Self::View(view) => view.delete(txn_id),
        }
    }

    fn delete_row<'a>(&'a self, txn_id: &'a TxnId, row: Row) -> TCBoxTryFuture<'a, ()> {
        match self {
            Self::Base(base) => base.delete_row(txn_id, row),
            Self::View(view) => view.delete_row(txn_id, row),
        }
    }

    fn group_by(&self, columns: Vec<ValueId>) -> TCResult<view::Aggregate> {
        match self {
            Self::Base(base) => base.group_by(columns),
            Self::View(view) => view.group_by(columns),
        }
    }

    fn index<'a>(
        &'a self,
        txn: Arc<Txn>,
        columns: Option<Vec<ValueId>>,
    ) -> TCBoxTryFuture<'a, index::ReadOnly> {
        match self {
            Self::Base(base) => base.index(txn, columns),
            Self::View(view) => view.index(txn, columns),
        }
    }

    fn key(&'_ self) -> &'_ [Column] {
        match self {
            Self::Base(base) => base.key(),
            Self::View(view) => view.key(),
        }
    }

    fn values(&'_ self) -> &'_ [Column] {
        match self {
            Self::Base(base) => base.values(),
            Self::View(view) => view.values(),
        }
    }

    fn limit(&self, limit: u64) -> TCResult<Arc<view::Limited>> {
        match self {
            Self::Base(base) => base.limit(limit),
            Self::View(view) => view.limit(limit),
        }
    }

    fn order_by(&self, columns: Vec<ValueId>, reverse: bool) -> TCResult<Table> {
        match self {
            Self::Base(base) => base.order_by(columns, reverse),
            Self::View(view) => view.order_by(columns, reverse),
        }
    }

    fn reversed(&self) -> TCResult<Table> {
        match self {
            Self::Base(base) => base.reversed(),
            Self::View(view) => view.reversed(),
        }
    }

    fn select(&self, columns: Vec<ValueId>) -> TCResult<view::Selection> {
        match self {
            Self::Base(base) => base.select(columns),
            Self::View(view) => view.select(columns),
        }
    }

    fn slice(&self, bounds: bounds::Bounds) -> TCResult<Table> {
        match self {
            Self::Base(base) => base.slice(bounds),
            Self::View(view) => view.slice(bounds),
        }
    }

    fn stream<'a>(self, txn_id: TxnId) -> TCBoxTryFuture<'a, Self::Stream> {
        match self {
            Self::Base(base) => base.stream(txn_id),
            Self::View(view) => view.stream(txn_id),
        }
    }

    fn validate_bounds(&self, bounds: &bounds::Bounds) -> TCResult<()> {
        match self {
            Self::Base(base) => base.validate_bounds(bounds),
            Self::View(view) => view.validate_bounds(bounds),
        }
    }

    fn validate_order(&self, order: &[ValueId]) -> TCResult<()> {
        match self {
            Self::Base(base) => base.validate_order(order),
            Self::View(view) => view.validate_order(order),
        }
    }

    fn update<'a>(self, txn: Arc<Txn>, value: Row) -> TCBoxTryFuture<'a, ()> {
        match self {
            Self::Base(base) => base.update(txn, value),
            Self::View(view) => view.update(txn, value),
        }
    }

    fn update_row(&self, txn_id: TxnId, row: Row, value: Row) -> TCBoxTryFuture<()> {
        match self {
            Self::Base(base) => base.update_row(txn_id, row, value),
            Self::View(view) => view.update_row(txn_id, row, value),
        }
    }
}

impl Transact for Table {
    fn commit<'a>(&'a self, txn_id: &'a TxnId) -> TCBoxFuture<'a, ()> {
        match self {
            Self::Base(base) => base.commit(txn_id),
            Self::View(view) => view.commit(txn_id),
        }
    }

    fn rollback<'a>(&'a self, txn_id: &'a TxnId) -> TCBoxFuture<'a, ()> {
        match self {
            Self::Base(base) => base.rollback(txn_id),
            Self::View(view) => view.rollback(txn_id),
        }
    }
}

impl From<TableBase> for Table {
    fn from(base: TableBase) -> Table {
        Table::Base(base)
    }
}

impl From<TableView> for Table {
    fn from(view: TableView) -> Table {
        Table::View(view)
    }
}

impl From<Table> for Collection {
    fn from(table: Table) -> Collection {
        match table {
            Table::Base(base) => Collection::Base(CollectionBase::Table(base)),
            _ => Collection::View(CollectionView::Table(table)),
        }
    }
}
