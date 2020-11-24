use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::fmt;
use std::iter::FromIterator;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::{self, StreamExt, TryStreamExt};
use futures::{future, join};

use crate::class::*;
use crate::collection::btree::{BTreeFile, BTreeInstance, BTreeRange};
use crate::collection::schema::{Column, IndexSchema, Row};
use crate::collection::{Collection, CollectionView};
use crate::error;
use crate::scalar::{label, Id, Link, PathSegment, TCPathBuf, Value};
use crate::transaction::{Transact, Txn, TxnId};

use super::bounds::{self, Bounds};
use super::index::TableIndex;
use super::{Table, TableInstance, TableType};

const ERR_AGGREGATE_SLICE: &str = "Table aggregate does not support slicing. \
Consider aggregating a slice of the source table.";
const ERR_AGGREGATE_NESTED: &str = "It doesn't make sense to aggregate an aggregate table view. \
Consider aggregating the source table directly.";
const ERR_LIMITED_ORDER: &str = "Cannot order a limited selection. \
Consider ordering the source or indexing the selection.";
const ERR_LIMITED_REVERSE: &str = "Cannot reverse a limited selection. \
Consider reversing a slice before limiting";

#[derive(Clone, Eq, PartialEq)]
pub enum TableViewType {
    Aggregate,
    IndexSlice,
    Limit,
    Merge,
    Selection,
    TableSlice,
}

impl Class for TableViewType {
    type Instance = TableView;
}

impl NativeClass for TableViewType {
    fn from_path(_path: &[PathSegment]) -> TCResult<Self> {
        Err(error::internal(crate::class::ERR_PROTECTED))
    }

    fn prefix() -> TCPathBuf {
        TableType::prefix()
    }
}

impl From<TableViewType> for Link {
    fn from(_tvt: TableViewType) -> Link {
        TableViewType::prefix().append(label("index")).into()
    }
}

impl fmt::Display for TableViewType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Aggregate => write!(f, "Table or Index Aggregate"),
            Self::IndexSlice => write!(f, "Index Slice"),
            Self::Limit => write!(f, "Table or Index Limit Selection"),
            Self::Merge => write!(f, "Table Merge Selection"),
            Self::Selection => write!(f, "Table or Index Column Selection"),
            Self::TableSlice => write!(f, "Table Slice"),
        }
    }
}

#[derive(Clone)]
pub enum TableView {
    Aggregate(Aggregate),
    IndexSlice(IndexSlice),
    Limit(Limited),
    Merge(Merged),
    Selection(Selection),
    TableSlice(TableSlice),
}

impl Instance for TableView {
    type Class = TableViewType;

    fn class(&self) -> Self::Class {
        match self {
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
impl TableInstance for TableView {
    type Stream = TCStream<Vec<Value>>;

    async fn count(&self, txn_id: TxnId) -> TCResult<u64> {
        match self {
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
            Self::Aggregate(aggregate) => aggregate.delete(txn_id).await,
            Self::IndexSlice(index_slice) => index_slice.delete(txn_id).await,
            Self::Limit(limited) => limited.delete(txn_id).await,
            Self::Merge(merged) => merged.delete(txn_id).await,
            Self::Selection(columns) => columns.delete(txn_id).await,
            Self::TableSlice(table_slice) => table_slice.delete(txn_id).await,
        }
    }

    async fn delete_row(&self, txn_id: &TxnId, row: Row) -> TCResult<()> {
        match self {
            Self::Aggregate(aggregate) => aggregate.delete_row(txn_id, row).await,
            Self::IndexSlice(index_slice) => index_slice.delete_row(txn_id, row).await,
            Self::Limit(limited) => limited.delete_row(txn_id, row).await,
            Self::Merge(merged) => merged.delete_row(txn_id, row).await,
            Self::Selection(columns) => columns.delete_row(txn_id, row).await,
            Self::TableSlice(table_slice) => table_slice.delete_row(txn_id, row).await,
        }
    }

    fn key(&'_ self) -> &'_ [Column] {
        match self {
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
            Self::Aggregate(aggregate) => aggregate.values(),
            Self::IndexSlice(index_slice) => index_slice.values(),
            Self::Limit(limited) => limited.values(),
            Self::Merge(merged) => merged.values(),
            Self::Selection(columns) => columns.values(),
            Self::TableSlice(table_slice) => table_slice.values(),
        }
    }

    fn order_by(&self, order: Vec<Id>, reverse: bool) -> TCResult<Table> {
        match self {
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
            Self::Aggregate(aggregate) => aggregate.reversed(),
            Self::IndexSlice(index_slice) => index_slice.reversed(),
            Self::Limit(limited) => limited.reversed(),
            Self::Merge(merged) => merged.reversed(),
            Self::Selection(columns) => columns.reversed(),
            Self::TableSlice(table_slice) => table_slice.reversed(),
        }
    }

    fn slice(&self, bounds: bounds::Bounds) -> TCResult<Table> {
        match self {
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
            Self::Aggregate(aggregate) => aggregate.stream(txn_id).await,
            Self::IndexSlice(index_slice) => index_slice.stream(txn_id).await,
            Self::Limit(limited) => limited.stream(txn_id).await,
            Self::Merge(merged) => merged.stream(txn_id).await,
            Self::Selection(columns) => columns.stream(txn_id).await,
            Self::TableSlice(table_slice) => table_slice.stream(txn_id).await,
        }
    }

    fn validate_bounds(&self, bounds: &bounds::Bounds) -> TCResult<()> {
        match self {
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
            Self::Aggregate(aggregate) => aggregate.update(txn, value).await,
            Self::IndexSlice(index_slice) => index_slice.update(txn, value).await,
            Self::Limit(limited) => limited.update(txn, value).await,
            Self::Merge(merged) => merged.update(txn, value).await,
            Self::Selection(columns) => columns.update(txn, value).await,
            Self::TableSlice(table_slice) => table_slice.update(txn, value).await,
        }
    }

    async fn update_row(&self, txn_id: TxnId, row: Row, value: Row) -> TCResult<()> {
        match self {
            Self::Aggregate(aggregate) => aggregate.update_row(txn_id, row, value).await,
            Self::IndexSlice(index_slice) => index_slice.update_row(txn_id, row, value).await,
            Self::Limit(limited) => limited.update_row(txn_id, row, value).await,
            Self::Merge(merged) => merged.update_row(txn_id, row, value).await,
            Self::Selection(columns) => columns.update_row(txn_id, row, value).await,
            Self::TableSlice(table_slice) => table_slice.update_row(txn_id, row, value).await,
        }
    }
}

#[async_trait]
impl Transact for TableView {
    async fn commit(&self, txn_id: &TxnId) {
        match self {
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
            Self::Aggregate(_) => (), // no-op
            Self::IndexSlice(index_slice) => index_slice.finalize(txn_id).await,
            Self::Limit(limited) => limited.finalize(txn_id).await,
            Self::Merge(merged) => merged.finalize(txn_id).await,
            Self::Selection(_) => (), // no-op
            Self::TableSlice(table_slice) => table_slice.finalize(txn_id).await,
        }
    }
}

impl From<Aggregate> for TableView {
    fn from(aggregate: Aggregate) -> Self {
        Self::Aggregate(aggregate)
    }
}

impl From<Selection> for TableView {
    fn from(selection: Selection) -> Self {
        Self::Selection(selection)
    }
}

impl From<Limited> for TableView {
    fn from(limited: Limited) -> Self {
        Self::Limit(limited)
    }
}

impl From<IndexSlice> for TableView {
    fn from(index_slice: IndexSlice) -> Self {
        Self::IndexSlice(index_slice)
    }
}

impl From<Merged> for TableView {
    fn from(merged: Merged) -> Self {
        Self::Merge(merged)
    }
}

impl From<TableSlice> for TableView {
    fn from(table_slice: TableSlice) -> Self {
        Self::TableSlice(table_slice)
    }
}

impl TryFrom<CollectionView> for TableView {
    type Error = error::TCError;

    fn try_from(view: CollectionView) -> TCResult<TableView> {
        match view {
            CollectionView::Table(Table::View(view)) => Ok(view.into_inner()),
            other => Err(error::bad_request("Expected TableView but found", other)),
        }
    }
}

impl From<TableView> for Collection {
    fn from(view: TableView) -> Collection {
        Collection::View(CollectionView::Table(Table::View(view.into())))
    }
}

#[derive(Clone)]
pub struct Aggregate {
    source: Box<Table>,
    columns: Vec<Id>,
}

impl Aggregate {
    pub fn new(source: Table, columns: Vec<Id>) -> TCResult<Aggregate> {
        let source = Box::new(source.order_by(columns.to_vec(), false)?);
        Ok(Aggregate { source, columns })
    }
}

impl Instance for Aggregate {
    type Class = TableViewType;

    fn class(&self) -> Self::Class {
        Self::Class::Aggregate
    }
}

#[async_trait]
impl TableInstance for Aggregate {
    type Stream = TCStream<Vec<Value>>;

    fn group_by(&self, _columns: Vec<Id>) -> TCResult<Aggregate> {
        Err(error::unsupported(ERR_AGGREGATE_NESTED))
    }

    fn key(&'_ self) -> &'_ [Column] {
        self.source.key()
    }

    fn values(&'_ self) -> &'_ [Column] {
        self.source.values()
    }

    fn order_by(&self, columns: Vec<Id>, reverse: bool) -> TCResult<Table> {
        let source = Box::new(self.source.order_by(columns, reverse)?);
        Ok(Aggregate {
            source,
            columns: self.columns.to_vec(),
        }
        .into())
    }

    fn reversed(&self) -> TCResult<Table> {
        let columns = self.columns.to_vec();
        let reversed = self
            .source
            .reversed()
            .map(Box::new)
            .map(|source| Aggregate { source, columns })?;
        Ok(reversed.into())
    }

    async fn stream(self, txn_id: TxnId) -> TCResult<Self::Stream> {
        let first = self
            .source
            .clone()
            .stream(txn_id.clone())
            .await?
            .next()
            .await;
        let first = if let Some(first) = first {
            first
        } else {
            let stream: TCStream<Vec<Value>> = Box::pin(stream::empty());
            return Ok(stream);
        };

        let left = stream::once(future::ready(first))
            .chain(self.source.clone().stream(txn_id.clone()).await?);
        let right = self.source.clone().stream(txn_id).await?;
        let aggregate = left.zip(right).filter_map(|(l, r)| {
            if l == r {
                future::ready(None)
            } else {
                future::ready(Some(r))
            }
        });
        let aggregate: TCStream<Vec<Value>> = Box::pin(aggregate);

        Ok(aggregate)
    }

    fn validate_bounds(&self, _bounds: &Bounds) -> TCResult<()> {
        Err(error::unsupported(ERR_AGGREGATE_SLICE))
    }

    fn validate_order(&self, order: &[Id]) -> TCResult<()> {
        self.source.validate_order(order)
    }
}

impl From<Aggregate> for Table {
    fn from(aggregate: Aggregate) -> Self {
        Self::View(TableView::from(aggregate).into())
    }
}

#[derive(Clone)]
pub struct IndexSlice {
    source: BTreeFile,
    schema: IndexSchema,
    bounds: Bounds,
    range: BTreeRange,
    reverse: bool,
}

impl IndexSlice {
    pub fn all(source: BTreeFile, schema: IndexSchema, reverse: bool) -> IndexSlice {
        IndexSlice {
            source,
            schema,
            bounds: bounds::all(),
            range: BTreeRange::default(),
            reverse,
        }
    }

    pub fn new(source: BTreeFile, schema: IndexSchema, bounds: Bounds) -> TCResult<IndexSlice> {
        let columns = schema.columns();

        assert!(source.schema() == &columns[..]);

        let bounds = bounds::validate(bounds, &columns)?;
        let range = bounds::btree_range(&bounds, &columns)?;

        Ok(IndexSlice {
            source,
            schema,
            bounds,
            range,
            reverse: false,
        })
    }

    pub fn schema(&'_ self) -> &'_ IndexSchema {
        &self.schema
    }

    pub fn into_reversed(mut self) -> IndexSlice {
        self.reverse = !self.reverse;
        self
    }

    pub async fn is_empty(&self, txn: &Txn) -> TCResult<bool> {
        let mut rows = self.clone().stream(txn.id().clone()).await?;
        Ok(rows.next().await.is_none())
    }

    pub fn slice_index(&self, bounds: Bounds) -> TCResult<IndexSlice> {
        let columns = self.schema().columns();
        let outer = bounds::btree_range(&self.bounds, &columns)?;
        let inner = bounds::btree_range(&bounds, &columns)?;

        if outer.contains(&inner, &self.schema.columns())? {
            let mut slice = self.clone();
            slice.bounds = bounds;
            Ok(slice)
        } else {
            Err(error::bad_request(
                &format!(
                    "IndexSlice with bounds {} does not contain",
                    bounds::format(&self.bounds)
                ),
                bounds::format(&bounds),
            ))
        }
    }
}

impl Instance for IndexSlice {
    type Class = TableViewType;

    fn class(&self) -> Self::Class {
        Self::Class::IndexSlice
    }
}

#[async_trait]
impl TableInstance for IndexSlice {
    type Stream = TCStream<Vec<Value>>;

    async fn count(&self, txn_id: TxnId) -> TCResult<u64> {
        self.source.len(txn_id, self.range.clone()).await
    }

    async fn delete(self, txn_id: TxnId) -> TCResult<()> {
        self.source.delete(&txn_id, self.range).await
    }

    fn key(&'_ self) -> &'_ [Column] {
        self.schema.key()
    }

    fn values(&'_ self) -> &'_ [Column] {
        self.schema.values()
    }

    fn order_by(&self, order: Vec<Id>, reverse: bool) -> TCResult<Table> {
        self.validate_order(&order)?;

        if reverse {
            self.reversed()
        } else {
            Ok(self.clone().into())
        }
    }

    fn reversed(&self) -> TCResult<Table> {
        Ok(self.clone().into_reversed().into())
    }

    async fn stream(self, txn_id: TxnId) -> TCResult<Self::Stream> {
        self.source
            .stream(txn_id.clone(), self.range.clone(), self.reverse)
            .await
    }

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()> {
        let schema = self.schema();
        let outer = bounds::btree_range(&self.bounds, &schema.columns())?;
        let inner = bounds::btree_range(&bounds, &schema.columns())?;
        outer.contains(&inner, &schema.columns()).map(|_| ())
    }

    fn validate_order(&self, order: &[Id]) -> TCResult<()> {
        if self.schema.starts_with(order) {
            Ok(())
        } else {
            Err(error::bad_request(
                &format!("Index with schema {} does not support order", &self.schema),
                Value::from_iter(order.to_vec()),
            ))
        }
    }

    async fn update(self, txn: Txn, value: Row) -> TCResult<()> {
        let key = self.schema.values_from_row(value, true)?;
        self.source.update(txn.id(), self.range, &key).await
    }
}

impl From<IndexSlice> for Table {
    fn from(index_slice: IndexSlice) -> Self {
        Self::View(TableView::from(index_slice).into())
    }
}

#[async_trait]
impl Transact for IndexSlice {
    async fn commit(&self, txn_id: &TxnId) {
        self.source.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.source.rollback(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.source.finalize(txn_id).await
    }
}

#[derive(Clone)]
pub struct Limited {
    source: Box<Table>,
    limit: u64,
}

impl Limited {
    pub fn new<T: Into<Table>>(source: T, limit: u64) -> Limited {
        let source = Box::new(source.into());
        Limited { source, limit }
    }
}

impl Instance for Limited {
    type Class = TableViewType;

    fn class(&self) -> Self::Class {
        Self::Class::Limit
    }
}

#[async_trait]
impl TableInstance for Limited {
    type Stream = TCStream<Vec<Value>>;

    async fn count(&self, txn_id: TxnId) -> TCResult<u64> {
        let source_count = self.source.count(txn_id).await?;
        Ok(u64::min(source_count, self.limit as u64))
    }

    async fn delete(self, txn_id: TxnId) -> TCResult<()> {
        let source = self.source.clone();
        let schema: IndexSchema = (source.key().to_vec(), source.values().to_vec()).into();

        self.stream(txn_id.clone())
            .await?
            .map(|row| schema.row_from_values(row))
            .map_ok(|row| source.delete_row(&txn_id, row))
            .try_buffer_unordered(2)
            .try_fold((), |_, _| future::ready(Ok(())))
            .await
    }

    fn key(&'_ self) -> &'_ [Column] {
        self.source.key()
    }

    fn values(&'_ self) -> &'_ [Column] {
        self.source.values()
    }

    fn order_by(&self, _order: Vec<Id>, _reverse: bool) -> TCResult<Table> {
        Err(error::unsupported(ERR_LIMITED_ORDER))
    }

    fn reversed(&self) -> TCResult<Table> {
        Err(error::unsupported(ERR_LIMITED_REVERSE))
    }

    async fn stream(self, txn_id: TxnId) -> TCResult<Self::Stream> {
        let rows = self.source.clone().stream(txn_id).await?;
        let rows: TCStream<Vec<Value>> = Box::pin(rows.take(self.limit as usize));
        Ok(rows)
    }

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()> {
        self.source.validate_bounds(bounds)
    }

    fn validate_order(&self, _order: &[Id]) -> TCResult<()> {
        Err(error::unsupported(ERR_LIMITED_ORDER))
    }

    async fn update(self, txn: Txn, value: Row) -> TCResult<()> {
        let source = self.source.clone();
        let schema: IndexSchema = (source.key().to_vec(), source.values().to_vec()).into();
        let txn_id = txn.id().clone();

        self.stream(txn_id.clone())
            .await?
            .map(|row| schema.row_from_values(row))
            .map_ok(|row| source.update_row(txn_id.clone(), row, value.clone()))
            .try_buffer_unordered(2)
            .try_fold((), |_, _| future::ready(Ok(())))
            .await
    }
}

impl From<Limited> for Table {
    fn from(limited: Limited) -> Table {
        Table::View(TableView::from(limited).into())
    }
}

#[async_trait]
impl Transact for Limited {
    async fn commit(&self, txn_id: &TxnId) {
        self.source.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.source.rollback(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.source.finalize(txn_id).await
    }
}

#[derive(Clone)]
pub enum MergeSource {
    Table(TableSlice),
    Merge(Arc<Merged>),
}

impl MergeSource {
    fn into_reversed(self) -> MergeSource {
        match self {
            Self::Table(table_slice) => Self::Table(table_slice.into_reversed()),
            Self::Merge(merged) => Self::Merge(merged.as_reversed()),
        }
    }

    fn slice(self, bounds: Bounds) -> TCResult<Table> {
        match self {
            Self::Table(table) => table.slice(bounds),
            Self::Merge(merged) => merged.slice(bounds),
        }
    }
}

#[async_trait]
impl Transact for MergeSource {
    async fn commit(&self, txn_id: &TxnId) {
        match self {
            Self::Table(table) => table.commit(txn_id).await,
            Self::Merge(merged) => merged.commit(txn_id).await,
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        match self {
            Self::Table(table) => table.rollback(txn_id).await,
            Self::Merge(merged) => merged.rollback(txn_id).await,
        }
    }

    async fn finalize(&self, txn_id: &TxnId) {
        match self {
            Self::Table(table) => table.finalize(txn_id).await,
            Self::Merge(merged) => merged.finalize(txn_id).await,
        }
    }
}

#[derive(Clone)]
pub struct Merged {
    left: MergeSource,
    right: IndexSlice,
}

impl Merged {
    pub fn new(left: MergeSource, right: IndexSlice) -> Merged {
        Merged { left, right }
    }

    fn as_reversed(&self) -> Arc<Self> {
        Arc::new(Merged {
            left: self.left.clone().into_reversed(),
            right: self.right.clone().into_reversed(),
        })
    }
}

impl Instance for Merged {
    type Class = TableViewType;

    fn class(&self) -> Self::Class {
        Self::Class::Merge
    }
}

#[async_trait]
impl TableInstance for Merged {
    type Stream = TCStream<Vec<Value>>;

    async fn delete(self, txn_id: TxnId) -> TCResult<()> {
        let schema: IndexSchema = (self.key().to_vec(), self.values().to_vec()).into();

        self.clone()
            .stream(txn_id.clone())
            .await?
            .map(|row| schema.row_from_values(row))
            .map_ok(|row| self.delete_row(&txn_id, row))
            .try_buffer_unordered(2)
            .try_fold((), |_, _| future::ready(Ok(())))
            .await
    }

    async fn delete_row(&self, txn_id: &TxnId, row: Row) -> TCResult<()> {
        match &self.left {
            MergeSource::Table(table) => table.delete_row(txn_id, row).await,
            MergeSource::Merge(merged) => merged.delete_row(txn_id, row).await,
        }
    }

    fn key(&'_ self) -> &'_ [Column] {
        match &self.left {
            MergeSource::Table(table) => table.key(),
            MergeSource::Merge(merged) => merged.key(),
        }
    }

    fn values(&'_ self) -> &'_ [Column] {
        match &self.left {
            MergeSource::Table(table) => table.values(),
            MergeSource::Merge(merged) => merged.values(),
        }
    }

    fn order_by(&self, columns: Vec<Id>, reverse: bool) -> TCResult<Table> {
        match &self.left {
            MergeSource::Merge(merged) => merged.order_by(columns, reverse),
            MergeSource::Table(table_slice) => table_slice.order_by(columns, reverse),
        }
    }

    fn reversed(&self) -> TCResult<Table> {
        Ok(Merged {
            left: self.left.clone().into_reversed(),
            right: self.right.clone().into_reversed(),
        }
        .into())
    }

    fn slice(&self, bounds: Bounds) -> TCResult<Table> {
        // TODO: reject bounds which lie outside the bounds of the table slice

        match &self.left {
            MergeSource::Merge(merged) => merged.slice(bounds),
            MergeSource::Table(table) => table.slice(bounds),
        }
    }

    async fn stream(self, txn_id: TxnId) -> TCResult<Self::Stream> {
        let key_columns = self.key().to_vec();
        let key_names: Vec<Id> = key_columns.iter().map(|c| c.name()).cloned().collect();
        let left = self.left.clone();
        let txn_id_clone = txn_id.clone();
        let rows = self
            .right
            .select(key_names)?
            .stream(txn_id.clone())
            .await?
            .map(move |key| bounds::from_key(key, &key_columns))
            .map(move |bounds| left.clone().slice(bounds))
            .map(|slice| slice.unwrap())
            .then(move |slice| slice.stream(txn_id_clone.clone()))
            .map(|stream| stream.unwrap())
            .flatten();

        let rows: TCStream<Vec<Value>> = Box::pin(rows);
        Ok(rows)
    }

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()> {
        match &self.left {
            MergeSource::Merge(merge) => merge.validate_bounds(bounds),
            MergeSource::Table(table) => table.validate_bounds(bounds),
        }
    }

    fn validate_order(&self, order: &[Id]) -> TCResult<()> {
        match &self.left {
            MergeSource::Merge(merge) => merge.validate_order(order),
            MergeSource::Table(table) => table.validate_order(order),
        }
    }

    async fn update(self, txn: Txn, value: Row) -> TCResult<()> {
        let schema: IndexSchema = (self.key().to_vec(), self.values().to_vec()).into();

        self.clone()
            .stream(txn.id().clone())
            .await?
            .map(|row| schema.row_from_values(row))
            .map_ok(|row| self.update_row(txn.id().clone(), row, value.clone()))
            .try_buffer_unordered(2)
            .try_fold((), |_, _| future::ready(Ok(())))
            .await
    }

    async fn update_row(&self, txn_id: TxnId, row: Row, value: Row) -> TCResult<()> {
        match &self.left {
            MergeSource::Table(table) => table.update_row(txn_id, row, value).await,
            MergeSource::Merge(merged) => merged.update_row(txn_id, row, value).await,
        }
    }
}

impl From<Merged> for Table {
    fn from(merged: Merged) -> Table {
        Table::View(TableView::from(merged).into())
    }
}

#[async_trait]
impl Transact for Merged {
    async fn commit(&self, txn_id: &TxnId) {
        join!(self.left.commit(txn_id), self.right.commit(txn_id));
    }

    async fn rollback(&self, txn_id: &TxnId) {
        join!(self.left.rollback(txn_id), self.right.rollback(txn_id));
    }

    async fn finalize(&self, txn_id: &TxnId) {
        join!(self.left.finalize(txn_id), self.right.finalize(txn_id));
    }
}

#[derive(Clone)]
pub struct Selection {
    source: Box<Table>,
    schema: IndexSchema,
    columns: Vec<Id>,
    indices: Vec<usize>,
}

impl Selection {
    pub fn new<T: Into<Table>>(source: T, columns: Vec<Id>) -> TCResult<Selection> {
        let source: Table = source.into();

        let column_set: HashSet<&Id> = columns.iter().collect();
        if column_set.len() != columns.len() {
            return Err(error::bad_request(
                "Tried to select duplicate column",
                columns
                    .iter()
                    .map(|name| name.to_string())
                    .collect::<Vec<String>>()
                    .join(", "),
            ));
        }

        let mut indices: Vec<usize> = Vec::with_capacity(columns.len());
        let mut schema: Vec<Column> = Vec::with_capacity(columns.len());
        let source_schema: IndexSchema = (source.key().to_vec(), source.values().to_vec()).into();

        let mut source_columns: HashMap<Id, Column> = source_schema.into();

        for (i, name) in columns.iter().enumerate() {
            let column = source_columns
                .remove(name)
                .ok_or_else(|| error::not_found(name))?;
            indices.push(i);
            schema.push(column);
        }

        Ok(Selection {
            source: Box::new(source),
            schema: (vec![], schema).into(),
            columns,
            indices,
        })
    }
}

impl Instance for Selection {
    type Class = TableViewType;

    fn class(&self) -> Self::Class {
        Self::Class::Selection
    }
}

#[async_trait]
impl TableInstance for Selection {
    type Stream = TCStream<Vec<Value>>;

    async fn count(&self, txn_id: TxnId) -> TCResult<u64> {
        self.source.clone().count(txn_id).await
    }

    fn key(&'_ self) -> &'_ [Column] {
        self.schema.key()
    }

    fn values(&'_ self) -> &'_ [Column] {
        self.schema.values()
    }

    fn order_by(&self, order: Vec<Id>, reverse: bool) -> TCResult<Table> {
        self.validate_order(&order)?;

        let source = self.source.order_by(order, reverse).map(Box::new)?;

        Ok(Selection {
            source,
            schema: self.schema.clone(),
            columns: self.columns.to_vec(),
            indices: self.indices.to_vec(),
        }
        .into())
    }

    fn reversed(&self) -> TCResult<Table> {
        self.source
            .reversed()?
            .select(self.columns.to_vec())
            .map(|s| s.into())
    }

    async fn stream(self, txn_id: TxnId) -> TCResult<Self::Stream> {
        let indices = self.indices.to_vec();
        let selected = self.source.stream(txn_id).await?.map(move |row| {
            let selection: Vec<Value> = indices.iter().map(|i| row[*i].clone()).collect();
            selection
        });
        let selected: TCStream<Vec<Value>> = Box::pin(selected);
        Ok(selected)
    }

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()> {
        let bounds_columns: HashSet<Id> = bounds.keys().cloned().collect();
        let selected: HashSet<Id> = self
            .schema
            .columns()
            .iter()
            .map(|c| c.name())
            .cloned()
            .collect();
        let mut unknown: HashSet<&Id> = selected.difference(&bounds_columns).collect();
        if !unknown.is_empty() {
            let unknown: Vec<String> = unknown.drain().map(|c| c.to_string()).collect();
            return Err(error::bad_request(
                "Tried to slice by unselected columns",
                unknown.join(", "),
            ));
        }

        self.source.validate_bounds(bounds)
    }

    fn validate_order(&self, order: &[Id]) -> TCResult<()> {
        let order_columns: HashSet<Id> = order.iter().cloned().collect();
        let selected: HashSet<Id> = self
            .schema
            .columns()
            .iter()
            .map(|c| c.name())
            .cloned()
            .collect();
        let mut unknown: HashSet<&Id> = selected.difference(&order_columns).collect();
        if !unknown.is_empty() {
            let unknown: Vec<String> = unknown.drain().map(|c| c.to_string()).collect();
            return Err(error::bad_request(
                "Tried to order by unselected columns",
                unknown.join(", "),
            ));
        }

        self.source.validate_order(order)
    }
}

impl From<Selection> for Table {
    fn from(selection: Selection) -> Self {
        Self::View(TableView::from(selection).into())
    }
}

#[derive(Clone)]
pub struct TableSlice {
    table: TableIndex,
    bounds: Bounds,
    reversed: bool,
}

impl TableSlice {
    pub fn new(table: TableIndex, bounds: Bounds) -> TCResult<TableSlice> {
        table.validate_bounds(&bounds)?;

        Ok(TableSlice {
            table,
            bounds,
            reversed: false,
        })
    }

    fn into_reversed(self) -> TableSlice {
        TableSlice {
            table: self.table,
            bounds: self.bounds,
            reversed: !self.reversed,
        }
    }
}

impl Instance for TableSlice {
    type Class = TableViewType;

    fn class(&self) -> Self::Class {
        Self::Class::TableSlice
    }
}

#[async_trait]
impl TableInstance for TableSlice {
    type Stream = TCStream<Vec<Value>>;

    async fn count(&self, txn_id: TxnId) -> TCResult<u64> {
        let index = self.table.supporting_index(&self.bounds)?;
        index.slice(self.bounds.clone())?.count(txn_id).await
    }

    async fn delete(self, txn_id: TxnId) -> TCResult<()> {
        let schema: IndexSchema = (self.key().to_vec(), self.values().to_vec()).into();

        self.clone()
            .stream(txn_id.clone())
            .await?
            .map(|row| schema.row_from_values(row))
            .map_ok(|row| self.delete_row(&txn_id, row))
            .try_buffer_unordered(2)
            .fold(Ok(()), |_, r| future::ready(r))
            .await
    }

    async fn delete_row(&self, txn_id: &TxnId, row: Row) -> TCResult<()> {
        self.table.delete_row(txn_id, row).await
    }

    fn key(&'_ self) -> &'_ [Column] {
        self.table.key()
    }

    fn values(&'_ self) -> &'_ [Column] {
        self.table.values()
    }

    fn order_by(&self, order: Vec<Id>, reverse: bool) -> TCResult<Table> {
        self.table.order_by(order, reverse)
    }

    fn reversed(&self) -> TCResult<Table> {
        let mut selection = self.clone();
        selection.reversed = true;
        Ok(selection.into())
    }

    fn slice(&self, bounds: Bounds) -> TCResult<Table> {
        self.validate_bounds(&bounds)?;
        self.table.slice(bounds)
    }

    async fn stream(self, txn_id: TxnId) -> TCResult<Self::Stream> {
        let slice = self.table.primary().slice(self.bounds.clone())?;

        if self.reversed {
            slice.reversed()?.stream(txn_id).await
        } else {
            slice.stream(txn_id).await
        }
    }

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()> {
        let index = self.table.supporting_index(&self.bounds)?;
        index
            .validate_slice_bounds(self.bounds.clone(), bounds.clone())
            .map(|_| ())
    }

    fn validate_order(&self, order: &[Id]) -> TCResult<()> {
        self.table.validate_order(order)
    }

    async fn update(self, txn: Txn, value: Row) -> TCResult<()> {
        let txn_id = txn.id().clone();
        let schema: IndexSchema = (self.key().to_vec(), self.values().to_vec()).into();
        self.clone()
            .stream(txn_id.clone())
            .await?
            .map(|row| schema.row_from_values(row))
            .map_ok(|row| self.update_row(txn_id.clone(), row, value.clone()))
            .try_buffer_unordered(2)
            .try_fold((), |_, _| future::ready(Ok(())))
            .await
    }

    async fn update_row(&self, txn_id: TxnId, row: Row, value: Row) -> TCResult<()> {
        self.table.update_row(txn_id, row, value).await
    }
}

impl From<TableSlice> for Table {
    fn from(table_slice: TableSlice) -> Table {
        Table::View(TableView::from(table_slice).into())
    }
}

#[async_trait]
impl Transact for TableSlice {
    async fn commit(&self, txn_id: &TxnId) {
        self.table.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.table.rollback(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.table.finalize(txn_id).await
    }
}
