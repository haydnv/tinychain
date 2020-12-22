use std::collections::{HashMap, HashSet};
use std::fmt;
use std::iter::FromIterator;

use async_trait::async_trait;
use futures::stream::{self, StreamExt, TryStreamExt};
use futures::{future, join};
use log::debug;

use crate::class::*;
use crate::collection::btree::{BTreeFile, BTreeInstance, BTreeRange};
use crate::collection::schema::{Column, IndexSchema, Row};
use crate::collection::Collection;
use crate::error;
use crate::general::{TCResult, TCTryStream};
use crate::scalar::{Id, Value};
use crate::transaction::{Transact, Txn, TxnId};

use super::bounds::Bounds;
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

#[derive(Clone)]
pub struct Aggregate<T: TableInstance> {
    source: Selection<T>,
}

impl<T: TableInstance> Instance for Aggregate<T> {
    type Class = TableType;

    fn class(&self) -> Self::Class {
        Self::Class::Aggregate
    }
}

#[async_trait]
impl<T: TableInstance> TableInstance for Aggregate<T> {
    type OrderBy = Aggregate<<T as TableInstance>::OrderBy>;
    type Reverse = Aggregate<<T as TableInstance>::Reverse>;
    type Slice = Table;

    fn into_table(self) -> Table {
        let source = self.source.into_table_selection();
        Table::Aggregate(Box::new(Aggregate { source }.into()))
    }

    fn group_by(self, _columns: Vec<Id>) -> TCResult<Aggregate<Self::OrderBy>> {
        Err(error::unsupported(ERR_AGGREGATE_NESTED))
    }

    fn key(&'_ self) -> &'_ [Column] {
        self.source.key()
    }

    fn values(&'_ self) -> &'_ [Column] {
        self.source.values()
    }

    fn order_by(self, columns: Vec<Id>, reverse: bool) -> TCResult<Self::OrderBy> {
        let source = self.source.order_by(columns, reverse)?;
        Ok(Aggregate { source })
    }

    fn reversed(self) -> TCResult<Self::Reverse> {
        self.source.reversed().map(|source| Aggregate { source })
    }

    async fn stream<'a>(&'a self, txn_id: &'a TxnId) -> TCResult<TCTryStream<'a, Vec<Value>>> {
        let first = self.source.stream(txn_id).await?.next().await;
        let first = if let Some(first) = first {
            first
        } else {
            let stream: TCTryStream<'_, Vec<Value>> = Box::pin(stream::empty());
            return Ok(stream);
        };

        let left =
            stream::once(future::ready(first.clone())).chain(self.source.stream(txn_id).await?);
        let right = self.source.stream(txn_id).await?;

        let aggregate = left
            .zip(right)
            .map(|(l, r)| Ok((l?, r?)))
            .try_filter_map(|(l, r)| {
                debug!("group {:?}, {:?}?", l, r);

                future::ready(if l == r { Ok(None) } else { Ok(Some(r)) })
            });

        let aggregate: TCTryStream<'a, Vec<Value>> =
            Box::pin(stream::once(future::ready(first)).chain(aggregate));

        Ok(aggregate)
    }

    fn validate_bounds(&self, _bounds: &Bounds) -> TCResult<()> {
        Err(error::unsupported(ERR_AGGREGATE_SLICE))
    }

    fn validate_order(&self, order: &[Id]) -> TCResult<()> {
        self.source.validate_order(order)
    }
}

impl<T: TableInstance> From<Aggregate<T>> for Collection {
    fn from(aggregate: Aggregate<T>) -> Collection {
        Collection::Table(aggregate.into_table())
    }
}

pub fn group_by<T: TableInstance>(
    source: T,
    columns: Vec<Id>,
) -> TCResult<Aggregate<<T as TableInstance>::OrderBy>> {
    let source = source.order_by(columns.to_vec(), false)?;
    let source = source.select(columns)?;
    Ok(Aggregate { source })
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
            bounds: Bounds::default(),
            range: BTreeRange::default(),
            reverse,
        }
    }

    pub fn new(source: BTreeFile, schema: IndexSchema, bounds: Bounds) -> TCResult<IndexSlice> {
        debug!("IndexSlice::new with bounds {}", bounds);
        let columns = schema.columns();

        assert!(source.schema() == &columns[..]);

        let bounds = bounds.validate(&columns)?;
        let range = bounds.clone().into_btree_range(&columns)?;

        Ok(IndexSlice {
            source,
            schema,
            bounds,
            range,
            reverse: false,
        })
    }

    pub fn bounds(&'_ self) -> &'_ Bounds {
        &self.bounds
    }

    pub fn schema(&'_ self) -> &'_ IndexSchema {
        &self.schema
    }

    pub fn into_reversed(mut self) -> IndexSlice {
        self.reverse = !self.reverse;
        self
    }

    pub async fn is_empty(&self, txn: &Txn) -> TCResult<bool> {
        let mut rows = self.stream(txn.id()).await?;
        Ok(rows.next().await.is_none())
    }

    pub fn slice_index(self, bounds: Bounds) -> TCResult<IndexSlice> {
        let columns = self.schema().columns();
        let outer = bounds.clone().into_btree_range(&columns)?;
        let inner = bounds.clone().into_btree_range(&columns)?;

        if outer.contains(&inner, &columns, self.source.collator()) {
            let mut slice = self;
            slice.bounds = bounds;
            Ok(slice)
        } else {
            Err(error::bad_request(
                &format!("IndexSlice with bounds {} does not contain", self.bounds),
                bounds,
            ))
        }
    }

    pub async fn stream_slice<'a>(
        &'a self,
        txn_id: &'a TxnId,
        bounds: Bounds,
        reverse: bool,
    ) -> TCResult<TCTryStream<'a, Vec<Value>>> {
        let reverse = self.reverse ^ reverse;
        let bounds = bounds.into_btree_range(&self.schema.columns())?;
        self.source.stream(txn_id, bounds, reverse).await
    }
}

impl Instance for IndexSlice {
    type Class = TableType;

    fn class(&self) -> Self::Class {
        Self::Class::IndexSlice
    }
}

#[async_trait]
impl TableInstance for IndexSlice {
    type OrderBy = Self;
    type Reverse = Self;
    type Slice = Table;

    fn into_table(self) -> Table {
        Table::IndexSlice(self.into())
    }

    async fn count(&self, txn_id: &TxnId) -> TCResult<u64> {
        self.source.len(txn_id, self.range.clone()).await
    }

    async fn delete(&self, txn_id: &TxnId) -> TCResult<()> {
        self.source.delete(txn_id, self.range.clone()).await
    }

    fn key(&'_ self) -> &'_ [Column] {
        self.schema.key()
    }

    fn values(&'_ self) -> &'_ [Column] {
        self.schema.values()
    }

    fn order_by(self, order: Vec<Id>, reverse: bool) -> TCResult<Self::OrderBy> {
        self.validate_order(&order)?;

        if reverse {
            self.reversed()
        } else {
            Ok(self.clone().into())
        }
    }

    fn reversed(self) -> TCResult<Self::Reverse> {
        Ok(self.into_reversed())
    }

    async fn stream<'a>(&'a self, txn_id: &'a TxnId) -> TCResult<TCTryStream<'a, Vec<Value>>> {
        debug!("IndexSlice::stream where {}", self.range);

        self.source
            .stream(txn_id, self.range.clone(), self.reverse)
            .await
    }

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()> {
        let schema = self.schema();
        let outer = bounds.clone().into_btree_range(&schema.columns())?;
        let inner = bounds.clone().into_btree_range(&schema.columns())?;

        if outer.contains(&inner, &schema.columns(), self.source.collator()) {
            Ok(())
        } else {
            Err(error::bad_request(
                "IndexSlice does not support bounds",
                bounds,
            ))
        }
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

    async fn update(&self, txn: &Txn, value: Row) -> TCResult<()> {
        let key = self.schema.values_from_row(value, true)?;
        self.source.update(txn.id(), self.range.clone(), &key).await
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

impl From<IndexSlice> for Collection {
    fn from(index: IndexSlice) -> Collection {
        Collection::Table(index.into())
    }
}

impl fmt::Display for IndexSlice {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "IndexSlice with bounds {}", self.bounds)
    }
}

#[derive(Clone)]
pub struct Limited {
    source: Box<Table>,
    limit: u64,
}

impl Limited {
    pub fn new<T: TableInstance>(source: T, limit: u64) -> Limited {
        let source = Box::new(source.into_table());
        Limited { source, limit }
    }
}

impl Instance for Limited {
    type Class = TableType;

    fn class(&self) -> Self::Class {
        Self::Class::Limit
    }
}

#[async_trait]
impl TableInstance for Limited {
    type OrderBy = Table;
    type Reverse = Table;
    type Slice = Table;

    fn into_table(self) -> Table {
        Table::Limit(self.into())
    }

    async fn count(&self, txn_id: &TxnId) -> TCResult<u64> {
        let source_count = self.source.count(txn_id).await?;
        Ok(u64::min(source_count, self.limit as u64))
    }

    async fn delete(&self, txn_id: &TxnId) -> TCResult<()> {
        let source = &self.source;
        let schema: IndexSchema = (source.key().to_vec(), source.values().to_vec()).into();

        let rows = self.stream(&txn_id).await?;

        rows.map(|row| row.and_then(|row| schema.row_from_values(row)))
            .map_ok(|row| source.delete_row(txn_id, row))
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

    fn order_by(self, _order: Vec<Id>, _reverse: bool) -> TCResult<Table> {
        Err(error::unsupported(ERR_LIMITED_ORDER))
    }

    fn reversed(self) -> TCResult<Table> {
        Err(error::unsupported(ERR_LIMITED_REVERSE))
    }

    async fn stream<'a>(&'a self, txn_id: &'a TxnId) -> TCResult<TCTryStream<'a, Vec<Value>>> {
        let rows = self.source.stream(txn_id).await?;
        let rows: TCTryStream<'_, Vec<Value>> = Box::pin(rows.take(self.limit as usize));
        Ok(rows)
    }

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()> {
        self.source.validate_bounds(bounds)
    }

    fn validate_order(&self, _order: &[Id]) -> TCResult<()> {
        Err(error::unsupported(ERR_LIMITED_ORDER))
    }

    async fn update(&self, txn: &Txn, value: Row) -> TCResult<()> {
        let source = &self.source;
        let schema: IndexSchema = (source.key().to_vec(), source.values().to_vec()).into();

        let rows = self.stream(txn.id()).await?;

        rows.map(|row| row.and_then(|row| schema.row_from_values(row)))
            .map_ok(|row| source.update_row(txn.id(), row, value.clone()))
            .try_buffer_unordered(2)
            .try_fold((), |_, _| future::ready(Ok(())))
            .await?;

        Ok(())
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

impl From<Limited> for Collection {
    fn from(limit: Limited) -> Collection {
        Collection::Table(limit.into())
    }
}

#[derive(Clone)]
pub enum MergeSource {
    Table(TableSlice),
    Merge(Box<Merged>),
}

impl MergeSource {
    fn bounds(&'_ self) -> &'_ Bounds {
        match self {
            Self::Table(table) => table.bounds(),
            Self::Merge(merged) => &merged.bounds,
        }
    }

    fn key(&'_ self) -> &'_ [Column] {
        match self {
            MergeSource::Table(table) => table.key(),
            MergeSource::Merge(merged) => merged.key(),
        }
    }

    fn into_reversed(self) -> MergeSource {
        match self {
            Self::Table(table_slice) => Self::Table(table_slice.into_reversed()),
            Self::Merge(merged) => Self::Merge(Box::new(merged.into_reversed())),
        }
    }

    fn slice(self, bounds: Bounds) -> TCResult<Merged> {
        match self {
            Self::Table(table) => table.slice(bounds),
            Self::Merge(merged) => merged.slice(bounds),
        }
    }

    pub async fn stream_slice<'a>(
        &'a self,
        txn_id: &'a TxnId,
        bounds: Bounds,
        reverse: bool,
    ) -> TCResult<TCTryStream<'a, Vec<Value>>> {
        match self {
            Self::Table(table) => table.stream_slice(txn_id, bounds, reverse).await,
            Self::Merge(merged) => merged.stream_slice(txn_id, bounds, reverse).await,
        }
    }

    fn source(&'_ self) -> &'_ TableIndex {
        match self {
            Self::Table(table_slice) => table_slice.source(),
            Self::Merge(merged) => merged.source(),
        }
    }

    fn into_source(self) -> TableIndex {
        match self {
            Self::Table(table_slice) => table_slice.into_source(),
            Self::Merge(merged) => merged.into_source(),
        }
    }

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()> {
        match self {
            Self::Table(table) => table.validate_bounds(bounds),
            Self::Merge(merged) => merged.validate_bounds(bounds),
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

impl fmt::Display for MergeSource {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Table(table) => write!(f, "MergeSource::Table({})", table),
            Self::Merge(merged) => write!(f, "MergeSource::Merge({})", merged),
        }
    }
}

#[derive(Clone)]
pub struct Merged {
    key_columns: Vec<Column>,
    left: MergeSource,
    right: IndexSlice,
    bounds: Bounds,
    keys: Selection<IndexSlice>,
}

impl Merged {
    pub fn new(left: MergeSource, right: IndexSlice) -> TCResult<Merged> {
        debug!("Merged::new({}, {})", &left, &right);
        let key_columns = left.key().to_vec();
        let key_names: Vec<Id> = key_columns.iter().map(|c| c.name()).cloned().collect();
        let keys = right.clone().select(key_names)?;

        left.source()
            .merge_bounds(vec![left.bounds().clone(), right.bounds().clone()])
            .map(|bounds| Merged {
                key_columns,
                left,
                right,
                bounds,
                keys,
            })
    }

    pub async fn stream_slice<'a>(
        &'a self,
        txn_id: &'a TxnId,
        bounds: Bounds,
        reverse: bool,
    ) -> TCResult<TCTryStream<'a, Vec<Value>>> {
        let bounds = self
            .source()
            .merge_bounds(vec![self.bounds.clone(), bounds])?;

        self.source().stream_slice(txn_id, bounds, reverse).await
    }

    fn into_reversed(self) -> Merged {
        let key_names = self
            .key_columns
            .iter()
            .map(|col| col.name())
            .cloned()
            .collect();

        let keys = Selection {
            source: self.right.clone().into_reversed(),
            schema: self.keys.schema.clone(),
            columns: key_names,
            indices: self.keys.indices.clone(),
        };

        Merged {
            key_columns: self.key_columns.to_vec(),
            left: self.left.into_reversed(),
            right: self.right.into_reversed(),
            bounds: self.bounds.clone(),
            keys,
        }
    }

    fn source(&'_ self) -> &'_ TableIndex {
        self.left.source()
    }

    fn into_source(self) -> TableIndex {
        self.left.into_source()
    }
}

impl Instance for Merged {
    type Class = TableType;

    fn class(&self) -> Self::Class {
        Self::Class::Merge
    }
}

#[async_trait]
impl TableInstance for Merged {
    type OrderBy = Self;
    type Reverse = Self;
    type Slice = Self;

    fn into_table(self) -> Table {
        Table::Merge(self.into())
    }

    async fn delete(&self, txn_id: &TxnId) -> TCResult<()> {
        let schema: IndexSchema = (self.key().to_vec(), self.values().to_vec()).into();

        let rows = self.stream(txn_id).await?;

        rows.map(|row| row.and_then(|row| schema.row_from_values(row)))
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
        self.left.key()
    }

    fn values(&'_ self) -> &'_ [Column] {
        match &self.left {
            MergeSource::Table(table) => table.values(),
            MergeSource::Merge(merged) => merged.values(),
        }
    }

    fn order_by(self, columns: Vec<Id>, reverse: bool) -> TCResult<Self::OrderBy> {
        match self.left {
            MergeSource::Merge(merged) => merged.order_by(columns, reverse),
            MergeSource::Table(table_slice) => table_slice.order_by(columns, reverse),
        }
    }

    fn reversed(self) -> TCResult<Self::Reverse> {
        Ok(self.into_reversed())
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        let bounds = self
            .source()
            .merge_bounds(vec![self.bounds.clone(), bounds])?;

        self.left.into_source().slice(bounds)
    }

    async fn stream<'a>(&'a self, txn_id: &'a TxnId) -> TCResult<TCTryStream<'a, Vec<Value>>> {
        debug!("Merged::stream from right {}", &self.right);
        debug!("left is {}", &self.left);

        let key_columns = &self.key_columns;
        let keys = self.keys.stream(txn_id).await?;

        let left = &self.left;
        let rows = keys
            .map_ok(move |key| Bounds::from_key(key, key_columns))
            .try_filter(move |bounds| future::ready(left.validate_bounds(bounds).is_ok()))
            .and_then(move |bounds| Box::pin(left.stream_slice(txn_id, bounds, false)))
            .try_flatten();

        let rows: TCTryStream<'a, Vec<Value>> = Box::pin(rows);
        Ok(rows)
    }

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()> {
        let bounds = self
            .source()
            .merge_bounds(vec![self.bounds.clone(), bounds.clone()])?;

        self.source().validate_bounds(&bounds)
    }

    fn validate_order(&self, order: &[Id]) -> TCResult<()> {
        match &self.left {
            MergeSource::Merge(merge) => merge.validate_order(order),
            MergeSource::Table(table) => table.validate_order(order),
        }
    }

    async fn update(&self, txn: &Txn, value: Row) -> TCResult<()> {
        let schema: IndexSchema = (self.key().to_vec(), self.values().to_vec()).into();

        let rows = self.stream(txn.id()).await?;

        rows.map(|row| row.and_then(|row| schema.row_from_values(row)))
            .map_ok(|row| self.update_row(txn.id(), row, value.clone()))
            .try_buffer_unordered(2)
            .try_fold((), |_, _| future::ready(Ok(())))
            .await
    }

    async fn update_row(&self, txn_id: &TxnId, row: Row, value: Row) -> TCResult<()> {
        match &self.left {
            MergeSource::Table(table) => table.update_row(txn_id, row, value).await,
            MergeSource::Merge(merged) => merged.update_row(txn_id, row, value).await,
        }
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

impl From<Merged> for Collection {
    fn from(merge: Merged) -> Collection {
        Collection::Table(merge.into_table())
    }
}

impl fmt::Display for Merged {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Merged({}, {})", &self.left, &self.right)
    }
}

#[derive(Clone)]
pub struct Selection<T> {
    source: T,
    schema: IndexSchema,
    columns: Vec<Id>,
    indices: Vec<usize>,
}

impl<T: TableInstance> Selection<T> {
    fn into_table_selection(self) -> Selection<Table> {
        let source = self.source.into_table();

        Selection {
            source,
            schema: self.schema,
            columns: self.columns,
            indices: self.indices,
        }
    }
}

impl<T: TableInstance> Selection<T> {
    pub fn new(source: T, columns: Vec<Id>) -> TCResult<Self> {
        let column_set: HashSet<&Id> = columns.iter().collect();
        if column_set.len() != columns.len() {
            return Err(error::bad_request(
                "Tried to select duplicate column",
                Value::from_iter(columns.into_iter()),
            ));
        }

        let mut indices: Vec<usize> = Vec::with_capacity(columns.len());
        let mut schema: Vec<Column> = Vec::with_capacity(columns.len());

        let source_columns = [source.key(), source.values()].concat();
        let source_indices: HashMap<&Id, usize> = source_columns
            .iter()
            .enumerate()
            .map(|(i, col)| (col.name(), i))
            .collect();

        for name in columns.iter() {
            let index = *source_indices
                .get(name)
                .ok_or(error::not_found(format!("Column {}", name)))?;

            indices.push(index);
            schema.push(source_columns[index].clone());
        }

        let schema = (vec![], schema).into();
        Ok(Selection {
            source,
            schema,
            columns,
            indices,
        })
    }
}

impl<T: Clone + Send + Sync> Instance for Selection<T> {
    type Class = TableType;

    fn class(&self) -> Self::Class {
        Self::Class::Selection
    }
}

#[async_trait]
impl<T: TableInstance> TableInstance for Selection<T> {
    type OrderBy = Selection<<T as TableInstance>::OrderBy>;
    type Reverse = Selection<<T as TableInstance>::Reverse>;
    type Slice = Selection<<T as TableInstance>::Slice>;

    fn into_table(self) -> Table {
        let this = Selection {
            source: self.source.into_table(),
            schema: self.schema,
            columns: self.columns,
            indices: self.indices,
        };

        Table::Selection(Box::new(this.into()))
    }

    async fn count(&self, txn_id: &TxnId) -> TCResult<u64> {
        self.source.clone().count(txn_id).await
    }

    fn key(&'_ self) -> &'_ [Column] {
        self.schema.key()
    }

    fn values(&'_ self) -> &'_ [Column] {
        self.schema.values()
    }

    fn order_by(self, order: Vec<Id>, reverse: bool) -> TCResult<Self::OrderBy> {
        self.validate_order(&order)?;

        let source = self.source.order_by(order, reverse)?;

        Ok(Selection {
            source,
            schema: self.schema,
            columns: self.columns,
            indices: self.indices,
        })
    }

    fn reversed(self) -> TCResult<Self::Reverse> {
        self.source.reversed()?.select(self.columns.to_vec())
    }

    async fn stream<'a>(&'a self, txn_id: &'a TxnId) -> TCResult<TCTryStream<'a, Vec<Value>>> {
        let indices = self.indices.to_vec();
        let selected = self.source.stream(txn_id).await?.map_ok(move |row| {
            let selection: Vec<Value> = indices.iter().map(|i| row[*i].clone()).collect();
            selection
        });
        let selected: TCTryStream<'a, Vec<Value>> = Box::pin(selected);
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

impl<T: TableInstance> From<Selection<T>> for Collection {
    fn from(selection: Selection<T>) -> Collection {
        Collection::Table(selection.into_table())
    }
}

#[derive(Clone)]
pub struct TableSlice {
    table: TableIndex,
    slice: IndexSlice,
}

impl TableSlice {
    pub fn new(table: TableIndex, bounds: Bounds) -> TCResult<TableSlice> {
        table.validate_bounds(&bounds)?;

        let index = table.supporting_index(&bounds)?;
        let slice = index.slice(bounds.clone())?;

        debug!("TableSlice::new w/bounds {}", bounds);
        Ok(TableSlice { table, slice })
    }

    pub fn bounds(&'_ self) -> &'_ Bounds {
        self.slice.bounds()
    }

    pub fn index_slice(self, bounds: Bounds) -> TCResult<IndexSlice> {
        self.slice.slice_index(bounds)
    }

    pub fn source(&'_ self) -> &'_ TableIndex {
        &self.table
    }

    pub fn into_source(self) -> TableIndex {
        self.table
    }

    pub async fn stream_slice<'a>(
        &'a self,
        txn_id: &'a TxnId,
        bounds: Bounds,
        reverse: bool,
    ) -> TCResult<TCTryStream<'a, Vec<Value>>> {
        self.slice.stream_slice(txn_id, bounds, reverse).await
    }

    fn into_reversed(self) -> TableSlice {
        TableSlice {
            table: self.table,
            slice: self.slice.into_reversed(),
        }
    }
}

impl Instance for TableSlice {
    type Class = TableType;

    fn class(&self) -> Self::Class {
        Self::Class::TableSlice
    }
}

#[async_trait]
impl TableInstance for TableSlice {
    type OrderBy = Merged;
    type Reverse = TableSlice;
    type Slice = Merged;

    fn into_table(self) -> Table {
        Table::TableSlice(self.into())
    }

    async fn count(&self, txn_id: &TxnId) -> TCResult<u64> {
        self.slice.count(txn_id).await
    }

    async fn delete(&self, txn_id: &TxnId) -> TCResult<()> {
        let schema: IndexSchema = (self.key().to_vec(), self.values().to_vec()).into();

        let rows = self.stream(&txn_id).await?;

        rows.map(|row| row.and_then(|row| schema.row_from_values(row)))
            .map_ok(|row| self.delete_row(&txn_id, row))
            .try_buffer_unordered(2)
            .fold(Ok(()), |_, r| future::ready(r))
            .await
    }

    async fn delete_row(&self, txn_id: &TxnId, row: Row) -> TCResult<()> {
        self.source().delete_row(txn_id, row).await
    }

    fn key(&'_ self) -> &'_ [Column] {
        self.source().key()
    }

    fn values(&'_ self) -> &'_ [Column] {
        self.source().values()
    }

    fn order_by(self, order: Vec<Id>, reverse: bool) -> TCResult<Self::OrderBy> {
        let bounds = self.slice.bounds;
        let table = self.table.order_by(order, reverse)?;
        table.slice(bounds)
    }

    fn reversed(self) -> TCResult<Self::Reverse> {
        Ok(Self {
            table: self.table,
            slice: self.slice.into_reversed(),
        })
    }

    fn slice(self, bounds: Bounds) -> TCResult<Merged> {
        let bounds = self
            .source()
            .merge_bounds(vec![self.slice.bounds().clone(), bounds])?;
        self.validate_bounds(&bounds)?;
        self.into_source().slice(bounds)
    }

    async fn stream<'a>(&'a self, txn_id: &'a TxnId) -> TCResult<TCTryStream<'a, Vec<Value>>> {
        debug!("TableSlice::stream");
        self.slice.stream(txn_id).await
    }

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()> {
        debug!("Table::validate_bounds {}", bounds);

        let index = self.source().supporting_index(self.slice.bounds())?;
        index
            .validate_slice_bounds(self.slice.bounds().clone(), bounds.clone())
            .map(|_| ())
    }

    fn validate_order(&self, order: &[Id]) -> TCResult<()> {
        self.source().validate_order(order)
    }

    async fn update(&self, txn: &Txn, value: Row) -> TCResult<()> {
        let schema: IndexSchema = (self.key().to_vec(), self.values().to_vec()).into();

        let rows = self.stream(txn.id()).await?;

        rows.map(|row| row.and_then(|row| schema.row_from_values(row)))
            .map_ok(|row| self.update_row(txn.id(), row, value.clone()))
            .try_buffer_unordered(2)
            .try_fold((), |_, _| future::ready(Ok(())))
            .await?;

        Ok(())
    }

    async fn update_row(&self, txn_id: &TxnId, row: Row, value: Row) -> TCResult<()> {
        self.source().update_row(txn_id, row, value).await
    }
}

#[async_trait]
impl Transact for TableSlice {
    async fn commit(&self, txn_id: &TxnId) {
        self.slice.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.slice.rollback(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.slice.finalize(txn_id).await
    }
}

impl From<TableSlice> for Collection {
    fn from(slice: TableSlice) -> Collection {
        Collection::Table(slice.into_table())
    }
}

impl fmt::Display for TableSlice {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "TableSlice with bounds {}", self.slice.bounds())
    }
}
