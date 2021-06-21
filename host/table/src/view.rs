use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;
use std::marker::PhantomData;

use async_trait::async_trait;
use futures::future::{self, TryFutureExt};
use futures::stream::{StreamExt, TryStreamExt};
use log::debug;

use tc_btree::{BTreeFile, BTreeInstance, Node};
use tc_error::*;
use tc_transact::fs::{Dir, File};
use tc_transact::{Transaction, TxnId};
use tc_value::Value;
use tcgeneric::{GroupStream, Id, Instance, TCTryStream};

use super::index::TableIndex;
use super::{Bounds, Column, IndexSchema, Row, Table, TableInstance, TableSchema, TableType};

const ERR_AGGREGATE_SLICE: &str = "Table aggregate does not support slicing. \
Consider aggregating a slice of the source table.";
const ERR_AGGREGATE_NESTED: &str = "It doesn't make sense to aggregate an aggregate table view. \
Consider aggregating the source table directly.";

const ERR_LIMITED_ORDER: &str = "Cannot order a limited selection. \
Consider ordering the source or indexing the selection.";
const ERR_LIMITED_REVERSE: &str = "Cannot reverse a limited selection. \
Consider reversing a slice before limiting";

#[derive(Clone)]
pub struct Aggregate<F, D, Txn, T> {
    source: Selection<F, D, Txn, T>,
    file: PhantomData<F>,
}

impl<F, D, Txn, T> Instance for Aggregate<F, D, Txn, T>
where
    Self: Send + Sync,
{
    type Class = TableType;

    fn class(&self) -> Self::Class {
        TableType::Aggregate
    }
}

#[async_trait]
impl<F: File<Node>, D: Dir, Txn: Transaction<D>, T: TableInstance<F, D, Txn>>
    TableInstance<F, D, Txn> for Aggregate<F, D, Txn, T>
{
    type OrderBy = Aggregate<F, D, Txn, <T as TableInstance<F, D, Txn>>::OrderBy>;
    type Reverse = Aggregate<F, D, Txn, <T as TableInstance<F, D, Txn>>::Reverse>;
    type Slice = Table<F, D, Txn>;

    fn group_by(self, _columns: Vec<Id>) -> TCResult<Aggregate<F, D, Txn, Self::OrderBy>> {
        Err(TCError::unsupported(ERR_AGGREGATE_NESTED))
    }

    fn key(&self) -> &[Column] {
        self.source.key()
    }

    fn values(&self) -> &'_ [Column] {
        self.source.values()
    }

    fn schema(&self) -> TableSchema {
        self.source.schema()
    }

    fn order_by(self, columns: Vec<Id>, reverse: bool) -> TCResult<Self::OrderBy> {
        let source = self.source.order_by(columns, reverse)?;
        Ok(Aggregate {
            source,
            file: PhantomData,
        })
    }

    fn reversed(self) -> TCResult<Self::Reverse> {
        let phantom_file = self.file;
        self.source.reversed().map(|source| Aggregate {
            source,
            file: phantom_file,
        })
    }

    async fn rows<'a>(self, txn_id: TxnId) -> TCResult<TCTryStream<'a, Vec<Value>>> {
        let grouped = self.source.rows(txn_id).map_ok(GroupStream::from).await?;
        let grouped: TCTryStream<Vec<Value>> = Box::pin(grouped);
        Ok(grouped)
    }

    fn validate_bounds(&self, _bounds: &Bounds) -> TCResult<()> {
        Err(TCError::unsupported(ERR_AGGREGATE_SLICE))
    }

    fn validate_order(&self, order: &[Id]) -> TCResult<()> {
        self.source.validate_order(order)
    }
}

impl<F: File<Node>, D: Dir, Txn: Transaction<D>, T: TableInstance<F, D, Txn>>
    From<Aggregate<F, D, Txn, T>> for Table<F, D, Txn>
{
    fn from(aggregate: Aggregate<F, D, Txn, T>) -> Table<F, D, Txn> {
        let source = Selection {
            source: aggregate.source.source.into(),
            schema: aggregate.source.schema,
            columns: aggregate.source.columns,
            indices: aggregate.source.indices,
            phantom: Phantom::default(),
        };

        Table::Aggregate(Box::new(Aggregate {
            source,
            file: aggregate.file,
        }))
    }
}

#[derive(Clone)]
pub struct IndexSlice<F, D, Txn> {
    source: BTreeFile<F, D, Txn>,
    schema: IndexSchema,
    bounds: Bounds,
    range: tc_btree::Range,
    reverse: bool,
}

impl<F: File<Node>, D: Dir, Txn: Transaction<D>> IndexSlice<F, D, Txn> {
    pub fn all(source: BTreeFile<F, D, Txn>, schema: IndexSchema, reverse: bool) -> Self {
        IndexSlice {
            source,
            schema,
            bounds: Bounds::default(),
            range: tc_btree::Range::default(),
            reverse,
        }
    }

    pub fn new(
        source: BTreeFile<F, D, Txn>,
        schema: IndexSchema,
        bounds: Bounds,
    ) -> TCResult<Self> {
        debug!("IndexSlice::new with bounds {}", bounds);
        let columns = schema.columns();

        assert!(source.schema() == &columns[..]);

        let bounds = bounds.validate(&columns)?;
        let range = bounds.clone().into_btree_range(&columns)?;

        debug!("bounds {} == range {:?}", bounds, range);
        assert_eq!(bounds.len(), range.len());

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

    pub fn into_reversed(mut self) -> Self {
        self.reverse = !self.reverse;
        self
    }

    pub async fn is_empty(&self, txn: &Txn) -> TCResult<bool> {
        self.source
            .clone()
            .slice(self.range.clone(), self.reverse)?
            .is_empty(*txn.id())
            .await
    }

    pub fn slice_index(self, bounds: Bounds) -> TCResult<Self> {
        let columns = self.schema().columns();
        let outer = bounds.clone().into_btree_range(&columns)?;
        let inner = bounds.clone().into_btree_range(&columns)?;

        if outer.contains(&inner, self.source.collator()) {
            let mut slice = self;
            slice.bounds = bounds;
            Ok(slice)
        } else {
            Err(TCError::bad_request(
                &format!("IndexSlice with bounds {} does not contain", self.bounds),
                bounds,
            ))
        }
    }

    pub async fn slice_rows<'a>(
        self,
        txn_id: TxnId,
        bounds: Bounds,
        reverse: bool,
    ) -> TCResult<TCTryStream<'a, Vec<Value>>> {
        let reverse = self.reverse ^ reverse;
        let range = bounds.into_btree_range(&self.schema.columns())?;
        self.source.slice(range, reverse)?.keys(txn_id).await
    }
}

impl<F, D, Txn> Instance for IndexSlice<F, D, Txn>
where
    Self: Send + Sync,
{
    type Class = TableType;

    fn class(&self) -> Self::Class {
        Self::Class::IndexSlice
    }
}

#[async_trait]
impl<F, D, Txn> TableInstance<F, D, Txn> for IndexSlice<F, D, Txn>
where
    F: File<Node>,
    D: Dir,
    Txn: Transaction<D>,
{
    type OrderBy = Self;
    type Reverse = Self;
    type Slice = Table<F, D, Txn>;

    async fn count(self, txn_id: TxnId) -> TCResult<u64> {
        self.source
            .slice(self.range.clone(), false)?
            .count(txn_id)
            .await
    }

    async fn delete(&self, txn_id: TxnId) -> TCResult<()> {
        self.source
            .clone()
            .slice(self.range.clone(), false)?
            .delete(txn_id)
            .await
    }

    fn key(&self) -> &[Column] {
        self.schema.key()
    }

    fn values(&self) -> &[Column] {
        self.schema.values()
    }

    fn schema(&self) -> TableSchema {
        self.schema.clone().into()
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

    async fn rows<'a>(self, txn_id: TxnId) -> TCResult<TCTryStream<'a, Vec<Value>>> {
        self.source
            .slice(self.range.clone(), self.reverse)?
            .keys(txn_id)
            .await
    }

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()> {
        let schema = self.schema();
        let outer = bounds.clone().into_btree_range(&schema.columns())?;
        let inner = bounds.clone().into_btree_range(&schema.columns())?;

        if outer.contains(&inner, self.source.collator()) {
            Ok(())
        } else {
            Err(TCError::bad_request(
                "IndexSlice does not support bounds",
                bounds,
            ))
        }
    }

    fn validate_order(&self, order: &[Id]) -> TCResult<()> {
        if self.schema.starts_with(order) {
            Ok(())
        } else {
            Err(TCError::bad_request(
                &format!("Index with schema {} does not support order", &self.schema),
                Value::from_iter(order.to_vec()),
            ))
        }
    }

    async fn update(&self, _txn: &Txn, _value: Row) -> TCResult<()> {
        unimplemented!()
    }
}

impl<F, D, Txn> From<IndexSlice<F, D, Txn>> for Table<F, D, Txn> {
    fn from(slice: IndexSlice<F, D, Txn>) -> Self {
        Self::IndexSlice(slice)
    }
}

#[derive(Clone)]
pub struct Limited<F, D, Txn> {
    source: Table<F, D, Txn>,
    limit: u64,
}

impl<F, D, Txn> Limited<F, D, Txn> {
    pub fn new(source: Table<F, D, Txn>, limit: u64) -> Self {
        Limited { source, limit }
    }
}

impl<F, D, Txn> Instance for Limited<F, D, Txn>
where
    Self: Send + Sync,
{
    type Class = TableType;

    fn class(&self) -> Self::Class {
        Self::Class::Limit
    }
}

#[async_trait]
impl<F: File<Node>, D: Dir, Txn: Transaction<D>> TableInstance<F, D, Txn> for Limited<F, D, Txn> {
    type OrderBy = Table<F, D, Txn>;
    type Reverse = Table<F, D, Txn>;
    type Slice = Table<F, D, Txn>;

    async fn count(self, txn_id: TxnId) -> TCResult<u64> {
        let source_count = self.source.count(txn_id).await?;
        Ok(u64::min(source_count, self.limit as u64))
    }

    async fn delete(&self, txn_id: TxnId) -> TCResult<()> {
        let source = &self.source;
        let schema: IndexSchema = (source.key().to_vec(), source.values().to_vec()).into();

        let rows = self.clone().rows(txn_id).await?;

        rows.map(|row| row.and_then(|row| schema.row_from_values(row)))
            .map_ok(|row| source.delete_row(txn_id, row))
            .try_buffer_unordered(2)
            .try_fold((), |_, _| future::ready(Ok(())))
            .await
    }

    fn key(&self) -> &[Column] {
        self.source.key()
    }

    fn values(&self) -> &[Column] {
        self.source.values()
    }

    fn schema(&self) -> TableSchema {
        self.source.schema()
    }

    fn order_by(self, _order: Vec<Id>, _reverse: bool) -> TCResult<Table<F, D, Txn>> {
        Err(TCError::unsupported(ERR_LIMITED_ORDER))
    }

    fn reversed(self) -> TCResult<Table<F, D, Txn>> {
        Err(TCError::unsupported(ERR_LIMITED_REVERSE))
    }

    async fn rows<'a>(self, txn_id: TxnId) -> TCResult<TCTryStream<'a, Vec<Value>>> {
        let rows = self.source.rows(txn_id).await?;
        let rows: TCTryStream<Vec<Value>> = Box::pin(rows.take(self.limit as usize));
        Ok(rows)
    }

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()> {
        self.source.validate_bounds(bounds)
    }

    fn validate_order(&self, _order: &[Id]) -> TCResult<()> {
        Err(TCError::unsupported(ERR_LIMITED_ORDER))
    }

    async fn update(&self, txn: &Txn, value: Row) -> TCResult<()> {
        let source = &self.source;
        let schema: IndexSchema = (source.key().to_vec(), source.values().to_vec()).into();

        let rows = self.clone().rows(*txn.id()).await?;

        rows.map(|row| row.and_then(|row| schema.row_from_values(row)))
            .map_ok(|row| source.update_row(*txn.id(), row, value.clone()))
            .try_buffer_unordered(2)
            .try_fold((), |_, _| future::ready(Ok(())))
            .await?;

        Ok(())
    }
}

impl<F, D, Txn> From<Limited<F, D, Txn>> for Table<F, D, Txn> {
    fn from(limited: Limited<F, D, Txn>) -> Self {
        Table::Limit(Box::new(limited))
    }
}

#[derive(Clone)]
pub enum MergeSource<F, D, Txn> {
    Table(TableSlice<F, D, Txn>),
    Merge(Box<Merged<F, D, Txn>>),
}

impl<F: File<Node>, D: Dir, Txn: Transaction<D>> MergeSource<F, D, Txn> {
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

    fn into_reversed(self) -> MergeSource<F, D, Txn> {
        match self {
            Self::Table(table_slice) => Self::Table(table_slice.into_reversed()),
            Self::Merge(merged) => Self::Merge(Box::new(merged.into_reversed())),
        }
    }

    pub async fn slice_rows<'a>(
        self,
        txn_id: TxnId,
        bounds: Bounds,
        reverse: bool,
    ) -> TCResult<TCTryStream<'a, Vec<Value>>> {
        match self {
            Self::Table(table) => table.slice_rows(txn_id, bounds, reverse).await,
            Self::Merge(merged) => merged.slice_rows(txn_id, bounds, reverse).await,
        }
    }

    fn source(&'_ self) -> &'_ TableIndex<F, D, Txn> {
        match self {
            Self::Table(table_slice) => table_slice.source(),
            Self::Merge(merged) => merged.source(),
        }
    }

    fn into_source(self) -> TableIndex<F, D, Txn> {
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

#[derive(Clone)]
pub struct Merged<F, D, Txn> {
    key_columns: Vec<Column>,
    left: MergeSource<F, D, Txn>,
    right: IndexSlice<F, D, Txn>,
    bounds: Bounds,
    keys: Selection<F, D, Txn, IndexSlice<F, D, Txn>>,
}

impl<F: File<Node>, D: Dir, Txn: Transaction<D>> Merged<F, D, Txn> {
    pub fn new(left: MergeSource<F, D, Txn>, right: IndexSlice<F, D, Txn>) -> TCResult<Self> {
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

    pub async fn slice_rows<'a>(
        self,
        txn_id: TxnId,
        bounds: Bounds,
        reverse: bool,
    ) -> TCResult<TCTryStream<'a, Vec<Value>>> {
        let bounds = self
            .source()
            .merge_bounds(vec![self.bounds.clone(), bounds])?;

        self.into_source().slice_rows(txn_id, bounds, reverse).await
    }

    fn into_reversed(self) -> Self {
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
            phantom: Phantom::default(),
        };

        Merged {
            key_columns: self.key_columns.to_vec(),
            left: self.left.into_reversed(),
            right: self.right.into_reversed(),
            bounds: self.bounds.clone(),
            keys,
        }
    }

    fn source(&'_ self) -> &'_ TableIndex<F, D, Txn> {
        self.left.source()
    }

    fn into_source(self) -> TableIndex<F, D, Txn> {
        self.left.into_source()
    }
}

impl<F, D, Txn> Instance for Merged<F, D, Txn>
where
    Self: Send + Sync,
{
    type Class = TableType;

    fn class(&self) -> Self::Class {
        Self::Class::Merge
    }
}

#[async_trait]
impl<F: File<Node>, D: Dir, Txn: Transaction<D>> TableInstance<F, D, Txn> for Merged<F, D, Txn> {
    type OrderBy = Self;
    type Reverse = Self;
    type Slice = Self;

    async fn delete(&self, txn_id: TxnId) -> TCResult<()> {
        let schema: IndexSchema = (self.key().to_vec(), self.values().to_vec()).into();

        let rows = self.clone().rows(txn_id).await?;

        rows.map(|row| row.and_then(|row| schema.row_from_values(row)))
            .map_ok(|row| self.delete_row(txn_id, row))
            .try_buffer_unordered(2)
            .try_fold((), |_, _| future::ready(Ok(())))
            .await
    }

    async fn delete_row(&self, txn_id: TxnId, row: Row) -> TCResult<()> {
        match &self.left {
            MergeSource::Table(table) => table.delete_row(txn_id, row).await,
            MergeSource::Merge(merged) => merged.delete_row(txn_id, row).await,
        }
    }

    fn key(&self) -> &[Column] {
        self.left.key()
    }

    fn values(&self) -> &[Column] {
        match &self.left {
            MergeSource::Table(table) => table.values(),
            MergeSource::Merge(merged) => merged.values(),
        }
    }

    fn schema(&self) -> TableSchema {
        match &self.left {
            MergeSource::Table(table) => table.schema(),
            MergeSource::Merge(merged) => merged.schema(),
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

    async fn rows<'a>(self, txn_id: TxnId) -> TCResult<TCTryStream<'a, Vec<Value>>> {
        let left = self.left;
        let left_clone = left.clone();
        let key_columns = self.key_columns;
        let keys = self.keys.clone().rows(txn_id).await?;

        let rows = keys
            .map_ok(move |key| Bounds::from_key(key, &key_columns))
            .try_filter(move |bounds| future::ready(left.validate_bounds(bounds).is_ok()))
            .and_then(move |bounds| Box::pin(left_clone.clone().slice_rows(txn_id, bounds, false)))
            .try_flatten();

        let rows: TCTryStream<Vec<Value>> = Box::pin(rows);
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

        let rows = self.clone().rows(*txn.id()).await?;

        rows.map(|row| row.and_then(|row| schema.row_from_values(row)))
            .map_ok(|row| self.update_row(*txn.id(), row, value.clone()))
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

impl<F, D, Txn> From<Merged<F, D, Txn>> for Table<F, D, Txn> {
    fn from(merged: Merged<F, D, Txn>) -> Self {
        Self::Merge(merged)
    }
}

#[derive(Clone)]
pub struct Selection<F, D, Txn, T> {
    source: T,
    schema: IndexSchema,
    columns: Vec<Id>,
    indices: Vec<usize>,
    phantom: Phantom<F, D, Txn>,
}

impl<F: File<Node>, D: Dir, Txn: Transaction<D>, T: TableInstance<F, D, Txn>>
    Selection<F, D, Txn, T>
{
    pub fn new(source: T, columns: Vec<Id>) -> TCResult<Self> {
        let column_set: HashSet<&Id> = columns.iter().collect();
        if column_set.len() != columns.len() {
            return Err(TCError::bad_request(
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
                .ok_or(TCError::not_found(format!("Column {}", name)))?;

            indices.push(index);
            schema.push(source_columns[index].clone());
        }

        let schema = (vec![], schema).into();
        Ok(Selection {
            source,
            schema,
            columns,
            indices,
            phantom: Phantom::default(),
        })
    }
}

impl<F, D, Txn, T> Instance for Selection<F, D, Txn, T>
where
    Self: Send + Sync,
{
    type Class = TableType;

    fn class(&self) -> TableType {
        TableType::Selection
    }
}

#[async_trait]
impl<F, D, Txn, T> TableInstance<F, D, Txn> for Selection<F, D, Txn, T>
where
    F: File<Node>,
    D: Dir,
    Txn: Transaction<D>,
    T: TableInstance<F, D, Txn>,
{
    type OrderBy = Selection<F, D, Txn, <T as TableInstance<F, D, Txn>>::OrderBy>;
    type Reverse = Selection<F, D, Txn, <T as TableInstance<F, D, Txn>>::Reverse>;
    type Slice = Selection<F, D, Txn, <T as TableInstance<F, D, Txn>>::Slice>;

    async fn count(self, txn_id: TxnId) -> TCResult<u64> {
        self.source.count(txn_id).await
    }

    fn key(&self) -> &[Column] {
        self.schema.key()
    }

    fn values(&self) -> &[Column] {
        self.schema.values()
    }

    fn schema(&self) -> TableSchema {
        let source = self.source.schema();
        let source = source.primary();

        let select = |columns: &[Column]| {
            columns
                .iter()
                .filter(|col| self.columns.contains(&col.name))
                .cloned()
                .collect()
        };

        let key = select(source.key());
        let values = select(source.values());
        IndexSchema::from((key, values)).into()
    }

    fn order_by(self, order: Vec<Id>, reverse: bool) -> TCResult<Self::OrderBy> {
        self.validate_order(&order)?;

        let source = self.source.order_by(order, reverse)?;

        Ok(Selection {
            source,
            schema: self.schema,
            columns: self.columns,
            indices: self.indices,
            phantom: Phantom::default(),
        })
    }

    fn reversed(self) -> TCResult<Self::Reverse> {
        self.source.reversed()?.select(self.columns.to_vec())
    }

    async fn rows<'a>(self, txn_id: TxnId) -> TCResult<TCTryStream<'a, Vec<Value>>> {
        let indices = self.indices.to_vec();
        let selected = self.source.rows(txn_id).await?.map_ok(move |row| {
            let selection: Vec<Value> = indices.iter().map(|i| row[*i].clone()).collect();
            selection
        });

        let selected: TCTryStream<Vec<Value>> = Box::pin(selected);
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
            return Err(TCError::bad_request(
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
            return Err(TCError::bad_request(
                "Tried to order by unselected columns",
                unknown.join(", "),
            ));
        }

        self.source.validate_order(order)
    }
}

impl<F, D, Txn, T> From<Selection<F, D, Txn, T>> for Table<F, D, Txn>
where
    F: File<Node>,
    D: Dir,
    Txn: Transaction<D>,
    T: TableInstance<F, D, Txn>,
{
    fn from(selection: Selection<F, D, Txn, T>) -> Self {
        Table::Selection(Box::new(Selection {
            source: selection.source.into(),
            schema: selection.schema,
            columns: selection.columns,
            indices: selection.indices,
            phantom: Phantom::default(),
        }))
    }
}

#[derive(Clone)]
pub struct TableSlice<F, D, Txn> {
    table: TableIndex<F, D, Txn>,
    slice: IndexSlice<F, D, Txn>,
}

impl<F: File<Node>, D: Dir, Txn: Transaction<D>> TableSlice<F, D, Txn> {
    pub fn new(table: TableIndex<F, D, Txn>, bounds: Bounds) -> TCResult<TableSlice<F, D, Txn>> {
        table.validate_bounds(&bounds)?;

        let index = table.supporting_index(&bounds)?;
        let slice = index.slice(bounds.clone())?;

        debug!("TableSlice::new w/bounds {}", bounds);
        Ok(TableSlice { table, slice })
    }

    pub fn bounds(&'_ self) -> &'_ Bounds {
        self.slice.bounds()
    }

    pub fn index_slice(self, bounds: Bounds) -> TCResult<IndexSlice<F, D, Txn>> {
        self.slice.slice_index(bounds)
    }

    pub fn source(&'_ self) -> &'_ TableIndex<F, D, Txn> {
        &self.table
    }

    pub fn into_source(self) -> TableIndex<F, D, Txn> {
        self.table
    }

    pub async fn slice_rows<'a>(
        self,
        txn_id: TxnId,
        bounds: Bounds,
        reverse: bool,
    ) -> TCResult<TCTryStream<'a, Vec<Value>>> {
        self.slice.slice_rows(txn_id, bounds, reverse).await
    }

    fn into_reversed(self) -> Self {
        TableSlice {
            table: self.table,
            slice: self.slice.into_reversed(),
        }
    }
}

impl<F, D, Txn> Instance for TableSlice<F, D, Txn>
where
    Self: Send + Sync,
{
    type Class = TableType;

    fn class(&self) -> Self::Class {
        Self::Class::TableSlice
    }
}

#[async_trait]
impl<F, D, Txn> TableInstance<F, D, Txn> for TableSlice<F, D, Txn>
where
    F: File<Node>,
    D: Dir,
    Txn: Transaction<D>,
{
    type OrderBy = Merged<F, D, Txn>;
    type Reverse = TableSlice<F, D, Txn>;
    type Slice = Merged<F, D, Txn>;

    async fn count(self, txn_id: TxnId) -> TCResult<u64> {
        self.slice.count(txn_id).await
    }

    async fn delete(&self, txn_id: TxnId) -> TCResult<()> {
        let schema: IndexSchema = (self.key().to_vec(), self.values().to_vec()).into();

        let rows = self.clone().rows(txn_id).await?;

        rows.map(|row| row.and_then(|row| schema.row_from_values(row)))
            .map_ok(|row| self.delete_row(txn_id, row))
            .try_buffer_unordered(2)
            .fold(Ok(()), |_, r| future::ready(r))
            .await
    }

    async fn delete_row(&self, txn_id: TxnId, row: Row) -> TCResult<()> {
        self.source().delete_row(txn_id, row).await
    }

    fn key(&self) -> &[Column] {
        self.source().key()
    }

    fn values(&self) -> &[Column] {
        self.source().values()
    }

    fn schema(&self) -> TableSchema {
        self.source().schema()
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

    fn slice(self, bounds: Bounds) -> TCResult<Merged<F, D, Txn>> {
        let bounds = self
            .source()
            .merge_bounds(vec![self.slice.bounds().clone(), bounds])?;

        self.validate_bounds(&bounds)?;

        self.into_source().slice(bounds)
    }

    async fn rows<'a>(self, txn_id: TxnId) -> TCResult<TCTryStream<'a, Vec<Value>>> {
        self.slice.rows(txn_id).await
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
        let txn_id = *txn.id();
        let schema: IndexSchema = (self.key().to_vec(), self.values().to_vec()).into();

        let rows = self.clone().rows(txn_id).await?;

        rows.map(|row| row.and_then(|row| schema.row_from_values(row)))
            .map_ok(|row| self.update_row(txn_id, row, value.clone()))
            .try_buffer_unordered(2)
            .try_fold((), |_, _| future::ready(Ok(())))
            .await?;

        Ok(())
    }

    async fn update_row(&self, txn_id: TxnId, row: Row, value: Row) -> TCResult<()> {
        self.source().update_row(txn_id, row, value).await
    }
}

impl<F, D, Txn> From<TableSlice<F, D, Txn>> for Table<F, D, Txn> {
    fn from(slice: TableSlice<F, D, Txn>) -> Self {
        Self::TableSlice(slice)
    }
}

pub fn group_by<F: File<Node>, D: Dir, Txn: Transaction<D>, T: TableInstance<F, D, Txn>>(
    source: T,
    columns: Vec<Id>,
) -> TCResult<Aggregate<F, D, Txn, <T as TableInstance<F, D, Txn>>::OrderBy>> {
    let source = source.order_by(columns.to_vec(), false)?;
    let source = source.select(columns)?;

    Ok(Aggregate {
        source,
        file: PhantomData,
    })
}

#[derive(Clone)]
struct Phantom<F, D, Txn> {
    file: PhantomData<F>,
    dir: PhantomData<D>,
    txn: PhantomData<Txn>,
}

impl<F: File<Node>, D: Dir, Txn: Transaction<D>> Default for Phantom<F, D, Txn> {
    fn default() -> Self {
        Self {
            file: PhantomData,
            dir: PhantomData,
            txn: PhantomData,
        }
    }
}
