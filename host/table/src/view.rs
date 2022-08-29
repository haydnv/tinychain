use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;
use std::marker::PhantomData;

use async_trait::async_trait;
use futures::future;
use futures::stream::{StreamExt, TryStreamExt};
use log::debug;

use tc_btree::{BTreeFile, BTreeInstance, Node};
use tc_error::*;
use tc_transact::fs::{DirLock, FileLock};
use tc_transact::{Transaction, TxnId};
use tc_value::Value;
use tcgeneric::{Id, Instance, TCBoxTryStream};

use super::index::TableIndex;
use super::{
    Bounds, Column, IndexSchema, Table, TableInstance, TableOrder, TableSchema, TableStream,
    TableType,
};

#[derive(Clone)]
pub struct IndexSlice<F, D, Txn> {
    source: BTreeFile<F, D, Txn>,
    schema: IndexSchema,
    bounds: Bounds,
    range: tc_btree::Range,
    reverse: bool,
}

impl<F: FileLock<Block = Node>, D: DirLock<File = F>, Txn: Transaction<D>> IndexSlice<F, D, Txn> {
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

        assert_eq!(source.schema(), &columns[..]);

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
    ) -> TCResult<TCBoxTryStream<'a, Vec<Value>>> {
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

impl<F, D, Txn> TableInstance for IndexSlice<F, D, Txn>
where
    F: FileLock<Block = Node>,
    D: DirLock<File = F>,
    Txn: Transaction<D>,
{
    fn key(&self) -> &[Column] {
        self.schema.key()
    }

    fn values(&self) -> &[Column] {
        self.schema.values()
    }

    fn schema(&self) -> TableSchema {
        self.schema.clone().into()
    }
}

impl<F, D, Txn> TableOrder for IndexSlice<F, D, Txn>
where
    F: FileLock<Block = Node>,
    D: DirLock<File = F>,
    Txn: Transaction<D>,
{
    type OrderBy = Self;
    type Reverse = Self;

    fn order_by(self, order: Vec<Id>, reverse: bool) -> TCResult<Self::OrderBy> {
        self.validate_order(&order)?;

        if reverse {
            self.reverse()
        } else {
            Ok(self.clone().into())
        }
    }

    fn reverse(self) -> TCResult<Self::Reverse> {
        Ok(self.into_reversed())
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
}

#[async_trait]
impl<F, D, Txn> TableStream for IndexSlice<F, D, Txn>
where
    F: FileLock<Block = Node>,
    D: DirLock<File = F>,
    Txn: Transaction<D>,
{
    type Limit = Limited<F, D, Txn>;
    type Selection = Selection<F, D, Txn, Self>;

    async fn count(self, txn_id: TxnId) -> TCResult<u64> {
        self.source
            .slice(self.range.clone(), false)?
            .count(txn_id)
            .await
    }

    fn limit(self, limit: u64) -> Self::Limit {
        Limited::new(self, limit)
    }

    fn select(self, columns: Vec<Id>) -> TCResult<Self::Selection> {
        Selection::new(self, columns)
    }

    async fn rows<'a>(self, txn_id: TxnId) -> TCResult<TCBoxTryStream<'a, Vec<Value>>> {
        self.source
            .slice(self.range.clone(), self.reverse)?
            .keys(txn_id)
            .await
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
    pub fn new<T: Into<Table<F, D, Txn>>>(source: T, limit: u64) -> Self {
        Limited {
            source: source.into(),
            limit,
        }
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

impl<F: FileLock<Block = Node>, D: DirLock<File = F>, Txn: Transaction<D>> TableInstance
    for Limited<F, D, Txn>
{
    fn key(&self) -> &[Column] {
        self.source.key()
    }

    fn values(&self) -> &[Column] {
        self.source.values()
    }

    fn schema(&self) -> TableSchema {
        self.source.schema()
    }
}

#[async_trait]
impl<F: FileLock<Block = Node>, D: DirLock<File = F>, Txn: Transaction<D>> TableStream
    for Limited<F, D, Txn>
{
    type Limit = Limited<F, D, Txn>;
    type Selection = Selection<F, D, Txn, Self>;

    async fn count(self, txn_id: TxnId) -> TCResult<u64> {
        let source_count = self.source.count(txn_id).await?;
        Ok(u64::min(source_count, self.limit as u64))
    }

    fn limit(self, limit: u64) -> Self::Limit {
        Limited::new(self, limit)
    }

    fn select(self, columns: Vec<Id>) -> TCResult<Self::Selection> {
        Selection::new(self, columns)
    }

    async fn rows<'a>(self, txn_id: TxnId) -> TCResult<TCBoxTryStream<'a, Vec<Value>>> {
        let rows = self.source.rows(txn_id).await?;
        let rows: TCBoxTryStream<Vec<Value>> = Box::pin(rows.take(self.limit as usize));
        Ok(rows)
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

impl<F: FileLock<Block = Node>, D: DirLock<File = F>, Txn: Transaction<D>> MergeSource<F, D, Txn> {
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

    fn reverse(self) -> TCResult<MergeSource<F, D, Txn>> {
        match self {
            Self::Table(table_slice) => table_slice.reverse().map(Self::Table),
            Self::Merge(merged) => merged.reverse().map(Box::new).map(Self::Merge),
        }
    }

    async fn slice_rows<'a>(
        self,
        txn_id: TxnId,
        bounds: Bounds,
        reverse: bool,
    ) -> TCResult<TCBoxTryStream<'a, Vec<Value>>> {
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
            Self::Table(table) => super::TableSlice::validate_bounds(table, bounds),
            Self::Merge(merged) => super::TableSlice::validate_bounds(&**merged, bounds),
        }
    }
}

/// A merge of multiple table indexes
#[derive(Clone)]
pub struct Merged<F, D, Txn> {
    left: MergeSource<F, D, Txn>,
    right: IndexSlice<F, D, Txn>,
    bounds: Bounds,
}

impl<F: FileLock<Block = Node>, D: DirLock<File = F>, Txn: Transaction<D>> Merged<F, D, Txn> {
    /// Create a new merge of the given `IndexSlice` with the given `MergeSource`.
    pub fn new(left: MergeSource<F, D, Txn>, right: IndexSlice<F, D, Txn>) -> TCResult<Self> {
        let bounds = left
            .source()
            .merge_bounds(vec![left.bounds().clone(), right.bounds().clone()])?;

        Ok(Self {
            left,
            right,
            bounds,
        })
    }

    fn source(&'_ self) -> &'_ TableIndex<F, D, Txn> {
        self.left.source()
    }

    fn into_source(self) -> TableIndex<F, D, Txn> {
        self.left.into_source()
    }

    /// Stream the rows within the given [`Bounds`] of this merge
    pub async fn slice_rows<'a>(
        self,
        txn_id: TxnId,
        bounds: Bounds,
        reverse: bool,
    ) -> TCResult<TCBoxTryStream<'a, Vec<Value>>> {
        let these_bounds = self.bounds;
        let source = self.left.into_source();
        let bounds = source.merge_bounds(vec![these_bounds, bounds])?;
        source.slice_rows(txn_id, bounds, reverse).await
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

impl<F: FileLock<Block = Node>, D: DirLock<File = F>, Txn: Transaction<D>> TableInstance
    for Merged<F, D, Txn>
{
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
}

#[async_trait]
impl<F: FileLock<Block = Node>, D: DirLock<File = F>, Txn: Transaction<D>> TableStream
    for Merged<F, D, Txn>
{
    type Limit = Limited<F, D, Txn>;
    type Selection = Selection<F, D, Txn, Self>;

    fn limit(self, limit: u64) -> Self::Limit {
        Limited::new(self, limit)
    }

    fn select(self, columns: Vec<Id>) -> TCResult<Self::Selection> {
        Selection::new(self, columns)
    }

    async fn rows<'a>(self, txn_id: TxnId) -> TCResult<TCBoxTryStream<'a, Vec<Value>>> {
        let key_columns = self.key().to_vec();
        let key_names = key_columns.iter().map(|col| &col.name).cloned().collect();
        let keys = self.right.select(key_names)?.rows(txn_id).await?;

        let left = self.left;
        let left_clone = left.clone();
        let merge = keys
            .map_ok(move |key| Bounds::from_key(key, &key_columns))
            .try_filter(move |bounds| future::ready(left.validate_bounds(bounds).is_ok()))
            .map_ok(move |bounds| Box::pin(left_clone.clone().slice_rows(txn_id, bounds, false)))
            .try_buffered(num_cpus::get())
            .try_flatten();

        Ok(Box::pin(merge))
    }
}

impl<F: FileLock<Block = Node>, D: DirLock<File = F>, Txn: Transaction<D>> TableOrder
    for Merged<F, D, Txn>
{
    type OrderBy = Self;
    type Reverse = Self;

    fn order_by(self, columns: Vec<Id>, reverse: bool) -> TCResult<Self::OrderBy> {
        match self.left {
            MergeSource::Merge(merged) => merged.order_by(columns, reverse),
            MergeSource::Table(table_slice) => table_slice.order_by(columns, reverse),
        }
    }

    fn reverse(self) -> TCResult<Self::Reverse> {
        Ok(Merged {
            left: self.left.reverse()?,
            right: self.right.reverse()?,
            bounds: self.bounds,
        })
    }

    fn validate_order(&self, order: &[Id]) -> TCResult<()> {
        match &self.left {
            MergeSource::Merge(merge) => merge.validate_order(order),
            MergeSource::Table(table) => table.validate_order(order),
        }
    }
}

impl<F: FileLock<Block = Node>, D: DirLock<File = F>, Txn: Transaction<D>> super::TableSlice
    for Merged<F, D, Txn>
{
    type Slice = Self;

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        let source = self.left.into_source();
        let bounds = source.merge_bounds(vec![self.bounds, bounds])?;
        source.slice(bounds)
    }

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()> {
        let bounds = self
            .source()
            .merge_bounds(vec![self.bounds.clone(), bounds.clone()])?;

        self.source().validate_bounds(&bounds)
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

    #[allow(dead_code)]
    phantom: Phantom<F, D, Txn>,
}

impl<F: FileLock<Block = Node>, D: DirLock<File = F>, Txn: Transaction<D>, T: TableInstance>
    Selection<F, D, Txn, T>
{
    pub fn new(source: T, columns: Vec<Id>) -> TCResult<Self> {
        let column_set: HashSet<&Id> = columns.iter().collect();
        let mut indices: Vec<usize> = Vec::with_capacity(columns.len());

        let source_columns = source.schema().primary().columns();
        let source_indices: HashMap<&Id, usize> = source_columns
            .iter()
            .enumerate()
            .map(|(i, col)| (&col.name, i))
            .collect();

        for name in columns.iter() {
            let index = *source_indices
                .get(name)
                .ok_or(TCError::not_found(format!("Column {}", name)))?;

            indices.push(index);
        }

        let key = source
            .key()
            .iter()
            .filter(|col| column_set.contains(&col.name))
            .cloned()
            .collect();

        let values = source
            .values()
            .iter()
            .filter(|col| column_set.contains(&col.name))
            .cloned()
            .collect();

        let schema = (key, values).into();

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

impl<F, D, Txn, T> TableInstance for Selection<F, D, Txn, T>
where
    F: FileLock<Block = Node>,
    D: DirLock<File = F>,
    Txn: Transaction<D>,
    T: TableInstance,
{
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
}

impl<F, D, Txn, T> TableOrder for Selection<F, D, Txn, T>
where
    F: FileLock<Block = Node>,
    D: DirLock<File = F>,
    Txn: Transaction<D>,
    T: TableOrder,
    <T as TableOrder>::Reverse: TableStream,
{
    type OrderBy = Selection<F, D, Txn, <T as TableOrder>::OrderBy>;
    type Reverse = <<T as TableOrder>::Reverse as TableStream>::Selection;

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

    fn reverse(self) -> TCResult<Self::Reverse> {
        self.source.reverse()?.select(self.columns)
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

#[async_trait]
impl<F, D, Txn, T> TableStream for Selection<F, D, Txn, T>
where
    F: FileLock<Block = Node>,
    D: DirLock<File = F>,
    Txn: Transaction<D>,
    T: TableStream,
    Table<F, D, Txn>: From<Self>,
{
    type Limit = Limited<F, D, Txn>;
    type Selection = Selection<F, D, Txn, Self>;

    async fn count(self, txn_id: TxnId) -> TCResult<u64> {
        self.source.count(txn_id).await
    }

    fn limit(self, limit: u64) -> Self::Limit {
        Limited::new(self, limit)
    }

    fn select(self, columns: Vec<Id>) -> TCResult<Self::Selection> {
        Selection::new(self, columns)
    }

    async fn rows<'a>(self, txn_id: TxnId) -> TCResult<TCBoxTryStream<'a, Vec<Value>>> {
        let indices = self.indices.to_vec();
        let selected = self.source.rows(txn_id).await?.map_ok(move |row| {
            let selection: Vec<Value> = indices.iter().map(|i| row[*i].clone()).collect();
            selection
        });

        let selected: TCBoxTryStream<Vec<Value>> = Box::pin(selected);
        Ok(selected)
    }
}

impl<F, D, Txn, T> From<Selection<F, D, Txn, T>> for Table<F, D, Txn>
where
    F: FileLock<Block = Node>,
    D: DirLock<File = F>,
    Txn: Transaction<D>,
    T: TableInstance,
    Table<F, D, Txn>: From<T>,
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

impl<F: FileLock<Block = Node>, D: DirLock<File = F>, Txn: Transaction<D>> TableSlice<F, D, Txn> {
    pub fn new(table: TableIndex<F, D, Txn>, bounds: Bounds) -> TCResult<TableSlice<F, D, Txn>> {
        super::TableSlice::validate_bounds(&table, &bounds)?;

        let index = table.supporting_index(&bounds)?;
        let slice = super::TableSlice::slice(index, bounds.clone())?;

        debug!("TableSlice::new w/bounds {}", bounds);
        Ok(TableSlice { table, slice })
    }

    pub fn bounds(&'_ self) -> &'_ Bounds {
        self.slice.bounds()
    }

    pub fn index_slice(self, bounds: Bounds) -> TCResult<IndexSlice<F, D, Txn>> {
        self.slice.slice_index(bounds)
    }

    pub fn into_source(self) -> TableIndex<F, D, Txn> {
        self.table
    }

    pub fn source(&'_ self) -> &'_ TableIndex<F, D, Txn> {
        &self.table
    }

    pub async fn slice_rows<'a>(
        self,
        txn_id: TxnId,
        bounds: Bounds,
        reverse: bool,
    ) -> TCResult<TCBoxTryStream<'a, Vec<Value>>> {
        self.slice.slice_rows(txn_id, bounds, reverse).await
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

impl<F, D, Txn> TableInstance for TableSlice<F, D, Txn>
where
    F: FileLock<Block = Node>,
    D: DirLock<File = F>,
    Txn: Transaction<D>,
{
    fn key(&self) -> &[Column] {
        self.source().key()
    }

    fn values(&self) -> &[Column] {
        self.source().values()
    }

    fn schema(&self) -> TableSchema {
        self.source().schema()
    }
}

impl<F, D, Txn> TableOrder for TableSlice<F, D, Txn>
where
    F: FileLock<Block = Node>,
    D: DirLock<File = F>,
    Txn: Transaction<D>,
{
    type OrderBy = Merged<F, D, Txn>;
    type Reverse = TableSlice<F, D, Txn>;

    fn order_by(self, order: Vec<Id>, reverse: bool) -> TCResult<Self::OrderBy> {
        let bounds = self.slice.bounds;
        let table = self.table.order_by(order, reverse)?;
        super::TableSlice::slice(table, bounds)
    }

    fn reverse(self) -> TCResult<Self::Reverse> {
        Ok(Self {
            table: self.table,
            slice: self.slice.into_reversed(),
        })
    }

    fn validate_order(&self, order: &[Id]) -> TCResult<()> {
        self.source().validate_order(order)
    }
}

#[async_trait]
impl<F, D, Txn> TableStream for TableSlice<F, D, Txn>
where
    F: FileLock<Block = Node>,
    D: DirLock<File = F>,
    Txn: Transaction<D>,
{
    type Limit = Limited<F, D, Txn>;
    type Selection = Selection<F, D, Txn, Self>;

    async fn count(self, txn_id: TxnId) -> TCResult<u64> {
        self.slice.count(txn_id).await
    }

    fn limit(self, limit: u64) -> Self::Limit {
        Limited::new(self, limit)
    }

    fn select(self, columns: Vec<Id>) -> TCResult<Self::Selection> {
        Selection::new(self, columns)
    }

    async fn rows<'a>(self, txn_id: TxnId) -> TCResult<TCBoxTryStream<'a, Vec<Value>>> {
        self.slice.rows(txn_id).await
    }
}

impl<F, D, Txn> super::TableSlice for TableSlice<F, D, Txn>
where
    F: FileLock<Block = Node>,
    D: DirLock<File = F>,
    Txn: Transaction<D>,
{
    type Slice = Merged<F, D, Txn>;

    fn slice(self, bounds: Bounds) -> TCResult<Merged<F, D, Txn>> {
        let slice_bounds = self.slice.bounds().clone();
        let source = self.into_source();
        let bounds = source.merge_bounds(vec![slice_bounds, bounds])?;
        source.validate_bounds(&bounds)?;
        source.slice(bounds)
    }

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()> {
        let index = self.source().supporting_index(self.slice.bounds())?;
        index
            .validate_slice_bounds(self.slice.bounds().clone(), bounds.clone())
            .map(|_| ())
    }
}

impl<F, D, Txn> From<TableSlice<F, D, Txn>> for Table<F, D, Txn> {
    fn from(slice: TableSlice<F, D, Txn>) -> Self {
        Self::TableSlice(slice)
    }
}

#[derive(Clone)]
struct Phantom<F, D, Txn> {
    file: PhantomData<F>,
    dir: PhantomData<D>,
    txn: PhantomData<Txn>,
}

impl<F: FileLock<Block = Node>, D: DirLock<File = F>, Txn: Transaction<D>> Default
    for Phantom<F, D, Txn>
{
    fn default() -> Self {
        Self {
            file: PhantomData,
            dir: PhantomData,
            txn: PhantomData,
        }
    }
}
