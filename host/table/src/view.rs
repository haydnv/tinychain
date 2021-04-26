use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;
use std::marker::PhantomData;

use async_trait::async_trait;
use futures::future::{self, TryFutureExt};
use futures::stream::{StreamExt, TryStreamExt};

use tc_btree::Node;
use tc_error::*;
use tc_transact::fs::{Dir, File};
use tc_transact::{Transaction, TxnId};
use tc_value::Value;
use tcgeneric::{GroupStream, Id, Instance, TCTryStream};

use super::{Bounds, Column, IndexSchema, Row, Table, TableInstance, TableType};

const ERR_AGGREGATE_SLICE: &str = "Table aggregate does not support slicing. \
Consider aggregating a slice of the source table.";
const ERR_AGGREGATE_NESTED: &str = "It doesn't make sense to aggregate an aggregate table view. \
Consider aggregating the source table directly.";

const ERR_LIMITED_ORDER: &str = "Cannot order a limited selection. \
Consider ordering the source or indexing the selection.";
const ERR_LIMITED_REVERSE: &str = "Cannot reverse a limited selection. \
Consider reversing a slice before limiting";

#[derive(Clone)]
pub struct Aggregate<F: File<Node>, D: Dir, Txn: Transaction<D>, T: TableInstance<F, D, Txn>> {
    source: Selection<F, D, Txn, T>,
    phantom_file: PhantomData<F>,
}

impl<F: File<Node>, D: Dir, Txn: Transaction<D>, T: TableInstance<F, D, Txn>> Instance
    for Aggregate<F, D, Txn, T>
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

    fn key(&'_ self) -> &'_ [Column] {
        self.source.key()
    }

    fn values(&'_ self) -> &'_ [Column] {
        self.source.values()
    }

    fn order_by(self, columns: Vec<Id>, reverse: bool) -> TCResult<Self::OrderBy> {
        let source = self.source.order_by(columns, reverse)?;
        Ok(Aggregate {
            source,
            phantom_file: PhantomData,
        })
    }

    fn reversed(self) -> TCResult<Self::Reverse> {
        let phantom_file = self.phantom_file;
        self.source.reversed().map(|source| Aggregate {
            source,
            phantom_file,
        })
    }

    async fn stream<'a>(&'a self, txn_id: &'a TxnId) -> TCResult<TCTryStream<'a, Vec<Value>>> {
        let grouped = self.source.stream(txn_id).map_ok(GroupStream::from).await?;
        let grouped: TCTryStream<'a, Vec<Value>> = Box::pin(grouped);
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
            phantom_file: PhantomData,
            phantom_dir: PhantomData,
            phantom_txn: PhantomData,
        };

        Table::Aggregate(Box::new(Aggregate {
            source,
            phantom_file: aggregate.phantom_file,
        }))
    }
}

#[derive(Clone)]
pub struct IndexSlice;

#[derive(Clone)]
pub struct Limited<F: File<Node>, D: Dir, Txn: Transaction<D>> {
    source: Table<F, D, Txn>,
    limit: u64,
}

impl<F: File<Node>, D: Dir, Txn: Transaction<D>> Limited<F, D, Txn> {
    pub fn new(source: Table<F, D, Txn>, limit: u64) -> Self {
        Limited { source, limit }
    }
}

impl<F: File<Node>, D: Dir, Txn: Transaction<D>> Instance for Limited<F, D, Txn> {
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

    fn order_by(self, _order: Vec<Id>, _reverse: bool) -> TCResult<Table<F, D, Txn>> {
        Err(TCError::unsupported(ERR_LIMITED_ORDER))
    }

    fn reversed(self) -> TCResult<Table<F, D, Txn>> {
        Err(TCError::unsupported(ERR_LIMITED_REVERSE))
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
        Err(TCError::unsupported(ERR_LIMITED_ORDER))
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

impl<F: File<Node>, D: Dir, Txn: Transaction<D>> From<Limited<F, D, Txn>> for Table<F, D, Txn> {
    fn from(limited: Limited<F, D, Txn>) -> Self {
        Table::Limit(Box::new(limited))
    }
}

#[derive(Clone)]
pub struct Merged;

#[derive(Clone)]
pub struct Selection<F: File<Node>, D: Dir, Txn: Transaction<D>, T: TableInstance<F, D, Txn>> {
    source: T,
    schema: IndexSchema,
    columns: Vec<Id>,
    indices: Vec<usize>,
    phantom_file: PhantomData<F>,
    phantom_dir: PhantomData<D>,
    phantom_txn: PhantomData<Txn>,
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
            phantom_file: PhantomData,
            phantom_dir: PhantomData,
            phantom_txn: PhantomData,
        })
    }
}

impl<F: File<Node>, D: Dir, Txn: Transaction<D>, T: TableInstance<F, D, Txn>> Instance
    for Selection<F, D, Txn, T>
{
    type Class = TableType;

    fn class(&self) -> TableType {
        TableType::Selection
    }
}

#[async_trait]
impl<F: File<Node>, D: Dir, Txn: Transaction<D>, T: TableInstance<F, D, Txn>>
    TableInstance<F, D, Txn> for Selection<F, D, Txn, T>
{
    type OrderBy = Selection<F, D, Txn, <T as TableInstance<F, D, Txn>>::OrderBy>;
    type Reverse = Selection<F, D, Txn, <T as TableInstance<F, D, Txn>>::Reverse>;
    type Slice = Selection<F, D, Txn, <T as TableInstance<F, D, Txn>>::Slice>;

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
            phantom_file: PhantomData,
            phantom_dir: PhantomData,
            phantom_txn: PhantomData,
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

impl<F: File<Node>, D: Dir, Txn: Transaction<D>, T: TableInstance<F, D, Txn>>
    From<Selection<F, D, Txn, T>> for Table<F, D, Txn>
{
    fn from(selection: Selection<F, D, Txn, T>) -> Self {
        Table::Selection(Box::new(Selection {
            source: selection.source.into(),
            schema: selection.schema,
            columns: selection.columns,
            indices: selection.indices,
            phantom_file: PhantomData,
            phantom_dir: PhantomData,
            phantom_txn: PhantomData,
        }))
    }
}

#[derive(Clone)]
pub struct TableSlice;

pub fn group_by<F: File<Node>, D: Dir, Txn: Transaction<D>, T: TableInstance<F, D, Txn>>(
    source: T,
    columns: Vec<Id>,
) -> TCResult<Aggregate<F, D, Txn, <T as TableInstance<F, D, Txn>>::OrderBy>> {
    let source = source.order_by(columns.to_vec(), false)?;
    let source = source.select(columns)?;
    Ok(Aggregate {
        source,
        phantom_file: PhantomData,
    })
}
