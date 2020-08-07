use std::collections::{HashMap, HashSet};
use std::convert::{TryFrom, TryInto};
use std::sync::Arc;

use futures::future;
use futures::stream::{self, StreamExt, TryStreamExt};

use crate::error;
use crate::state::btree::{BTree, BTreeRange};
use crate::transaction::{Txn, TxnId};
use crate::value::{TCBoxTryFuture, TCResult, TCStream, Value, ValueId};

use super::index::TableBase;
use super::schema::{Bounds, Column, Row, Schema};
use super::{Selection, Table};

const ERR_NESTED_AGGREGATE: &str = "It doesn't make sense to aggregate an aggregate table view; \
consider aggregating the source table directly";

#[derive(Clone)]
pub struct Aggregate {
    source: Box<Table>,
    columns: Vec<ValueId>,
}

impl Aggregate {
    pub async fn new(source: Table, txn_id: TxnId, columns: Vec<ValueId>) -> TCResult<Aggregate> {
        let source = Box::new(source.order_by(&txn_id, columns.to_vec(), false).await?);
        Ok(Aggregate { source, columns })
    }
}

impl Selection for Aggregate {
    type Stream = TCStream<Vec<Value>>;

    fn group_by<'a>(
        &'a self,
        _txn_id: TxnId,
        _columns: Vec<ValueId>,
    ) -> TCBoxTryFuture<'a, Aggregate> {
        Box::pin(future::ready(Err(error::unsupported(ERR_NESTED_AGGREGATE))))
    }

    fn order_by<'a>(
        &'a self,
        txn_id: &'a TxnId,
        columns: Vec<ValueId>,
        reverse: bool,
    ) -> TCBoxTryFuture<'a, Table> {
        Box::pin(async move {
            let source = Box::new(self.source.order_by(txn_id, columns, reverse).await?);
            Ok(Aggregate {
                source,
                columns: self.columns.to_vec(),
            }
            .into())
        })
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

    fn schema(&'_ self) -> &'_ Schema {
        self.source.schema()
    }

    fn stream<'a>(self, txn_id: TxnId) -> TCBoxTryFuture<'a, Self::Stream> {
        Box::pin(async move {
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
        })
    }

    fn validate_bounds<'a>(
        &'a self,
        _txn_id: &'a TxnId,
        _bounds: &'a Bounds,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            Err(error::unsupported("Table aggregate does not support slicing, consider aggregating a slice of the source table"))
        })
    }

    fn validate_order<'a>(
        &'a self,
        txn_id: &'a TxnId,
        order: &'a [ValueId],
    ) -> TCBoxTryFuture<'a, ()> {
        self.source.validate_order(txn_id, order)
    }
}

#[derive(Clone)]
pub struct ColumnSelection {
    source: Box<Table>,
    schema: Schema,
    columns: Vec<ValueId>,
    indices: Vec<usize>,
}

impl<T: Into<Table>> TryFrom<(T, Vec<ValueId>)> for ColumnSelection {
    type Error = error::TCError;

    fn try_from(params: (T, Vec<ValueId>)) -> TCResult<ColumnSelection> {
        let (source, columns) = params;
        let source: Table = source.into();

        let column_set: HashSet<&ValueId> = columns.iter().collect();
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
        let mut source_columns: HashMap<ValueId, Column> = source.schema().clone().into();

        for (i, name) in columns.iter().enumerate() {
            let column = source_columns
                .remove(name)
                .ok_or_else(|| error::not_found(name))?;
            indices.push(i);
            schema.push(column);
        }

        Ok(ColumnSelection {
            source: Box::new(source),
            schema: (vec![], schema).into(),
            columns,
            indices,
        })
    }
}

impl Selection for ColumnSelection {
    type Stream = TCStream<Vec<Value>>;

    fn count(&self, txn_id: TxnId) -> TCBoxTryFuture<u64> {
        Box::pin(async move { self.source.clone().count(txn_id).await })
    }

    fn order_by<'a>(
        &'a self,
        txn_id: &'a TxnId,
        order: Vec<ValueId>,
        reverse: bool,
    ) -> TCBoxTryFuture<'a, Table> {
        Box::pin(async move {
            self.validate_order(txn_id, &order).await?;

            let source = self
                .source
                .order_by(txn_id, order, reverse)
                .await
                .map(Box::new)?;

            Ok(ColumnSelection {
                source,
                schema: self.schema.clone(),
                columns: self.columns.to_vec(),
                indices: self.indices.to_vec(),
            }
            .into())
        })
    }

    fn reversed(&self) -> TCResult<Table> {
        self.source
            .reversed()?
            .select(self.columns.to_vec())
            .map(|s| s.into())
    }

    fn schema(&'_ self) -> &'_ Schema {
        &self.schema
    }

    fn stream<'a>(self, txn_id: TxnId) -> TCBoxTryFuture<'a, Self::Stream> {
        Box::pin(async move {
            let indices = self.indices.to_vec();
            let selected = self.source.clone().stream(txn_id).await?.map(move |row| {
                let selection: Vec<Value> = indices.iter().map(|i| row[*i].clone()).collect();
                selection
            });
            let selected: TCStream<Vec<Value>> = Box::pin(selected);
            Ok(selected)
        })
    }

    fn validate_bounds<'a>(
        &'a self,
        txn_id: &'a TxnId,
        bounds: &'a Bounds,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let bounds_columns: HashSet<ValueId> = bounds.keys().cloned().collect();
            let selected: HashSet<ValueId> = self.schema.column_names();
            let mut unknown: HashSet<&ValueId> = selected.difference(&bounds_columns).collect();
            if !unknown.is_empty() {
                let unknown: Vec<String> = unknown.drain().map(|c| c.to_string()).collect();
                return Err(error::bad_request(
                    "Tried to slice by unselected columns",
                    unknown.join(", "),
                ));
            }

            self.source.validate_bounds(txn_id, bounds).await
        })
    }

    fn validate_order<'a>(
        &'a self,
        txn_id: &'a TxnId,
        order: &'a [ValueId],
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let order_columns: HashSet<ValueId> = order.iter().cloned().collect();
            let selected: HashSet<ValueId> = self.schema().column_names();
            let mut unknown: HashSet<&ValueId> = selected.difference(&order_columns).collect();
            if !unknown.is_empty() {
                let unknown: Vec<String> = unknown.drain().map(|c| c.to_string()).collect();
                return Err(error::bad_request(
                    "Tried to order by unselected columns",
                    unknown.join(", "),
                ));
            }

            self.source.validate_order(txn_id, order).await
        })
    }
}

#[derive(Clone)]
pub struct IndexSlice {
    source: Arc<BTree>,
    schema: Schema,
    bounds: Bounds,
    range: BTreeRange,
    reverse: bool,
}

impl IndexSlice {
    pub fn all(source: Arc<BTree>, schema: Schema, reverse: bool) -> IndexSlice {
        IndexSlice {
            source,
            schema,
            bounds: Bounds::all(),
            range: BTreeRange::all(),
            reverse,
        }
    }

    pub fn new(source: Arc<BTree>, schema: Schema, bounds: Bounds) -> TCResult<IndexSlice> {
        assert!(source.schema() == &schema.clone().into());
        schema.validate_bounds(&bounds)?;

        let range: BTreeRange = bounds.clone().try_into_btree_range(&schema)?;

        Ok(IndexSlice {
            source,
            schema,
            bounds,
            range,
            reverse: false,
        })
    }

    pub fn into_reversed(mut self) -> IndexSlice {
        self.reverse = !self.reverse;
        self
    }

    pub fn slice_index(&self, bounds: Bounds) -> TCResult<IndexSlice> {
        let schema = self.schema();
        let outer = self.bounds.clone().try_into_btree_range(schema)?;
        let inner = bounds.clone().try_into_btree_range(schema)?;
        if outer.contains(&inner, schema.data_types())? {
            let mut slice = self.clone();
            slice.bounds = bounds;
            Ok(slice)
        } else {
            Err(error::bad_request(
                &format!("IndexSlice with bounds {} does not contain", &self.bounds),
                bounds,
            ))
        }
    }
}

impl Selection for IndexSlice {
    type Stream = TCStream<Vec<Value>>;

    fn count(&self, txn_id: TxnId) -> TCBoxTryFuture<u64> {
        self.source.clone().len(txn_id, self.range.clone().into())
    }

    fn delete<'a>(self, txn_id: TxnId) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move { self.source.delete(&txn_id, self.range.into()).await })
    }

    fn order_by<'a>(
        &'a self,
        _txn_id: &'a TxnId,
        order: Vec<ValueId>,
        reverse: bool,
    ) -> TCBoxTryFuture<'a, Table> {
        let result = if self.schema.starts_with(&order) {
            if reverse {
                self.reversed()
            } else {
                Ok(self.clone().into())
            }
        } else {
            let order: Vec<String> = order.iter().map(String::from).collect();
            Err(error::bad_request(
                &format!("Index with schema {} does not support order", &self.schema),
                order.join(", "),
            ))
        };

        Box::pin(future::ready(result))
    }

    fn reversed(&self) -> TCResult<Table> {
        Ok(self.clone().into_reversed().into())
    }

    fn schema(&'_ self) -> &'_ Schema {
        &self.schema
    }

    fn stream<'a>(self, txn_id: TxnId) -> TCBoxTryFuture<'a, Self::Stream> {
        Box::pin(async move {
            self.source
                .clone()
                .slice(txn_id.clone(), self.range.clone().into())
                .await
        })
    }

    fn update<'a>(self, txn: Arc<Txn>, value: Row) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            self.source
                .update(
                    txn.id(),
                    &self.range.into(),
                    &self.schema.row_into_values(value, true)?,
                )
                .await
        })
    }

    fn validate_bounds<'a>(
        &'a self,
        _txn_id: &'a TxnId,
        bounds: &'a Bounds,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let schema = self.schema();
            let outer = self.bounds.clone().try_into_btree_range(schema)?;
            let inner = bounds.clone().try_into_btree_range(schema)?;
            outer.contains(&inner, schema.data_types()).map(|_| ())
        })
    }

    fn validate_order<'a>(
        &'a self,
        _txn_id: &'a TxnId,
        order: &'a [ValueId],
    ) -> TCBoxTryFuture<'a, ()> {
        let result = if self.schema.starts_with(order) {
            Ok(())
        } else {
            let order: Vec<String> = order.iter().map(String::from).collect();
            Err(error::bad_request(
                &format!("Index with schema {} does not support order", &self.schema),
                order.join(", "),
            ))
        };

        Box::pin(future::ready(result))
    }
}

#[derive(Clone)]
pub struct Limited {
    source: Box<Table>,
    limit: usize,
}

impl TryFrom<(Table, u64)> for Limited {
    type Error = error::TCError;

    fn try_from(params: (Table, u64)) -> TCResult<Limited> {
        let (source, limit) = params;
        let limit: usize = limit.try_into().map_err(|_| {
            error::internal("This host architecture does not support a 64-bit stream limit")
        })?;

        Ok(Limited {
            source: Box::new(source),
            limit,
        })
    }
}

impl Selection for Limited {
    type Stream = TCStream<Vec<Value>>;

    fn count(&self, txn_id: TxnId) -> TCBoxTryFuture<u64> {
        Box::pin(async move {
            let source_count = self.source.count(txn_id).await?;
            Ok(u64::min(source_count, self.limit as u64))
        })
    }

    fn delete<'a>(self, txn_id: TxnId) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let source = self.source.clone();
            let schema = source.schema().clone();
            self.stream(txn_id.clone())
                .await?
                .map(|row| schema.values_into_row(row))
                .map_ok(|row| source.delete_row(&txn_id, row))
                .try_buffer_unordered(2)
                .try_fold((), |_, _| future::ready(Ok(())))
                .await
        })
    }

    fn order_by<'a>(
        &'a self,
        _txn_id: &'a TxnId,
        _order: Vec<ValueId>,
        _reverse: bool,
    ) -> TCBoxTryFuture<Table> {
        Box::pin(future::ready(Err(error::unsupported("Cannot order a limited selection, consider ordering the source or indexing the selection"))))
    }

    fn reversed(&self) -> TCResult<Table> {
        Err(error::unsupported(
            "Cannot reverse a limited selection, consider reversing a slice before limiting",
        ))
    }

    fn schema(&'_ self) -> &'_ Schema {
        self.source.schema()
    }

    fn stream<'a>(self, txn_id: TxnId) -> TCBoxTryFuture<'a, Self::Stream> {
        Box::pin(async move {
            let rows = self.source.clone().stream(txn_id).await?;
            let rows: TCStream<Vec<Value>> = Box::pin(rows.take(self.limit));
            Ok(rows)
        })
    }

    fn validate_bounds<'a>(
        &'a self,
        txn_id: &'a TxnId,
        bounds: &'a Bounds,
    ) -> TCBoxTryFuture<'a, ()> {
        self.source.validate_bounds(txn_id, bounds)
    }

    fn validate_order<'a>(
        &'a self,
        _txn_id: &'a TxnId,
        _order: &'a [ValueId],
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(future::ready(Err(error::unsupported("Cannot order a limited selection, consider ordering the source or indexing the selection"))))
    }

    fn update<'a>(self, txn: Arc<Txn>, value: Row) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let source = self.source.clone();
            let schema = source.schema().clone();
            let txn_id = txn.id().clone();
            self.stream(txn_id.clone())
                .await?
                .map(|row| schema.values_into_row(row))
                .map_ok(|row| source.update_row(txn_id.clone(), row, value.clone()))
                .try_buffer_unordered(2)
                .try_fold((), |_, _| future::ready(Ok(())))
                .await
        })
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

    fn slice<'a>(self, txn_id: TxnId, bounds: Bounds) -> TCBoxTryFuture<'a, Table> {
        Box::pin(async move {
            match self {
                Self::Table(table) => table.slice(&txn_id, bounds).await,
                Self::Merge(merged) => merged.slice(&txn_id, bounds).await,
            }
        })
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

    fn as_reversed(self: Arc<Self>) -> Arc<Self> {
        Arc::new(Merged {
            left: self.left.clone().into_reversed(),
            right: self.right.clone().into_reversed(),
        })
    }
}

impl Selection for Merged {
    type Stream = TCStream<Vec<Value>>;

    fn delete<'a>(self, txn_id: TxnId) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let schema = self.schema();
            self.clone()
                .stream(txn_id.clone())
                .await?
                .map(|values| schema.values_into_row(values))
                .map_ok(|row| self.delete_row(&txn_id, row))
                .try_buffer_unordered(2)
                .try_fold((), |_, _| future::ready(Ok(())))
                .await
        })
    }

    fn delete_row<'a>(&'a self, txn_id: &'a TxnId, row: Row) -> TCBoxTryFuture<'a, ()> {
        match &self.left {
            MergeSource::Table(table) => table.delete_row(txn_id, row),
            MergeSource::Merge(merged) => merged.delete_row(txn_id, row),
        }
    }

    fn order_by<'a>(
        &'a self,
        txn_id: &'a TxnId,
        columns: Vec<ValueId>,
        reverse: bool,
    ) -> TCBoxTryFuture<'a, Table> {
        match &self.left {
            MergeSource::Merge(merged) => merged.order_by(txn_id, columns, reverse),
            MergeSource::Table(table_slice) => table_slice.order_by(txn_id, columns, reverse),
        }
    }

    fn reversed(&self) -> TCResult<Table> {
        Ok(Merged {
            left: self.left.clone().into_reversed(),
            right: self.right.clone().into_reversed(),
        }
        .into())
    }

    fn schema(&'_ self) -> &'_ Schema {
        match &self.left {
            MergeSource::Table(table) => table.schema(),
            MergeSource::Merge(merged) => merged.schema(),
        }
    }

    fn slice<'a>(&'a self, txn_id: &'a TxnId, bounds: Bounds) -> TCBoxTryFuture<'a, Table> {
        // TODO: reject bounds which lie outside the bounds of the table slice

        match &self.left {
            MergeSource::Merge(merged) => merged.slice(txn_id, bounds),
            MergeSource::Table(table) => table.slice(txn_id, bounds),
        }
    }

    fn stream<'a>(self, txn_id: TxnId) -> TCBoxTryFuture<'a, Self::Stream> {
        Box::pin(async move {
            let schema = self.schema().clone();
            let key_names = schema.key_names();
            let left = self.left.clone();
            let key_into_bounds = move |key| schema.key_into_bounds(key);
            let txn_id_clone = txn_id.clone();
            let rows = self
                .right
                .select(key_names)?
                .stream(txn_id.clone())
                .await?
                .map(key_into_bounds)
                .then(move |key| left.clone().slice(txn_id.clone(), key))
                .map(|slice| slice.unwrap())
                .then(move |slice| slice.stream(txn_id_clone.clone()))
                .map(|stream| stream.unwrap())
                .flatten();

            let rows: TCStream<Vec<Value>> = Box::pin(rows);
            Ok(rows)
        })
    }

    fn validate_bounds<'a>(
        &'a self,
        txn_id: &'a TxnId,
        bounds: &'a Bounds,
    ) -> TCBoxTryFuture<'a, ()> {
        match &self.left {
            MergeSource::Merge(merge) => merge.validate_bounds(txn_id, bounds),
            MergeSource::Table(table) => table.validate_bounds(txn_id, bounds),
        }
    }

    fn validate_order<'a>(
        &'a self,
        txn_id: &'a TxnId,
        order: &'a [ValueId],
    ) -> TCBoxTryFuture<'a, ()> {
        match &self.left {
            MergeSource::Merge(merge) => merge.validate_order(txn_id, order),
            MergeSource::Table(table) => table.validate_order(txn_id, order),
        }
    }

    fn update<'a>(self, txn: Arc<Txn>, value: Row) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let schema = self.schema();
            self.clone()
                .stream(txn.id().clone())
                .await?
                .map(|values| schema.values_into_row(values))
                .map_ok(|row| self.update_row(txn.id().clone(), row, value.clone()))
                .try_buffer_unordered(2)
                .try_fold((), |_, _| future::ready(Ok(())))
                .await
        })
    }

    fn update_row(&self, txn_id: TxnId, row: Row, value: Row) -> TCBoxTryFuture<()> {
        match &self.left {
            MergeSource::Table(table) => table.update_row(txn_id, row, value),
            MergeSource::Merge(merged) => merged.update_row(txn_id, row, value),
        }
    }
}

#[derive(Clone)]
pub struct TableSlice {
    table: TableBase,
    bounds: Bounds,
    reversed: bool,
}

impl TableSlice {
    pub async fn new(table: TableBase, txn_id: &TxnId, bounds: Bounds) -> TCResult<TableSlice> {
        table.validate_bounds(txn_id, &bounds).await?;

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

impl Selection for TableSlice {
    type Stream = TCStream<Vec<Value>>;

    fn count(&self, txn_id: TxnId) -> TCBoxTryFuture<u64> {
        Box::pin(async move {
            let index = self.table.supporting_index(&txn_id, &self.bounds).await?;
            index
                .slice(&txn_id, self.bounds.clone())
                .await?
                .count(txn_id)
                .await
        })
    }

    fn delete<'a>(self, txn_id: TxnId) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let schema = self.schema().clone();
            self.clone()
                .stream(txn_id.clone())
                .await?
                .map(|row| schema.values_into_row(row))
                .map_ok(|row| self.delete_row(&txn_id, row))
                .try_buffer_unordered(2)
                .fold(Ok(()), |_, r| future::ready(r))
                .await
        })
    }

    fn delete_row<'a>(&'a self, txn_id: &'a TxnId, row: Row) -> TCBoxTryFuture<'a, ()> {
        self.table.delete_row(txn_id, row)
    }

    fn order_by<'a>(
        &'a self,
        txn_id: &'a TxnId,
        order: Vec<ValueId>,
        reverse: bool,
    ) -> TCBoxTryFuture<'a, Table> {
        self.table.order_by(txn_id, order, reverse)
    }

    fn reversed(&self) -> TCResult<Table> {
        let mut selection = self.clone();
        selection.reversed = true;
        Ok(selection.into())
    }

    fn schema(&'_ self) -> &'_ Schema {
        self.table.schema()
    }

    fn slice<'a>(&'a self, txn_id: &'a TxnId, bounds: Bounds) -> TCBoxTryFuture<'a, Table> {
        Box::pin(async move {
            self.validate_bounds(txn_id, &bounds).await?;
            self.table.slice(txn_id, bounds).await
        })
    }

    fn stream<'a>(self, txn_id: TxnId) -> TCBoxTryFuture<'a, Self::Stream> {
        Box::pin(async move {
            let slice = self
                .table
                .primary()
                .slice(&txn_id, self.bounds.clone())
                .await?;
            slice.stream(txn_id).await
        })
    }

    fn validate_bounds<'a>(
        &'a self,
        txn_id: &'a TxnId,
        bounds: &'a Bounds,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let index = self.table.supporting_index(txn_id, &self.bounds).await?;
            index
                .validate_schema_bounds(self.bounds.clone(), bounds.clone())
                .map(|_| ())
        })
    }

    fn validate_order<'a>(
        &'a self,
        txn_id: &'a TxnId,
        order: &'a [ValueId],
    ) -> TCBoxTryFuture<'a, ()> {
        self.table.validate_order(txn_id, order)
    }

    fn update<'a>(self, txn: Arc<Txn>, value: Row) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let txn_id = txn.id().clone();
            let schema = self.schema().clone();
            self.clone()
                .stream(txn_id.clone())
                .await?
                .map(|row| schema.values_into_row(row))
                .map_ok(|row| self.update_row(txn_id.clone(), row, value.clone()))
                .try_buffer_unordered(2)
                .try_fold((), |_, _| future::ready(Ok(())))
                .await
        })
    }

    fn update_row(&self, txn_id: TxnId, row: Row, value: Row) -> TCBoxTryFuture<()> {
        self.table.update_row(txn_id, row, value)
    }
}
