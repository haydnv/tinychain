use std::collections::{HashMap, HashSet};
use std::convert::{TryFrom, TryInto};
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future;
use futures::stream::{Stream, StreamExt, TryStreamExt};

use crate::error;
use crate::state::btree::{BTree, BTreeRange};
use crate::transaction::{Txn, TxnId};
use crate::value::{TCResult, TCStream, Value, ValueId};

use super::index::TableBase;
use super::schema::{Bounds, Column, Row, Schema};
use super::{Selection, Table};

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

#[async_trait]
impl Selection for ColumnSelection {
    type Stream = TCStream<Vec<Value>>;

    async fn count(&self, txn_id: TxnId) -> TCResult<u64> {
        self.source.clone().count(txn_id).await
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

    async fn stream(&self, txn_id: TxnId) -> TCResult<Self::Stream> {
        let indices = self.indices.to_vec();
        let selected = self.source.clone().stream(txn_id).await?.map(move |row| {
            let selection: Vec<Value> = indices.iter().map(|i| row[*i].clone()).collect();
            selection
        });

        Ok(Box::pin(selected))
    }

    async fn validate(&self, txn_id: &TxnId, bounds: &Bounds) -> TCResult<()> {
        let bounds_columns: HashSet<&ValueId> = bounds.keys().collect();
        let selected: HashSet<&ValueId> = self.schema.column_names();
        let mut unknown: HashSet<&&ValueId> = selected.difference(&bounds_columns).collect();
        if !unknown.is_empty() {
            return Err(error::bad_request(
                "Tried to slice by unselected columns",
                unknown
                    .drain()
                    .map(|c| c.to_string())
                    .collect::<Vec<String>>()
                    .join(", "),
            ));
        }

        self.source.validate(txn_id, bounds).await
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

#[async_trait]
impl Selection for IndexSlice {
    type Stream = TCStream<Vec<Value>>;

    async fn count(&self, txn_id: TxnId) -> TCResult<u64> {
        self.source
            .clone()
            .len(txn_id, self.range.clone().into())
            .await
    }

    async fn delete(self, txn_id: TxnId) -> TCResult<()> {
        self.source.delete(&txn_id, self.range.into()).await
    }

    fn reversed(&self) -> TCResult<Table> {
        Ok(self.clone().into_reversed().into())
    }

    fn schema(&'_ self) -> &'_ Schema {
        &self.schema
    }

    async fn stream(&self, txn_id: TxnId) -> TCResult<Self::Stream> {
        self.source
            .clone()
            .slice(txn_id.clone(), self.range.clone().into())
            .await
    }

    async fn update(self, txn: Arc<Txn>, value: Row) -> TCResult<()> {
        self.source
            .update(
                txn.id(),
                &self.range.into(),
                &self.schema.row_into_values(value, true)?,
            )
            .await
    }

    async fn validate(&self, _txn_id: &TxnId, bounds: &Bounds) -> TCResult<()> {
        let schema = self.schema();
        let outer = self.bounds.clone().try_into_btree_range(schema)?;
        let inner = bounds.clone().try_into_btree_range(schema)?;
        outer.contains(&inner, schema.data_types()).map(|_| ())
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

#[async_trait]
impl Selection for Limited {
    type Stream = TCStream<Vec<Value>>;

    async fn count(&self, txn_id: TxnId) -> TCResult<u64> {
        let source_count = self.source.count(txn_id).await?;
        Ok(u64::min(source_count, self.limit as u64))
    }

    async fn delete(self, txn_id: TxnId) -> TCResult<()> {
        let source = self.source.clone();
        let schema = source.schema().clone();
        self.stream(txn_id.clone())
            .await?
            .map(|row| Ok(source.delete_row(&txn_id, schema.values_into_row(row)?)))
            .try_buffer_unordered(2)
            .fold(Ok(()), |_, r| future::ready(r))
            .await
    }

    fn reversed(&self) -> TCResult<Table> {
        Err(error::unsupported(
            "Cannot reverse a limited selection, consider reversing a slice before limiting",
        ))
    }

    fn schema(&'_ self) -> &'_ Schema {
        self.source.schema()
    }

    async fn stream(&self, txn_id: TxnId) -> TCResult<Self::Stream> {
        let rows = self.source.clone().stream(txn_id).await?;

        Ok(Box::pin(rows.take(self.limit)))
    }

    async fn validate(&self, txn_id: &TxnId, bounds: &Bounds) -> TCResult<()> {
        self.source.validate(txn_id, bounds).await
    }

    async fn update(self, txn: Arc<Txn>, value: Row) -> TCResult<()> {
        let source = self.source.clone();
        let schema = source.schema().clone();
        let txn_id = txn.id().clone();
        self.stream(txn_id.clone())
            .await?
            .map(|row| {
                Ok(source.update_row(txn_id.clone(), schema.values_into_row(row)?, value.clone()))
            })
            .try_buffer_unordered(2)
            .fold(Ok(()), |_, r| future::ready(r))
            .await
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
        table.validate(txn_id, &bounds).await?;
        Ok(TableSlice {
            table,
            bounds,
            reversed: false,
        })
    }
}

#[async_trait]
impl Selection for TableSlice {
    type Stream = Pin<Box<dyn Stream<Item = Vec<Value>> + Send + Sync + Unpin>>;

    async fn count(&self, txn_id: TxnId) -> TCResult<u64> {
        let index = self.table.supporting_index(&txn_id, &self.bounds).await?;
        index
            .slice(&txn_id, self.bounds.clone())
            .await?
            .count(txn_id)
            .await
    }

    async fn delete(self, txn_id: TxnId) -> TCResult<()> {
        let schema = self.schema().clone();
        self.stream(txn_id.clone())
            .await?
            .map(|row| schema.values_into_row(row))
            .map_ok(|row| self.delete_row(&txn_id, row))
            .try_buffer_unordered(2)
            .fold(Ok(()), |_, r| future::ready(r))
            .await
    }

    async fn delete_row(&self, txn_id: &TxnId, row: Row) -> TCResult<()> {
        self.table.delete_row(txn_id, row).await
    }

    fn reversed(&self) -> TCResult<Table> {
        let mut selection = self.clone();
        selection.reversed = true;
        Ok(selection.into())
    }

    fn schema(&'_ self) -> &'_ Schema {
        self.table.schema()
    }

    async fn slice(&self, txn_id: &TxnId, bounds: Bounds) -> TCResult<Table> {
        self.validate(txn_id, &bounds).await?;
        self.table.slice(txn_id, bounds).await
    }

    async fn stream(&self, txn_id: TxnId) -> TCResult<Self::Stream> {
        let left = Arc::new(self.table.primary().clone());
        let right = self.table.supporting_index(&txn_id, &self.bounds).await?;
        right.validate(&txn_id, &self.bounds).await?;

        let rows = right
            .stream(txn_id.clone())
            .await?
            .then(move |key| left.clone().get_by_key(txn_id.clone(), key))
            .filter(|row| future::ready(row.is_some()))
            .map(|row| row.unwrap());

        Ok(Box::pin(rows))
    }

    async fn validate(&self, txn_id: &TxnId, bounds: &Bounds) -> TCResult<()> {
        let index = self.table.supporting_index(txn_id, &self.bounds).await?;
        index
            .validate_bounds(self.bounds.clone(), bounds.clone())
            .map(|_| ())
    }

    async fn update(self, txn: Arc<Txn>, value: Row) -> TCResult<()> {
        let txn_id = txn.id().clone();
        let schema = self.schema().clone();
        self.stream(txn_id.clone())
            .await?
            .map(|row| schema.values_into_row(row))
            .map_ok(|row| self.update_row(txn_id.clone(), row, value.clone()))
            .try_buffer_unordered(2)
            .fold(Ok(()), |_, r| future::ready(r))
            .await
    }

    async fn update_row(&self, txn_id: TxnId, row: Row, value: Row) -> TCResult<()> {
        self.table.update_row(txn_id, row, value).await
    }
}
