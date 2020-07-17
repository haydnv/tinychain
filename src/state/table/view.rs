use std::collections::{HashMap, HashSet};
use std::convert::{TryFrom, TryInto};
use std::ops::Bound;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future;
use futures::stream::{StreamExt, TryStreamExt};

use crate::error;
use crate::state::btree::{BTree, BTreeRange};
use crate::transaction::{Txn, TxnId};
use crate::value::{TCResult, TCStream, Value, ValueId};

use super::schema::{Bounds, Column, ColumnBound, Row, Schema};
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

    pub fn new(source: Arc<BTree>, schema: Schema, mut bounds: Bounds) -> TCResult<IndexSlice> {
        use Bound::*;
        assert!(source.schema() == &schema.clone().into());
        schema.validate_bounds(&bounds)?;

        let mut start = Vec::with_capacity(bounds.len());
        let mut end = Vec::with_capacity(bounds.len());
        let column_names: Vec<&ValueId> = schema.column_names();
        for name in &column_names[0..bounds.len()] {
            let bound = bounds.remove(&name).ok_or_else(|| error::not_found(name))?;
            match bound {
                ColumnBound::Is(value) => {
                    start.push(Included(value.clone()));
                    end.push(Included(value));
                }
                ColumnBound::In(s, e) => {
                    start.push(s);
                    end.push(e);
                }
            }
        }
        let range = (start, end).into();

        Ok(IndexSlice {
            source,
            schema,
            bounds,
            range,
            reverse: false,
        })
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
        let mut slice = self.clone();
        slice.reverse = true;
        Ok(slice.into())
    }

    fn schema(&'_ self) -> &'_ Schema {
        &self.schema
    }

    async fn stream(&self, txn_id: TxnId) -> TCResult<Self::Stream> {
        self.source
            .clone()
            .slice(txn_id, self.range.clone().into())
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

    async fn validate(&self, _txn_id: &TxnId, _bounds: &Bounds) -> TCResult<()> {
        Err(error::not_implemented())
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
pub struct Merged {
    left: Box<Table>,
    right: Box<Table>,
}

#[async_trait]
impl Selection for Merged {
    type Stream = TCStream<Vec<Value>>;

    fn reversed(&self) -> TCResult<Table> {
        let left = Box::new(self.left.reversed()?);
        let right = Box::new(self.right.reversed()?);
        Ok(Merged { left, right }.into())
    }

    fn schema(&'_ self) -> &'_ Schema {
        self.left.schema()
    }

    async fn stream(&self, _txn_id: TxnId) -> TCResult<Self::Stream> {
        Err(error::not_implemented())
    }

    async fn validate(&self, txn_id: &TxnId, bounds: &Bounds) -> TCResult<()> {
        self.left.validate(txn_id, bounds).await
    }

    async fn update(self, txn: Arc<Txn>, value: Row) -> TCResult<()> {
        let source = self.left.clone();
        let schema = source.schema().clone();
        let txn_id = txn.id().clone();
        self.stream(txn_id.clone())
            .await?
            .map(|row| {
                Ok(source.update_row(txn_id.clone(), schema.values_into_row(row)?, value.clone()))
            })
            .try_buffer_unordered(2u32 as usize)
            .fold(Ok(()), |_, r| future::ready(r))
            .await
    }
}

impl From<(Table, Table)> for Merged {
    fn from(tables: (Table, Table)) -> Merged {
        let (left, right) = tables;
        Merged {
            left: Box::new(left),
            right: Box::new(right),
        }
    }
}
