use std::collections::{HashMap, HashSet};
use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future;
use futures::stream::{StreamExt, TryStreamExt};

use crate::error;
use crate::transaction::{Txn, TxnId};
use crate::value::{TCResult, TCStream, Value, ValueId};

use super::{Bounds, Column, Row, Schema, Selection};

pub struct ColumnSelection<T: Selection> {
    source: Arc<T>,
    schema: Schema,
    indices: Vec<usize>,
}

impl<T: Selection> TryFrom<(Arc<T>, Vec<ValueId>)> for ColumnSelection<T> {
    type Error = error::TCError;

    fn try_from(params: (Arc<T>, Vec<ValueId>)) -> TCResult<ColumnSelection<T>> {
        let (source, columns) = params;

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
        let mut source_columns: HashMap<ValueId, Column> = source
            .schema()
            .columns()
            .into_iter()
            .map(|c| (c.name.clone(), c))
            .collect();

        for (i, name) in columns.iter().enumerate() {
            let column = source_columns
                .remove(name)
                .ok_or_else(|| error::bad_request("No such column", name))?;
            indices.push(i);
            schema.push(column);
        }

        Ok(ColumnSelection {
            source,
            schema: (vec![], schema).into(),
            indices,
        })
    }
}

#[async_trait]
impl<T: Selection + 'static> Selection for ColumnSelection<T> {
    type Stream = TCStream<Vec<Value>>;

    async fn count(self: Arc<Self>, txn_id: TxnId) -> TCResult<u64> {
        self.source.clone().count(txn_id).await
    }

    fn schema(&'_ self) -> &'_ Schema {
        &self.schema
    }

    async fn stream(self: Arc<Self>, txn_id: TxnId) -> TCResult<Self::Stream> {
        let indices = self.indices.to_vec();
        let selected = self.source.clone().stream(txn_id).await?.map(move |row| {
            let selection: Vec<Value> = indices.iter().map(|i| row[*i].clone()).collect();
            selection
        });

        Ok(Box::pin(selected))
    }

    fn validate(&self, bounds: &Bounds) -> TCResult<()> {
        let bounds_columns: HashSet<&ValueId> = bounds.keys().collect();
        let selected: HashSet<&ValueId> = self.schema.column_names().into_iter().collect();
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

        self.source.validate(bounds)
    }
}

impl<T: Selection> fmt::Display for ColumnSelection<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "table/view/column_selection")
    }
}

pub struct Limited<T: Selection> {
    source: Arc<T>,
    limit: usize,
}

impl<T: Selection> TryFrom<(Arc<T>, u64)> for Limited<T> {
    type Error = error::TCError;

    fn try_from(params: (Arc<T>, u64)) -> TCResult<Limited<T>> {
        let (source, limit) = params;
        let limit: usize = limit.try_into().map_err(|_| {
            error::internal("This host architecture does not support a 64-bit stream limit")
        })?;

        Ok(Limited { source, limit })
    }
}

#[async_trait]
impl<T: Selection> Selection for Limited<T> {
    type Stream = TCStream<Vec<Value>>;

    async fn count(self: Arc<Self>, txn_id: TxnId) -> TCResult<u64> {
        let source_count = self.source.clone().count(txn_id).await?;
        Ok(u64::min(source_count, self.limit as u64))
    }

    async fn delete(self: Arc<Self>, txn_id: TxnId) -> TCResult<()> {
        let source = self.source.clone();
        let schema = source.schema().clone();
        self.stream(txn_id.clone())
            .await?
            .map(|row| Ok(source.delete_row(&txn_id, schema.values_into_row(row)?)))
            .try_buffer_unordered(2)
            .fold(Ok(()), |_, r| future::ready(r))
            .await
    }

    fn schema(&'_ self) -> &'_ Schema {
        self.source.schema()
    }

    async fn stream(self: Arc<Self>, txn_id: TxnId) -> TCResult<Self::Stream> {
        let rows = self.source.clone().stream(txn_id).await?;

        Ok(Box::pin(rows.take(self.limit)))
    }

    fn validate(&self, bounds: &Bounds) -> TCResult<()> {
        self.source.validate(bounds)
    }

    async fn update(self: Arc<Self>, txn: Arc<Txn>, value: Row) -> TCResult<()> {
        let source = self.source.clone();
        let schema = source.schema().clone();
        let txn_id = txn.id().clone();
        self.stream(txn_id.clone())
            .await?
            .map(|row| {
                Ok(source.clone().update_row(
                    txn_id.clone(),
                    schema.values_into_row(row)?,
                    value.clone(),
                ))
            })
            .try_buffer_unordered(2)
            .fold(Ok(()), |_, r| future::ready(r))
            .await
    }
}

impl<T: Selection> fmt::Display for Limited<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "table/view/limited")
    }
}

pub struct Sliced<T: Selection> {
    source: Arc<T>,
    bounds: Bounds,
}

impl<T: Selection> TryFrom<(Arc<T>, Bounds)> for Sliced<T> {
    type Error = error::TCError;

    fn try_from(params: (Arc<T>, Bounds)) -> TCResult<Sliced<T>> {
        let (source, bounds) = params;
        source.validate(&bounds)?;
        Ok(Sliced { source, bounds })
    }
}

impl<T: Selection> fmt::Display for Sliced<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "table/view/sliced")
    }
}

pub struct Sorted<T: Selection> {
    source: Arc<T>,
    columns: Vec<ValueId>,
    reverse: bool,
}

impl<T: Selection> TryFrom<(Arc<T>, Vec<ValueId>, bool)> for Sorted<T> {
    type Error = error::TCError;

    fn try_from(params: (Arc<T>, Vec<ValueId>, bool)) -> TCResult<Sorted<T>> {
        let (source, columns, reverse) = params;
        source.schema().validate_columns(&columns)?;
        Ok(Sorted {
            source,
            columns,
            reverse,
        })
    }
}

impl<T: Selection> fmt::Display for Sorted<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "table/view/sorted")
    }
}
