use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::StreamExt;

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

    async fn delete(self: Arc<Self>, _txn_id: TxnId) -> TCResult<()> {
        Err(error::method_not_allowed(
            "ColumnSelection does not support deletion, try deleting from the source table",
        ))
    }

    fn schema(&'_ self) -> &'_ Schema {
        &self.schema
    }

    fn slice(self: Arc<Self>, bounds: Bounds) -> TCResult<Arc<Slice<Self>>> {
        self.validate(&bounds)?;

        Ok(Arc::new(Slice {
            source: self,
            bounds,
        }))
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

    async fn update(self: Arc<Self>, txn: Arc<Txn>, value: Row) -> TCResult<()> {
        self.source.clone().update(txn, value).await
    }
}

pub struct Derived<T: Selection, M: Fn(Row) -> Value> {
    source: Arc<T>,
    name: ValueId,
    map: M,
}

impl<T: Selection, M: Fn(Row) -> Value> From<(Arc<T>, ValueId, M)> for Derived<T, M> {
    fn from(params: (Arc<T>, ValueId, M)) -> Derived<T, M> {
        let (source, name, map) = params;
        Derived { source, name, map }
    }
}

pub struct Filtered<T: Selection, F: Fn(Row) -> bool> {
    source: Arc<T>,
    filter: F,
}

impl<T: Selection, F: Fn(Row) -> bool> From<(Arc<T>, F)> for Filtered<T, F> {
    fn from(params: (Arc<T>, F)) -> Filtered<T, F> {
        let (source, filter) = params;
        Filtered { source, filter }
    }
}

pub struct Limit<T: Selection> {
    source: Arc<T>,
    limit: u64,
}

impl<T: Selection> From<(Arc<T>, u64)> for Limit<T> {
    fn from(params: (Arc<T>, u64)) -> Limit<T> {
        Limit {
            source: params.0,
            limit: params.1,
        }
    }
}

pub struct Slice<T: Selection> {
    source: Arc<T>,
    bounds: Bounds,
}

impl<T: Selection> TryFrom<(Arc<T>, Bounds)> for Slice<T> {
    type Error = error::TCError;

    fn try_from(params: (Arc<T>, Bounds)) -> TCResult<Slice<T>> {
        let (source, bounds) = params;
        source.validate(&bounds)?;
        Ok(Slice { source, bounds })
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
