use std::convert::TryFrom;
use std::sync::Arc;

use async_trait::async_trait;

use crate::error;
use crate::transaction::{Txn, TxnId};
use crate::value::{TCResult, TCStream, Value, ValueId};

use super::{Bounds, Row, Schema, Selection};

pub struct Aggregate<T: Selection> {
    source: Arc<T>,
    columns: Vec<ValueId>,
}

impl<T: Selection> From<(Arc<T>, Vec<ValueId>)> for Aggregate<T> {
    fn from(params: (Arc<T>, Vec<ValueId>)) -> Aggregate<T> {
        Aggregate {
            source: params.0,
            columns: params.1,
        }
    }
}

pub struct ColumnSelection<T: Selection> {
    source: Arc<T>,
    columns: Vec<ValueId>,
    schema: Schema,
}

impl<T: Selection> TryFrom<(Arc<T>, Vec<ValueId>)> for ColumnSelection<T> {
    type Error = error::TCError;

    fn try_from(params: (Arc<T>, Vec<ValueId>)) -> TCResult<ColumnSelection<T>> {
        let schema = params.0.schema().subset(params.1.iter().collect())?;

        Ok(ColumnSelection {
            source: params.0,
            columns: params.1,
            schema,
        })
    }
}

#[async_trait]
impl<T: Selection> Selection for ColumnSelection<T> {
    type Stream = TCStream<Vec<Value>>;

    async fn count(self: Arc<Self>, _txn_id: TxnId) -> TCResult<u64> {
        Err(error::not_implemented())
    }

    async fn delete(self: Arc<Self>, _txn_id: TxnId) -> TCResult<()> {
        Err(error::not_implemented())
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

    async fn stream(self: Arc<Self>, _txn_id: TxnId) -> TCResult<Self::Stream> {
        Err(error::not_implemented())
    }

    fn validate(&self, bounds: &Bounds) -> TCResult<()> {
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
