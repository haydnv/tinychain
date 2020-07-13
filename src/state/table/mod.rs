use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;

use crate::error;
use crate::transaction::TxnId;
use crate::value::{TCResult, Value, ValueId};
use crate::value::class::ValueType;

type Row = HashMap<ValueId, Value>;

#[derive(Clone)]
pub struct Column {
    pub name: ValueId,
    pub dtype: ValueType,
}

struct Schema {
    key: Vec<Column>,
    value: Vec<Column>,
}

impl Schema {
    fn columns(&self) -> Vec<Column> {
        [&self.key[..], &self.value[..]].concat().into_iter().collect()
    }

    fn len(&self) -> usize {
        self.key.len() + self.value.len()
    }
}

struct Bounds {}

impl fmt::Display for Bounds {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "NOT IMPLEMENTED")
    }
}

#[async_trait]
trait Selection: Sized + Send + Sync {
    async fn count(self: Arc<Self>, txn_id: &TxnId) -> TCResult<u64>;

    async fn delete(self: Arc<Self>, txn_id: &TxnId) -> TCResult<()>;

    fn derive<M: Fn(Row) -> Value>(self: Arc<Self>, name: ValueId, map: M) -> Derived<Self, M> {
        Derived { source: self, name, map }
    }

    fn filter<F: Fn(Row) -> bool>(self: Arc<Self>, filter: F) -> Filtered<Self, F> {
        Filtered { source: self, filter }
    }

    fn group_by(self: Arc<Self>, columns: Vec<ValueId>) -> Aggregate<Self> {
        Aggregate { source: self, columns }
    }

    async fn index(self: Arc<Self>, _columns: Option<Vec<ValueId>>) -> TCResult<ReadOnlyIndex> {
        Err(error::not_implemented())
    }

    fn limit(self: Arc<Self>, limit: u64) -> Limit<Self> {
        Limit { source: self, limit }
    }

    fn order_by(self: Arc<Self>, columns: Vec<ValueId>, reverse: bool) -> Sorted<Self> {
        Sorted { source: self, columns, reverse }
    }

    fn schema(&self) -> Schema;

    fn slice(self: Arc<Self>, bounds: Bounds) -> TCResult<Slice<Self>> {
        if self.supports(&bounds) {
            Ok(Slice { source: self, bounds })
        } else {
            Err(error::bad_request("Invalid bounds", bounds))
        }
    }

    fn supports(&self, bounds: &Bounds) -> bool;

    async fn update(self: Arc<Self>, value: Row) -> TCResult<()>;
}

struct Aggregate<T: Selection> {
    source: Arc<T>,
    columns: Vec<ValueId>,
}

struct Derived<T: Selection, M: Fn(Row) -> Value> {
    source: Arc<T>,
    name: ValueId,
    map: M,
}

struct Filtered<T: Selection, F: Fn(Row) -> bool> {
    source: Arc<T>,
    filter: F,
}

struct Limit<T: Selection> {
    source: Arc<T>,
    limit: u64
}

struct ReadOnlyIndex {}

struct Slice<T: Selection> {
    source: Arc<T>,
    bounds: Bounds,
}

struct Sorted<T: Selection> {
    source: Arc<T>,
    columns: Vec<ValueId>,
    reverse: bool
}
