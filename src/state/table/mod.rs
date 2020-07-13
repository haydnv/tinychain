use std::collections::{hash_map, HashMap, HashSet};
use std::fmt;
use std::ops::Bound;
use std::sync::Arc;

use async_trait::async_trait;

use crate::error;
use crate::transaction::TxnId;
use crate::value::class::{Impl, ValueType};
use crate::value::{TCResult, Value, ValueId};

mod index;

type Row = HashMap<ValueId, Value>;

#[derive(Clone)]
pub struct Column {
    pub name: ValueId,
    pub dtype: ValueType,
}

pub enum ColumnBound {
    Is(Value),
    In(Bound<Value>, Bound<Value>),
}

impl ColumnBound {
    fn expect<M: fmt::Display>(&self, dtype: ValueType, err_context: &M) -> TCResult<()> {
        match self {
            Self::Is(value) => value.expect(dtype, err_context),
            Self::In(start, end) => match start {
                Bound::Included(value) => value.expect(dtype, err_context),
                Bound::Excluded(value) => value.expect(dtype, err_context),
                Bound::Unbounded => Ok(()),
            }
            .and_then(|_| match end {
                Bound::Included(value) => value.expect(dtype, err_context),
                Bound::Excluded(value) => value.expect(dtype, err_context),
                Bound::Unbounded => Ok(()),
            }),
        }
    }
}

impl fmt::Display for ColumnBound {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Is(value) => write!(f, "{}", value),
            Self::In(Bound::Unbounded, Bound::Unbounded) => write!(f, "[...]"),
            Self::In(start, end) => {
                match start {
                    Bound::Unbounded => write!(f, "[...")?,
                    Bound::Included(value) => write!(f, "[{},", value)?,
                    Bound::Excluded(value) => write!(f, "({},", value)?,
                };
                match end {
                    Bound::Unbounded => write!(f, "...]"),
                    Bound::Included(value) => write!(f, "{}]", value),
                    Bound::Excluded(value) => write!(f, "{})", value),
                }
            }
        }
    }
}

pub struct Bounds(HashMap<ValueId, ColumnBound>);

impl Bounds {
    fn iter(&self) -> hash_map::Iter<ValueId, ColumnBound> {
        self.0.iter()
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

impl fmt::Display for Bounds {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{{{}}}",
            self.0
                .iter()
                .map(|(k, v)| format!("{}: {}", k, v))
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}

struct Schema {
    key: Vec<Column>,
    value: Vec<Column>,
}

impl Schema {
    fn columns(&self) -> Vec<Column> {
        [&self.key[..], &self.value[..]]
            .concat()
            .into_iter()
            .collect()
    }

    fn column_names(&'_ self) -> HashSet<&'_ ValueId> {
        self.key
            .iter()
            .map(|c| &c.name)
            .chain(self.value.iter().map(|c| &c.name))
            .collect()
    }

    fn len(&self) -> usize {
        self.key.len() + self.value.len()
    }

    fn validate(&self, bounds: &Bounds) -> TCResult<()> {
        let column_names = self.column_names();
        for name in bounds.0.keys() {
            if !column_names.contains(name) {
                return Err(error::bad_request("No such column", name));
            }
        }

        Ok(())
    }
}

#[async_trait]
trait Selection: Sized + Send + Sync {
    async fn count(self: Arc<Self>, txn_id: TxnId) -> TCResult<u64>;

    async fn delete(self: Arc<Self>, txn_id: TxnId) -> TCResult<()>;

    fn derive<M: Fn(Row) -> Value>(self: Arc<Self>, name: ValueId, map: M) -> Derived<Self, M> {
        Derived {
            source: self,
            name,
            map,
        }
    }

    fn filter<F: Fn(Row) -> bool>(self: Arc<Self>, filter: F) -> Filtered<Self, F> {
        Filtered {
            source: self,
            filter,
        }
    }

    fn group_by(self: Arc<Self>, columns: Vec<ValueId>) -> Aggregate<Self> {
        Aggregate {
            source: self,
            columns,
        }
    }

    async fn index(self: Arc<Self>, _columns: Option<Vec<ValueId>>) -> TCResult<index::ReadOnly> {
        Err(error::not_implemented())
    }

    fn limit(self: Arc<Self>, limit: u64) -> Limit<Self> {
        Limit {
            source: self,
            limit,
        }
    }

    fn order_by(self: Arc<Self>, columns: Vec<ValueId>, reverse: bool) -> Sorted<Self> {
        Sorted {
            source: self,
            columns,
            reverse,
        }
    }

    fn select(self: Arc<Self>, columns: Vec<ValueId>) -> ColumnSelection<Self> {
        ColumnSelection {
            source: self,
            columns,
        }
    }

    fn schema(&'_ self) -> &'_ Schema;

    fn slice(self: Arc<Self>, bounds: Bounds) -> TCResult<Slice<Self>> {
        self.validate(&bounds)?;

        Ok(Slice {
            source: self,
            bounds,
        })
    }

    fn validate(&self, bounds: &Bounds) -> TCResult<()>;

    async fn update(self: Arc<Self>, txn_id: TxnId, value: Row) -> TCResult<()>;
}

struct Aggregate<T: Selection> {
    source: Arc<T>,
    columns: Vec<ValueId>,
}

struct ColumnSelection<T: Selection> {
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
    limit: u64,
}

struct Slice<T: Selection> {
    source: Arc<T>,
    bounds: Bounds,
}

struct Sorted<T: Selection> {
    source: Arc<T>,
    columns: Vec<ValueId>,
    reverse: bool,
}
