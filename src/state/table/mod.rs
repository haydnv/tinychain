use std::collections::{hash_map, HashMap, HashSet};
use std::fmt;
use std::ops::Bound;
use std::sync::Arc;

use async_trait::async_trait;
use futures::Stream;

use crate::error;
use crate::state::btree;
use crate::transaction::{Txn, TxnId};
use crate::value::class::{Impl, ValueType};
use crate::value::{TCResult, TCStream, Value, ValueId};

mod index;

type Row = HashMap<ValueId, Value>;

#[derive(Clone)]
pub struct Column {
    pub name: ValueId,
    pub dtype: ValueType,
    pub max_len: Option<usize>,
}

impl From<Column> for btree::Column {
    fn from(column: Column) -> btree::Column {
        (column.name, column.dtype, column.max_len).into()
    }
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

#[derive(Clone)]
pub struct Schema {
    key: Vec<Column>,
    value: Vec<Column>,
}

impl Schema {
    pub fn columns(&self) -> Vec<Column> {
        [&self.key[..], &self.value[..]]
            .concat()
            .into_iter()
            .collect()
    }

    pub fn column_names(&'_ self) -> HashSet<&'_ ValueId> {
        self.key
            .iter()
            .map(|c| &c.name)
            .chain(self.value.iter().map(|c| &c.name))
            .collect()
    }

    pub fn len(&self) -> usize {
        self.key.len() + self.value.len()
    }

    pub fn subset(&self, key_columns: HashSet<&ValueId>) -> TCResult<Schema> {
        let key: Vec<Column> = self
            .key
            .iter()
            .filter(|c| key_columns.contains(&c.name))
            .cloned()
            .collect();
        let value: Vec<Column> = self
            .columns()
            .iter()
            .filter(|c| !key_columns.contains(&c.name))
            .cloned()
            .collect();
        Ok((key, value).into())
    }

    pub fn validate(&self, bounds: &Bounds) -> TCResult<()> {
        let column_names = self.column_names();
        for name in bounds.0.keys() {
            if !column_names.contains(name) {
                return Err(error::bad_request("No such column", name));
            }
        }

        Ok(())
    }

    fn into_row(&self, mut values: Vec<Value>) -> TCResult<Row> {
        if values.len() > self.len() {
            return Err(error::bad_request(
                "Too many values provided for a row with schema",
                self,
            ));
        }

        let mut row = HashMap::new();
        for (column, value) in self.columns()[0..values.len()].iter().zip(values.drain(..)) {
            row.insert(column.name.clone(), value);
        }
        Ok(row)
    }
}

impl From<(Vec<Column>, Vec<Column>)> for Schema {
    fn from(kv: (Vec<Column>, Vec<Column>)) -> Schema {
        Schema {
            key: kv.0,
            value: kv.1,
        }
    }
}

impl From<Schema> for btree::Schema {
    fn from(source: Schema) -> btree::Schema {
        source
            .columns()
            .iter()
            .cloned()
            .map(|c| c.into())
            .collect::<Vec<btree::Column>>()
            .into()
    }
}

impl fmt::Display for Schema {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "[{}]",
            self.columns()
                .iter()
                .map(|c| format!("{}: {}", c.name, c.dtype))
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}

#[async_trait]
pub trait Selection: Sized + Send + Sync {
    type Stream: Stream<Item = Vec<Value>> + Send + Sync;

    async fn count(self: Arc<Self>, txn_id: TxnId) -> TCResult<u64>;

    async fn delete(self: Arc<Self>, txn_id: TxnId) -> TCResult<()>;

    fn derive<M: Fn(Row) -> Value>(
        self: Arc<Self>,
        name: ValueId,
        map: M,
    ) -> Arc<Derived<Self, M>> {
        Arc::new(Derived {
            source: self,
            name,
            map,
        })
    }

    fn filter<F: Fn(Row) -> bool>(self: Arc<Self>, filter: F) -> Arc<Filtered<Self, F>> {
        Arc::new(Filtered {
            source: self,
            filter,
        })
    }

    fn group_by(self: Arc<Self>, columns: Vec<ValueId>) -> Arc<Aggregate<Self>> {
        Arc::new(Aggregate {
            source: self,
            columns,
        })
    }

    async fn index(
        self: Arc<Self>,
        txn: Arc<Txn>,
        columns: Option<Vec<ValueId>>,
    ) -> TCResult<Arc<index::ReadOnly>> {
        index::ReadOnly::copy_from(self, txn, columns)
            .await
            .map(Arc::new)
    }

    fn limit(self: Arc<Self>, limit: u64) -> Arc<Limit<Self>> {
        Arc::new(Limit {
            source: self,
            limit,
        })
    }

    fn order_by(self: Arc<Self>, columns: Vec<ValueId>, reverse: bool) -> Arc<Sorted<Self>> {
        Arc::new(Sorted {
            source: self,
            columns,
            reverse,
        })
    }

    fn select(self: Arc<Self>, columns: Vec<ValueId>) -> TCResult<Arc<ColumnSelection<Self>>> {
        let schema = self.schema().subset(columns.iter().collect())?;

        Ok(Arc::new(ColumnSelection {
            source: self,
            columns,
            schema,
        }))
    }

    fn schema(&'_ self) -> &'_ Schema;

    fn slice(self: Arc<Self>, bounds: Bounds) -> TCResult<Arc<Slice<Self>>> {
        self.validate(&bounds)?;

        Ok(Arc::new(Slice {
            source: self,
            bounds,
        }))
    }

    async fn stream(self: Arc<Self>, txn_id: TxnId) -> TCResult<Self::Stream>;

    fn validate(&self, bounds: &Bounds) -> TCResult<()>;

    async fn update(self: Arc<Self>, txn_id: TxnId, value: Row) -> TCResult<()>;
}

pub struct Aggregate<T: Selection> {
    source: Arc<T>,
    columns: Vec<ValueId>,
}

pub struct ColumnSelection<T: Selection> {
    source: Arc<T>,
    columns: Vec<ValueId>,
    schema: Schema,
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

    async fn update(self: Arc<Self>, txn_id: TxnId, value: Row) -> TCResult<()> {
        self.source.clone().update(txn_id, value).await
    }
}

pub struct Derived<T: Selection, M: Fn(Row) -> Value> {
    source: Arc<T>,
    name: ValueId,
    map: M,
}

pub struct Filtered<T: Selection, F: Fn(Row) -> bool> {
    source: Arc<T>,
    filter: F,
}

pub struct Limit<T: Selection> {
    source: Arc<T>,
    limit: u64,
}

pub struct Slice<T: Selection> {
    source: Arc<T>,
    bounds: Bounds,
}

pub struct Sorted<T: Selection> {
    source: Arc<T>,
    columns: Vec<ValueId>,
    reverse: bool,
}
