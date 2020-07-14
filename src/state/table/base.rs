use std::collections::{hash_map, HashMap, HashSet};
use std::fmt;
use std::ops::Bound;

use crate::error;
use crate::state::btree;
use crate::value::class::{Impl, ValueType};
use crate::value::{TCResult, Value, ValueId};

pub type Row = HashMap<ValueId, Value>;

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
    pub fn expect<M: fmt::Display>(&self, dtype: ValueType, err_context: &M) -> TCResult<()> {
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
    pub fn iter(&self) -> hash_map::Iter<ValueId, ColumnBound> {
        self.0.iter()
    }

    pub fn len(&self) -> usize {
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

    pub fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()> {
        let column_names = self.column_names();
        for name in bounds.0.keys() {
            if !column_names.contains(name) {
                return Err(error::bad_request("No such column", name));
            }
        }

        Ok(())
    }

    pub fn validate_columns(&self, columns: &[ValueId]) -> TCResult<()> {
        let valid_columns = self.column_names();
        for column in columns {
            if !valid_columns.contains(column) {
                return Err(error::bad_request("No such column", column));
            }
        }

        Ok(())
    }

    pub fn into_row(&self, mut values: Vec<Value>) -> TCResult<Row> {
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
