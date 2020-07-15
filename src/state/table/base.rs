use std::collections::{hash_map, HashMap, HashSet};
use std::fmt;
use std::iter::FromIterator;
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

#[derive(Clone)]
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

#[derive(Clone)]
pub struct Bounds(HashMap<ValueId, ColumnBound>);

impl Bounds {
    pub fn iter(&self) -> hash_map::Iter<ValueId, ColumnBound> {
        self.0.iter()
    }

    pub fn keys(&self) -> hash_map::Keys<ValueId, ColumnBound> {
        self.0.keys()
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

    pub fn column_names<'a, F: FromIterator<&'a ValueId>>(&'a self) -> F {
        self.key
            .iter()
            .map(|c| &c.name)
            .chain(self.value.iter().map(|c| &c.name))
            .collect()
    }

    pub fn key(&self, row: &Row) -> TCResult<Vec<Value>> {
        let mut key = Vec::with_capacity(self.key.len());
        for column in &self.key {
            if let Some(value) = row.get(&column.name) {
                value.expect(column.dtype, format!("for table schema {}", self))?;
                key.push(value.clone())
            } else {
                return Err(error::bad_request(
                    "Row has no value for key column",
                    &column.name,
                ));
            }
        }

        Ok(key)
    }

    pub fn key_columns(&'_ self) -> &'_ [Column] {
        &self.key
    }

    pub fn key_names(&self) -> Vec<ValueId> {
        self.key.iter().map(|c| &c.name).cloned().collect()
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
        let column_names: HashSet<&ValueId> = self.column_names();
        for name in bounds.0.keys() {
            if !column_names.contains(name) {
                return Err(error::not_found(name));
            }
        }

        Ok(())
    }

    pub fn validate_columns(&self, columns: &[ValueId]) -> TCResult<()> {
        let valid_columns: HashSet<&ValueId> = self.column_names();
        for column in columns {
            if !valid_columns.contains(column) {
                return Err(error::not_found(column));
            }
        }

        Ok(())
    }

    pub fn validate_row_partial(&self, row: &Row) -> TCResult<()> {
        let columns: HashMap<ValueId, ValueType> = self
            .columns()
            .drain(..)
            .map(|c| (c.name, c.dtype))
            .collect();
        for (col_name, value) in row {
            if let Some(dtype) = columns.get(col_name) {
                value.expect(*dtype, format!("for table with schema {}", self))?;
            } else {
                return Err(error::not_found(col_name));
            }
        }

        Ok(())
    }

    pub fn validate_row(&self, row: &Row) -> TCResult<()> {
        let expected: HashSet<&ValueId> = self.column_names();
        let actual: HashSet<&ValueId> = row.keys().collect();
        let mut missing: Vec<&&ValueId> = expected.difference(&actual).collect();
        let mut extra: Vec<&&ValueId> = actual.difference(&expected).collect();

        if !missing.is_empty() {
            return Err(error::bad_request(
                "Row is missing columns",
                missing
                    .drain(..)
                    .map(|c| (*c).to_string())
                    .collect::<Vec<String>>()
                    .join(", "),
            ));
        }

        if !extra.is_empty() {
            return Err(error::bad_request(
                "Row contains unrecognized columns",
                extra
                    .drain(..)
                    .map(|c| (*c).to_string())
                    .collect::<Vec<String>>()
                    .join(", "),
            ));
        }

        self.validate_row_partial(row)
    }

    pub fn row_into_values(&self, mut row: Row, reject_extras: bool) -> TCResult<Vec<Value>> {
        let mut key = Vec::with_capacity(self.len());
        for column in self.columns() {
            let value = row
                .remove(&column.name)
                .ok_or_else(|| error::bad_request("Missing value for column", &column.name))?;
            value.expect(column.dtype, format!("for table with schema {}", self))?;
            key.push(value);
        }

        if reject_extras && !row.is_empty() {
            return Err(error::bad_request(
                &format!(
                    "Unrecognized columns (`{}`) for schema",
                    row.keys()
                        .map(|c| c.to_string())
                        .collect::<Vec<String>>()
                        .join("`, `")
                ),
                self,
            ));
        }

        Ok(key)
    }

    pub fn values_into_row(&self, mut values: Vec<Value>) -> TCResult<Row> {
        if values.len() > self.len() {
            return Err(error::bad_request(
                "Too many values provided for a row with schema",
                self,
            ));
        }

        let mut row = HashMap::new();
        for (column, value) in self.columns()[0..values.len()].iter().zip(values.drain(..)) {
            value.expect(column.dtype, format!("for table with schema {}", self))?;
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

impl From<Schema> for HashMap<ValueId, Column> {
    fn from(mut schema: Schema) -> HashMap<ValueId, Column> {
        schema
            .key
            .drain(..)
            .chain(schema.value.drain(..))
            .map(|c| (c.name.clone(), c))
            .collect()
    }
}

impl From<Schema> for Vec<ValueId> {
    fn from(mut schema: Schema) -> Vec<ValueId> {
        schema
            .key
            .drain(..)
            .chain(schema.value.drain(..))
            .map(|c| c.name)
            .collect()
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
