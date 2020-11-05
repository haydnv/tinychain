use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;

use crate::class::{Instance, TCResult};
use crate::error;
use crate::scalar::*;

pub type Row = HashMap<ValueId, Value>;

#[derive(Clone, PartialEq)]
pub struct Column {
    name: ValueId,
    dtype: ValueType,
    max_len: Option<usize>,
}

impl Column {
    pub fn name(&'_ self) -> &'_ ValueId {
        &self.name
    }

    pub fn dtype(&'_ self) -> &'_ ValueType {
        &self.dtype
    }

    pub fn max_len(&'_ self) -> &'_ Option<usize> {
        &self.max_len
    }
}

impl<Id: Into<ValueId>> From<(Id, NumberType)> for Column {
    fn from(column: (Id, NumberType)) -> Column {
        let (name, dtype) = column;
        let name: ValueId = name.into();
        let dtype: ValueType = dtype.into();
        let max_len = None;

        Column {
            name,
            dtype,
            max_len,
        }
    }
}

impl From<(ValueId, ValueType)> for Column {
    fn from(column: (ValueId, ValueType)) -> Column {
        let (name, dtype) = column;
        let max_len = None;

        Column {
            name,
            dtype,
            max_len,
        }
    }
}

impl From<(ValueId, ValueType, usize)> for Column {
    fn from(column: (ValueId, ValueType, usize)) -> Column {
        let (name, dtype, size) = column;
        let max_len = Some(size);

        Column {
            name,
            dtype,
            max_len,
        }
    }
}

impl TryCastFrom<Value> for Column {
    fn can_cast_from(value: &Value) -> bool {
        value.matches::<(ValueId, ValueType)>() || value.matches::<(ValueId, ValueType, u64)>()
    }

    fn opt_cast_from(value: Value) -> Option<Column> {
        if value.matches::<(ValueId, ValueType)>() {
            let (name, dtype) = value.opt_cast_into().unwrap();
            Some(Column {
                name,
                dtype,
                max_len: None,
            })
        } else if value.matches::<(ValueId, ValueType, u64)>() {
            let (name, dtype, max_len) = value.opt_cast_into().unwrap();
            Some(Column {
                name,
                dtype,
                max_len: Some(max_len),
            })
        } else {
            None
        }
    }
}

impl fmt::Display for Column {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.max_len {
            Some(max_len) => write!(f, "{}: {}({})", self.name, self.dtype, max_len),
            None => write!(f, "{}: {}", self.name, self.dtype),
        }
    }
}

pub type RowSchema = Vec<Column>;

#[derive(Clone)]
pub struct IndexSchema {
    key: RowSchema,
    values: RowSchema,
}

impl IndexSchema {
    pub fn columns(&self) -> Vec<Column> {
        [&self.key[..], &self.values[..]].concat()
    }

    pub fn data_types(&self) -> Vec<ValueType> {
        self.key
            .iter()
            .map(|c| c.dtype)
            .chain(self.values.iter().map(|c| c.dtype))
            .collect()
    }

    pub fn key(&'_ self) -> &'_ [Column] {
        &self.key
    }

    pub fn values(&'_ self) -> &'_ [Column] {
        &self.values
    }

    pub fn len(&self) -> usize {
        self.key.len() + self.values.len()
    }

    pub fn key_from_row(&self, row: &Row) -> TCResult<Vec<Value>> {
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

    pub fn starts_with(&self, expected: &[ValueId]) -> bool {
        let actual: Vec<ValueId> = self.columns().iter().map(|c| c.name()).cloned().collect();
        for (a, e) in actual[0..expected.len()].iter().zip(expected.iter()) {
            if a != e {
                return false;
            }
        }

        true
    }

    pub fn subset(&self, key_columns: HashSet<&ValueId>) -> TCResult<IndexSchema> {
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

    pub fn validate_columns(&self, columns: &[ValueId]) -> TCResult<()> {
        let valid_columns: HashSet<ValueId> =
            self.columns().iter().map(|c| c.name()).cloned().collect();
        for column in columns {
            if !valid_columns.contains(column) {
                return Err(error::not_found(column));
            }
        }

        Ok(())
    }

    pub fn validate_key(&self, key: &[Value]) -> TCResult<()> {
        if key.len() != self.key.len() {
            let key_columns: Vec<String> = self.key.iter().map(|c| c.to_string()).collect();
            return Err(error::bad_request(
                "Invalid key, expected",
                format!("[{}]", key_columns.join(", ")),
            ));
        }

        for (val, col) in key.iter().zip(self.key.iter()) {
            if !val.is_a(col.dtype) {
                return Err(error::bad_request(
                    &format!("Expected {} for column {}, found", col.dtype, col.name),
                    val,
                ));
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
        let expected: HashSet<ValueId> = self.columns().iter().map(|c| c.name()).cloned().collect();
        let actual: HashSet<ValueId> = row.keys().cloned().collect();
        let mut missing: Vec<&ValueId> = expected.difference(&actual).collect();
        let mut extra: Vec<&ValueId> = actual.difference(&expected).collect();

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

impl From<(Vec<Column>, Vec<Column>)> for IndexSchema {
    fn from(schema: (Vec<Column>, Vec<Column>)) -> IndexSchema {
        let (key, values) = schema;
        IndexSchema { key, values }
    }
}

impl TryCastFrom<Value> for IndexSchema {
    fn can_cast_from(value: &Value) -> bool {
        value.matches::<(Vec<Column>, Vec<Column>)>()
    }

    fn opt_cast_from(value: Value) -> Option<IndexSchema> {
        if let Some((key, values)) = value.opt_cast_into() {
            Some(IndexSchema { key, values })
        } else {
            None
        }
    }
}

impl From<IndexSchema> for HashMap<ValueId, Column> {
    fn from(mut schema: IndexSchema) -> HashMap<ValueId, Column> {
        schema
            .key
            .drain(..)
            .chain(schema.values.drain(..))
            .map(|c| (c.name.clone(), c))
            .collect()
    }
}

impl From<IndexSchema> for Vec<ValueId> {
    fn from(mut schema: IndexSchema) -> Vec<ValueId> {
        schema
            .key
            .drain(..)
            .chain(schema.values.drain(..))
            .map(|c| c.name)
            .collect()
    }
}

impl From<IndexSchema> for RowSchema {
    fn from(mut schema: IndexSchema) -> RowSchema {
        schema
            .key
            .drain(..)
            .chain(schema.values.drain(..))
            .collect()
    }
}

impl fmt::Display for IndexSchema {
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

#[derive(Clone)]
pub struct TableSchema {
    primary: IndexSchema,
    indices: BTreeMap<ValueId, Vec<ValueId>>,
}

impl TableSchema {
    pub fn indices(&'_ self) -> &'_ BTreeMap<ValueId, Vec<ValueId>> {
        &self.indices
    }

    pub fn primary(&'_ self) -> &'_ IndexSchema {
        &self.primary
    }
}

impl From<IndexSchema> for TableSchema {
    fn from(schema: IndexSchema) -> TableSchema {
        TableSchema {
            primary: schema,
            indices: BTreeMap::new(),
        }
    }
}

impl<I: Iterator<Item = (ValueId, Vec<ValueId>)>> From<(IndexSchema, I)> for TableSchema {
    fn from(schema: (IndexSchema, I)) -> TableSchema {
        TableSchema {
            primary: schema.0,
            indices: schema.1.collect(),
        }
    }
}

impl TryCastFrom<Value> for TableSchema {
    fn can_cast_from(value: &Value) -> bool {
        value.matches::<(IndexSchema, Vec<(ValueId, Vec<ValueId>)>)>()
    }

    fn opt_cast_from(value: Value) -> Option<TableSchema> {
        if let Some((primary, indices)) = value.opt_cast_into() {
            let indices: Vec<(ValueId, Vec<ValueId>)> = indices;
            let indices = indices.into_iter().collect();
            Some(TableSchema { primary, indices })
        } else {
            None
        }
    }
}
