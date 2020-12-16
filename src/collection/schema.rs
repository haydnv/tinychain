use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;

use log::debug;

use crate::error::{self, TCResult};
use crate::scalar::*;

pub type Row = HashMap<Id, Value>;

impl TryCastFrom<Map> for Row {
    fn can_cast_from(object: &Map) -> bool {
        object.values().all(Value::can_cast_from)
    }

    fn opt_cast_from(object: Map) -> Option<Row> {
        let mut row = Row::new();

        for (id, value) in object.into_iter() {
            if let Some(value) = value.opt_cast_into() {
                row.insert(id, value);
            } else {
                return None;
            }
        }

        Some(row)
    }
}

#[derive(Clone, PartialEq)]
pub struct Column {
    name: Id,
    dtype: ValueType,
    max_len: Option<usize>,
}

impl Column {
    pub fn name(&'_ self) -> &'_ Id {
        &self.name
    }

    pub fn dtype(&self) -> ValueType {
        self.dtype
    }

    pub fn max_len(&'_ self) -> &'_ Option<usize> {
        &self.max_len
    }
}

impl<I: Into<Id>> From<(I, NumberType)> for Column {
    fn from(column: (I, NumberType)) -> Column {
        let (name, dtype) = column;
        let name: Id = name.into();
        let dtype: ValueType = dtype.into();
        let max_len = None;

        Column {
            name,
            dtype,
            max_len,
        }
    }
}

impl From<(Id, ValueType)> for Column {
    fn from(column: (Id, ValueType)) -> Column {
        let (name, dtype) = column;
        let max_len = None;

        Column {
            name,
            dtype,
            max_len,
        }
    }
}

impl From<(Id, ValueType, usize)> for Column {
    fn from(column: (Id, ValueType, usize)) -> Column {
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
        debug!("Column::can_cast_from {}?", value);

        value.matches::<(Id, ValueType)>() || value.matches::<(Id, ValueType, u64)>()
    }

    fn opt_cast_from(value: Value) -> Option<Column> {
        if value.matches::<(Id, ValueType)>() {
            let (name, dtype) = value.opt_cast_into().unwrap();
            Some(Column {
                name,
                dtype,
                max_len: None,
            })
        } else if value.matches::<(Id, ValueType, u64)>() {
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

impl<'a> From<&'a Column> for (&'a Id, ValueType) {
    fn from(col: &'a Column) -> (&'a Id, ValueType) {
        (&col.name, col.dtype)
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

    pub fn key(&'_ self) -> &'_ [Column] {
        &self.key
    }

    pub fn values(&'_ self) -> &'_ [Column] {
        &self.values
    }

    pub fn len(&self) -> usize {
        self.key.len() + self.values.len()
    }

    pub fn key_values_from_row(&self, mut row: Row) -> TCResult<(Vec<Value>, Vec<Value>)> {
        if self.len() != row.len() {
            return Err(error::bad_request("Invalid row for schema", self));
        }

        let mut key = Vec::with_capacity(self.key().len());
        for col in self.key() {
            let value = row.remove(col.name()).ok_or(error::not_found(col.name()))?;
            key.push(value);
        }

        let mut values = Vec::with_capacity(self.values().len());
        for col in self.values() {
            let value = row.remove(col.name()).ok_or(error::not_found(col.name()))?;
            values.push(value);
        }

        Ok((key, values))
    }

    pub fn row_from_key_values(&self, key: Vec<Value>, values: Vec<Value>) -> TCResult<Row> {
        assert_eq!(key.len(), self.key.len());
        assert_eq!(values.len(), self.values.len());

        let mut row = key;
        row.extend(values);
        self.row_from_values(row)
    }

    pub fn row_from_values(&self, values: Vec<Value>) -> TCResult<Row> {
        assert_eq!(values.len(), self.len());

        let mut row = HashMap::new();
        for (column, value) in self.columns().into_iter().zip(values.into_iter()) {
            let value = column.dtype.try_cast(value)?;
            row.insert(column.name, value);
        }

        Ok(row)
    }
    pub fn starts_with(&self, expected: &[Id]) -> bool {
        let schema = self.columns();
        if expected.len() > schema.len() {
            return false;
        }

        let actual: Vec<Id> = schema.iter().map(|c| c.name()).cloned().collect();
        for (a, e) in actual[0..expected.len()].iter().zip(expected.iter()) {
            if a != e {
                return false;
            }
        }

        true
    }

    pub fn subset(&self, key_columns: HashSet<&Id>) -> TCResult<IndexSchema> {
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

    pub fn validate_columns(&self, columns: &[Id]) -> TCResult<()> {
        let valid_columns: HashSet<Id> = self.columns().iter().map(|c| c.name()).cloned().collect();
        for column in columns {
            if !valid_columns.contains(column) {
                return Err(error::not_found(column));
            }
        }

        Ok(())
    }

    pub fn validate_key(&self, key: Vec<Value>) -> TCResult<Vec<Value>> {
        if key.len() != self.key.len() {
            let key_columns: Vec<String> = self.key.iter().map(|c| c.to_string()).collect();
            return Err(error::bad_request(
                format!("Invalid key {}, expected", Value::Tuple(key)),
                format!("[{}]", key_columns.join(", ")),
            ));
        }

        let mut validated = Vec::with_capacity(key.len());
        for (val, col) in key.into_iter().zip(self.key.iter()) {
            let value = col.dtype.try_cast(val)?;
            validated.push(value);
        }

        Ok(validated)
    }

    pub fn validate_row_partial(&self, row: Row) -> TCResult<Row> {
        let mut validated = Row::new();
        let columns: HashMap<Id, ValueType> = self
            .columns()
            .into_iter()
            .map(|c| (c.name, c.dtype))
            .collect();

        for (col_name, value) in row.into_iter() {
            let dtype = columns
                .get(&col_name)
                .ok_or(error::bad_request("No such column", &col_name))?;
            let value = dtype.try_cast(value)?;
            validated.insert(col_name, value);
        }

        Ok(validated)
    }

    pub fn validate_row(&self, row: Row) -> TCResult<Row> {
        let expected: HashSet<Id> = self.columns().iter().map(|c| c.name()).cloned().collect();
        let actual: HashSet<Id> = row.keys().cloned().collect();
        let missing: Vec<&Id> = expected.difference(&actual).collect();
        let extra: Vec<&Id> = actual.difference(&expected).collect();

        if !missing.is_empty() {
            return Err(error::bad_request(
                "Row is missing columns",
                missing
                    .into_iter()
                    .map(|c| (*c).to_string())
                    .collect::<Vec<String>>()
                    .join(", "),
            ));
        }

        if !extra.is_empty() {
            return Err(error::bad_request(
                "Row contains unrecognized columns",
                extra
                    .into_iter()
                    .map(|c| (*c).to_string())
                    .collect::<Vec<String>>()
                    .join(", "),
            ));
        }

        self.validate_row_partial(row)
    }

    pub fn values_from_row(&self, mut row: Row, reject_extras: bool) -> TCResult<Vec<Value>> {
        let mut key = Vec::with_capacity(self.len());
        for column in self.columns() {
            let value = row
                .remove(&column.name)
                .ok_or_else(|| error::bad_request("Missing value for column", &column.name))?;
            let value = column.dtype.try_cast(value)?;
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

impl From<IndexSchema> for HashMap<Id, Column> {
    fn from(schema: IndexSchema) -> HashMap<Id, Column> {
        schema
            .key
            .into_iter()
            .chain(schema.values.into_iter())
            .map(|c| (c.name.clone(), c))
            .collect()
    }
}

impl From<IndexSchema> for Vec<Id> {
    fn from(schema: IndexSchema) -> Vec<Id> {
        schema
            .key
            .into_iter()
            .chain(schema.values.into_iter())
            .map(|c| c.name)
            .collect()
    }
}

impl From<IndexSchema> for RowSchema {
    fn from(schema: IndexSchema) -> RowSchema {
        schema
            .key
            .into_iter()
            .chain(schema.values.into_iter())
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
    indices: BTreeMap<Id, Vec<Id>>,
}

impl TableSchema {
    pub fn indices(&'_ self) -> &'_ BTreeMap<Id, Vec<Id>> {
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

impl<I: Iterator<Item = (Id, Vec<Id>)>> From<(IndexSchema, I)> for TableSchema {
    fn from(schema: (IndexSchema, I)) -> TableSchema {
        TableSchema {
            primary: schema.0,
            indices: schema.1.collect(),
        }
    }
}

impl TryCastFrom<Value> for TableSchema {
    fn can_cast_from(value: &Value) -> bool {
        value.matches::<(IndexSchema, Vec<(Id, Vec<Id>)>)>() || value.matches::<IndexSchema>()
    }

    fn opt_cast_from(value: Value) -> Option<TableSchema> {
        if value.matches::<(IndexSchema, Vec<(Id, Vec<Id>)>)>() {
            let (primary, indices): (IndexSchema, Vec<(Id, Vec<Id>)>) =
                value.opt_cast_into().unwrap();
            let indices = indices.into_iter().collect();
            Some(TableSchema { primary, indices })
        } else if value.matches::<IndexSchema>() {
            let primary = value.opt_cast_into().unwrap();
            let indices = BTreeMap::new();
            Some(TableSchema { primary, indices })
        } else {
            None
        }
    }
}
