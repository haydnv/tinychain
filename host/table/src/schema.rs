use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;

use destream::en;
use safecast::*;

use tc_error::*;
use tc_value::{Value, ValueType};
use tcgeneric::{Id, Map};

pub use tc_btree::Column;

pub type Row = Map<Value>;
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

    pub fn column_names(&self) -> impl Iterator<Item = &Id> {
        self.key
            .iter()
            .map(|col| &col.name)
            .chain(self.values.iter().map(|col| &col.name))
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
            return Err(TCError::bad_request("Invalid row for schema", self));
        }

        let mut key = Vec::with_capacity(self.key().len());
        for col in self.key() {
            let value = row
                .remove(col.name())
                .ok_or(TCError::not_found(col.name()))?;
            key.push(value);
        }

        let mut values = Vec::with_capacity(self.values().len());
        for col in self.values() {
            let value = row
                .remove(col.name())
                .ok_or(TCError::not_found(col.name()))?;
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

        Ok(row.into())
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
                return Err(TCError::not_found(column));
            }
        }

        Ok(())
    }

    pub fn validate_key(&self, key: Vec<Value>) -> TCResult<Vec<Value>> {
        if key.len() != self.key.len() {
            let key_columns: Vec<String> = self.key.iter().map(|c| c.to_string()).collect();
            return Err(TCError::bad_request(
                format!("Invalid key {}, expected", Value::Tuple(key.into())),
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
                .ok_or(TCError::bad_request("No such column", &col_name))?;

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
            return Err(TCError::bad_request(
                "Row is missing columns",
                missing
                    .into_iter()
                    .map(|c| (*c).to_string())
                    .collect::<Vec<String>>()
                    .join(", "),
            ));
        }

        if !extra.is_empty() {
            return Err(TCError::bad_request(
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
                .ok_or_else(|| TCError::bad_request("Missing value for column", &column.name))?;
            let value = column.dtype.try_cast(value)?;
            key.push(value);
        }

        if reject_extras && !row.is_empty() {
            return Err(TCError::bad_request(
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

impl From<IndexSchema> for Map<Column> {
    fn from(schema: IndexSchema) -> Map<Column> {
        schema
            .key
            .into_iter()
            .chain(schema.values.into_iter())
            .map(|c| (c.name.clone(), c))
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

impl<'en> en::IntoStream<'en> for IndexSchema {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        (self.key, self.values).into_stream(encoder)
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
    pub fn new<I: IntoIterator<Item = (Id, Vec<Id>)>>(primary: IndexSchema, indices: I) -> Self {
        Self {
            primary,
            indices: indices.into_iter().collect(),
        }
    }

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

impl<'en> en::IntoStream<'en> for TableSchema {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        (self.primary, self.indices).into_stream(encoder)
    }
}
