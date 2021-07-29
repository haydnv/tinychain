use std::collections::{HashMap, HashSet};
use std::fmt;
use std::iter::FromIterator;

use async_trait::async_trait;
use destream::{de, en};
use futures::TryFutureExt;
use safecast::*;

use tc_error::*;
use tc_value::{Value, ValueType};
use tcgeneric::{Id, Map, Tuple};

use super::{Key, Values};

pub use tc_btree::Column;

/// A `Table` row
pub type Row = Map<Value>;

/// The schema of a `Table` row
pub type RowSchema = Vec<Column>;

/// The schema of a `Table` index
#[derive(Clone, Eq, PartialEq)]
pub struct IndexSchema {
    key: RowSchema,
    values: RowSchema,
}

impl IndexSchema {
    /// Return a list of the columns in this schema.
    pub fn columns(&self) -> Vec<Column> {
        [&self.key[..], &self.values[..]].concat()
    }

    /// Iterate over the names of the columns in this schema.
    pub fn column_names(&self) -> impl Iterator<Item = &Id> {
        self.key
            .iter()
            .map(|col| &col.name)
            .chain(self.values.iter().map(|col| &col.name))
    }

    /// Return a slice of the columns in this schema's key.
    pub fn key(&self) -> &[Column] {
        &self.key
    }

    /// Return a slice of the columns in this schema's values.
    pub fn values(&self) -> &[Column] {
        &self.values
    }

    /// Return the number of columns in this schema.
    pub fn len(&self) -> usize {
        self.key.len() + self.values.len()
    }

    /// Given a [`Row`], return its key.
    pub fn key_from_row(&self, row: &Row) -> TCResult<Key> {
        let mut key = Vec::with_capacity(self.key().len());
        for col in self.key() {
            let value = row
                .get(col.name())
                .cloned()
                .ok_or(TCError::not_found(format!(
                    "value of row {} at column {}",
                    row, col.name
                )))?;

            key.push(value);
        }

        Ok(key)
    }

    /// Given a [`Row`], return a `(key, values)` tuple.
    pub fn key_values_from_row(
        &self,
        mut row: Row,
        reject_extras: bool,
    ) -> TCResult<(Key, Values)> {
        if reject_extras && self.len() != row.len() {
            return Err(TCError::unsupported(format!(
                "invalid row for schema {}: {}",
                self,
                Map::from(row)
            )));
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

    /// Given a [`Row`], return a `(key, values)` tuple.
    pub fn key_values_from_tuple(&self, tuple: Tuple<Value>) -> TCResult<(Vec<Value>, Vec<Value>)> {
        if self.len() != tuple.len() {
            return Err(TCError::unsupported(format!(
                "{} is not a valid row for schema {}",
                tuple, self
            )));
        }

        let mut key = tuple.into_inner();
        let values = key.split_off(self.key.len());
        Ok((key, values))
    }

    /// Given a `key` and `values`, return a [`Row`].
    pub fn row_from_key_values(&self, key: Key, values: Values) -> TCResult<Row> {
        assert_eq!(key.len(), self.key.len());
        assert_eq!(values.len(), self.values.len());

        let mut row = key;
        row.extend(values);
        self.row_from_values(row)
    }

    /// Given a list of `Value`s, return a [`Row`].
    pub fn row_from_values(&self, values: Values) -> TCResult<Row> {
        assert_eq!(values.len(), self.len());

        let mut row = Map::new();
        for (column, value) in self.columns().into_iter().zip(values.into_iter()) {
            let value = column.dtype.try_cast(value)?;
            row.insert(column.name, value);
        }

        Ok(row)
    }

    /// Return `true` if this schema starts with the given slice of column names.
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

    /// Return the `IndexSchema` needed to index the given columns.
    pub fn auxiliary(&self, key: &[Id]) -> TCResult<IndexSchema> {
        let subset: HashSet<&Id> = key.iter().collect();

        let mut columns: HashMap<Id, Column> = self
            .columns()
            .iter()
            .cloned()
            .map(|c| (c.name().clone(), c))
            .collect();

        let key = key.iter().map(|col_name| {
            columns
                .remove(col_name)
                .ok_or_else(|| TCError::not_found(col_name))
        });

        let values = self
            .key()
            .iter()
            .filter(|c| !subset.contains(c.name()))
            .cloned()
            .map(Ok);

        let key = key.chain(values).collect::<TCResult<Vec<Column>>>()?;
        Ok((key, vec![]).into())
    }

    /// Return an error if this schema does not support ordering by the given columns.
    pub fn validate_columns(&self, columns: &[Id]) -> TCResult<()> {
        let valid_columns: HashSet<Id> = self.columns().iter().map(|c| c.name()).cloned().collect();
        for column in columns {
            if !valid_columns.contains(column) {
                return Err(TCError::not_found(column));
            }
        }

        Ok(())
    }

    /// Return an error if the given key does not match this schema.
    #[inline]
    pub fn validate_key(&self, key: Key) -> TCResult<Key> {
        if key.len() != self.key.len() {
            return Err(TCError::unsupported(format!(
                "invalid key {} for schema {}",
                Tuple::from(key),
                self
            )));
        }

        let mut validated = Vec::with_capacity(key.len());
        for (val, col) in key.into_iter().zip(self.key.iter()) {
            let value = col.dtype.try_cast(val)?;
            validated.push(value);
        }

        Ok(validated)
    }

    /// Return an error if the given values do not match this schema.
    #[inline]
    pub fn validate_values(&self, values: Values) -> TCResult<Key> {
        if values.len() != self.values.len() {
            return Err(TCError::unsupported(format!(
                "invalid values {} for schema {}",
                Tuple::from(values),
                Tuple::<&Column>::from_iter(&self.values)
            )));
        }

        let mut validated = Vec::with_capacity(values.len());
        for (val, col) in values.into_iter().zip(self.values.iter()) {
            let value = col.dtype.try_cast(val)?;
            validated.push(value);
        }

        Ok(validated)
    }

    /// Return an error if the given [`Row`] has any extra fields or incompatible values.
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

    /// Return an error if the given [`Row`] does not have a compatible value for every column.
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

    /// Given a [`Row`], return an ordered list of [`Value`]s.
    pub fn values_from_row(&self, mut row: Row, reject_extras: bool) -> TCResult<Vec<Value>> {
        let mut key = Vec::with_capacity(self.len());
        for column in self.columns() {
            let value = row
                .remove(&column.name)
                .ok_or_else(|| TCError::bad_request("missing value for column", &column.name))?;

            let value = column.dtype.try_cast(value)?;
            key.push(value);
        }

        if reject_extras && !row.is_empty() {
            return Err(TCError::bad_request(
                &format!(
                    "unrecognized columns (`{}`) for schema",
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

impl CastFrom<IndexSchema> for Value {
    fn cast_from(schema: IndexSchema) -> Self {
        Value::Tuple(
            vec![
                Value::from_iter(schema.key),
                Value::from_iter(schema.values),
            ]
            .into(),
        )
    }
}

#[async_trait]
impl de::FromStream for IndexSchema {
    type Context = ();

    async fn from_stream<D: de::Decoder>(cxt: (), decoder: &mut D) -> Result<Self, D::Error> {
        de::FromStream::from_stream(cxt, decoder)
            .map_ok(|(key, values)| Self { key, values })
            .await
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

/// The schema of a `Table`.
#[derive(Clone, Eq, PartialEq)]
pub struct TableSchema {
    primary: IndexSchema,
    indices: Vec<(Id, Vec<Id>)>,
}

impl TableSchema {
    /// Construct a new `Table` schema.
    pub fn new<I: IntoIterator<Item = (Id, Vec<Id>)>>(primary: IndexSchema, indices: I) -> Self {
        Self {
            primary,
            indices: indices.into_iter().collect(),
        }
    }

    /// Return a list of index names and the names of the columns they index.
    pub fn indices(&self) -> &[(Id, Vec<Id>)] {
        &self.indices
    }

    /// Return the [`IndexSchema`] of this `TableSchema`'s primary index.
    pub fn primary(&self) -> &IndexSchema {
        &self.primary
    }
}

#[async_trait]
impl de::FromStream for TableSchema {
    type Context = ();

    async fn from_stream<D: de::Decoder>(cxt: (), decoder: &mut D) -> Result<Self, D::Error> {
        de::FromStream::from_stream(cxt, decoder)
            .map_ok(|(primary, indices)| Self { primary, indices })
            .await
    }
}

impl<'en> en::IntoStream<'en> for TableSchema {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        (self.primary, self.indices).into_stream(encoder)
    }
}

impl From<IndexSchema> for TableSchema {
    fn from(schema: IndexSchema) -> TableSchema {
        TableSchema {
            primary: schema,
            indices: vec![],
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
            let indices = vec![];
            Some(TableSchema { primary, indices })
        } else {
            None
        }
    }
}

impl CastFrom<TableSchema> for Value {
    fn cast_from(schema: TableSchema) -> Self {
        let indices = schema
            .indices
            .into_iter()
            .map(|(id, col_names)| (Value::from(id), Tuple::<Value>::from_iter(col_names)))
            .map(|(id, col_names)| Value::Tuple(vec![id, col_names.into()].into()));

        Self::Tuple(vec![schema.primary.cast_into(), Value::from_iter(indices)].into())
    }
}

impl fmt::Display for TableSchema {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "primary: {}", self.primary)?;
        if !self.indices.is_empty() {
            writeln!(f, "indices:")?;
            for (name, columns) in &self.indices {
                writeln!(f, "{}: {}", name, Tuple::<&Id>::from_iter(columns))?;
            }
        }

        Ok(())
    }
}
