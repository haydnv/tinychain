use std::collections::HashMap;
use std::fmt;

use async_hash::{Digest, Hash, Output};
use async_trait::async_trait;
use destream::{de, en};
use safecast::{CastFrom, TryCastInto};

use tc_error::*;
use tc_value::Value;
use tcgeneric::Id;

use crate::btree::{BTreeSchema as IndexSchema, Column};
use crate::table::{Key, Range};

#[derive(Clone, Eq, PartialEq)]
pub struct TableSchema {
    key: Vec<Id>,
    values: Vec<Id>,
    primary: IndexSchema,
    indices: Vec<(String, IndexSchema)>,
}

impl TableSchema {
    pub(super) fn from_index(primary: IndexSchema) -> Self {
        Self {
            key: vec![],
            values: primary.iter().map(|col| &col.name).cloned().collect(),
            primary,
            indices: vec![],
        }
    }

    pub(super) fn range_from_key(&self, key: Key) -> TCResult<Range> {
        if key.len() != self.key.len() {
            return Err(bad_request!(
                "invalid key for table with schema {:?}: {:?}",
                self,
                key
            ));
        }

        let mut range = HashMap::with_capacity(key.len());
        for (val, col) in key.into_iter().zip(&self.primary) {
            let val = val
                .into_type(col.dtype)
                .ok_or_else(|| bad_request!("key has an invalid value for column {}", col.name))?;

            range.insert(col.name.clone(), val.into());
        }

        Ok(range.into())
    }

    /// Try to construct a [`TableSchema`] from its [`Value`] representation.
    pub fn try_cast_from_value(value: Value) -> TCResult<Self> {
        let ((key, values), indices): ((Vec<Column>, Vec<Column>), Vec<(String, Vec<Id>)>) =
            value.try_cast_into(|v| bad_request!("invalid table schema: {}", v))?;

        let key_names = key.iter().map(|col| &col.name).cloned().collect();
        let value_names = values.iter().map(|col| &col.name).cloned().collect();

        let mut primary = Vec::with_capacity(key.len() + values.len());
        primary.extend(key);
        primary.extend(values);

        let columns: HashMap<_, _> = primary.iter().map(|col| (&col.name, col)).collect();

        let indices = indices
            .into_iter()
            .map(|(index_name, column_names)| {
                let columns = column_names
                    .into_iter()
                    .map(|name| {
                        columns.get(&name).map(|col| *col).cloned().ok_or_else(|| {
                            bad_request!(
                                "index {} specified nonexistent column {}",
                                index_name,
                                name
                            )
                        })
                    })
                    .collect::<TCResult<Vec<Column>>>()?;

                let index_schema = IndexSchema::new(columns)?;
                Ok((index_name, index_schema))
            })
            .collect::<TCResult<Vec<_>>>()?;

        let primary = IndexSchema::new(primary)?;

        Ok(Self {
            key: key_names,
            values: value_names,
            primary,
            indices,
        })
    }
}

impl b_table::Schema for TableSchema {
    type Id = Id;
    type Error = TCError;
    type Value = Value;
    type Index = IndexSchema;

    fn key(&self) -> &[Self::Id] {
        &self.key
    }

    fn values(&self) -> &[Self::Id] {
        &self.values
    }

    fn primary(&self) -> &Self::Index {
        &self.primary
    }

    fn auxiliary(&self) -> &[(String, Self::Index)] {
        &self.indices
    }

    fn validate_key(&self, key: Vec<Self::Value>) -> Result<Vec<Self::Value>, Self::Error> {
        if key.len() == self.key.len() {
            key.into_iter()
                .zip(&self.primary)
                .map(|(val, col)| {
                    val.into_type(col.dtype).ok_or_else(|| {
                        bad_request!("invalid value for column {} in row key", col.name)
                    })
                })
                .collect()
        } else {
            Err(bad_request!(
                "{:?} is not a valid key for a table with schema {:?} and key columns {:?}",
                key,
                self.primary,
                self.key
            ))
        }
    }

    fn validate_values(&self, values: Vec<Self::Value>) -> Result<Vec<Self::Value>, Self::Error> {
        if values.len() == self.values.len() {
            values
                .into_iter()
                .zip(self.primary.iter().skip(self.key.len()))
                .map(|(val, col)| {
                    val.into_type(col.dtype).ok_or_else(|| {
                        bad_request!("invalid value for column {} in row key", col.name)
                    })
                })
                .collect()
        } else {
            Err(bad_request!(
                "{:?} are not valid values for a table with schema {:?} and value columns {:?}",
                values,
                self.primary,
                self.values
            ))
        }
    }
}

impl<'a, D: Digest> Hash<D> for &'a TableSchema {
    fn hash(self) -> Output<D> {
        Hash::<D>::hash((&self.primary, &self.indices))
    }
}

impl CastFrom<TableSchema> for Value {
    fn cast_from(schema: TableSchema) -> Self {
        let key_len = schema.key.len();

        let mut columns = schema.primary.into_iter();

        let mut key = Vec::with_capacity(key_len);
        for _ in 0..key_len {
            let column = columns.next().expect("column");
            key.push(Value::cast_from(column));
        }
        let key = Value::Tuple(key.into());

        let values = columns.map(Value::cast_from).collect::<Value>();

        let indices = schema
            .indices
            .into_iter()
            .map(|(name, schema)| {
                let len = b_table::b_tree::Schema::len(&schema) - key_len;
                let columns = schema
                    .into_iter()
                    .take(len)
                    .map(|col| col.name)
                    .map(Value::from)
                    .collect::<Value>();

                Value::Tuple(vec![name.into(), columns].into())
            })
            .collect::<Value>();

        Value::Tuple(vec![vec![key, values].into(), indices].into())
    }
}

impl<'en> en::IntoStream<'en> for TableSchema {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        Value::cast_from(self).into_stream(encoder)
    }
}

#[async_trait]
impl de::FromStream for TableSchema {
    type Context = ();

    async fn from_stream<D: de::Decoder>(cxt: (), decoder: &mut D) -> Result<Self, D::Error> {
        let ((key, values), indices): ((Vec<Column>, Vec<Column>), Vec<(String, Vec<Id>)>) =
            de::FromStream::from_stream(cxt, decoder).await?;

        let key_names = key.iter().map(|col| &col.name).cloned().collect();
        let value_names = values.iter().map(|col| &col.name).cloned().collect();

        let mut primary = Vec::with_capacity(key.len() + values.len());
        primary.extend(key);
        primary.extend(values);

        let columns: HashMap<_, _> = primary.iter().map(|col| (&col.name, col)).collect();

        let indices = indices
            .into_iter()
            .map(|(name, column_names)| {
                let columns = column_names
                    .into_iter()
                    .map(|name| {
                        columns.get(&name).map(|col| *col).cloned().ok_or_else(|| {
                            de::Error::invalid_value(name, "there is no such column to index")
                        })
                    })
                    .collect::<Result<Vec<Column>, D::Error>>()?;

                IndexSchema::new(columns)
                    .map(|schema| (name, schema))
                    .map_err(de::Error::custom)
            })
            .collect::<Result<Vec<(String, IndexSchema)>, D::Error>>()?;

        let primary = IndexSchema::new(primary).map_err(de::Error::custom)?;

        Ok(Self {
            key: key_names,
            values: value_names,
            primary,
            indices,
        })
    }
}

impl fmt::Debug for TableSchema {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.primary, f)
    }
}
