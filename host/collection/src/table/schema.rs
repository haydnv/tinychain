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

        let mut columns = HashMap::with_capacity(key.len() + values.len());
        columns.extend(key.iter().map(|col| (&col.name, col)));
        columns.extend(values.iter().map(|col| (&col.name, col)));

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
                    .chain(key.iter().cloned().map(Ok))
                    .collect::<TCResult<Vec<Column>>>()?;

                let index_schema = IndexSchema::new(columns)?;
                Ok((index_name, index_schema))
            })
            .collect::<TCResult<Vec<_>>>()?;

        let mut primary = Vec::with_capacity(key.len() + values.len());
        primary.extend(key);
        primary.extend(values);

        let primary = IndexSchema::new(primary)?;

        Ok(Self {
            key: key_names,
            values: value_names,
            primary,
            indices,
        })
    }

    #[inline]
    fn pack(self) -> ((Vec<Column>, Vec<Column>), Vec<(String, Vec<Id>)>) {
        let key_len = self.key.len();

        let mut columns = self.primary.into_iter();

        let mut key = Vec::with_capacity(key_len);
        for _ in 0..key_len {
            let column = columns.next().expect("column");
            key.push(column);
        }

        let values = columns.collect();

        let indices = self
            .indices
            .into_iter()
            .map(|(name, schema)| {
                let len = b_table::b_tree::Schema::len(&schema) - key_len;
                let columns = schema.into_iter().take(len).map(|col| col.name).collect();

                (name, columns)
            })
            .collect();

        ((key, values), indices)
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
        let ((key, values), indices) = schema.pack();

        let key = key.into_iter().map(Value::cast_from).collect::<Value>();
        let values = values.into_iter().map(Value::cast_from).collect::<Value>();

        let indices = indices
            .into_iter()
            .map(|(name, columns)| {
                (
                    Value::from(name),
                    columns
                        .into_iter()
                        .map(Value::from)
                        .collect::<Value>(),
                )
            })
            .map(|(name, columns)| Value::Tuple(vec![name, columns].into()))
            .collect::<Value>();

        Value::Tuple(vec![vec![key, values].into(), indices].into())
    }
}

impl<'en> en::IntoStream<'en> for TableSchema {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        self.pack().into_stream(encoder)
    }
}

#[async_trait]
impl de::FromStream for TableSchema {
    type Context = ();

    async fn from_stream<D: de::Decoder>(cxt: (), decoder: &mut D) -> Result<Self, D::Error> {
        let value = Value::from_stream(cxt, decoder).await?;
        Self::try_cast_from_value(value).map_err(de::Error::custom)
    }
}

impl fmt::Debug for TableSchema {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.primary, f)
    }
}
