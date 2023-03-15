use std::fmt;

use async_hash::{Digest, Hash, Output};
use destream::en;
use tc_error::{bad_request, TCError};
use tc_value::Value;

use tcgeneric::Id;

use crate::btree::{Column, Schema as BTreeSchema};

#[derive(Clone, Eq, PartialEq)]
pub struct Schema {
    key: Vec<Id>,
    values: Vec<Id>,
    primary: BTreeSchema,
    indices: Vec<(String, BTreeSchema)>,
}

impl b_table::Schema for Schema {
    type Id = Id;
    type Error = TCError;
    type Value = Value;
    type Index = BTreeSchema;

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

impl<'a, D: Digest> Hash<D> for &'a Schema {
    fn hash(self) -> Output<D> {
        Hash::<D>::hash((&self.primary, &self.indices))
    }
}

impl<'en> en::IntoStream<'en> for Schema {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        let key_len = self.key.len();

        let mut columns = self.primary.into_iter();

        let mut key = Vec::with_capacity(key_len);
        for i in 0..key_len {
            key.push(columns.next().expect("column"));
        }

        let values = columns.collect::<Vec<Column>>();

        let indices = self
            .indices
            .into_iter()
            .map(|(name, schema)| {
                let len = b_table::b_tree::Schema::len(&schema) - key_len;
                let columns = schema
                    .into_iter()
                    .take(len)
                    .map(|col| col.name)
                    .collect::<Vec<_>>();

                (name, columns)
            })
            .collect::<Vec<(String, Vec<Id>)>>();

        ((key, values), indices).into_stream(encoder)
    }
}

impl fmt::Debug for Schema {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.primary, f)
    }
}
