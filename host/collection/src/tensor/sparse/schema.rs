use std::{fmt, io};

use tc_error::*;
use tc_value::Number;

use crate::tensor::{Axes, Shape};

use super::BLOCK_SIZE;

#[derive(Clone, Eq, PartialEq)]
pub struct IndexSchema {
    columns: Axes,
}

impl IndexSchema {
    pub fn new(columns: Axes) -> Self {
        Self { columns }
    }
}

impl b_table::b_tree::Schema for IndexSchema {
    type Error = TCError;
    type Value = Number;

    fn block_size(&self) -> usize {
        BLOCK_SIZE
    }

    fn len(&self) -> usize {
        self.columns.len()
    }

    fn order(&self) -> usize {
        12
    }

    fn validate(&self, key: Vec<Self::Value>) -> Result<Vec<Self::Value>, Self::Error> {
        if key.len() == self.len() {
            Ok(key)
        } else {
            Err(io::Error::new(io::ErrorKind::InvalidData, "wrong number of values").into())
        }
    }
}

impl b_table::IndexSchema for IndexSchema {
    type Id = usize;

    fn columns(&self) -> &[Self::Id] {
        &self.columns
    }

    fn extract_key(&self, key: &[Self::Value], other: &Self) -> Vec<Self::Value> {
        debug_assert_eq!(key.len(), self.columns.len());
        other.columns.iter().copied().map(|x| key[x]).collect()
    }
}

impl fmt::Debug for IndexSchema {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("sparse tensor index schema")
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct Schema {
    primary: IndexSchema,
    auxiliary: Vec<(String, IndexSchema)>,
    shape: Shape,
}

impl Schema {
    pub fn new(shape: Shape) -> Self {
        let primary = IndexSchema::new((0..shape.len() + 1).into_iter().collect());
        let mut auxiliary = Vec::with_capacity(shape.len());
        for x in 0..shape.len() {
            let mut columns = Vec::with_capacity(shape.len());
            columns.push(x);

            for xo in 0..shape.len() {
                if xo != x {
                    columns.push(xo);
                }
            }

            let index_schema = IndexSchema::new(columns);
            auxiliary.push((x.to_string(), index_schema));
        }

        Self {
            primary,
            auxiliary,
            shape,
        }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }
}

impl b_table::Schema for Schema {
    type Id = usize;
    type Error = TCError;
    type Value = Number;
    type Index = IndexSchema;

    fn key(&self) -> &[Self::Id] {
        &self.primary.columns[..self.shape.len()]
    }

    fn values(&self) -> &[Self::Id] {
        &self.primary.columns[self.shape.len()..]
    }

    fn primary(&self) -> &Self::Index {
        &self.primary
    }

    fn auxiliary(&self) -> &[(String, IndexSchema)] {
        &self.auxiliary
    }

    fn validate_key(&self, key: Vec<Self::Value>) -> Result<Vec<Self::Value>, Self::Error> {
        if key.len() == self.shape.len() {
            Ok(key)
        } else {
            let cause = io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid key: {:?}", key),
            );

            Err(cause.into())
        }
    }

    fn validate_values(&self, values: Vec<Self::Value>) -> Result<Vec<Self::Value>, Self::Error> {
        if values.len() == 1 {
            Ok(values)
        } else {
            let cause = io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid values: {:?}", values),
            );

            Err(cause.into())
        }
    }
}

impl fmt::Debug for Schema {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("sparse tensor schema")
    }
}
