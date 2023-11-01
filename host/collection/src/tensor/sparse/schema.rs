//! Tensor schema types

use itertools::Itertools;
use std::{fmt, io};

use tc_error::*;
use tc_value::Number;

use crate::tensor::Shape;

use super::BLOCK_SIZE;

#[derive(Clone, Eq, PartialEq)]
pub struct IndexSchema {
    columns: Vec<usize>,
}

impl IndexSchema {
    pub fn new(columns: Vec<usize>) -> Self {
        Self { columns }
    }
}

impl b_table::BTreeSchema for IndexSchema {
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

    fn validate_key(&self, key: Vec<Self::Value>) -> Result<Vec<Self::Value>, Self::Error> {
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
}

impl fmt::Debug for IndexSchema {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "sparse tensor index on axes {:?}", self.columns)
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
        let primary = IndexSchema::new((0..=shape.len()).into_iter().collect());

        let auxiliary = if shape.len() <= 4 {
            let mut permutations = (0..shape.len()).permutations(shape.len());

            assert_eq!(
                permutations.next().as_ref().map(|cols| cols.as_slice()),
                Some(&primary.columns[..shape.len()])
            );

            permutations
                .map(|axes| {
                    let name = axes.iter().join("-");
                    let index_schema = IndexSchema::new(axes);
                    (name, index_schema)
                })
                .collect()
        } else {
            Vec::default()
        };

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
