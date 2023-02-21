use std::convert::TryFrom;
use std::fmt;

use async_hash::Hash;
use destream::de::Error;
use destream::en;
use log::debug;
use safecast::{CastFrom, CastInto, TryCastFrom, TryCastInto};
use sha2::digest::Output;
use sha2::Sha256;

#[cfg(feature = "btree")]
use tc_btree::BTreeType;
use tc_error::*;
#[cfg(feature = "table")]
use tc_table::TableType;
#[cfg(feature = "tensor")]
use tc_tensor::TensorType;
use tc_value::Value;
use tcgeneric::{Class, NativeClass, PathSegment, TCPath, TCPathBuf};

use crate::scalar::{OpRef, Scalar, TCRef};

use super::PREFIX;

/// The [`Class`] of a `Collection`.
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum CollectionType {
    #[cfg(feature = "btree")]
    BTree(BTreeType),
    #[cfg(feature = "table")]
    Table(TableType),
    #[cfg(feature = "tensor")]
    Tensor(TensorType),
}

impl Class for CollectionType {}

impl NativeClass for CollectionType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        debug!("CollectionType::from_path {}", TCPath::from(path));

        if path.len() > 2 && &path[0..2] == &PREFIX[..] {
            match path[2].as_str() {
                #[cfg(feature = "btree")]
                "btree" => BTreeType::from_path(path).map(Self::BTree),
                #[cfg(feature = "table")]
                "table" => TableType::from_path(path).map(Self::Table),
                #[cfg(feature = "tensor")]
                "tensor" => TensorType::from_path(path).map(Self::Tensor),
                _ => None,
            }
        } else {
            None
        }
    }

    fn path(&self) -> TCPathBuf {
        match self {
            #[cfg(feature = "btree")]
            Self::BTree(btt) => btt.path(),
            #[cfg(feature = "table")]
            Self::Table(tt) => tt.path(),
            #[cfg(feature = "tensor")]
            Self::Tensor(tt) => tt.path(),

            _ => unimplemented!("no collection flags enabled")
        }
    }
}

#[cfg(feature = "btree")]
impl From<BTreeType> for CollectionType {
    fn from(btt: BTreeType) -> Self {
        Self::BTree(btt)
    }
}

#[cfg(feature = "table")]
impl From<TableType> for CollectionType {
    fn from(tt: TableType) -> Self {
        Self::Table(tt)
    }
}

#[cfg(feature = "tensor")]
impl From<TensorType> for CollectionType {
    fn from(tt: TensorType) -> Self {
        Self::Tensor(tt)
    }
}

impl fmt::Display for CollectionType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            #[cfg(feature = "btree")]
            Self::BTree(btt) => fmt::Display::fmt(btt, f),
            #[cfg(feature = "table")]
            Self::Table(tt) => fmt::Display::fmt(tt, f),
            #[cfg(feature = "tensor")]
            Self::Tensor(tt) => fmt::Display::fmt(tt, f),

            _ => unimplemented!("no collection flags enabled")
        }
    }
}

/// The schema of a `Collection`.
#[derive(Clone, Eq, PartialEq)]
pub enum CollectionSchema {
    #[cfg(feature = "btree")]
    BTree(tc_btree::RowSchema),
    #[cfg(feature = "table")]
    Table(tc_table::TableSchema),
    #[cfg(feature = "tensor")]
    Dense(tc_tensor::Schema),
    #[cfg(feature = "tensor")]
    Sparse(tc_tensor::Schema),
}

impl CollectionSchema {
    pub fn from_scalar(tc_ref: TCRef) -> TCResult<Self> {
        match tc_ref {
            TCRef::Op(op_ref) => match op_ref {
                OpRef::Get((class, schema)) => {
                    let class = TCPathBuf::try_from(class)?;
                    let class = CollectionType::from_path(&class)
                        .ok_or_else(|| TCError::invalid_type(class, "a Collection class"))?;

                    fn expect_value(scalar: Scalar) -> TCResult<Value> {
                        Value::try_cast_from(scalar, |s| {
                            TCError::invalid_type(s, "a Value for a Chain schema")
                        })
                    }

                    match class {
                        #[cfg(feature = "btree")]
                        CollectionType::BTree(_) => {
                            let schema = expect_value(schema)?;

                            let schema = schema
                                .try_cast_into(|s| TCError::invalid_value(s, "a BTree schema"))?;

                            Ok(Self::BTree(schema))
                        }
                        #[cfg(feature = "table")]
                        CollectionType::Table(_) => {
                            let schema = expect_value(schema)?;

                            let schema = schema
                                .try_cast_into(|s| TCError::invalid_value(s, "a Table schema"))?;

                            Ok(Self::Table(schema))
                        }

                        #[cfg(feature = "tensor")]
                        CollectionType::Tensor(tt) => {
                            let schema = expect_value(schema)?;
                            let schema = schema
                                .try_cast_into(|v| TCError::invalid_value(v, "a Tensor schema"))?;

                            match tt {
                                TensorType::Dense => Ok(Self::Dense(schema)),
                                TensorType::Sparse => Ok(Self::Sparse(schema)),
                            }
                        }
                    }
                }
                other => Err(TCError::invalid_value(other, "a Collection schema")),
            },
            other => Err(TCError::invalid_value(other, "a Collection schema")),
        }
    }
}

impl Hash<Sha256> for CollectionSchema {
    fn hash(self) -> Output<Sha256> {
        async_hash::Hash::<Sha256>::hash(&TCRef::cast_from(self))
    }
}

impl<'en> en::IntoStream<'en> for CollectionSchema {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        use destream::en::EncodeMap;

        match self {
            #[cfg(feature = "btree")]
            Self::BTree(schema) => {
                let mut map = encoder.encode_map(Some(1))?;
                map.encode_entry(BTreeType::default().path(), (schema,))?;
                map.end()
            }

            #[cfg(feature = "table")]
            Self::Table(schema) => {
                let mut map = encoder.encode_map(Some(1))?;
                map.encode_entry(TableType::default().path(), (schema,))?;
                map.end()
            }

            #[cfg(feature = "tensor")]
            Self::Dense(schema) | Self::Sparse(schema) => {
                let mut map = encoder.encode_map(Some(1))?;
                map.encode_entry(TensorType::Dense.path(), (schema,))?;
                map.end()
            }
        }
    }
}

impl<'en> en::ToStream<'en> for CollectionSchema {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        use destream::en::EncodeMap;

        match self {
            #[cfg(feature = "btree")]
            Self::BTree(schema) => {
                let mut map = encoder.encode_map(Some(1))?;
                map.encode_entry(BTreeType::default().path(), (schema,))?;
                map.end()
            }

            #[cfg(feature = "table")]
            Self::Table(schema) => {
                let mut map = encoder.encode_map(Some(1))?;
                map.encode_entry(TableType::default().path(), (schema,))?;
                map.end()
            }

            #[cfg(feature = "tensor")]
            Self::Dense(schema) | Self::Sparse(schema) => {
                let mut map = encoder.encode_map(Some(1))?;
                map.encode_entry(TensorType::Dense.path(), (schema,))?;
                map.end()
            }

            _ => unimplemented!("no collection flags enabled")
        }
    }
}

impl CastFrom<CollectionSchema> for TCRef {
    fn cast_from(schema: CollectionSchema) -> TCRef {
        let class: CollectionType = match schema {
            #[cfg(feature = "btree")]
            CollectionSchema::BTree(_) => BTreeType::default().into(),
            #[cfg(feature = "table")]
            CollectionSchema::Table(_) => TableType::default().into(),
            #[cfg(feature = "tensor")]
            CollectionSchema::Dense(_) => TensorType::Dense.into(),
            #[cfg(feature = "tensor")]
            CollectionSchema::Sparse(_) => TensorType::Sparse.into(),
        };

        let schema: Value = match schema {
            #[cfg(feature = "btree")]
            CollectionSchema::BTree(schema) => {
                Value::Tuple(schema.into_iter().map(Value::from).collect())
            }
            #[cfg(feature = "table")]
            CollectionSchema::Table(schema) => schema.cast_into(),
            #[cfg(feature = "tensor")]
            CollectionSchema::Dense(schema) => schema.cast_into(),
            #[cfg(feature = "tensor")]
            CollectionSchema::Sparse(schema) => schema.cast_into(),

            _ => unimplemented!("no collection flags enabled")
        };

        TCRef::Op(OpRef::Get((class.path().into(), schema.into())))
    }
}

impl fmt::Display for CollectionSchema {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            #[cfg(feature = "btree")]
            Self::BTree(schema) => write!(f, "{:?}", schema),
            #[cfg(feature = "table")]
            Self::Table(schema) => fmt::Display::fmt(schema, f),
            #[cfg(feature = "tensor")]
            Self::Dense(schema) => fmt::Display::fmt(schema, f),
            #[cfg(feature = "tensor")]
            Self::Sparse(schema) => fmt::Display::fmt(schema, f),

            _ => unimplemented!("no collection flags enabled")
        }
    }
}
