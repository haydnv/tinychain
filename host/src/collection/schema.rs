use std::convert::TryFrom;
use std::fmt;

use async_hash::Hash;
use destream::en;
use log::debug;
use safecast::{CastFrom, CastInto, TryCastFrom, TryCastInto};
use sha2::digest::Output;
use sha2::Sha256;

use tc_btree::BTreeType;
use tc_error::*;
use tc_table::TableType;
#[cfg(feature = "tensor")]
use tc_tensor::TensorType;
use tc_value::Value;
use tcgeneric::{Class, NativeClass, PathSegment, TCPath, TCPathBuf};

use crate::scalar::{OpRef, Scalar, TCRef};

use super::PREFIX;

/// The [`Class`] of a [`Collection`].
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum CollectionType {
    BTree(BTreeType),
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
                "btree" => BTreeType::from_path(path).map(Self::BTree),
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
            Self::BTree(btt) => btt.path(),
            Self::Table(tt) => tt.path(),
            #[cfg(feature = "tensor")]
            Self::Tensor(tt) => tt.path(),
        }
    }
}

impl From<BTreeType> for CollectionType {
    fn from(btt: BTreeType) -> Self {
        Self::BTree(btt)
    }
}

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
            Self::BTree(btt) => fmt::Display::fmt(btt, f),
            Self::Table(tt) => fmt::Display::fmt(tt, f),
            #[cfg(feature = "tensor")]
            Self::Tensor(tt) => fmt::Display::fmt(tt, f),
        }
    }
}

/// The schema of a [`Chain`] whose [`Subject`] is a `Collection`.
#[derive(Clone, Eq, PartialEq)]
pub enum CollectionSchema {
    BTree(tc_btree::RowSchema),
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
                        .ok_or_else(|| TCError::bad_request("invalid Collection type", class))?;

                    fn expect_value(scalar: Scalar) -> TCResult<Value> {
                        Value::try_cast_from(scalar, |s| {
                            TCError::bad_request("expected a Value for chain schema, not", s)
                        })
                    }

                    match class {
                        CollectionType::BTree(_) => {
                            let schema = expect_value(schema)?;

                            let schema = schema.try_cast_into(|s| {
                                TCError::bad_request("invalid BTree schema", s)
                            })?;

                            Ok(Self::BTree(schema))
                        }
                        CollectionType::Table(_) => {
                            let schema = expect_value(schema)?;

                            let schema = schema.try_cast_into(|s| {
                                TCError::bad_request("invalid Table schema", s)
                            })?;

                            Ok(Self::Table(schema))
                        }

                        #[cfg(feature = "tensor")]
                        CollectionType::Tensor(tt) => {
                            let schema = expect_value(schema)?;
                            let schema = schema.try_cast_into(|v| {
                                TCError::bad_request("invalid Tensor schema", v)
                            })?;

                            match tt {
                                TensorType::Dense => Ok(Self::Dense(schema)),
                                TensorType::Sparse => Ok(Self::Sparse(schema)),
                            }
                        }
                    }
                }
                other => Err(TCError::bad_request("invalid Collection schema", other)),
            },
            other => Err(TCError::bad_request("invalid Collection schema", other)),
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
            Self::BTree(schema) => {
                let mut map = encoder.encode_map(Some(1))?;
                map.encode_entry(BTreeType::default().path(), (schema,))?;
                map.end()
            }

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
            Self::BTree(schema) => {
                let mut map = encoder.encode_map(Some(1))?;
                map.encode_entry(BTreeType::default().path(), (schema,))?;
                map.end()
            }

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

impl CastFrom<CollectionSchema> for TCRef {
    fn cast_from(schema: CollectionSchema) -> TCRef {
        let class: CollectionType = match schema {
            CollectionSchema::BTree(_) => BTreeType::default().into(),
            CollectionSchema::Table(_) => TableType::default().into(),
            #[cfg(feature = "tensor")]
            CollectionSchema::Dense(_) => TensorType::Dense.into(),
            #[cfg(feature = "tensor")]
            CollectionSchema::Sparse(_) => TensorType::Sparse.into(),
        };

        let schema = match schema {
            CollectionSchema::BTree(schema) => {
                Value::Tuple(schema.into_iter().map(Value::from).collect())
            }
            CollectionSchema::Table(schema) => schema.cast_into(),
            #[cfg(feature = "tensor")]
            CollectionSchema::Dense(schema) => schema.cast_into(),
            #[cfg(feature = "tensor")]
            CollectionSchema::Sparse(schema) => schema.cast_into(),
        };

        TCRef::Op(OpRef::Get((class.path().into(), schema.into())))
    }
}

impl fmt::Display for CollectionSchema {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BTree(schema) => write!(f, "{:?}", schema),
            Self::Table(schema) => fmt::Display::fmt(schema, f),
            #[cfg(feature = "tensor")]
            Self::Dense(schema) => fmt::Display::fmt(schema, f),
            #[cfg(feature = "tensor")]
            Self::Sparse(schema) => fmt::Display::fmt(schema, f),
        }
    }
}
