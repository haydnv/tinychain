use std::fmt;

use async_hash::{Digest, Hash, Output};
use destream::en;
use safecast::{as_type, CastFrom, CastInto, TryCastFrom};

use tc_error::*;
use tc_value::Value;
use tcgeneric::{NativeClass, TCPathBuf};

use crate::btree::{BTreeSchema, BTreeType};
use crate::table::{TableSchema, TableType};
use crate::tensor::{Schema as TensorSchema, TensorType};
use crate::CollectionType;

/// The schema of a `Collection`.
#[derive(Clone, Eq, PartialEq)]
pub enum Schema {
    BTree(BTreeSchema),
    Table(TableSchema),
    Dense(TensorSchema),
    Sparse(TensorSchema),
}

as_type!(Schema, BTree, BTreeSchema);
as_type!(Schema, Table, TableSchema);

impl<D: Digest> Hash<D> for Schema {
    fn hash(self) -> Output<D> {
        async_hash::Hash::<D>::hash(&self)
    }
}

impl<'a, D: Digest> Hash<D> for &'a Schema {
    fn hash(self) -> Output<D> {
        match self {
            Schema::BTree(schema) => Hash::<D>::hash((BTreeType::default().path(), schema)),
            Schema::Table(schema) => Hash::<D>::hash((TableType::default().path(), schema)),
            Schema::Dense(schema) => Hash::<D>::hash((TensorType::Dense.path(), schema)),
            Schema::Sparse(schema) => Hash::<D>::hash((TensorType::Sparse.path(), schema)),
        }
    }
}

impl TryFrom<(TCPathBuf, Value)> for Schema {
    type Error = TCError;

    fn try_from(value: (TCPathBuf, Value)) -> Result<Self, Self::Error> {
        let (classpath, schema) = value;

        let class = CollectionType::from_path(&classpath)
            .ok_or_else(|| bad_request!("invalid collection type: {}", classpath))?;

        match class {
            CollectionType::BTree(_) => BTreeSchema::try_cast_from_value(schema).map(Self::BTree),
            CollectionType::Table(_) => TableSchema::try_cast_from_value(schema).map(Self::Table),
            CollectionType::Tensor(class) => {
                let schema = TensorSchema::try_cast_from(schema, |v| {
                    bad_request!("invalid Tensor schema: {v:?}")
                })?;

                match class {
                    TensorType::Dense => Ok(Self::Dense(schema)),
                    TensorType::Sparse => Ok(Self::Sparse(schema)),
                }
            }
        }
    }
}

impl<'en> en::IntoStream<'en> for Schema {
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

            Self::Dense(schema) | Self::Sparse(schema) => {
                let mut map = encoder.encode_map(Some(1))?;
                map.encode_entry(TensorType::Dense.path(), (schema,))?;
                map.end()
            }
        }
    }
}

impl CastFrom<Schema> for Value {
    fn cast_from(schema: Schema) -> Self {
        match schema {
            Schema::BTree(schema) => schema.cast_into(),
            Schema::Table(schema) => schema.cast_into(),
            Schema::Dense(schema) => schema.cast_into(),
            Schema::Sparse(schema) => schema.cast_into(),
        }
    }
}

impl fmt::Debug for Schema {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BTree(schema) => fmt::Debug::fmt(schema, f),
            Self::Table(schema) => fmt::Debug::fmt(schema, f),
            Self::Dense(schema) => fmt::Debug::fmt(schema, f),
            Self::Sparse(schema) => fmt::Debug::fmt(schema, f),
        }
    }
}
