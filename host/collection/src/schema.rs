use std::fmt;

use destream::en;
use safecast::CastFrom;
#[cfg(feature = "tensor")]
use safecast::TryCastFrom;
#[cfg(feature = "btree")]
use safecast::{as_type, CastInto};

use tc_error::*;
use tc_transact::hash::{Digest, Hash, Output};
use tc_value::Value;
#[cfg(feature = "btree")]
use tcgeneric::NativeClass;
use tcgeneric::TCPathBuf;

#[cfg(feature = "btree")]
use crate::btree::{BTreeSchema, BTreeType};
#[cfg(feature = "table")]
use crate::table::{TableSchema, TableType};
#[cfg(feature = "tensor")]
use crate::tensor::{Schema as TensorSchema, TensorType};
#[cfg(feature = "btree")]
use crate::CollectionType;

/// The schema of a `Collection`.
#[cfg(feature = "btree")]
#[derive(Clone, Eq, PartialEq)]
pub enum Schema {
    #[cfg(feature = "btree")]
    BTree(BTreeSchema),
    #[cfg(feature = "table")]
    Table(TableSchema),
    #[cfg(feature = "tensor")]
    Dense(TensorSchema),
    #[cfg(feature = "tensor")]
    Sparse(TensorSchema),
}

/// The schema of a `Collection`.
#[cfg(not(feature = "btree"))]
#[derive(Clone, Eq, PartialEq)]
pub struct Schema;

#[cfg(feature = "btree")]
as_type!(Schema, BTree, BTreeSchema);
#[cfg(feature = "table")]
as_type!(Schema, Table, TableSchema);

impl<D: Digest> Hash<D> for Schema {
    fn hash(self) -> Output<D> {
        Hash::<D>::hash(&self)
    }
}

impl<'a, D: Digest> Hash<D> for &'a Schema {
    fn hash(self) -> Output<D> {
        #[cfg(feature = "btree")]
        match self {
            Schema::BTree(schema) => Hash::<D>::hash((BTreeType::default().path(), schema)),
            #[cfg(feature = "table")]
            Schema::Table(schema) => Hash::<D>::hash((TableType::default().path(), schema)),
            #[cfg(feature = "tensor")]
            Schema::Dense(schema) => Hash::<D>::hash((TensorType::Dense.path(), schema)),
            #[cfg(feature = "tensor")]
            Schema::Sparse(schema) => Hash::<D>::hash((TensorType::Sparse.path(), schema)),
        }

        #[cfg(not(feature = "btree"))]
        tc_transact::hash::default_hash::<D>()
    }
}

impl TryFrom<(TCPathBuf, Value)> for Schema {
    type Error = TCError;

    fn try_from(value: (TCPathBuf, Value)) -> Result<Self, Self::Error> {
        #[allow(unused)]
        let (classpath, schema) = value;

        #[cfg(feature = "btree")]
        let class = CollectionType::from_path(&classpath)
            .ok_or_else(|| bad_request!("invalid collection type: {}", classpath))?;

        #[cfg(feature = "btree")]
        match class {
            CollectionType::BTree(_) => BTreeSchema::try_cast_from_value(schema).map(Self::BTree),
            #[cfg(feature = "table")]
            CollectionType::Table(_) => TableSchema::try_cast_from_value(schema).map(Self::Table),
            #[cfg(feature = "tensor")]
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

        #[cfg(not(feature = "btree"))]
        Ok(Self)
    }
}

impl<'en> en::IntoStream<'en> for Schema {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        use destream::en::EncodeMap;

        #[allow(unused_mut)]
        let mut map = encoder.encode_map(Some(1))?;

        #[cfg(feature = "btree")]
        match self {
            Self::BTree(schema) => {
                map.encode_entry(BTreeType::default().path(), (schema,))?;
            }

            #[cfg(feature = "table")]
            Self::Table(schema) => {
                map.encode_entry(TableType::default().path(), (schema,))?;
            }

            #[cfg(feature = "tensor")]
            Self::Dense(schema) | Self::Sparse(schema) => {
                map.encode_entry(TensorType::Dense.path(), (schema,))?;
            }
        }

        map.end()
    }
}

impl CastFrom<Schema> for Value {
    #[allow(unused)]
    fn cast_from(schema: Schema) -> Self {
        #[cfg(feature = "btree")]
        match schema {
            Schema::BTree(schema) => schema.cast_into(),
            #[cfg(feature = "table")]
            Schema::Table(schema) => schema.cast_into(),
            #[cfg(feature = "tensor")]
            Schema::Dense(schema) => schema.cast_into(),
            #[cfg(feature = "tensor")]
            Schema::Sparse(schema) => schema.cast_into(),
        }

        #[cfg(not(feature = "btree"))]
        Value::default()
    }
}

impl fmt::Debug for Schema {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        #[cfg(feature = "btree")]
        match self {
            Self::BTree(schema) => fmt::Debug::fmt(schema, f),
            #[cfg(feature = "table")]
            Self::Table(schema) => fmt::Debug::fmt(schema, f),
            #[cfg(feature = "tensor")]
            Self::Dense(schema) => fmt::Debug::fmt(schema, f),
            #[cfg(feature = "tensor")]
            Self::Sparse(schema) => fmt::Debug::fmt(schema, f),
        }

        #[cfg(not(feature = "btree"))]
        f.write_str("mock collection schema")
    }
}
