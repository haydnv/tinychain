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
use tcgeneric::NativeClass;
use tcgeneric::TCPathBuf;

#[cfg(feature = "btree")]
use crate::btree::{BTreeSchema, BTreeType};
#[cfg(feature = "table")]
use crate::table::{TableSchema, TableType};
#[cfg(feature = "tensor")]
use crate::tensor::{Schema as TensorSchema, TensorType};
use crate::CollectionType;

/// The schema of a `Collection`.
#[derive(Clone, Eq, PartialEq)]
pub enum Schema {
    Null,
    #[cfg(feature = "btree")]
    BTree(BTreeSchema),
    #[cfg(feature = "table")]
    Table(TableSchema),
    #[cfg(feature = "tensor")]
    Dense(TensorSchema),
    #[cfg(feature = "tensor")]
    Sparse(TensorSchema),
}

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
        match self {
            Schema::Null => tc_transact::hash::default_hash::<D>(),
            #[cfg(feature = "btree")]
            Schema::BTree(schema) => Hash::<D>::hash((BTreeType::default().path(), schema)),
            #[cfg(feature = "table")]
            Schema::Table(schema) => Hash::<D>::hash((TableType::default().path(), schema)),
            #[cfg(feature = "tensor")]
            Schema::Dense(schema) => Hash::<D>::hash((TensorType::Dense.path(), schema)),
            #[cfg(feature = "tensor")]
            Schema::Sparse(schema) => Hash::<D>::hash((TensorType::Sparse.path(), schema)),
        }
    }
}

impl TryFrom<(TCPathBuf, Value)> for Schema {
    type Error = TCError;

    fn try_from(value: (TCPathBuf, Value)) -> Result<Self, Self::Error> {
        #[allow(unused_variables)]
        let (classpath, schema) = value;

        let class = CollectionType::from_path(&classpath)
            .ok_or_else(|| bad_request!("invalid collection type: {}", classpath))?;

        match class {
            CollectionType::Null => Ok(Schema::Null),
            #[cfg(feature = "btree")]
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
    }
}

impl<'en> en::IntoStream<'en> for Schema {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        use destream::en::EncodeMap;

        let mut map = encoder.encode_map(Some(1))?;

        match self {
            Self::Null => {
                map.encode_entry(CollectionType::Null.path(), ())?;
            }

            #[cfg(feature = "btree")]
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
    fn cast_from(schema: Schema) -> Self {
        match schema {
            Schema::Null => Value::None,
            #[cfg(feature = "btree")]
            Schema::BTree(schema) => schema.cast_into(),
            #[cfg(feature = "table")]
            Schema::Table(schema) => schema.cast_into(),
            #[cfg(feature = "tensor")]
            Schema::Dense(schema) => schema.cast_into(),
            #[cfg(feature = "tensor")]
            Schema::Sparse(schema) => schema.cast_into(),
        }
    }
}

impl fmt::Debug for Schema {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Schema::Null => f.write_str("null collection schema"),
            #[cfg(feature = "btree")]
            Self::BTree(schema) => fmt::Debug::fmt(schema, f),
            #[cfg(feature = "table")]
            Self::Table(schema) => fmt::Debug::fmt(schema, f),
            #[cfg(feature = "tensor")]
            Self::Dense(schema) => fmt::Debug::fmt(schema, f),
            #[cfg(feature = "tensor")]
            Self::Sparse(schema) => fmt::Debug::fmt(schema, f),
        }
    }
}
