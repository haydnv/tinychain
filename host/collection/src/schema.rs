use std::fmt;

use async_hash::{Digest, Hash, Output};
use destream::en;
use safecast::as_type;

use tcgeneric::NativeClass;

use crate::btree::{BTreeSchema, BTreeType};
use crate::table::{TableSchema, TableType};
use crate::tensor::{Schema as TensorSchema, TensorType};

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
