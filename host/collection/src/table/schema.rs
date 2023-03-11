use std::fmt;

use async_hash::{Digest, Hash, Output};
use destream::en;

use tcgeneric::Id;

use crate::btree::Schema as BTreeSchema;

#[derive(Clone, Eq, PartialEq)]
pub struct Schema {
    primary: BTreeSchema,
    indices: Vec<(Id, BTreeSchema)>,
}

impl Schema {
    /// Construct a new `Table` schema.
    pub fn new<I: IntoIterator<Item = (Id, BTreeSchema)>>(
        primary: BTreeSchema,
        indices: I,
    ) -> Self {
        Self {
            primary,
            indices: indices.into_iter().collect(),
        }
    }
}

impl<'a, D: Digest> Hash<D> for &'a Schema {
    fn hash(self) -> Output<D> {
        Hash::<D>::hash((&self.primary, &self.indices))
    }
}

impl<'en> en::IntoStream<'en> for Schema {
    fn into_stream<E: en::Encoder<'en>>(self, _encoder: E) -> Result<E::Ok, E::Error> {
        todo!()
    }
}

impl<'en> en::ToStream<'en> for Schema {
    fn to_stream<E: en::Encoder<'en>>(&'en self, _encoder: E) -> Result<E::Ok, E::Error> {
        todo!()
    }
}

impl fmt::Debug for Schema {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.primary, f)
    }
}
