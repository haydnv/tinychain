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
