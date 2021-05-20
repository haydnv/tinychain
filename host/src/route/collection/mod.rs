use tcgeneric::PathSegment;

use crate::collection::Collection;

use super::{Handler, Route};

mod btree;
mod table;

#[cfg(feature = "tensor")]
mod tensor;

impl Route for Collection {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        match self {
            Self::BTree(btree) => btree.route(path),
            Self::Table(table) => table.route(path),

            #[cfg(feature = "tensor")]
            Self::Tensor(tensor) => tensor.route(path),
        }
    }
}
