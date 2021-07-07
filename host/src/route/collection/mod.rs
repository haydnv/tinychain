use tcgeneric::PathSegment;

use crate::collection::{Collection, CollectionType};

use super::{Handler, Route};

#[cfg(feature = "tensor")]
pub use tensor::{EinsumHandler, EINSUM};

mod btree;
mod table;

#[cfg(feature = "tensor")]
mod tensor;

impl Route for CollectionType {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        match self {
            Self::BTree(btt) => btt.route(path),
            Self::Table(tt) => tt.route(path),
            #[cfg(feature = "tensor")]
            Self::Tensor(tt) => tt.route(path),
        }
    }
}

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
