use tcgeneric::{label, Label, PathSegment};

use crate::collection::{Collection, CollectionType};

use super::{Handler, Route};

mod btree;
mod table;

#[cfg(feature = "tensor")]
mod tensor;

pub const PREFIX: Label = label("collection");

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

pub struct Static;

impl Route for Static {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.is_empty() {
            return None;
        }

        #[cfg(feature = "tensor")]
        if path[0] == tensor::PREFIX {
            return tensor::Static.route(&path[1..]);
        }

        None
    }
}
