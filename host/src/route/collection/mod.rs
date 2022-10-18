use safecast::CastInto;

use tc_btree::BTreeInstance;
use tc_table::TableInstance;
use tc_value::Value;
use tcgeneric::{PathSegment, Tuple};

use crate::collection::{Collection, CollectionBase, CollectionType};
use crate::route::GetHandler;

use super::{Handler, Route};

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

struct SchemaHandler<'a> {
    collection: &'a Collection,
}

impl<'a> Handler<'a> for SchemaHandler<'a> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                key.expect_none()?;

                let schema: Value = match self.collection {
                    Collection::BTree(btree) => btree
                        .schema()
                        .to_vec()
                        .into_iter()
                        .collect::<Tuple<Value>>()
                        .into(),

                    Collection::Table(table) => table.schema().clone().cast_into(),

                    #[cfg(feature = "tensor")]
                    Collection::Tensor(tensor) => tensor.schema().clone().cast_into(),
                };

                Ok(schema.into())
            })
        }))
    }
}

impl<'a> From<&'a Collection> for SchemaHandler<'a> {
    fn from(collection: &'a Collection) -> Self {
        Self { collection }
    }
}

impl Route for Collection {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        let child_handler = match self {
            Self::BTree(btree) => btree.route(path),
            Self::Table(table) => table.route(path),
            #[cfg(feature = "tensor")]
            Self::Tensor(tensor) => tensor.route(path),
        };

        if child_handler.is_some() {
            return child_handler;
        }

        if path.len() == 1 {
            match path[0].as_str() {
                "schema" => Some(Box::new(SchemaHandler::from(self))),
                _ => None,
            }
        } else {
            None
        }
    }
}

impl Route for CollectionBase {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        match self {
            Self::BTree(btree) => btree.route(path),
            Self::Table(table) => table.route(path),
            #[cfg(feature = "tensor")]
            Self::Dense(dense) => dense.route(path),
            #[cfg(feature = "tensor")]
            Self::Sparse(sparse) => sparse.route(path),
        }
    }
}

pub(super) struct Static;

impl Route for Static {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.is_empty() {
            return None;
        }

        match path[0].as_str() {
            "btree" => btree::Static.route(&path[1..]),
            "table" => table::Static.route(&path[1..]),
            #[cfg(feature = "tensor")]
            "tensor" => tensor::Static.route(&path[1..]),
            _ => None,
        }
    }
}
