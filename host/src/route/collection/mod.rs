use safecast::CastInto;

use tc_collection::btree::BTreeInstance;
use tc_collection::table::TableInstance;
use tc_collection::tensor::TensorInstance;
use tc_collection::{CollectionType, TensorBase};
use tc_value::Value;
use tcgeneric::{PathSegment, TCPath};

use crate::collection::{Collection, CollectionBase};
use crate::route::GetHandler;

use super::{Handler, Route};

mod btree;
mod table;
// mod tensor;

impl Route for CollectionType {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        match self {
            Self::BTree(btt) => btt.route(path),
            Self::Table(tt) => tt.route(path),
            Self::Tensor(tt) => todo!("route {tt:?}"),
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
                    Collection::BTree(btree) => btree.schema().clone().cast_into(),

                    Collection::Table(table) => table.schema().clone().cast_into(),

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
        log::debug!("Collection::route {}", TCPath::from(path));

        let child_handler: Option<Box<dyn Handler<'a> + 'a>> = match self {
            Self::BTree(btree) => btree.route(path),
            Self::Table(table) => table.route(path),
            Self::Tensor(tensor) => todo!("route {tensor:?}"),
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
            Self::Tensor(tensor) => match tensor {
                TensorBase::Dense(dense) => todo!("route {dense:?}"),
                TensorBase::Sparse(sparse) => todo!("route {sparse:?}"),
            },
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
            "tensor" => todo!("route tensor static"),
            _ => None,
        }
    }
}
