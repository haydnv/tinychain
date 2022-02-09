use std::convert::TryInto;

use futures::TryFutureExt;
use safecast::{CastInto, TryCastInto};

use tc_btree::BTreeInstance;
use tc_error::TCError;
use tc_table::TableInstance;
use tc_transact::Transaction;
use tc_value::Value;
use tcgeneric::{PathSegment, Tuple};

use crate::collection::{Collection, CollectionMap, CollectionType};
use crate::route::{GetHandler, PutHandler};
use crate::state::State;

use super::{Handler, Route};

mod btree;
mod table;

#[cfg(feature = "tensor")]
mod tensor;

impl Route for CollectionType {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        match self {
            Self::BTree(btt) => btt.route(path),
            Self::Map => None,
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

                    Collection::Map(_map) => unimplemented!(),

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

struct CollectionMapHandler {
    collection: CollectionMap,
}

impl<'a> Handler<'a> for CollectionMapHandler {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if key.is_none() {
                    return Ok(State::Collection(self.collection.clone().into()));
                }

                let id =
                    key.try_cast_into(|v| TCError::bad_request("invalid Id for CollectionMap", v))?;

                self.collection
                    .get(*txn.id(), &id)
                    .map_ok(State::from)
                    .await
            })
        }))
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key, state| {
            Box::pin(async move {
                let id =
                    key.try_cast_into(|v| TCError::bad_request("invalid Id for CollectionMap", v))?;

                let collection = state.try_into()?;
                self.collection.put(*txn.id(), id, collection).await
            })
        }))
    }
}

impl From<CollectionMap> for CollectionMapHandler {
    fn from(collection: CollectionMap) -> Self {
        Self { collection }
    }
}

impl Route for CollectionMap {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.is_empty() {
            Some(Box::new(CollectionMapHandler::from(self.clone())))
        } else {
            unimplemented!()
        }
    }
}

impl Route for Collection {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        let child_handler = match self {
            Self::BTree(btree) => btree.route(path),
            Self::Map(map) => map.route(path),
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
