use futures::TryFutureExt;
use safecast::{Match, TryCastInto};

use tc_btree::{BTreeInstance, Range, RangeExt};
use tc_error::TCError;
use tc_transact::Transaction;
use tc_value::Value;
use tcgeneric::PathSegment;

use crate::collection::{BTree, Collection};
use crate::route::{GetHandler, Handler, PutHandler, Route};
use crate::state::State;

struct CountHandler<'a> {
    btree: &'a BTree,
}

impl<'a> Handler<'a> for CountHandler<'a> {
    fn get(self: Box<Self>) -> Option<GetHandler<'a>> {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if key.is_some() {
                    return Err(TCError::bad_request(
                        "BTree::count does not accept a key (call BTree::slice first)",
                        key,
                    ));
                }

                self.btree.count(*txn.id()).map_ok(State::from).await
            })
        }))
    }
}

impl<'a> From<&'a BTree> for CountHandler<'a> {
    fn from(btree: &'a BTree) -> Self {
        Self { btree }
    }
}

struct InsertHandler<'a> {
    btree: &'a BTree,
}

impl<'a> Handler<'a> for InsertHandler<'a> {
    fn put(self: Box<Self>) -> Option<PutHandler<'a>> {
        Some(Box::new(|txn, key, value| {
            Box::pin(async move {
                if key.is_some() {
                    return Err(TCError::bad_request(
                        "BTree::insert does not support an explicit key",
                        key,
                    ));
                }

                let key = value.try_cast_into(|v| TCError::bad_request("invalid BTree key", v))?;
                self.btree.insert(*txn.id(), key).await
            })
        }))
    }
}

impl<'a> From<&'a BTree> for InsertHandler<'a> {
    fn from(btree: &'a BTree) -> Self {
        Self { btree }
    }
}

struct SliceHandler {
    btree: BTree,
}

impl<'a> Handler<'a> for SliceHandler {
    fn get(self: Box<Self>) -> Option<GetHandler<'a>> {
        Some(Box::new(|_txn, range| {
            Box::pin(async move {
                let invalid_range =
                    |v: &Value| TCError::bad_request("invalid range for BTree slice", v);

                let (range, reverse): (Value, bool) = if range.matches::<(Value, bool)>() {
                    range.try_cast_into(invalid_range)?
                } else {
                    (range.try_cast_into(invalid_range)?, false)
                };

                let range = Range::try_cast_from(range)?;
                let slice = self.btree.slice(range, reverse);

                Ok(Collection::BTree(slice).into())
            })
        }))
    }
}

impl From<BTree> for SliceHandler {
    fn from(btree: BTree) -> Self {
        Self { btree }
    }
}

impl Route for BTree {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.is_empty() {
            Some(Box::new(SliceHandler::from(self.clone())))
        } else if path.len() == 1 {
            match path[0].as_str() {
                "count" => Some(Box::new(CountHandler::from(self))),
                "insert" => Some(Box::new(InsertHandler::from(self))),
                _ => None,
            }
        } else {
            None
        }
    }
}
