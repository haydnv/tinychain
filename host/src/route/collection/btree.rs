use futures::TryFutureExt;
use safecast::{Match, TryCastFrom, TryCastInto};

use tc_btree::{BTreeInstance, Range};
use tc_error::*;
use tc_transact::Transaction;
use tc_value::Value;
use tcgeneric::{label, PathSegment, Tuple};

use crate::collection::{BTree, Collection};
use crate::route::{DeleteHandler, GetHandler, Handler, PostHandler, PutHandler, Route};
use crate::scalar::Scalar;
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

struct SliceHandler<'a> {
    btree: &'a BTree,
}

impl<'a> Handler<'a> for SliceHandler<'a> {
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

                let range = cast_into_range(Scalar::Value(range))?;
                let slice = self.btree.clone().slice(range, reverse);

                Ok(Collection::BTree(slice).into())
            })
        }))
    }

    fn post(self: Box<Self>) -> Option<PostHandler<'a>> {
        Some(Box::new(|_txn, mut params| {
            Box::pin(async move {
                let reverse = params.or_default(&label("reverse").into())?;
                let range = params.or_default(&label("range").into())?;
                let range = cast_into_range(range)?;
                let slice = self.btree.clone().slice(range, reverse);
                Ok(Collection::BTree(slice).into())
            })
        }))
    }

    fn delete(self: Box<Self>) -> Option<DeleteHandler<'a>> {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if key.is_some() {
                    return Err(TCError::bad_request(
                        "BTree::delete does not accept a key (call BTree::slice first)",
                        key,
                    ));
                }

                self.btree.delete(*txn.id()).await
            })
        }))
    }
}

impl<'a> From<&'a BTree> for SliceHandler<'a> {
    fn from(btree: &'a BTree) -> Self {
        Self { btree }
    }
}

impl Route for BTree {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.is_empty() {
            Some(Box::new(SliceHandler::from(self)))
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

fn cast_into_range(scalar: Scalar) -> TCResult<Range> {
    if scalar.is_none() {
        return Ok(Range::default());
    } else if let Scalar::Value(value) = scalar {
        return match value {
            Value::Tuple(prefix) => Ok(Range::with_prefix(prefix.into_inner())),
            value => Ok(Range::with_prefix(vec![value])),
        };
    };

    let mut prefix: Tuple<Scalar> =
        scalar.try_cast_into(|s| TCError::bad_request("invalid BTree range", s))?;

    if !prefix.is_empty() && tc_value::Range::can_cast_from(prefix.last().unwrap()) {
        let range = tc_value::Range::opt_cast_from(prefix.pop().unwrap()).unwrap();
        let prefix =
            prefix.try_cast_into(|v| TCError::bad_request("invalid BTree range prefix", v))?;

        Ok((prefix, range.start.into(), range.end.into()).into())
    } else {
        let prefix =
            prefix.try_cast_into(|v| TCError::bad_request("invalid BTree range prefix", v))?;

        Ok(Range::with_prefix(prefix))
    }
}
