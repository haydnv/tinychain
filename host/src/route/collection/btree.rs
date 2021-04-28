use futures::TryFutureExt;
use safecast::{Match, TryCastFrom, TryCastInto};

use tc_btree::{BTreeInstance, Range};
use tc_error::*;
use tc_transact::Transaction;
use tc_value::Value;
use tcgeneric::{label, PathSegment, Tuple};

use crate::collection::{BTree, BTreeFile, Collection};
use crate::route::{DeleteHandler, GetHandler, Handler, PostHandler, PutHandler, Route};
use crate::scalar::Scalar;
use crate::state::State;

struct CountHandler<'a, T> {
    btree: &'a T,
}

impl<'a, T: BTreeInstance> Handler<'a> for CountHandler<'a, T> {
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

impl<'a, T> From<&'a T> for CountHandler<'a, T> {
    fn from(btree: &'a T) -> Self {
        Self { btree }
    }
}

struct BTreeHandler<'a, T> {
    btree: &'a T,
}

impl<'a, T: BTreeInstance> Handler<'a> for BTreeHandler<'a, T>
where
    BTree: From<T::Slice>,
{
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
                let slice = self.btree.clone().slice(range, reverse)?;

                Ok(Collection::BTree(slice.into()).into())
            })
        }))
    }

    fn put(self: Box<Self>) -> Option<PutHandler<'a>> {
        Some(Box::new(|txn, key, value| {
            Box::pin(async move {
                if key.is_some() {
                    return Err(TCError::bad_request(
                        "BTree::insert does not support an explicit key",
                        key,
                    ));
                }

                let key =
                    Value::try_cast_from(value, |s| TCError::bad_request("invalid BTree key", s))?;
                let key = key.try_cast_into(|v| TCError::bad_request("invalid BTree key", v))?;

                self.btree.insert(*txn.id(), key).await
            })
        }))
    }

    fn post(self: Box<Self>) -> Option<PostHandler<'a>> {
        Some(Box::new(|_txn, mut params| {
            Box::pin(async move {
                let reverse = params.or_default(&label("reverse").into())?;
                let range = params.or_default(&label("range").into())?;
                let range = cast_into_range(range)?;
                let slice = self.btree.clone().slice(range, reverse)?;
                Ok(Collection::BTree(slice.into()).into())
            })
        }))
    }

    fn delete(self: Box<Self>) -> Option<DeleteHandler<'a>> {
        Some(Box::new(|txn, range| {
            Box::pin(async move {
                let range = cast_into_range(Scalar::Value(range))?;
                self.btree
                    .clone()
                    .slice(range, false)?
                    .delete(*txn.id())
                    .await
            })
        }))
    }
}

impl<'a, T> From<&'a T> for BTreeHandler<'a, T> {
    fn from(btree: &'a T) -> Self {
        Self { btree }
    }
}

impl Route for BTree {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        route(self, path)
    }
}

impl Route for BTreeFile {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        route(self, path)
    }
}

#[inline]
fn route<'a, T: BTreeInstance>(
    btree: &'a T,
    path: &'a [PathSegment],
) -> Option<Box<dyn Handler<'a> + 'a>>
where
    BTree: From<T::Slice>,
{
    if path.is_empty() {
        Some(Box::new(BTreeHandler::from(btree)))
    } else if path.len() == 1 {
        match path[0].as_str() {
            "count" => Some(Box::new(CountHandler::from(btree))),
            _ => None,
        }
    } else {
        None
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
