use std::iter::FromIterator;

use futures::{TryFutureExt, TryStreamExt};
use safecast::{Match, TryCastFrom, TryCastInto};

use tc_btree::{BTreeInstance, BTreeType, Range};
use tc_error::*;
use tc_transact::fs::Dir;
use tc_transact::Transaction;
use tc_value::Value;
use tcgeneric::{label, Map, PathSegment};

use crate::collection::{BTree, BTreeFile, Collection};
use crate::route::{DeleteHandler, GetHandler, Handler, PostHandler, PutHandler, Route};
use crate::scalar::Scalar;
use crate::state::State;

struct CreateHandler;

impl<'a> Handler<'a> for CreateHandler {
    fn get(self: Box<Self>) -> Option<GetHandler<'a>> {
        Some(Box::new(|txn, value| {
            Box::pin(async move {
                let schema = tc_btree::RowSchema::try_cast_from(value, |v| {
                    TCError::bad_request("invalid BTree schema", v)
                })?;

                let file = txn
                    .context()
                    .create_file_tmp(*txn.id(), BTreeType::default())
                    .await?;

                BTreeFile::create(file, schema, *txn.id())
                    .map_ok(Collection::from)
                    .map_ok(State::from)
                    .await
            })
        }))
    }
}

impl Route for BTreeType {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.is_empty() && self == &Self::File {
            Some(Box::new(CreateHandler))
        } else {
            None
        }
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
                let range = cast_into_range(Scalar::Value(range))?;
                let slice = self.btree.clone().slice(range, false)?;
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

                if let State::Collection(Collection::BTree(value)) = value {
                    let keys = value.keys(*txn.id()).await?;
                    self.btree.try_insert_from(*txn.id(), keys).await
                } else if value.matches::<Value>() {
                    let value = Value::opt_cast_from(value).unwrap();
                    let value =
                        value.try_cast_into(|v| TCError::bad_request("invalid BTree key", v))?;

                    self.btree.insert(*txn.id(), value).await
                } else {
                    Err(TCError::bad_request("invalid BTree key", value))
                }
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

struct FirstHandler<'a, T> {
    btree: &'a T,
}

impl<'a, T: BTreeInstance> Handler<'a> for FirstHandler<'a, T> {
    fn get(self: Box<Self>) -> Option<GetHandler<'a>> {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if key.is_some() {
                    return Err(TCError::bad_request(
                        "BTree::first does not accept a key",
                        key,
                    ));
                }

                let mut keys = self.btree.clone().keys(*txn.id()).await?;
                if let Some(values) = keys.try_next().await? {
                    let names = self.btree.schema().iter().map(|col| col.name()).cloned();
                    Ok(Map::from_iter(names.zip(values.into_iter().map(State::from))).into())
                } else {
                    Err(TCError::not_found("this BTree is empty"))
                }
            })
        }))
    }
}

impl<'a, T> From<&'a T> for FirstHandler<'a, T> {
    fn from(btree: &'a T) -> Self {
        Self { btree }
    }
}

struct ReverseHandler<T> {
    btree: T,
}

impl<'a, T: BTreeInstance + 'a> Handler<'a> for ReverseHandler<T>
where
    BTree: From<<T as BTreeInstance>::Slice>,
{
    fn get(self: Box<Self>) -> Option<GetHandler<'a>> {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                if key.is_some() {
                    return Err(TCError::bad_request(
                        "BTree::reverse does not accept a key",
                        key,
                    ));
                }

                let reversed = self.btree.slice(Range::default(), true)?;
                Ok(Collection::from(BTree::from(reversed)).into())
            })
        }))
    }
}

impl<T> From<T> for ReverseHandler<T> {
    fn from(btree: T) -> Self {
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
            "first" => Some(Box::new(FirstHandler::from(btree))),
            "reverse" => Some(Box::new(ReverseHandler::from(btree.clone()))),
            _ => None,
        }
    } else {
        None
    }
}

#[inline]
fn cast_into_range(scalar: Scalar) -> TCResult<Range> {
    if scalar.is_none() {
        return Ok(Range::default());
    } else if let Scalar::Value(value) = scalar {
        return match value {
            Value::Tuple(prefix) => Ok(Range::with_prefix(prefix.into_inner())),
            value => Ok(Range::with_prefix(vec![value])),
        };
    };

    let mut prefix: Vec<Value> =
        scalar.try_cast_into(|s| TCError::bad_request("invalid BTree range", s))?;

    if !prefix.is_empty() && tc_value::Range::can_cast_from(prefix.last().unwrap()) {
        let range = tc_value::Range::opt_cast_from(prefix.pop().unwrap()).unwrap();
        Ok((prefix, range.start.into(), range.end.into()).into())
    } else {
        Ok(Range::with_prefix(prefix))
    }
}
