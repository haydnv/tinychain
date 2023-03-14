use std::iter::FromIterator;
use std::ops::Bound;

use destream::de::Error;
use futures::{future, StreamExt, TryFutureExt, TryStreamExt};
use safecast::{Match, TryCastFrom, TryCastInto};

use tc_collection::btree::{BTreeInstance, BTreeType, BTreeWrite, Column, Range, Schema};
use tc_error::*;
use tc_transact::fs::{Dir, Persist};
use tc_transact::Transaction;
use tc_value::Value;
use tcgeneric::{label, Map, PathSegment, Tuple};

use crate::collection::{BTree, BTreeFile, Collection};
use crate::route::{DeleteHandler, GetHandler, Handler, PostHandler, PutHandler, Route};
use crate::scalar::Scalar;
use crate::state::State;
use crate::stream::{Source, TCStream};

impl Route for BTreeType {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if self == &Self::default() {
            Static.route(path)
        } else {
            None
        }
    }
}

struct CopyHandler;

impl<'a> Handler<'a> for CopyHandler {
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, mut params| {
            Box::pin(async move {
                let schema: Value = params.require(&label("schema").into())?;
                let schema = cast_into_schema(schema)?;

                let source: TCStream = params.require(&label("source").into())?;
                params.expect_empty()?;

                let txn_id = *txn.id();

                let store = {
                    let mut dir = txn.context().write().await;
                    let (_, cache) = dir.create_dir_unique()?;
                    Dir::load(*txn.id(), cache).await?
                };

                let keys = source.into_stream(txn.clone()).await?;
                let keys = keys
                    .map(|r| {
                        r.and_then(|state| {
                            Value::try_cast_from(state, |s| TCError::unexpected(s, "a BTree key"))
                        })
                    })
                    .map(|r| {
                        r.and_then(|value| {
                            value.try_cast_into(|v| TCError::unexpected(v, "a BTree key"))
                        })
                    });

                let btree = BTreeFile::create(txn_id, schema, store).await?;
                btree.try_insert_from(*txn.id(), keys).await?;

                Ok(State::Collection(btree.into()))
            })
        }))
    }
}

struct CreateHandler;

impl<'a> Handler<'a> for CreateHandler {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, value| {
            Box::pin(async move {
                let schema = cast_into_schema(value)?;
                let store = {
                    let mut dir = txn.context().write().await;
                    let (_, cache) = dir.create_dir_unique()?;
                    Dir::load(*txn.id(), cache).await?
                };

                BTreeFile::create(*txn.id(), schema, store)
                    .map_ok(Collection::from)
                    .map_ok(State::from)
                    .await
            })
        }))
    }
}

struct BTreeHandler<'a, T> {
    btree: &'a T,
}

impl<'a, T: BTreeInstance + BTreeWrite> Handler<'a> for BTreeHandler<'a, T>
where
    BTree: From<T::Slice>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, range| {
            Box::pin(async move {
                let range = cast_into_range(Scalar::Value(range))?;
                let slice = self.btree.clone().slice(range, false)?;
                Ok(Collection::BTree(slice.into()).into())
            })
        }))
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key, value| {
            Box::pin(async move {
                if key.is_some() {
                    return Err(bad_request!(
                        "BTree::insert does not support an explicit key {}",
                        key,
                    ));
                }

                if let State::Collection(Collection::BTree(value)) = value {
                    let keys = value.keys(*txn.id()).await?;
                    self.btree.try_insert_from(*txn.id(), keys).await
                } else if value.matches::<Value>() {
                    let value = Value::opt_cast_from(value).expect("value");
                    let value = value.try_cast_into(|v| TCError::unexpected(v, "a BTree key"))?;

                    self.btree.insert(*txn.id(), value).await
                } else {
                    Err(TCError::unexpected(value, "a BTree key"))
                }
            })
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
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

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, range| {
            Box::pin(async move {
                let range = cast_into_range(Scalar::Value(range))?;
                self.btree.clone().delete(*txn.id(), range).await
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
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if key.is_some() {
                    return Err(bad_request!(
                        "BTree::count does not accept a key {} (call BTree::slice first)",
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
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if key.is_some() {
                    return Err(bad_request!("BTree::first does not accept a key {}", key));
                }

                let mut keys = self.btree.clone().keys(*txn.id()).await?;
                if let Some(values) = keys.try_next().await? {
                    let names = self.btree.schema().iter().map(|col| col.name()).cloned();
                    Ok(
                        Map::<State>::from_iter(names.zip(values.into_iter().map(State::from)))
                            .into(),
                    )
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

impl<'a, T> Handler<'a> for ReverseHandler<T>
where
    T: BTreeInstance + 'a,
    BTree: From<<T as BTreeInstance>::Slice>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                if key.is_some() {
                    return Err(bad_request!("BTree::reverse does not accept a key {}", key));
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

struct StreamHandler<T> {
    btree: T,
}

impl<'a, T> Handler<'a> for StreamHandler<T>
where
    T: BTreeInstance + 'a,
    BTree: From<T>,
    BTree: From<T::Slice>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                if key.is_none() {
                    Ok(TCStream::from(BTree::from(self.btree)).into())
                } else {
                    let range = cast_into_range(Scalar::Value(key))?;
                    let slice = self.btree.slice(range, false)?;
                    Ok(TCStream::from(BTree::from(slice)).into())
                }
            })
        }))
    }
}

impl<T> From<T> for StreamHandler<T> {
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
fn route<'a, T>(btree: &'a T, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>>
where
    T: BTreeInstance + BTreeWrite + 'a,
    BTree: From<T>,
    BTree: From<T::Slice>,
{
    if path.is_empty() {
        Some(Box::new(BTreeHandler::from(btree)))
    } else if path.len() == 1 {
        match path[0].as_str() {
            "count" => Some(Box::new(CountHandler::from(btree))),
            "first" => Some(Box::new(FirstHandler::from(btree))),
            "keys" => Some(Box::new(StreamHandler::from(btree.clone()))),
            "reverse" => Some(Box::new(ReverseHandler::from(btree.clone()))),
            _ => None,
        }
    } else {
        None
    }
}

pub struct Static;

impl Route for Static {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.is_empty() {
            Some(Box::new(CreateHandler))
        } else if path == &["copy_from"] {
            Some(Box::new(CopyHandler))
        } else {
            None
        }
    }
}

#[inline]
fn cast_into_range(scalar: Scalar) -> TCResult<Range> {
    let mut range = if scalar.is_none() {
        return Ok(Range::default());
    } else {
        Tuple::<Scalar>::try_cast_from(scalar, |s| bad_request!("invalid BTree selector: {:?}", s))?
    };

    if range
        .last()
        .expect("bounds")
        .matches::<(Bound<Value>, Bound<Value>)>()
    {
        let bounds: (Bound<Value>, Bound<Value>) = range
            .pop()
            .expect("bounds")
            .opt_cast_into()
            .expect("bounds");

        let prefix = range
            .into_iter()
            .map(|v| {
                Value::try_cast_from(v, |v| {
                    bad_request!("invalid value for BTree range: {:?}", v)
                })
            })
            .collect::<TCResult<_>>()?;

        Ok(Range::with_bounds(prefix, bounds))
    } else {
        let prefix = range
            .into_iter()
            .map(|v| {
                Value::try_cast_from(v, |v| {
                    bad_request!("invalid value for BTree range: {:?}", v)
                })
            })
            .collect::<TCResult<_>>()?;

        Ok(Range::from_prefix(prefix))
    }
}

#[inline]
fn cast_into_schema(schema: Value) -> TCResult<Schema> {
    let columns = if let Value::Tuple(columns) = schema {
        columns
            .into_iter()
            .map(|col| {
                Column::try_cast_from(col, |v| bad_request!("invalid column schema: {:?}", v))
            })
            .collect::<TCResult<Vec<Column>>>()
    } else {
        Err(bad_request!("invalid BTree schema: {:?}", schema))
    }?;

    Schema::new(columns)
}
